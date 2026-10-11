"""Finite-difference validation + training for VN v3 (<=200 lines).

Checks dL/dG (inter-layer matrix), dL/dconcept, dL/dW1 against central
differences, then trains.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from conceptwave3.predict import Predictor, Adam, normalise
from conceptwave3.field3 import Field3
from conceptwave3.train3 import Codec3, mse_term, evaluate

LOG2 = float(np.log(2.0))


def total_loss(field, F, codec, emb_t, tgt, lam):
    field.reset()
    field.seed_from(emb_t)
    field.settle(F, grad=True)
    lg = codec.logits(field)
    l, hg = codec.head_grads(field, tgt, lg)
    nxt = field._last[0][2]
    ml, _ = mse_term(field, codec, nxt, tgt, lam)
    return float(l + ml)


def grads(field, F, codec, emb_t, tgt, lam):
    field.reset()
    field.seed_from(emb_t)
    field.settle(F, grad=True)
    lg = codec.logits(field)
    l, hg = codec.head_grads(field, tgt, lg)
    da, dcx = codec.pooled_bwd(field, hg["h"])
    nxt = field._last[0][2]
    ml, dn = mse_term(field, codec, nxt, tgt, lam)
    bg = field.backward(F, da, dcx, dn)
    return bg, float(l + ml)


def fd_G(field, F, codec, emb_t, tgt, lam, nprobe=4):
    bg, _ = grads(field, F, codec, emb_t, tgt, lam)
    g = bg["G"].astype(np.float64)
    out = []
    flat = np.argsort(-np.abs(g).ravel())[:nprobe]
    for pos in flat:
        i, j = divmod(int(pos), g.shape[1])
        ana = float(g[i, j])
        best = None
        for eps in [1e-3, 1e-2, 5e-2]:
            o = field.G[i, j]
            field.G[i, j] = o + eps
            lp = total_loss(field, F, codec, emb_t, tgt, lam)
            field.G[i, j] = o - eps
            lm = total_loss(field, F, codec, emb_t, tgt, lam)
            field.G[i, j] = o
            num = (lp - lm) / (2 * eps)
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
            if best is None or rel < best["relerr"]:
                best = {"eps": eps, "fd": round(float(num), 8),
                        "ana": round(ana, 8), "relerr": round(float(rel), 6)}
        out.append({"i": i, "j": j, **best})
    return out


def fd_concept(field, F, codec, emb_t, tgt, lam, nprobe=4):
    bg, _ = grads(field, F, codec, emb_t, tgt, lam)
    g = bg["concept"].astype(np.float64)
    out = []
    for pos in np.argsort(-np.abs(g).ravel())[:nprobe]:
        l, i, d = np.unravel_index(int(pos), g.shape)
        ana = float(g[l, i, d])
        best = None
        for eps in [1e-3, 1e-2, 5e-2]:
            o = field.concept[l, i, d]
            field.concept[l, i, d] = o + eps
            field.renorm()
            lp = total_loss(field, F, codec, emb_t, tgt, lam)
            field.concept[l, i, d] = o - eps
            field.renorm()
            lm = total_loss(field, F, codec, emb_t, tgt, lam)
            field.concept[l, i, d] = o
            field.renorm()
            num = (lp - lm) / (2 * eps)
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
            if best is None or rel < best["relerr"]:
                best = {"eps": eps, "fd": round(float(num), 8),
                        "ana": round(ana, 8), "relerr": round(float(rel), 6)}
        out.append({"layer": int(l), "row": int(i), "dim": int(d), **best})
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    dc = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-3
    tag = sys.argv[6] if len(sys.argv) > 6 else "v3"
    out = f"{tag}_probe.json"

    txt = open("tinyshakespeare.txt", encoding="utf-8").read()
    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], np.int64)
    cut = int(len(ids) * 0.9)
    tr, va = ids[:cut], ids[cut:]
    V = len(chars)

    field = Field3(n=n, dc=dc, seed=0)
    F = Predictor(dc=dc, hidden=64, seed=0)
    codec = Codec3(V, dc, seed=0)

    t0 = time.time()
    fdg = fd_G(field, F, codec, codec.emb[int(ids[0])], int(ids[1]), lam)
    fdc = fd_concept(field, F, codec, codec.emb[int(ids[0])], int(ids[1]), lam)
    print(f"FD done {time.time()-t0:.1f}s", flush=True)

    adam = Adam(lr=lr)
    losses, trace = [], []
    vlen = min(int(sys.argv[7]) if len(sys.argv) > 7 else 1200, len(va) - 1)
    b0, a0 = evaluate(field, F, codec, va[:vlen])
    t0 = time.time()
    for t in range(min(steps, len(tr) - 1)):
        tgt = int(tr[t + 1])
        if t % 1000 == 0:
            field.reset()
        field.seed_from(codec.emb[int(tr[t])])
        field.settle(F, grad=True)
        lg = codec.logits(field)
        loss, hg = codec.head_grads(field, tgt, lg)
        if field._last is not None:
            da, dcx = codec.pooled_bwd(field, hg["h"])
            nxt = field._last[0][2]
            ml, dn = mse_term(field, codec, nxt, tgt, lam)
            bg = field.backward(F, da, dcx, dn)
            adam.step({
                "concept": (field.concept, bg["concept"]),
                "emb": (codec.emb, hg["emb"]),
                "bias": (codec.bias, hg["bias"]),
                "G": (field.G, bg["G"]),
                "W1": (F.W1, bg["F"]["W1"]), "b1": (F.b1, bg["F"]["b1"]),
                "W2": (F.W2, bg["F"]["W2"]), "b2": (F.b2, bg["F"]["b2"])})
            field.renorm()
            codec.renorm()
        losses.append(loss)
        if (t + 1) % 2000 == 0:
            tb = float(np.mean(losses[-2000:])) / LOG2
            trace.append({"n": t + 1, "train_bits": round(tb, 4)})
            print(f"{tag} n={t+1} bits={tb:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    field.reset()
    b1, a1 = evaluate(field, F, codec, va[:vlen])

    sims = {}
    for ch in list("Fae Th"):
        if ch in stoi:
            field.reset()
            field.seed_from(codec.emb[stoi[ch]])
            field.settle(F)
            s, lst = field.concept_sim_top(codec.emb[stoi[ch]], 5)
            sims[ch] = s
    allsim = [sims[c] for c in sims]

    params = int(field.concept.size + field.G.size
                 + F.W1.size + F.b1.size + F.W2.size + F.b2.size
                 + codec.emb.size + codec.bias.size)
    res = {
        "tag": tag, "config": {"N": n, "dc": dc, "steps": steps, "lam": lam,
                               "lr": lr, "layers": field.L, "V": V},
        "fd_G": fdg, "fd_G_max_relerr": max(r["relerr"] for r in fdg),
        "fd_concept": fdc,
        "fd_concept_max_relerr": max(r["relerr"] for r in fdc),
        "val_bits_before": round(b0, 4), "val_bits_after": round(b1, 4),
        "val_acc_before": round(a0, 4), "val_acc_after": round(a1, 4),
        "train_trace": trace,
        "concept_sim_top5_by_char": {k: round(float(v), 4)
                                     for k, v in sims.items()},
        "concept_sim_top5_mean": round(float(np.mean(allsim)), 4),
        "sparsity": round(field.sparsity(), 4),
        "params": params,
        "sec": round(time.time() - t0, 1),
    }
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("fd_G", "fd_concept", "train_trace")},
                     indent=2))


if __name__ == "__main__":
    main()
