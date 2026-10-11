"""Chunked training for VN v3: run K steps, save state, resume next call.

The sandbox kills any process that outlives one tool call, so long runs have to
be stitched from chunks whose state is written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from conceptwave3.predict import Predictor, Adam
from conceptwave3.field3 import Field3
from conceptwave3.train3 import Codec3, mse_term, evaluate

LOG2 = float(np.log(2.0))
STATE = "v3_state.npz"
LOG = "v3_progress.json"


def load_data():
    txt = open("tinyshakespeare.txt", encoding="utf-8").read()
    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], np.int64)
    cut = int(len(ids) * 0.9)
    return ids[:cut], ids[cut:], chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dc", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--vlen", type=int, default=1200)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    tr, va, chars = load_data()
    V = len(chars)
    field = Field3(n=a.n, dc=a.dc, seed=0)
    F = Predictor(dc=a.dc, hidden=64, seed=0)
    codec = Codec3(V, a.dc, seed=0)
    adam = Adam(lr=a.lr)
    pos, hist_log = 0, []

    if not a.fresh and os.path.exists(STATE):
        z = np.load(STATE, allow_pickle=True)
        field.concept = z["concept"]
        field.G = z["G"]
        F.W1, F.b1, F.W2, F.b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        codec.emb, codec.bias = z["emb"], z["bias"]
        adam.m = {k: z["m_" + k] for k in ("concept", "emb", "bias", "G",
                                           "W1", "b1", "W2", "b2")}
        adam.v = {k: z["v_" + k] for k in ("concept", "emb", "bias", "G",
                                           "W1", "b1", "W2", "b2")}
        adam.t = int(z["t"])
        pos = int(z["pos"])
        field.renorm()
        codec.renorm()
        if os.path.exists(LOG):
            hist_log = json.load(open(LOG))["log"]
        print(f"resumed from pos={pos}", flush=True)

    losses = []
    t0 = time.time()
    for k in range(a.steps):
        t = (pos + k) % (len(tr) - 1)
        tgt = int(tr[t + 1])
        if k % 1000 == 0:
            field.reset()
        field.seed_from(codec.emb[int(tr[t])])
        field.settle(F, grad=True)
        lg = codec.logits(field)
        loss, hg = codec.head_grads(field, tgt, lg)
        if field._last is not None:
            da, dcx = codec.pooled_bwd(field, hg["h"])
            nxt = field._last[0][2]
            ml, dn = mse_term(field, codec, nxt, tgt, a.lam)
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
        if (k + 1) % 2000 == 0:
            print(f"  +{k+1} bits={np.mean(losses[-2000:])/LOG2:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    pos += a.steps

    field.reset()
    vb, ab = evaluate(field, F, codec, va[:a.vlen])
    sims = {}
    for ch in list("Fae Th"):
        if ch in {c: i for i, c in enumerate(chars)}:
            stoi = {c: i for i, c in enumerate(chars)}
            field.reset()
            field.seed_from(codec.emb[stoi[ch]])
            field.settle(F)
            s, _ = field.concept_sim_top(codec.emb[stoi[ch]], 5)
            sims[ch] = round(float(s), 4)
    hist_log.append({"pos": pos, "val_bits": round(vb, 4),
                     "val_acc": round(ab, 4)})

    np.savez(STATE, concept=field.concept, G=field.G,
             W1=F.W1, b1=F.b1, W2=F.W2, b2=F.b2,
             emb=codec.emb, bias=codec.bias, t=adam.t, pos=pos,
             **{f"m_{k}": v for k, v in adam.m.items()},
             **{f"v_{k}": v for k, v in adam.v.items()})
    json.dump({"log": hist_log, "sims": sims,
               "val_bits": vb, "val_acc": ab,
               "sparsity": field.sparsity()}, open(LOG, "w"), indent=2)
    print(json.dumps({"pos": pos, "val_bits": round(vb, 4),
                      "val_acc": round(ab, 4), "sims": sims,
                      "sparsity": round(field.sparsity(), 4)}, indent=2))


if __name__ == "__main__":
    main()
