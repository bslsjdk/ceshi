"""Probe: does `concept` actually receive gradient and actually move?

deepseek's first requirement for v3:

    "concept_i is a parameter, trained through the prediction loss.
     CHECK: does concept have gradient after training?"

Three independent measurements:
  A. analytic gradient norm from field.backward()  (is it non-zero?)
  B. finite-difference check of that gradient       (is it correct?)
  C. actual parameter drift after N adam steps      (does it move?)
  D. cosine drift: how far each concept vector rotated from init
Also reports concept_sim_top5 and why it may be negative.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from conceptwave2.predict import Predictor, Adam, normalise
from conceptwave2.field2 import Field2
from conceptwave2.train2 import Codec2, mse_term

LOG2 = float(np.log(2.0))


def fd_check(field, F, codec, ids, lam=0.1, eps=1e-2, nprobe=6, seed=0):
    """Compare analytic dL/dconcept against central finite differences."""
    rng = np.random.default_rng(seed)
    t = 0
    field.reset()
    field.seed_from(codec.emb[int(ids[t])])
    field.settle(F)
    aff = codec.aff(field)
    lg = codec.logits(field, aff)
    loss0, hg = codec.head_grads(field, int(ids[t + 1]), lg, aff)
    srcs, nxt, nn, M, top = field._last
    ml, d_nxt, d_emb = mse_term(field, codec, nxt, srcs, int(ids[t + 1]), lam)
    bg = field.backward(F, hg["act"], d_nxt)
    g_ana = bg["concept"] + hg["cn"]

    # only probe rows that are actually sources (others have zero flow)
    rows = np.unique(np.concatenate([srcs, np.flatnonzero(
        np.abs(g_ana).sum(1) > 0)[:nprobe]]))[:nprobe]
    out = []
    for i in rows:
        for d in range(min(3, field.dc)):
            orig = field.concept[i, d]
            field.concept[i, d] = orig + eps
            field.renorm()
            field.reset()
            field.seed_from(codec.emb[int(ids[t])])
            field.settle(F)
            a1 = codec.aff(field)
            l1, _ = codec.head_grads(field, int(ids[t + 1]),
                                     codec.logits(field, a1), a1)
            s1, x1, _, _, _ = field._last
            m1, _, _ = mse_term(field, codec, x1, s1, int(ids[t + 1]), lam)

            field.concept[i, d] = orig - eps
            field.renorm()
            field.reset()
            field.seed_from(codec.emb[int(ids[t])])
            field.settle(F)
            a2 = codec.aff(field)
            l2, _ = codec.head_grads(field, int(ids[t + 1]),
                                     codec.logits(field, a2), a2)
            s2, x2, _, _, _ = field._last
            m2, _, _ = mse_term(field, codec, x2, s2, int(ids[t + 1]), lam)

            field.concept[i, d] = orig
            field.renorm()
            num = ((l1 + m1) - (l2 + m2)) / (2 * eps)
            ana = float(g_ana[i, d])
            denom = max(abs(num), abs(ana), 1e-9)
            out.append({"i": int(i), "d": int(d), "fd": round(float(num), 8),
                        "ana": round(ana, 8),
                        "relerr": round(abs(num - ana) / denom, 6)})
    return out, float(np.abs(g_ana).sum()), float(np.linalg.norm(g_ana))


def main():
    dc = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    out = sys.argv[5] if len(sys.argv) > 5 else "probe_concept.json"

    txt = open("tinyshakespeare.txt", encoding="utf-8").read()
    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], np.int64)
    cut = int(len(ids) * 0.9)
    tr, va = ids[:cut], ids[cut:]
    V = len(chars)

    field = Field2(n=n, dc=dc, seed=0)
    F = Predictor(dc=dc, hidden=64, seed=0)
    codec = Codec2(V, dc, seed=0)
    adam = Adam(lr=3e-3)

    # ---- A/B: gradient exists and is correct (before any training) ----
    fd, gsum, gnorm = fd_check(field, F, codec, tr, lam=lam)
    rel = [r["relerr"] for r in fd]

    # ---- C/D: drift after real training ----
    c0 = field.concept.copy()
    bits0, acc0, _, _ = run_eval(field, F, codec, va[:2000])
    losses = []
    for t in range(min(steps, len(tr) - 1)):
        tgt = int(tr[t + 1])
        if t % 1000 == 0:
            field.reset()
        field.seed_from(codec.emb[int(tr[t])])
        field.settle(F)
        aff = codec.aff(field)
        lg = codec.logits(field, aff)
        loss, hg = codec.head_grads(field, tgt, lg, aff)
        if field._last is not None:
            srcs, nxt, nn, M, top = field._last
            ml, d_nxt, d_emb = mse_term(field, codec, nxt, srcs, tgt, lam)
            bg = field.backward(F, hg["act"], d_nxt)
            adam.step({
                "concept": (field.concept, bg["concept"] + hg["cn"]),
                "emb": (codec.emb, hg["emb"] + d_emb),
                "bias": (codec.bias, hg["bias"]),
                "W1": (F.W1, bg["F"]["W1"]), "b1": (F.b1, bg["F"]["b1"]),
                "W2": (F.W2, bg["F"]["W2"]), "b2": (F.b2, bg["F"]["b2"])})
            field.renorm()
            codec.renorm()
        losses.append(loss)
        if (t + 1) % 500 == 0:
            print(f"  step {t+1} bits={np.mean(losses[-500:])/LOG2:.4f}",
                  flush=True)

    c1 = field.concept
    drift = float(np.abs(c1 - c0).mean())
    cos = float((normalise(c0) * normalise(c1)).sum(1).mean())
    moved = float(np.linalg.norm(c1 - c0, axis=1).mean())

    bits1, acc1, _, _ = run_eval(field, F, codec, va[:2000])

    # ---- concept_sim_top5 for a few injected chars ----
    sims = {}
    for ch in list("Fae Th"):
        if ch in stoi:
            field.reset()
            field.seed_from(codec.emb[stoi[ch]])
            field.settle(F)
            s, lst = field.concept_sim_top(codec.emb[stoi[ch]], 5)
            sims[ch] = {"mean": round(s, 4), "list": lst}

    res = {
        "config": {"N": n, "dc": dc, "steps": steps, "lam": lam, "V": V},
        "A_gradient_abs_sum": round(gsum, 6),
        "A_gradient_norm": round(gnorm, 6),
        "B_fd_relerr_max": round(max(rel), 6),
        "B_fd_relerr_median": round(float(np.median(rel)), 6),
        "B_fd_samples": fd[:6],
        "C_param_drift_mean_abs": round(drift, 8),
        "C_param_move_l2_mean": round(moved, 8),
        "D_cosine_vs_init_mean": round(cos, 6),
        "bits_before": round(bits0, 4), "bits_after": round(bits1, 4),
        "acc_before": round(acc0, 4), "acc_after": round(acc1, 4),
        "concept_sim_top5_by_char": sims,
        "sparsity": round(field.sparsity(), 4),
    }
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2)[:2600])


def run_eval(field, F, codec, ids, hist=1000):
    tot, acc, cnt = 0.0, 0, 0
    for t in range(len(ids) - 1):
        tgt = int(ids[t + 1])
        if t % hist == 0:
            field.reset()
        field.seed_from(codec.emb[int(ids[t])])
        field.settle(F)
        lg = codec.logits(field)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        tot += float(-np.log(p[tgt] + 1e-12))
        acc += int(np.argmax(lg) == tgt)
        cnt += 1
    return tot / max(cnt, 1) / LOG2, acc / max(cnt, 1), None, None


if __name__ == "__main__":
    main()
