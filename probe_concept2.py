"""Focused finite-difference check on the LARGEST-|grad| concept rows.

The first FD probe picked rows with |grad| ~ 4e-5, below float resolution of
the loss.  This one picks the top rows by |analytic grad| and sweeps eps so we
can see whether the error is a real bug or just float noise.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from conceptwave2.predict import Predictor, normalise
from conceptwave2.field2 import Field2
from conceptwave2.train2 import Codec2, mse_term

LOG2 = float(np.log(2.0))


def loss_at(field, F, codec, emb_t, tgt, lam):
    field.reset()
    field.seed_from(emb_t)
    field.settle(F)
    aff = codec.aff(field)
    lg = codec.logits(field, aff)
    l, _ = codec.head_grads(field, tgt, lg, aff)
    s, x, _, _, _ = field._last
    m, _, _ = mse_term(field, codec, x, s, tgt, lam)
    return l + m


def main():
    dc, n, lam = 32, 1000, 0.1
    txt = open("tinyshakespeare.txt", encoding="utf-8").read()
    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], np.int64)
    V = len(chars)

    field = Field2(n=n, dc=dc, seed=0)
    F = Predictor(dc=dc, hidden=64, seed=0)
    codec = Codec2(V, dc, seed=0)

    t = 0
    tgt = int(ids[t + 1])
    emb_t = codec.emb[int(ids[t])]

    field.reset()
    field.seed_from(emb_t)
    field.settle(F)
    aff = codec.aff(field)
    lg = codec.logits(field, aff)
    loss0, hg = codec.head_grads(field, tgt, lg, aff)
    srcs, nxt, nn, M, top = field._last
    ml, d_nxt, d_emb = mse_term(field, codec, nxt, srcs, tgt, lam)
    bg = field.backward(F, hg["act"], d_nxt)
    g = (bg["concept"] + hg["cn"]).astype(np.float64)

    order = np.argsort(-np.abs(g).sum(1))[:5]
    rows = []
    for i in order:
        d = int(np.argmax(np.abs(g[i])))
        ana = float(g[i, d])
        best = None
        for eps in [1e-3, 1e-2, 5e-2, 1e-1]:
            orig = field.concept[i, d]
            field.concept[i, d] = orig + eps
            field.renorm()
            lp = loss_at(field, F, codec, emb_t, tgt, lam)
            field.concept[i, d] = orig - eps
            field.renorm()
            lm = loss_at(field, F, codec, emb_t, tgt, lam)
            field.concept[i, d] = orig
            field.renorm()
            num = (lp - lm) / (2 * eps)
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
            if best is None or rel < best["relerr"]:
                best = {"eps": eps, "fd": round(float(num), 8),
                        "ana": round(ana, 8), "relerr": round(float(rel), 6)}
        rows.append({"row": int(i), "dim": d, "grad_row_abs_sum":
                     round(float(np.abs(g[i]).sum()), 6), **best})

    res = {"rows": rows,
           "grad_total_abs_sum": round(float(np.abs(g).sum()), 4),
           "grad_max_row": round(float(np.abs(g).sum(1).max()), 6),
           "n_rows_with_grad": int((np.abs(g).sum(1) > 1e-9).sum()),
           "n_sources": int(len(srcs)),
           "loss": round(float(loss0 + ml), 6)}
    json.dump(res, open("probe_concept2.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
