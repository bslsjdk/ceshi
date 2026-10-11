"""Chunked trainer for concept wave v1.1 (<=200 lines).

Long runs die when one command outlives its slot, so training is driven in
chunks; every chunk reloads codec + Adam moments from disk.  ``--eval`` runs
the validation slice and prints the exp-1 metrics without training.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, "/data/workspace/ceshi")

from conceptwave.encode import CharCodec
from conceptwave.field import ConceptField
from conceptwave.kernels import cosine, normalise
from conceptwave.train import run
from vn_mobile.io.adam import Adam
from vn_mobile.io.vocab import build_vocab, encode, load_text, split

VARIANTS = {
    "v1": dict(signed=False, nonlinear=False, psi_max=0.0, norm_global=True),
    "v11": dict(signed=True, nonlinear=True, psi_max=10.0, norm_global=False),
}


def load_data(seed=0):
    text = load_text()
    sym, stoi = build_vocab(text)
    tr_txt, va_txt = split(text)
    return len(sym), encode(tr_txt, stoi), encode(va_txt, stoi)


def build(V, seed, variant, dorm):
    f = ConceptField.create(1000, 32, 1, 8, seed=seed, dormancy=dorm,
                            **VARIANTS[variant])
    codec = CharCodec(V, 32, seed=seed)
    return f, codec


def restore(codec, opt, state):
    if not os.path.exists(state):
        return 0
    z = np.load(state, allow_pickle=True)
    codec.emb = z["emb"]; codec.bias = z["bias"]
    codec.scale = np.float32(z["scale"]); opt.t = int(z["t"])
    for k in ("emb", "bias", "scale"):
        if f"m_{k}" in z:
            opt.m[k] = z[f"m_{k}"]; opt.v[k] = z[f"v_{k}"]
    return int(z["done"])


def save(codec, opt, done, state):
    d = {"emb": codec.emb, "bias": codec.bias,
         "scale": np.float32(codec.scale), "t": opt.t, "done": done}
    for k in ("emb", "bias", "scale"):
        d[f"m_{k}"] = opt.m[k]; d[f"v_{k}"] = opt.v[k]
    np.savez(state, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="v11", choices=["v1", "v11"])
    ap.add_argument("--dorm", type=float, default=0.3)
    ap.add_argument("--state", default="cw_state.npz")
    ap.add_argument("--curve", default="cw_curve.json")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--val_limit", type=int, default=3000)
    ap.add_argument("--hist", type=int, default=1000)
    a = ap.parse_args()

    V, tr, va = load_data(a.seed)
    f, codec = build(V, a.seed, a.variant, a.dorm)
    opt = Adam(lr=a.lr)
    done = restore(codec, opt, a.state)
    t0 = time.perf_counter()

    if a.eval:
        b, acc, _ = run(f, codec, va[:a.val_limit], None, hist=a.hist,
                          rounds=12)
        fa = float(np.mean([_frac(f, codec, int(t)) for t in va[:150]]))
        dom = float(np.mean([_dom(f, codec, int(t)) for t in va[:80]]))
        print("EVAL " + json.dumps(dict(variant=a.variant, tokens=done,
              hist=a.hist, val_bits=round(float(b), 4),
              val_acc=round(float(acc), 4),
              frac_active=round(fa, 4), concept_sim_top5=round(dom, 4))))
        return

    trb, acc, curve = run(f, codec, tr[done:done + a.len], opt, hist=1000,
                          rounds=12, log_every=max(1, a.len // 2), tag=a.variant)
    done += a.len
    save(codec, opt, done, a.state)
    old = json.load(open(a.curve)) if os.path.exists(a.curve) else []
    old += [dict(done=done, **c) for c in curve]
    json.dump(old, open(a.curve, "w"), indent=2)
    print("CHUNK " + json.dumps(dict(variant=a.variant, done=done,
          train_bits=curve[-1]["train_bits"] if curve else None,
          secs=round(time.perf_counter() - t0, 1),
          rss_mb=round(resource.getrusage(
              resource.RUSAGE_SELF).ru_maxrss / 1024, 1))))


def _frac(f, codec, tok):
    f.set_context(codec.ctx_vector([tok]))
    codec.inject_char(f, tok)
    f.settle(max_rounds=12)
    return f.frac_active()


def _dom(f, codec, tok):
    f.set_context(codec.ctx_vector([tok]))
    codec.inject_char(f, tok)
    f.settle(max_rounds=12)
    top = np.argsort(-f.energy())[:5]
    return float(cosine(f.cn[top],
                        normalise(codec.emb[tok].reshape(1, -1))).mean())


if __name__ == "__main__":
    main()
