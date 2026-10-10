"""Chunked trainer for the concept wave (<=200 lines).

Long runs get killed when a single command outlives its slot, so training is
driven in chunks and every chunk re-loads the codec + Adam moments from disk.
State file holds emb / bias / scale and the Adam first/second moments.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/data/workspace/ceshi")

from conceptwave.encode import CharCodec
from conceptwave.field import ConceptField
from conceptwave.train import run
from vn_mobile.io.adam import Adam
from vn_mobile.io.vocab import build_vocab, encode, load_text, split

STATE = "cw_state.npz"
CURVE = "cw_curve.json"


def load_data():
    text = load_text()
    sym, stoi = build_vocab(text)
    tr_txt, va_txt = split(text)
    return len(sym), encode(tr_txt, stoi), encode(va_txt, stoi)


def restore(codec, opt, keys):
    try:
        z = np.load(STATE, allow_pickle=True)
    except FileNotFoundError:
        return 0
    codec.emb = z["emb"]; codec.bias = z["bias"]
    codec.scale = np.float32(z["scale"]); opt.t = int(z["t"])
    for k in keys:
        if f"m_{k}" in z:
            opt.m[k] = z[f"m_{k}"]; opt.v[k] = z[f"v_{k}"]
    return int(z["done"])


def save(codec, opt, done, keys):
    d = {"emb": codec.emb, "bias": codec.bias,
         "scale": np.float32(codec.scale), "t": opt.t, "done": done}
    for k in keys:
        d[f"m_{k}"] = opt.m[k]; d[f"v_{k}"] = opt.v[k]
    np.savez(STATE, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=30000)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--dorm", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    V, tr, va = load_data()
    keys = ["emb", "bias", "scale"]
    f = ConceptField.create(1000, 32, 1, 8, seed=a.seed, dormancy=a.dorm)
    codec = CharCodec(V, 32, seed=a.seed)
    opt = Adam(lr=a.lr)
    done = 0 if a.reset else restore(codec, opt, keys)
    if a.reset:
        import os
        if os.path.exists(STATE):
            os.remove(STATE)
    if a.val:
        fb, fa, _ = run(f, codec, va[:a.val], None, hist=1000,
                        rounds=a.rounds)
        print(json.dumps(dict(val_bits=round(float(fb), 4),
                              val_acc=round(float(fa), 4),
                              frac_active=round(float(f.frac_active()), 4))))
        return
    seg = tr[done:done + a.len]
    b, acc, curve = run(f, codec, seg, opt, hist=1000, rounds=a.rounds,
                        log_every=15000, tag="cw")
    done += len(seg)
    save(codec, opt, done, keys)
    try:
        hist = json.load(open(CURVE))
    except Exception:
        hist = []
    hist.extend(curve)
    json.dump(hist, open(CURVE, "w"))
    print(json.dumps(dict(done=done, train_bits=round(float(b), 4),
                          last=curve[-1] if curve else None)))


if __name__ == "__main__":
    main()
