"""Finite-difference check for the hand-written BPTT backward pass.

A hand-written recurrent backward pass is exactly the kind of thing that can
be subtly wrong and still look plausible, so the gradients are checked against
a numerical estimate before any training run is trusted.

The field is stateful, so each perturbed evaluation replays the whole window
from a snapshot with a freshly seeded generator -- the same noise, the same
starting state, one parameter changed.
"""
from __future__ import annotations

import json

import numpy as np

from train6c import forward, inject
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.io.bptt import backward_window
from vn_mobile.io.lowrank import LowRankHead
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split

EPS = 1e-3
SEED = 0
EPS_LIST = [1e-3, 1e-2, 5e-2, 2e-1]


def snapshot(pop):
    return pop.energy.copy(), pop.phase.copy(), pop.state.copy()


def restore(pop, snap):
    pop.energy[:] = snap[0]
    pop.phase[:] = snap[1]
    pop.state[:] = snap[2]


def total_loss(pop, head, block, members, seq, start, T, rng_seed, vocab,
               inj_e, inj_h) -> float:
    rng = np.random.default_rng(rng_seed)
    win = forward(pop, head, block, members, seq, start, T, rng, inj_e, inj_h,
                  vocab, hist=1, decay=0.6)
    return float(np.sum(win.losses))


def main() -> None:
    text = load_text()
    symbols, stoi = build_vocab(text)
    V = len(symbols)
    tr_txt, _ = split(text)
    tr = encode(tr_txt, stoi)
    n_vn, T = 200, 4

    pop = build_population(n=n_vn, degree=8, seed=SEED,
                           omega_mix=(0.8, 0.3, 1.0))
    block = block_of(n_vn, V)
    members = [np.flatnonzero(block == v) for v in range(V)]
    head = LowRankHead.make(V, DIM, rank=16, seed=SEED)
    inj_e = np.zeros(n_vn, dtype=np.float32)
    inj_h = np.zeros((n_vn, DIM), dtype=np.float32)

    snap = snapshot(pop)
    rng = np.random.default_rng(SEED)
    win = forward(pop, head, block, members, tr, 0, T, rng, inj_e, inj_h, V,
                  hist=1, decay=0.6)
    base = float(np.sum(win.losses))
    g = backward_window(pop, head, win, block, V, DIM)

    out = {}
    rng_probe = np.random.default_rng(1234)
    for name, analytic, tensor in (("u", g["dw"][0], head.u),
                                   ("p", g["dw"][1], head.p),
                                   ("bias", g["db"], pop.bias)):
        idx = tuple(int(rng_probe.integers(0, tensor.shape[d]))
                    for d in range(tensor.ndim))
        old = float(tensor[idx])
        ana = float(analytic.reshape(tensor.shape)[idx])
        rows = []
        for eps in EPS_LIST:
            restore(pop, snap)
            tensor[idx] = old + eps
            hi = total_loss(pop, head, block, members, tr, 0, T, SEED, V,
                            inj_e, inj_h)
            restore(pop, snap)
            tensor[idx] = old - eps
            lo = total_loss(pop, head, block, members, tr, 0, T, SEED, V,
                            inj_e, inj_h)
            tensor[idx] = old
            num = (hi - lo) / (2 * eps)
            den = max(abs(num), abs(ana), 1e-8)
            rows.append(dict(eps=eps, numeric=round(float(num), 9),
                             rel_err=round(abs(ana - num) / den, 4)))
        out[name] = dict(idx=list(idx), analytic=round(ana, 9), scan=rows)
    out["base_loss"] = round(base, 6)
    print(json.dumps(out, indent=2))
    with open("gradcheck6c.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
