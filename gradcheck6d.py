"""Finite-difference check for the zone-shared adapter gradients (6A2).

The adapters are now inside the unrolled recurrence, so unlike 6C their
gradient has a *time* component: perturbing A or B changes h at every tick
after the perturbation, not just the tick that feeds the readout.  A gradient
that ignored that would still pass a one-tick test and would silently train
the wrong thing, so the probe here unrolls the same T ticks the trainer does.

The probe reads a single scalar: the mean cross-entropy over the window.
Central differences are taken against the analytic gradient for increasing
step sizes; float32 rounding dominates at small steps, so what matters is the
relative error *falling* as the step grows, not being small at one step.
"""
from __future__ import annotations

import json

import numpy as np

from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import step_energy, step_phase
from vn_mobile.io.bptt import (backward_window, finite, pooled_flat,
                               step_state_x)
from vn_mobile.io.lowrank import LowRankHead, softmax
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split

T = 6


def build(n=120, zones=4, seed=0):
    text = load_text()
    symbols, stoi = build_vocab(text)
    V = len(symbols)
    tr = encode(split(text)[0], stoi)
    pop = build_population(n=n, degree=8, seed=seed,
                           omega_mix=(0.8, 0.3, 1.0), n_zones=zones)
    block = block_of(n, V)
    members = [np.flatnonzero(block == v) for v in range(V)]
    head = LowRankHead.make(V, DIM, rank=8, seed=seed)
    return pop, head, block, members, tr, V


def snapshot(pop):
    return (pop.energy.copy(), pop.phase.copy(), pop.state.copy())


def restore(pop, snap):
    pop.energy[:] = snap[0]
    pop.phase[:] = snap[1]
    pop.state[:] = snap[2]
    pop.WM = None


def loss_of(pop, head, block, members, seq, V, t0=0):
    """Unroll T ticks from the *current* field state; leaves the field moved.

    Callers must snapshot/restore around this: the field is deliberately
    persistent, so a second call starts wherever the first one ended, and a
    finite difference taken that way measures the wrong thing.
    """
    rng = np.random.default_rng(7)
    inj_e = np.zeros(pop.n, dtype=np.float32)
    inj_h = np.zeros((pop.n, DIM), dtype=np.float32)
    from vn_mobile.io.bptt import Window
    win = Window()
    t = t0
    for _ in range(T):
        inj_e[:] = 0.0
        inj_h[:] = 0.0
        inj_e[members[int(seq[t % seq.size])]] += 1.0
        inj_h[members[int(seq[t % seq.size])]] += 1.0
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        x, th = step_state_x(pop, inj_h)
        probs = softmax(head.logits(pooled_flat(pop, block, V)))
        nxt = int(seq[(t + 1) % seq.size])
        win.xs.append(x)
        win.ths.append(th)
        win.flats.append(pooled_flat(pop, block, V))
        win.probs.append(probs)
        win.energies.append(pop.energy.copy())
        win.targets.append(nxt)
        win.losses.append(float(-np.log(max(float(probs[nxt]), 1e-9))))
        t += 1
    return float(np.mean(win.losses)), win


def main() -> None:
    pop, head, block, members, tr, V = build()
    snap = snapshot(pop)
    base, win = loss_of(pop, head, block, members, tr, V)
    restore(pop, snap)
    g = backward_window(pop, head, win, block, V, DIM)
    assert finite(g), "analytic gradient went non-finite"
    rows = []
    for name, analytic in (("adapter_b", g["gb"]), ("adapter_a", g["ga"]),
                           ("bias", g["db"])):
        param = {"adapter_b": pop.adapter_b, "adapter_a": pop.adapter_a,
                 "bias": pop.bias}[name]
        idx = tuple(int(v) for v in np.unravel_index(
            int(np.argmax(np.abs(analytic))), analytic.shape))
        for eps in (1e-3, 1e-2, 1e-1):
            flat = param.reshape(-1)
            k = int(np.ravel_multi_index(idx, param.shape))
            old = float(flat[k])
            flat[k] = old + eps
            pop.WM = None
            hi, _ = loss_of(pop, head, block, members, tr, V)
            restore(pop, snap)
            flat[k] = old - eps
            pop.WM = None
            lo, _ = loss_of(pop, head, block, members, tr, V)
            restore(pop, snap)
            flat[k] = old
            pop.WM = None
            num = (hi - lo) / (2.0 * eps)
            if abs(num) < 1e-12:
                continue
            ana = float(analytic[idx])
            rows.append(dict(name=name, idx=str(idx), eps=eps,
                             analytic=round(ana, 8), numeric=round(num, 8),
                             rel_err=round(abs(num - ana) /
                                           max(1e-12, abs(num) + abs(ana)), 6)))
    print(json.dumps(dict(base_loss=round(base, 6), rows=rows), indent=2))
    with open("gradcheck6d.json", "w") as fh:
        json.dump(dict(base_loss=base, rows=rows), fh, indent=2)


if __name__ == "__main__":
    main()
