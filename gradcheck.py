"""Is the adapters-only gradient actually the gradient?

Central finite differences on a handful of adapter entries, compared against
what train_char applies.  If these disagree the whole of 6A is noise, and no
amount of learning-rate tuning will fix it.
"""
from __future__ import annotations

import json

import numpy as np

from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import ALPHA, neighbour_mean, step_state
from vn_mobile.io.charhead import CharHead, softmax, train_char
from vn_mobile.io.vocab import block_of


def loss_of(pop, block, head, x, target: int, inj_h=None) -> float:
    """Loss after one tick from the CURRENT state.

    The first version of this helper forgot to re-run step_state, so changing
    an adapter had no effect on an already-computed state and every numeric
    gradient came out exactly 0.0.  Adapters act through the next tick, so the
    tick has to be replayed.
    """
    from vn_mobile.core.update import read_output, step_state
    from vn_mobile.io.vocab import pool_by_block
    snap = pop.state.copy()
    tmp = pop if inj_h is None else None
    if inj_h is not None:
        step_state(pop, inj_h)
    pooled = pool_by_block(read_output(pop), block, head.w.shape[0])
    probs = softmax(head.logits(pooled))
    loss = float(-np.log(max(float(probs[target]), 1e-9)))
    pop.state[:] = snap
    return loss


def main() -> None:
    V, N = 8, 32
    pop = build_population(n=N, degree=6, seed=1)
    block = block_of(N, V)
    head = CharHead.make(V, DIM, seed=1, scale=0.05)
    rng = np.random.default_rng(1)
    inj_h = np.zeros((N, DIM), dtype=np.float32)
    inj_h[block == 3] = 1.0
    step_state(pop, inj_h)
    x = inj_h + neighbour_mean(pop, pop.state)

    target = 5
    eps = 1e-3
    rows = []
    for name, arr in (("adapter_b", pop.adapter_b), ("adapter_a", pop.adapter_a)):
        flat = arr.reshape(-1)
        idx = np.random.default_rng(2).choice(flat.size, 6, replace=False)
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            pop.WM = None
            lp = loss_of(pop, block, head, x, target, inj_h)
            flat[i] = old - eps
            pop.WM = None
            lm = loss_of(pop, block, head, x, target, inj_h)
            flat[i] = old
            pop.WM = None
            numeric = (lp - lm) / (2 * eps)
            # what train_char would apply: dW = -lr * grad
            snap_b = pop.adapter_b.copy()
            snap_a = pop.adapter_a.copy()
            train_char(pop, block, head, target, 1.0, ALPHA, x)
            applied = float((pop.adapter_b - snap_b).reshape(-1)[i]
                            if name == "adapter_b"
                            else (pop.adapter_a - snap_a).reshape(-1)[i])
            pop.adapter_b[:] = snap_b
            pop.adapter_a[:] = snap_a
            pop.WM = None
            rows.append(dict(term=name, flat_index=int(i),
                             numeric=round(numeric, 8),
                             analytic=round(-applied, 8),
                             ratio=(round(-applied / numeric, 4)
                                    if abs(numeric) > 1e-10 else None)))
    bad = [r for r in rows if r["ratio"] is None
           or not (0.8 < abs(r["ratio"]) < 1.25)]
    out = dict(rows=rows, mismatches=len(bad),
               verdict=("gradient OK" if not bad else "GRADIENT WRONG"))
    print(json.dumps(out, indent=2))
    with open("gradcheck.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
