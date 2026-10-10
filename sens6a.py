"""Which parameters can actually move the loss (stage 6A)?

Adapter A/B are a low-rank perturbation on a frozen W: at N(0, 0.01) the B@A
term is ~5e-5 of W, and the measured effect on the loss is ~3e-7 -- smaller
than float noise.  Before tuning anything, this measures the sensitivity of
the loss to each candidate trainable set, so 6A does not spend a million
steps optimising something that cannot move.
"""
from __future__ import annotations

import json

import numpy as np

from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import ALPHA, neighbour_mean, step_energy, step_phase, step_state
from vn_mobile.io.charhead import CharHead, softmax, train_char
from vn_mobile.io.vocab import block_of, pool_by_block
from vn_mobile.core.update import read_output


def sensitivity(pop, block, head, inj_h, target, arrays, eps=1e-2):
    """|dL| when every entry of each candidate array gets +eps."""
    def L():
        snap = pop.state.copy()
        step_state(pop, inj_h)
        pooled = pool_by_block(read_output(pop), block, head.w.shape[0])
        p = softmax(head.logits(pooled))
        v = float(-np.log(max(float(p[target]), 1e-9)))
        pop.state[:] = snap
        return v
    base = L()
    out = {}
    for name, arr in arrays.items():
        old = arr.copy()
        arr[:] = old + eps
        pop.WM = None
        out[name] = abs(L() - base)
        arr[:] = old
        pop.WM = None
    return base, out


def main() -> None:
    V, N = 65, 640
    rows = []
    for init in (0.01, 0.1, 0.3):
        for scale_w in (0.01, 0.05):
            pop = build_population(n=N, degree=8, seed=2)
            pop.adapter_a = (np.random.default_rng(3).normal(
                0.0, init, size=pop.adapter_a.shape)).astype(np.float32)
            pop.adapter_b = (np.random.default_rng(4).normal(
                0.0, init, size=pop.adapter_b.shape)).astype(np.float32)
            pop.WM = None
            block = block_of(N, V)
            head = CharHead.make(V, DIM, seed=2, scale=scale_w)
            rng = np.random.default_rng(2)
            inj_e = np.zeros(N, dtype=np.float32)
            inj_h = np.zeros((N, DIM), dtype=np.float32)
            inj_e[block == 3] = 1.0
            inj_h[block == 3] = 1.0
            step_energy(pop, rng, injection=inj_e)
            step_phase(pop)
            step_state(pop, inj_h)
            base, sens = sensitivity(
                pop, block, head, inj_h, 5,
                dict(adapter_a=pop.adapter_a, adapter_b=pop.adapter_b,
                     head_w=head.w, bias=pop.bias, scale=pop.scale))
            rows.append(dict(adapter_init=init, head_scale=scale_w,
                             loss=round(base, 5),
                             **{k: round(float(v), 9) for k, v in sens.items()}))
    print(json.dumps(rows, indent=2))
    with open("sens6a.json", "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
