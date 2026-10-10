"""Per-step cost, measured in layers so the remaining budget is visible.

Optimisation target was < 200 us/step.  This script reports where the time
actually goes instead of a single blended number:

    layer 1  main field only      step_energy + step_phase + step_state
    layer 2  + trial-zone tick    TZ neurons keep their own wave field alive
    layer 3  + sampling           shadow read, train_step, evaluate (every 10)
    layer 4  full run_5a          everything, including cull rounds and timers
"""
from __future__ import annotations

import json
import time

import numpy as np

from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.trial import SAMPLE_EVERY
from vn_mobile.core.tzbuild import make_trial_zone, tick_trial
from vn_mobile.core.update import step_energy, step_phase, step_state

N = 1000
WARM = 500
MEAS = 3000


def _bench(fn, warm=WARM, meas=MEAS):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(meas):
        fn()
    return round((time.perf_counter() - t0) / meas * 1e6, 1)


def main() -> None:
    rng = np.random.default_rng(0)
    pop = build_population(n=N, degree=8, seed=0)
    inj_h = np.zeros((N, DIM), dtype=np.float32)
    inj_e = np.zeros(N, dtype=np.float32)

    def layer1():
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        step_state(pop, inj_h)

    tz = make_trial_zone(n_bn=1000, n_main=1000, n_zones=20, seed=3)

    def layer2():
        layer1()
        tick_trial(tz, rng)

    prev = np.random.rand(N).astype(np.float32)
    cur = np.random.rand(N).astype(np.float32)
    bn_idx = np.arange(N)

    def layer3():
        layer2()
        if rng.integers(SAMPLE_EVERY) == 0:
            tz.shadow_energy(pop.energy)
            tz.train_step(prev, cur)
            tz.evaluate(prev, cur)

    out = dict(layer1_main_field=_bench(layer1),
               layer2_plus_tz=_bench(layer2),
               layer3_plus_sampling=_bench(layer3))

    from stages5a_5d import run_5a
    run_5a(steps=100, log_every=0)
    t0 = time.perf_counter()
    run_5a(steps=2000, log_every=0)
    out["layer4_full_run5a"] = round((time.perf_counter() - t0) / 2000 * 1e6, 1)
    out["delta_tz"] = round(out["layer2_plus_tz"] - out["layer1_main_field"], 1)
    out["delta_sampling"] = round(out["layer3_plus_sampling"]
                                  - out["layer2_plus_tz"], 1)
    print(json.dumps(out, indent=2))
    with open("bench_step.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
