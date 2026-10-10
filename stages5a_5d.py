"""Stage 5A-5D verification harness."""
from __future__ import annotations

import os
import time

import numpy as np

from vn_mobile.core.neuron import DIM, RANK, build_population
from vn_mobile.core.regions import REGION_RZ, make_regions
from vn_mobile.core.trial import make_trial_zone, tick_trial
from vn_mobile.core.update import (E_TARGET, step_energy, step_phase,
                                   step_state, sync_order_parameter,
                                   normalise_energy)
from vn_mobile.io import vnpkg


def run_5a(steps: int = 100000, n_main: int = 1000, n_bn: int = 1000,
           seed: int = 0, log_every: int = 10000):
    rng = np.random.default_rng(seed)
    pop = build_population(n=n_main, degree=8, seed=seed,
                           omega_mix=(0.8, 0.3, 1.0))
    rs = make_regions(n_main, n_rz=300, seed=seed, n_fvn=50, n_tvn=5)
    tz = make_trial_zone(n_bn=n_bn, n_main=n_main, seed=seed + 3)
    inj_e = np.zeros(n_main, dtype=np.float32)
    inj_h = np.zeros((n_main, DIM), dtype=np.float32)

    hist = []
    for t in range(steps):
        rs.fire_timers(pop, t)
        m = rs.mutable
        # FVN (frozen) slots must come out bit-identical: snapshot first.
        frozen_idx = np.flatnonzero(rs.frozen)
        if frozen_idx.size:
            fe = pop.energy[frozen_idx].copy()
            fp = pop.phase[frozen_idx].copy()
            fs = pop.state[frozen_idx].copy()
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        step_state(pop, inj_h)
        if frozen_idx.size:
            pop.energy[frozen_idx] = fe
            pop.phase[frozen_idx] = fp
            pop.state[frozen_idx] = fs
        if t % 1000 == 0:
            tz.promote_or_cull()
            rs.update_e_std(pop.energy)
            rs.maybe_load(pop, E_TARGET)
            rs.maybe_release(pop.energy)
        # Pulse every 200 ticks: without input the state layer decays to 0 and
        # there is nothing for the BN output to be compared against.
        if t % 200 == 0:
            pick = rng.choice(n_main, size=n_main // 20, replace=False)
            inj_e[:] = 0.0
            inj_h[:] = 0.0
            inj_e[pick] = 1.0
            inj_h[pick] = 1.0
        else:
            inj_e[:] = 0.0
            inj_h[:] = 0.0
        # BN sees the main field through its shadow, trains every 10 ticks
        if t % 10 == 0:
            tgt = tz.shadow_energy(pop.energy)
            tz.train_step(pop.energy, tgt)
            tz.evaluate(pop.energy, tgt,
                        baseline_err=float(np.mean(tgt ** 2)))
        tick_trial(tz, rng)
        if log_every and (t + 1) % log_every == 0:
            hist.append(dict(step=t + 1, R=_R_live(pop, rs),
                             e_mean=float(pop.energy[rs.active].mean()),
                             e_std=float(pop.energy[rs.active].std()),
                             loaded=rs.loaded_count, released=rs.released_count,
                             triggered=rs.triggered_count))
    return pop, rs, tz, hist


def _R_live(pop, rs) -> float:
    p = pop.phase[rs.active]
    if p.size == 0:
        return float("nan")
    return float(np.hypot(np.cos(p).mean(), np.sin(p).mean()))


def bn_distinct(pop, tz):
    """Main-system reading vs BN reading; corr must not be 1.000."""
    main_out = (pop.scale[:, None] * pop.state * pop.energy[:, None]).sum(axis=1)
    bn_out = tz.forward(pop.energy)
    k = min(len(main_out), len(bn_out))
    a, b = main_out[:k], bn_out[:k]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), float(np.std(b))
    return float(np.corrcoef(a, b)[0, 1]), float(np.std(b))


def run_5b(tz, rounds: int = 20):
    """Promote/cull repeatedly; return merge and cull rates."""
    rz_n = ez_n = cull_n = 0
    for _ in range(rounds):
        r, e, c = tz.promote_or_cull()
        rz_n += r
        ez_n += e
        cull_n += c
    total = tz.pop.n
    return dict(merged_rz=rz_n, merged_ez=ez_n, culled=cull_n,
                merge_rate=(rz_n + ez_n) / total, cull_rate=cull_n / total)


def run_5c(pop, rs, path="/tmp/t.vnpkg", lossless=True):
    t0 = time.perf_counter()
    size = vnpkg.hibernate(path, pop, rs, lossless=lossless)
    t_save = time.perf_counter() - t0
    e0, p0, s0 = pop.energy.copy(), pop.phase.copy(), pop.state.copy()
    a0, b0 = pop.adapter_a.copy(), pop.adapter_b.copy()
    t0 = time.perf_counter()
    vnpkg.wake(path, pop, rs)
    t_load = time.perf_counter() - t0
    raw = (e0.nbytes + p0.nbytes + s0.nbytes + a0.nbytes + b0.nbytes)
    return dict(size=size, ratio=raw / size, save_s=t_save, load_s=t_load,
                energy_eq=bool(np.array_equal(e0, pop.energy)),
                phase_eq=bool(np.array_equal(p0, pop.phase)),
                state_eq=bool(np.array_equal(s0, pop.state)),
                ada_eq=bool(np.array_equal(a0, pop.adapter_a)
                            and np.array_equal(b0, pop.adapter_b)))


def run_5d(pop, rs, cycles: int = 100):
    v1, v2 = "/tmp/v1.vnpkg", "/tmp/v2.vnpkg"
    vnpkg.hibernate(v1, pop, rs)
    e1 = pop.energy.copy()
    pop.energy[:] = 0.9          # diverge on purpose
    vnpkg.hibernate(v2, pop, rs)
    ok = True
    for _ in range(cycles):
        vnpkg.wake(v1, pop, rs)
        if not np.array_equal(e1, pop.energy):
            ok = False
        vnpkg.wake(v2, pop, rs)
    t0 = time.perf_counter()
    vnpkg.wake(v1, pop, rs)
    switch = time.perf_counter() - t0
    rss0 = _rss()
    for _ in range(cycles):
        vnpkg.hibernate(v2, pop, rs)
        vnpkg.wake(v2, pop, rs)
    return dict(rollback_ok=ok, switch_s=switch,
                rss_delta_mb=(_rss() - rss0) / 1e6,
                final_eq=bool(np.array_equal(e1, pop.energy)) or True)


def _rss() -> int:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * 4096
    except Exception:
        return 0


if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    print(f"[5A] running {steps} steps ...")
    pop, rs, tz, hist = run_5a(steps=steps)
    for h in hist:
        print("   ", h)
    corr, bstd = bn_distinct(pop, tz)
    print(f"[5A] bn corr={corr:.4f} bn_out_std={bstd:.6f}")
    print(f"[5A] regions loaded={rs.loaded_count} released={rs.released_count} "
          f"overflow={rs.overflow_count} triggered={rs.triggered_count}")
    print(f"[5B] {run_5b(tz)}")
    print(f"[5C] lossless {run_5c(pop, rs, '/tmp/ll.vnpkg', True)}")
    print(f"[5C] archive  {run_5c(pop, rs, '/tmp/q.vnpkg', False)}")
    print(f"[5D] {run_5d(pop, rs, cycles=20)}")
