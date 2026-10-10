"""Stage 5C (vnpkg persistence) and 5D (hibernate/wake, versioning).

Split out of the 5A harness to keep both files under the 200-line budget.
"""
from __future__ import annotations

import time

import numpy as np

from vn_mobile.io import vnpkg


def run_5c(pop, rs, path="/tmp/t.vnpkg", lossless=True) -> dict:
    """5C keeps two modes: hibernate is judged on byte equality only, archive
    on ratio only -- random float32 caps out near 1.09x, so >3x requires
    quantisation and the two criteria cannot both hold for one file."""
    size = vnpkg.hibernate(path, pop, rs, lossless=lossless)
    e0, p0, s0 = pop.energy.copy(), pop.phase.copy(), pop.state.copy()
    a0, b0 = pop.adapter_a.copy(), pop.adapter_b.copy()
    t0 = time.perf_counter()
    vnpkg.wake(path, pop, rs)
    t_load = time.perf_counter() - t0
    raw = e0.nbytes + p0.nbytes + s0.nbytes + a0.nbytes + b0.nbytes
    return dict(size=size, ratio=round(raw / size, 3), load_s=round(t_load, 4),
                energy_eq=bool(np.array_equal(e0, pop.energy)),
                phase_eq=bool(np.array_equal(p0, pop.phase)),
                state_eq=bool(np.array_equal(s0, pop.state)),
                ada_eq=bool(np.array_equal(a0, pop.adapter_a)
                            and np.array_equal(b0, pop.adapter_b)))


def run_5d(pop, rs, cycles: int = 100) -> dict:
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
    return dict(rollback_ok=ok, switch_s=round(switch, 4),
                rss_delta_mb=round((_rss() - rss0) / 1e6, 3))


def _rss() -> int:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * 4096
    except Exception:
        return 0
