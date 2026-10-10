"""Optimisation-pass tests: every rewrite must stay numerically equivalent.

The speed work replaced three pieces of the update path -- per-edge trig with
two sparse matvecs, the segmented reduce with a CSR product, and Wx + B(Ax)
with one fused [N, DIM, DIM] product.  These tests pin each rewrite against
the straightforward version it replaced, so a future edit cannot silently
change the dynamics.

The 5A criteria themselves are measured in ``diag100k.py``; they vary across
seeds (between_zone_std was 0.0114 at seed 0 / 100k and 0.0052 at seed 1 /
20k) so they are not asserted here at a tight threshold.
"""
from __future__ import annotations

import numpy as np

from vn_mobile.core import update as U
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.tzbuild import make_trial_zone
from vn_mobile.core.zstats import zone_median_std


def _dense_adj(pop):
    """Force the pre-optimisation adjacency path."""
    return None, None


def test_sparse_energy_matches_per_edge_form():
    pop = build_population(n=256, degree=8, seed=5)
    rng = np.random.default_rng(5)
    for _ in range(20):
        U.step_energy(pop, rng)
        U.step_phase(pop)
    c, s = np.cos(pop.phase), np.sin(pop.phase)
    _, Aw = U._adjacency(pop)
    fast = c * (Aw @ (c * pop.energy)) + s * (Aw @ (s * pop.energy))
    d = pop.phase[pop.dst] - pop.phase[pop.src]
    slow = np.bincount(pop.src, weights=pop.wedge * np.cos(d)
                       * pop.energy[pop.dst], minlength=pop.n)
    assert np.allclose(fast, slow, atol=1e-4)


def test_sparse_phase_matches_per_edge_form():
    pop = build_population(n=256, degree=8, seed=6)
    rng = np.random.default_rng(6)
    for _ in range(20):
        U.step_energy(pop, rng)
        U.step_phase(pop)
    c, s = np.cos(pop.phase), np.sin(pop.phase)
    A1, _ = U._adjacency(pop)
    fast = c * (A1 @ s) - s * (A1 @ c)
    d = pop.phase[pop.dst] - pop.phase[pop.src]
    slow = np.bincount(pop.src, weights=np.sin(d), minlength=pop.n)
    assert np.allclose(fast, slow, atol=1e-4)


def test_aggregate_matches_segmented_reduce():
    pop = build_population(n=128, degree=8, seed=7)
    vals = np.random.rand(pop.n, DIM).astype(np.float32)
    got = U.aggregate(pop, vals)
    want = U._aggregate_dense(pop, vals)
    assert np.allclose(got, want, atol=1e-5)


def test_fused_matrix_matches_wx_plus_bax():
    pop = build_population(n=64, degree=8, seed=8)
    x = np.random.rand(pop.n, DIM).astype(np.float32)
    fused = np.matmul(U._fused(pop), x[:, :, None])[:, :, 0]
    mid = np.matmul(pop.adapter_a, x[:, :, None])[:, :, 0]
    slow = (x @ pop.W.T + np.matmul(pop.adapter_b, mid[:, :, None])[:, :, 0])
    assert np.allclose(fused, slow, atol=1e-4)


def test_zone_median_std_matches_numpy():
    rng = np.random.default_rng(9)
    z = np.repeat(np.arange(20), 50).astype(np.int64)
    v = rng.normal(size=z.size).astype(np.float32)
    med, std = zone_median_std(v, z)
    for u in range(20):
        m = z == u
        assert abs(float(med[m][0]) - float(np.median(v[m]))) < 1e-5
        assert abs(float(std[m][0]) - float(np.std(v[m]))) < 1e-4


def test_zone_median_std_handles_ragged_groups():
    z = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    v = np.array([3.0, 1.0, 2.0, 9.0, 5.0, 0.0, 4.0, 8.0, 6.0],
                 dtype=np.float32)
    med, _ = zone_median_std(v, z)
    # med is broadcast back per member, so index by a member of each group.
    assert abs(float(med[0]) - 2.0) < 1e-5      # zone 0: median(3,1,2)
    assert abs(float(med[3]) - 7.0) < 1e-5      # zone 1: mean(5,9), even count
    assert abs(float(med[5]) - 5.0) < 1e-5      # zone 2: (4+6)/2


def test_dense_fallback_stays_importable_without_scipy():
    """The numpy path must still produce the same field if scipy is absent."""
    pop = build_population(n=64, degree=8, seed=10)
    saved = U._adjacency
    U._adjacency = _dense_adj
    try:
        rng = np.random.default_rng(10)
        inj = np.zeros((pop.n, DIM), dtype=np.float32)
        for _ in range(30):
            U.step_energy(pop, rng)
            U.step_phase(pop)
            U.step_state(pop, inj)
        got = pop.energy.copy()
    finally:
        U._adjacency = saved
    pop2 = build_population(n=64, degree=8, seed=10)
    rng2 = np.random.default_rng(10)
    inj2 = np.zeros((pop2.n, DIM), dtype=np.float32)
    for _ in range(30):
        U.step_energy(pop2, rng2)
        U.step_phase(pop2)
        U.step_state(pop2, inj2)
    assert np.allclose(got, pop2.energy, atol=1e-4)


def test_training_invalidates_the_fused_cache():
    tz = make_trial_zone(n_bn=64, n_main=256, n_zones=4, seed=11)
    prev = np.random.rand(64).astype(np.float32)
    cur = np.random.rand(64).astype(np.float32)
    U._fused(tz.pop)
    assert tz.pop.WM is not None
    tz.train_step(prev, cur)
    assert tz.pop.WM is None      # stale after the adapter update
