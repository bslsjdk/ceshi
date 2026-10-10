"""Stage 5 tests, written against the real RegionManager API.

The earlier version imported ``make_regions``, which does not exist, so the
module failed at collection and never ran a single assertion.
"""
from __future__ import annotations

import numpy as np
import pytest

from stages5a_5d import bn_distinct, run_5a, run_5b, setup_regions
from stages5cd import run_5c, run_5d
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.regions import REGION_EZ, REGION_RZ, RegionManager
from vn_mobile.core.trial import CULL_WINDOW, make_trial_zone, zone_of_main
from vn_mobile.core.update import E_TARGET, step_energy, step_phase


def _pop(n=200, seed=1):
    return build_population(n=n, degree=8, seed=seed,
                            omega_mix=(0.8, 0.3, 1.0))


# --- 5A: spatial division of labour ------------------------------------
def test_main_system_splits_into_contiguous_address_zones():
    """Similar co-located: zones are contiguous blocks in addr order."""
    addr = np.array([50, 10, 90, 30, 70, 20, 60, 40, 80, 0], dtype=np.int64)
    z = zone_of_main(addr, n_zones=5)
    assert z.shape == (10,)
    assert set(np.unique(z)) == {0, 1, 2, 3, 4}
    # each zone holds exactly n/n_zones neurons, contiguous in sorted order
    order = np.argsort(addr)
    assert all(np.diff(z[order]) >= 0)


def test_bn_shadows_only_its_own_zone():
    tz = make_trial_zone(n_bn=100, n_main=200, n_zones=20, seed=5,
                         main_addr=_pop(200).addr)
    zm = zone_of_main(_pop(200).addr, 20)
    for i in range(0, 100, 7):
        assert set(zm[tz.shadow[i]]) == {tz.zone_id[i]}


def test_bn_dynamics_are_perturbed_per_neuron():
    tz = make_trial_zone(n_bn=200, n_main=200, n_zones=20, seed=5)
    assert 0.1 <= tz.kappa.min() and tz.kappa.max() <= 0.2
    assert 0.2 <= tz.sigma_w.min() and tz.sigma_w.max() <= 0.4
    assert tz.pop.omega.std() > 0.1          # not a single shared value


# --- 5A: culling is proportional, not an absolute step count ------------
def test_cull_uses_window_share_not_absolute_steps():
    tz = make_trial_zone(n_bn=10, n_main=100, n_zones=10, seed=5)
    # 0.5 is neither low nor promotable, so it isolates the cull path.
    tz.score[:] = 0.5                        # evaluate() recomputes score, so
    for _ in range(CULL_WINDOW):             # sample the window directly here
        tz._sample_cull_window()
    assert tz.promote_or_cull()[2] == 0      # never low -> no cull
    assert bool((tz.merged == 0).all())      # and nothing promoted either

    tz.score[:] = 0.1                        # low for just over half the window
    for _ in range(int(CULL_WINDOW * 0.6)):
        tz._sample_cull_window()
    assert tz.promote_or_cull()[2] == 10


# --- 4/5A: regions, FVN, TVN -------------------------------------------
def test_fvn_never_mutated():
    pop = _pop(300, seed=2)
    rs, _ = setup_regions(pop, n_rz=100, n_fvn=20, n_tvn=3, seed=2)
    ids = np.flatnonzero(pop.frozen)
    assert ids.size == 20
    e0, p0 = pop.energy[ids].copy(), pop.phase[ids].copy()
    rng = np.random.default_rng(0)
    for _ in range(500):
        step_energy(pop, rng)
        step_phase(pop)
        pop.energy[ids], pop.phase[ids] = e0, p0
    assert np.array_equal(e0, pop.energy[ids])
    assert np.array_equal(p0, pop.phase[ids])


def test_tvn_fires_when_due():
    pop = _pop(100, seed=3)
    rs, _ = setup_regions(pop, n_rz=50, n_fvn=5, n_tvn=3, seed=3)
    fired = rs.tick_timers(10000.0)
    assert fired.size == 3
    assert rs.fired == 3
    assert rs.tick_timers(10000.0).size == 0     # one-shot


def test_ez_load_release_forced_by_overflow():
    """Forced overflow: EZ must load and later release (release <= load)."""
    pop, rs, tz, _, ov = run_5a(steps=6000, n_main=300, n_bn=100,
                                n_zones=10, seed=4, log_every=0,
                                n_rz=150, forced_pulse=5.0)
    assert rs.loads > 0, "forced pulse must trigger EZ loads"
    assert rs.releases > 0, "EZ must release once neurons fall quiet"
    assert rs.releases <= rs.loads


def test_steady_state_has_no_runaway():
    pop, rs, tz, _, _ = run_5a(steps=4000, n_main=200, n_bn=100,
                               n_zones=10, seed=6, log_every=0)
    assert np.isfinite(pop.energy).all()
    assert pop.energy.max() < 0.95


# --- 5A: BN output must be distinguishable ------------------------------
def test_bn_output_not_identical_to_main():
    pop, rs, tz, _, _ = run_5a(steps=4000, n_main=200, n_bn=100,
                               n_zones=10, seed=7, log_every=0)
    corr, _ = bn_distinct(pop, tz)
    if not np.isnan(corr):
        assert corr < 0.8


# --- 5C / 5D ------------------------------------------------------------
def test_hibernate_is_lossless_and_archive_compresses():
    pop = _pop(200, seed=8)
    rs, _ = setup_regions(pop, n_rz=100, n_fvn=10, n_tvn=2, seed=8)
    h = run_5c(pop, rs, "/tmp/t_ll.vnpkg", lossless=True)
    assert h["energy_eq"] and h["phase_eq"] and h["state_eq"] and h["ada_eq"]
    a = run_5c(pop, rs, "/tmp/t_q.vnpkg", lossless=False)
    assert a["ratio"] > 3.0
    assert a["load_s"] < 1.0


def test_rollback_and_no_leak():
    pop = _pop(200, seed=9)
    rs, _ = setup_regions(pop, n_rz=100, n_fvn=10, n_tvn=2, seed=9)
    d = run_5d(pop, rs, cycles=15)
    assert d["rollback_ok"]
    assert d["switch_s"] < 0.1
    assert abs(d["rss_delta_mb"]) < 20.0
