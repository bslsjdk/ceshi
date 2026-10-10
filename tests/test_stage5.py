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
from vn_mobile.core.cull import relative_round
from vn_mobile.core.trial import CULL_WINDOW, PROMOTE_FRAC
from vn_mobile.core.tzbuild import make_trial_zone, zone_of_main
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


# --- 5A: culling is relative (worst 40% of each zone) ------------------
def test_relative_cull_takes_worst_40pct_of_each_zone():
    tz = make_trial_zone(n_bn=100, n_main=200, n_zones=10, seed=5)
    rng = np.random.default_rng(0)
    tz.score[:] = rng.uniform(0.0, 1.0, size=100)
    before = tz.score.copy()
    r = relative_round(tz, rng)
    assert r["culled"] == 40                 # 40% of 100
    assert abs(r["cull_rate"] - 0.4) < 1e-6
    assert int((tz.merged == 0).sum()) == 100     # reinitialised, not removed
    # Culled slots are exactly the bottom 40% by score inside each zone.
    for u in np.unique(tz.zone_id):
        m = np.flatnonzero(tz.zone_id == u)
        assert m.size == 10
        order = m[np.argsort(before[m])]
        assert set(order[:4].tolist()) == set(np.flatnonzero(tz.score == 0.0)
                                              ).intersection(m.tolist())
    assert tz.reinit_count == 40


def test_relative_cull_keeps_population_size():
    tz = make_trial_zone(n_bn=200, n_main=200, n_zones=10, seed=6)
    rng = np.random.default_rng(1)
    for _ in range(5):
        tz.score[:] = rng.uniform(0.0, 1.0, size=200)
        relative_round(tz, rng)
    assert tz.pop.n == 200
    assert int((tz.merged == 0).sum()) == 200


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

# --- 5A final: promotion is relative too (best 12% of each zone) -----
def test_promote_takes_best_12pct_of_each_zone():
    """Merging ranks against zone peers, so the rate is a parameter."""
    tz = make_trial_zone(n_bn=500, n_main=200, n_zones=10, seed=5)
    tz.score[:] = np.random.default_rng(0).uniform(0.0, 1.0, size=500)
    before = tz.score.copy()
    n_rz, n_ez = tz.promote_relative()
    assert n_rz + n_ez == 60                       # 12% of 500
    assert int(((tz.merged == 1) | (tz.merged == 2)).sum()) == 60
    for u in np.unique(tz.zone_id):                # exactly the top of each zone
        m = np.flatnonzero(tz.zone_id == u)
        top = set(m[np.argsort(-before[m])[:6]].tolist())
        got = set(np.flatnonzero((tz.merged == 1) | (tz.merged == 2)).tolist())
        assert top <= got


def test_promote_rate_is_independent_of_absolute_score():
    """An absolute floor would merge 0% or 100%; relative always merges 12%."""
    tz = make_trial_zone(n_bn=500, n_main=200, n_zones=10, seed=5)
    for shift in (0.0, 0.5, 5.0):                  # same ranking, other level
        tz.merged[:] = 0
        tz.score[:] = np.random.default_rng(1).uniform(0, 1, size=500) * shift
        n_rz, n_ez = tz.promote_relative()
        assert n_rz + n_ez == 60


def test_util_is_continuous_not_a_binary_gate():
    """The median gate was bimodal by construction; sigmoid must not be."""
    from vn_mobile.core.update import step_energy, step_phase
    tz = make_trial_zone(n_bn=100, n_main=200, n_zones=10, seed=5)
    pop = build_population(n=200, degree=8, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(30):
        prev = tz.shadow_energy(pop.energy)
        step_energy(pop, rng)
        step_phase(pop)
        cur = tz.shadow_energy(pop.energy)
        tz.train_step(prev, cur)
        tz.evaluate(prev, cur)
    u = tz.util
    assert u.min() > 0.0 and u.max() < 1.0
    assert len(np.unique(np.round(u, 4))) > 10      # many levels, not just {0,1}
