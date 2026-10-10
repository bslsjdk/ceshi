"""Stage 5A-5D verification harness.

Rewritten against the real RegionManager API.  The previous version called
``make_regions`` / ``fire_timers`` / ``maybe_load``, none of which exist, so it
could not even be imported -- the "46 passed" reported for it was not real.
"""
from __future__ import annotations

import numpy as np

from vn_mobile.core.neuron import DIM, MODE_SAFE, build_population
from vn_mobile.core.regions import REGION_RZ, RegionManager, split_resident
from vn_mobile.core.trial import make_trial_zone, tick_trial, zone_of_main
from vn_mobile.core.update import E_TARGET, step_energy, step_phase, step_state
from vn_mobile.io import vnpkg
from stages5cd import run_5c, run_5d

# Prediction horizon in BN samples (one sample every 10 ticks).  Predicting the
# very next sample is too easy: every BN nails it, scores saturate and the
# merge rate collapses to 0%.  5 samples = 50 ticks makes the task hard enough
# that BN quality actually separates.
HORIZON = 5


def setup_regions(pop, n_rz: int = 300, n_fvn: int = 50, n_tvn: int = 5,
                  seed: int = 0) -> tuple[RegionManager, np.ndarray]:
    """RZ with frozen FVN, the rest as loadable EVN, plus a few timers."""
    rs = RegionManager(pop)
    rz_ids, other = split_resident(pop.n, n_rz, seed=seed)
    rs.mark_fvn(rz_ids[:n_fvn], MODE_SAFE)
    rs.mark_evn(other)
    tv = rz_ids[n_fvn:n_fvn + n_tvn]
    rs.set_timer(tv, np.linspace(500.0, 5000.0, tv.size).astype(np.float32))
    return rs, rz_ids


def run_5a(steps: int = 100000, n_main: int = 1000, n_bn: int = 1000,
           n_zones: int = 20, seed: int = 0, log_every: int = 10000,
           n_rz: int = 300, forced_pulse: float = 0.0):
    """``forced_pulse`` > 0 holds RZ neurons hot so EZ overflow fires.

    It is applied every tick, not on pulse ticks: a one-shot injection decays
    below the overflow band long before the next 1000-tick check.
    """
    rng = np.random.default_rng(seed)
    pop = build_population(n=n_main, degree=8, seed=seed,
                           omega_mix=(0.8, 0.3, 1.0))
    rs, rz_ids = setup_regions(pop, n_rz=n_rz, seed=seed)
    tz = make_trial_zone(n_bn=n_bn, n_main=n_main, n_zones=n_zones,
                         seed=seed + 3, main_addr=pop.addr)
    inj_e = np.zeros(n_main, dtype=np.float32)
    inj_h = np.zeros((n_main, DIM), dtype=np.float32)
    fz = np.flatnonzero(pop.frozen)
    # Difficulty gradient: each zone is driven at a different amplitude so some
    # are genuinely harder to predict.  Without it every BN faces the same
    # task and scores identically.
    zm = zone_of_main(pop.addr, n_zones)
    amp = rng.uniform(0.2, 1.5, size=n_zones).astype(np.float32)
    hot = np.setdiff1d(rz_ids, np.flatnonzero(pop.frozen))[:20]
    buf = np.zeros((HORIZON, n_bn), dtype=np.float32)
    hpos = filled = 0
    hist, mse0, mse_last, ov_total = [], None, float("nan"), 0

    for t in range(steps):
        rs.tick_timers(float(t))
        fe, fp, fs = (pop.energy[fz].copy(), pop.phase[fz].copy(),
                      pop.state[fz].copy())
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        step_state(pop, inj_h)
        # FVN slots must come out bit-identical.
        pop.energy[fz], pop.phase[fz], pop.state[fz] = fe, fp, fs
        if t % 1000 == 0:
            tz.promote_or_cull()
            rs.update_threshold(E_TARGET)
            ov = rs.overflow_ids(E_TARGET)
            ov_total += int(ov.size)
            if ov.size:
                rs.load(ov.size)
            rs.release()
        # Pulse every 200 ticks: with no input the state layer decays to 0 and
        # there is nothing for the BN output to be compared against.
        inj_e[:] = 0.0
        inj_h[:] = 0.0
        if t % 200 == 0:
            pick = rng.choice(n_main, size=n_main // 20, replace=False)
            inj_e[pick] = amp[zm[pick]]
            inj_h[pick] = 1.0
        elif t % 10:
            inj_e[:] = rng.normal(0.0, 0.01, size=n_main) * amp[zm]
        if forced_pulse > 0.0:
            inj_e[hot] = forced_pulse
        if t % 10 == 0:
            cur = tz.shadow_energy(pop.energy)
            buf[hpos] = cur
            hpos = (hpos + 1) % HORIZON
            filled += 1
            if filled > HORIZON:
                old = buf[hpos]          # oldest sample = prediction input
                base = float(np.mean(cur ** 2))
                loss = tz.train_step(old, cur)
                if mse0 is None:
                    mse0 = loss
                mse_last = loss
                tz.evaluate(old, cur, baseline_err=base)
        tick_trial(tz, rng)
        if log_every and (t + 1) % log_every == 0:
            hist.append(dict(step=t + 1, R=_R_live(pop),
                             e_mean=float(pop.energy.mean()),
                             e_std=float(pop.energy.std()),
                             loads=rs.loads, releases=rs.releases,
                             fired=rs.fired))
    tz.mse0, tz.mse_last = mse0, mse_last
    return pop, rs, tz, hist, ov_total


def _R_live(pop) -> float:
    return float(np.hypot(np.cos(pop.phase).mean(), np.sin(pop.phase).mean()))


def bn_distinct(pop, tz):
    """Main-system reading vs BN reading; corr must not be 1.000."""
    main_out = (pop.scale[:, None] * pop.state * pop.energy[:, None]).sum(axis=1)
    bn_out = tz.forward(pop.energy)
    k = min(len(main_out), len(bn_out))
    a, b = main_out[:k], bn_out[:k]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), float(np.std(b))
    return float(np.corrcoef(a, b)[0, 1]), float(np.std(b))


def run_5b(tz) -> dict:
    """Cumulative merge/cull read off the live merged flags."""
    m = tz.merged
    return dict(merged_rz=int((m == 2).sum()), merged_ez=int((m == 1).sum()),
                culled=int((m == -1).sum()),
                merge_rate=round(float(((m == 1) | (m == 2)).mean()), 4),
                cull_rate=round(float((m == -1).mean()), 4))


def score_report(tz) -> dict:
    """Stage 5A rerun criteria: spread, division of labour, MSE drop."""
    drop = float("nan")
    if tz.mse0:
        drop = (tz.mse0 - tz.mse_last) / tz.mse0
    return dict(score_std=round(float(tz.score.std()), 4),
                score_mean=round(float(tz.score.mean()), 4),
                mse_drop=round(drop, 4), **tz.zone_stats())


if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    print(f"[5A] running {steps} steps ...")
    pop, rs, tz, hist, ov_total = run_5a(steps=steps)
    for h in hist:
        print("   ", h)
    corr, bstd = bn_distinct(pop, tz)
    print(f"[5A] bn corr={corr:.4f} bn_out_std={bstd:.6f}")
    print(f"[5A] regions {rs.stats()} overflow_hits={ov_total}")
    print(f"[5A] {score_report(tz)}")
    print(f"[5B] {run_5b(tz)}")
    print(f"[5C] hibernate {run_5c(pop, rs, '/tmp/ll.vnpkg', True)}")
    print(f"[5C] archive   {run_5c(pop, rs, '/tmp/q.vnpkg', False)}")
    print(f"[5D] {run_5d(pop, rs, cycles=20)}")
