"""Stage 5A-5D verification harness.

Rewritten against the real RegionManager API.  The previous version called
``make_regions`` / ``fire_timers`` / ``maybe_load``, none of which exist, so it
could not even be imported -- the "46 passed" reported for it was not real.
"""
from __future__ import annotations

import numpy as np

from vn_mobile.core.neuron import DIM, MODE_SAFE, build_population
from vn_mobile.core.cull import CULL_ROUND_STEPS, relative_round
from vn_mobile.core.trial import PROMOTE_FRAC
from vn_mobile.core.regions import REGION_RZ, RegionManager, split_resident
from vn_mobile.core.tzbuild import make_trial_zone, tick_trial, zone_of_main
from vn_mobile.core.update import E_TARGET, step_energy, step_phase, step_state
from vn_mobile.io import vnpkg
from stages5cd import run_5c, run_5d

# Prediction lag in BN samples (one sample every 10 ticks).  Predicting the very
# next sample is too easy: every BN nails it, scores saturate and the merge rate
# collapses to 0%.  Each zone now predicts a DIFFERENT number of ticks ahead --
# zone 0 -> 25 ticks, zone 19 -> 120 -- so difficulty, and therefore score,
# varies between zones and not only within them.
SAMPLE_EVERY = 10


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
           n_rz: int = 300, forced_pulse: float = 0.0,
           lr: float | None = None, amp_lo: float = 0.2, amp_hi: float = 1.5,
           promote_mode: str = "terminal", util_norm: str = "zone",
           hz_lo: float = 0.5, hz_step: float = 0.1):
    """``promote_mode``: ``terminal`` culls every 2000-step round and merges the
    best 12% once at the end, so the terminal merge rate equals the parameter.
    ``rolling`` merges every round too; merged slots leave for good while culled
    slots return after reinit, so the terminal rate climbs regardless.
    """
    # forced_pulse > 0 holds RZ neurons hot so EZ overflow fires; it is applied
    # every tick because a one-shot injection decays below the overflow band
    # long before the next 1000-tick check.
    rng = np.random.default_rng(seed)
    pop = build_population(n=n_main, degree=8, seed=seed,
                           omega_mix=(0.8, 0.3, 1.0))
    rs, rz_ids = setup_regions(pop, n_rz=n_rz, seed=seed)

    tz = make_trial_zone(n_bn=n_bn, n_main=n_main, n_zones=n_zones,
                         seed=seed + 3, main_addr=pop.addr,
                         util_norm=util_norm, hz_lo=hz_lo, hz_step=hz_step)
    if lr is not None:
        tz.lr = float(lr)
    tz.promote_round_log = []
    inj_e = np.zeros(n_main, dtype=np.float32)
    inj_h = np.zeros((n_main, DIM), dtype=np.float32)
    fz = np.flatnonzero(pop.frozen)
    # Difficulty gradient: each zone is driven at a different amplitude so
    # some are genuinely harder to predict.
    zm = zone_of_main(pop.addr, n_zones)
    amp = rng.uniform(amp_lo, amp_hi, size=n_zones).astype(np.float32)
    tz.zone_amp = amp          # diagnostics: is zone spread drive or horizon?
    hot = np.setdiff1d(rz_ids, np.flatnonzero(pop.frozen))[:20]
    max_hz = int(tz.hz_bn.max()) + 1      # ring must exceed the longest lag
    buf = np.zeros((max_hz, n_bn), dtype=np.float32)
    bn_idx = np.arange(n_bn)
    hpos = filled = 0
    hist, mse0, mse_last, ov_total = [], None, float("nan"), 0
    cull_log = []

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
            rs.update_threshold(E_TARGET)
            ov = rs.overflow_ids(E_TARGET)
            ov_total += int(ov.size)
            if ov.size:
                rs.load(ov.size)
            rs.release()
        if t > 0 and t % CULL_ROUND_STEPS == 0:
            pf = PROMOTE_FRAC if promote_mode == "rolling" else 0.0
            cull_log.append(relative_round(tz, rng, promote_frac=pf))
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
        if t % SAMPLE_EVERY == 0:
            cur = tz.shadow_energy(pop.energy)
            buf[hpos] = cur
            hpos = (hpos + 1) % max_hz
            filled += 1
            if filled > max_hz:
                # Per-BN lag: buf[(hpos-1-hz) % max_hz] is that BN's input.
                old = buf[(hpos - 1 - tz.hz_bn) % max_hz, bn_idx]
                loss = tz.train_step(old, cur)
                if mse0 is None:
                    mse0 = loss
                mse_last = loss
                tz.evaluate(old, cur)
        tick_trial(tz, rng)
        if log_every and (t + 1) % log_every == 0:
            hist.append(dict(step=t + 1, R=_R_live(pop),
                             e_mean=float(pop.energy.mean()),
                             e_std=float(pop.energy.std()),
                             loads=rs.loads, releases=rs.releases,
                             fired=rs.fired))
    # Promotion is a single terminal decision: promoting every round is
    # cumulative, so the rate marches to ~50% regardless of the threshold.
    tz.promote_or_cull()
    tz.mse0, tz.mse_last = mse0, mse_last
    tz.cull_log = cull_log
    return pop, rs, tz, hist, ov_total


def _R_live(pop) -> float:
    """Kuramoto order parameter: 1 = fully locked, 0 = incoherent."""
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
    """Merge/cull read off the live merged flags, both accounting senses.

    ``merge_rate`` is the terminal fraction carrying a merge flag; it equals
    PROMOTE_FRAC in terminal mode.  ``merge_rate_per_round`` is the mean over
    selection rounds of (promoted / alive-at-round-start).
    """
    m = tz.merged
    rounds = tz.promote_round_log or []
    per = [p / a for p, a in rounds if a]
    return dict(merged_rz=int((m == 2).sum()), merged_ez=int((m == 1).sum()),
                culled=int((m == -1).sum()),
                merge_rate=round(float(((m == 1) | (m == 2)).mean()), 4),
                cull_rate=round(float((m == -1).mean()), 4),
                promote_events=tz.promote_events,
                merge_rate_per_round=(round(float(np.mean(per)), 4)
                                      if per else None),
                rounds_with_promotion=len(per))


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
