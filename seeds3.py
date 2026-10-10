"""Task 1.1: is the between-zone spread real, or one lucky seed?

A single seed gave between_zone_std = 0.0114 against a 0.01 floor - too close
to call.  This script runs three seeds and reports each value plus the mean,
so the criterion can be judged on a distribution instead of a point sample.

It also prints, per seed, the two quantities the spread could come from:
zone horizon (the intended gradient) and zone drive amplitude (a confound).
"""
from __future__ import annotations

import json
import time

import numpy as np

from stages5a_5d import run_5a, score_report

STEPS = 100000
SEEDS = (0, 1, 2)


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def main() -> None:
    rows = []
    for sd in SEEDS:
        t0 = time.perf_counter()
        pop, rs, tz, hist, ov = run_5a(steps=STEPS, seed=sd, log_every=0)
        el = time.perf_counter() - t0
        sr = score_report(tz)
        z = np.asarray(tz.zone_id, dtype=np.int64)
        alive = tz.merged == 0
        s = np.asarray(tz.score, dtype=np.float64)
        ids = np.unique(z[alive])
        means = np.array([s[(z == u) & alive].mean() for u in ids])
        hz = np.array([np.asarray(tz.hz_bn, np.float64)[(z == u) & alive].mean()
                       for u in ids])
        amp = getattr(tz, "zone_amp", None)
        # zone_amp is one value per zone, not per BN.
        amp_m = (np.asarray(amp, np.float64)[ids] if amp is not None
                 and len(np.asarray(amp)) == len(ids) else None)
        rows.append(dict(
            seed=sd, wall_s=round(el, 1),
            us_per_step=round(el / STEPS * 1e6, 1),
            between_zone_std=sr["between_zone_std"],
            within_zone_std=sr["within_zone_std"],
            score_std=sr["score_std"], score_mean=sr["score_mean"],
            mse_drop=sr["mse_drop"],
            horizon_corr=_corr(means, -hz),
            amp_corr=_corr(means, amp_m) if amp_m is not None else None,
            nan=bool(not np.isfinite(pop.energy).all()),
            e_mean=round(float(pop.energy.mean()), 4),
        ))
        print(json.dumps(rows[-1]), flush=True)

    vals = [r["between_zone_std"] for r in rows]
    out = dict(
        seeds=list(SEEDS), steps=STEPS, rows=rows,
        between_zone_std_mean=round(float(np.mean(vals)), 4),
        between_zone_std_min=round(float(np.min(vals)), 4),
        between_zone_std_max=round(float(np.max(vals)), 4),
        all_above_001=bool(all(v > 0.01 for v in vals)),
        mean_above_0012=bool(np.mean(vals) > 0.012),
    )
    print(json.dumps(out, indent=2))
    with open("seeds3.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
