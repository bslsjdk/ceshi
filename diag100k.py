"""100k-step 5A closing run: criteria table, per-step cost, memory.

Stage 5A closing replaced the dropped ``between > within`` criterion with two
direct ones:

    between_zone_std > 0.01                     zones really differ
    corr(zone_mean_score, -zone_horizon) > 0.7  longer look-ahead scores lower

The run also records where the between-zone spread actually comes from: every
zone is driven at its own amplitude, so if score tracks amplitude rather than
horizon, the spread comes from the random drive level and not from the task
gradient the horizon was meant to create.
"""
import json
import resource
import time

import numpy as np

from stages5a_5d import bn_distinct, run_5a, run_5b, score_report
from stages5cd import run_5c, run_5d

STEPS = 100000
t0 = time.perf_counter()
pop, rs, tz, hist, ov = run_5a(steps=STEPS, log_every=0)
el = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
z = tz.zone_id
alive = tz.merged == 0
s = tz.score.astype(np.float64)


def zone_mean(vals, mask=None):
    v = np.asarray(vals, dtype=np.float64)
    zz = np.asarray(z, dtype=np.int64)
    if mask is not None:
        v, zz = v[mask], zz[mask]
    ids = np.unique(zz)
    return np.array([v[zz == u].mean() for u in ids])


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


means = zone_mean(s, alive)
hz = zone_mean(tz.hz_bn, alive)
amp = getattr(tz, "zone_amp", None)
amp_z = np.asarray(amp, dtype=np.float64) if amp is not None else None
corr, bstd = bn_distinct(pop, tz)
sr, b5 = score_report(tz), run_5b(tz)
arr = sum(v.nbytes for v in vars(pop).values() if isinstance(v, np.ndarray))
arr += sum(v.nbytes for v in vars(tz.pop).values()
           if isinstance(v, np.ndarray))
out = dict(
    wall_s=round(el, 1), us_per_step=round(el / STEPS * 1e6, 1),
    rss_mb=round(rss, 1), arrays_mb=round(arr / 1048576.0, 3),
    nan=bool(not np.isfinite(pop.energy).all()),
    e_max=round(float(pop.energy.max()), 4),
    e_mean=round(float(pop.energy.mean()), 4),
    e_std=round(float(pop.energy.std()), 4),
    top_frac=round(float((pop.energy > 0.95).mean()), 6),
    bn_corr=round(corr, 4),
    loads=rs.loads, releases=rs.releases, fired=rs.fired,
    cull_rate=tz.cull_log[-1]["cull_rate"] if tz.cull_log else None,
    cull_rounds=tz.cull_rounds, reinit=tz.reinit_count,
    bn_alive=int(alive.sum()),
    merge_rate=b5["merge_rate"], cull_rate_terminal=b5["cull_rate"],
    score_std=sr["score_std"], score_mean=sr["score_mean"],
    mse_drop=sr["mse_drop"],
    between_zone_std=sr["between_zone_std"],
    within_zone_std=sr["within_zone_std"],
    horizon_corr=sr["horizon_corr"],
    horizon_corr_local=round(_corr(means, -hz), 4),
    amp_corr=(round(_corr(means, amp_z), 4) if amp_z is not None else None),
    hz_range=[int(hz.min()), int(hz.max())],
    n_zones=int(len(means)),
    hibernate=run_5c(pop, rs, "/tmp/d_ll.vnpkg", True),
    archive=run_5c(pop, rs, "/tmp/d_q.vnpkg", False),
    d5=run_5d(pop, rs, cycles=20),
)
print(json.dumps(out, indent=2, default=str))
with open("diag100k.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
