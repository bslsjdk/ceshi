"""100k-step run with a between/within breakdown that separates slots
reinitialised in the final selection round from slots that had time to mature.
"""
import json, numpy as np, time, resource
from stages5a_5d import run_5a, run_5b, score_report, bn_distinct
from stages5cd import run_5c, run_5d

t0 = time.perf_counter()
pop, rs, tz, hist, ov = run_5a(steps=100000, log_every=0)
el = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
z = tz.zone_id
alive = tz.merged == 0
s = tz.score.astype(np.float64)


def bw(mask):
    ids = [u for u in np.unique(z[mask]) if (mask & (z == u)).sum() > 1]
    if not ids:
        return float("nan"), float("nan")
    means = np.array([s[mask & (z == u)].mean() for u in ids])
    return float(means.std()), float(np.mean([s[mask & (z == u)].std()
                                              for u in ids]))


bt_a, wt_a = bw(alive)
mature = alive & (tz.samples >= 100)
bt_m, wt_m = bw(mature) if mature.sum() > 20 else (float("nan"),) * 2
fresh = alive & (tz.samples < 100)
corr, bstd = bn_distinct(pop, tz)
sr, b5 = score_report(tz), run_5b(tz)

# component breakdown
parts = {}
if tz.last_parts:
    for k, v in tz.last_parts.items():
        v = np.asarray(v, float)
        ids = [u for u in np.unique(z[alive]) if (alive & (z == u)).sum() > 1]
        parts[k] = dict(
            std=round(float(v.std()), 4),
            within=round(float(np.mean([v[alive & (z == u)].std()
                                        for u in ids])), 4),
            between=round(float(np.array([v[alive & (z == u)].mean()
                                          for u in ids]).std()), 4))
out = dict(
    wall_s=round(el, 1), us_per_step=round(el / 100000 * 1e6, 1),
    rss_mb=round(rss, 1),
    nan=bool(not np.isfinite(pop.energy).all()),
    e_max=round(float(pop.energy.max()), 4),
    e_mean=round(float(pop.energy.mean()), 4),
    e_std=round(float(pop.energy.std()), 4),
    top_frac=round(float((pop.energy > 0.95).mean()), 6),
    bn_corr=round(corr, 4),
    loads=rs.loads, releases=rs.releases, fired=rs.fired,
    cull_rate=tz.cull_log[-1]["cull_rate"], cull_rounds=tz.cull_rounds,
    reinit=tz.reinit_count, bn_alive=int(alive.sum()),
    merge_rate=b5["merge_rate"], promote_events=b5["promote_events"],
    merged_rz=b5["merged_rz"], merged_ez=b5["merged_ez"],
    score_std=sr["score_std"], score_mean=sr["score_mean"],
    mse_drop=sr["mse_drop"],
    between_all=round(bt_a, 4), within_all=round(wt_a, 4),
    bw_all=round(bt_a / wt_a, 3),
    between_mature=round(bt_m, 4), within_mature=round(wt_m, 4),
    bw_mature=round(bt_m / wt_m, 3) if wt_m == wt_m else None,
    n_mature=int(mature.sum()), n_fresh=int(fresh.sum()),
    parts=parts,
    hibernate=run_5c(pop, rs, "/tmp/d_ll.vnpkg", True),
    archive=run_5c(pop, rs, "/tmp/d_q.vnpkg", False),
    d5=run_5d(pop, rs, cycles=20),
)
print(json.dumps(out, indent=2, default=str))
open("diag100k.json", "w").write(json.dumps(out, indent=2, default=str))
