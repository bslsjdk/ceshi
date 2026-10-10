"""Final 100k-step validation for stage 5A (final spec).

Records the agreed criteria plus raw sampling of time and memory.
"""
from __future__ import annotations

import json
import resource
import time

import numpy as np

from stages5a_5d import (bn_distinct, run_5a, run_5b, score_report,
                         setup_regions)
from stages5cd import run_5c, run_5d
from vn_mobile.core.neuron import build_population

t0 = time.perf_counter()
pop, rs, tz, hist, ov = run_5a(steps=100000, log_every=20000)
el = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
arr = sum(v.nbytes for v in vars(pop).values() if isinstance(v, np.ndarray))
arr += sum(v.nbytes for v in vars(tz).values() if isinstance(v, np.ndarray))

corr, bstd = bn_distinct(pop, tz)
sr, b5 = score_report(tz), run_5b(tz)
out = dict(
    steps=100000, wall_s=round(el, 1), us_per_step=round(el / 100000 * 1e6, 1),
    rss_mb=round(rss, 1), arrays_mb=round(arr / 1048576, 3),
    nan=bool(not np.isfinite(pop.energy).all()),
    e_max=round(float(pop.energy.max()), 4),
    e_mean=round(float(pop.energy.mean()), 4),
    e_std=round(float(pop.energy.std()), 4),
    top_frac=round(float((pop.energy > 0.95).mean()), 6),
    bn_corr=round(corr, 4), bn_out_std=round(bstd, 6),
    loads=rs.loads, releases=rs.releases, fired=rs.fired,
    cull_rate=tz.cull_log[-1]["cull_rate"], cull_rounds=tz.cull_rounds,
    cull_events=tz.cull_events, reinit=tz.reinit_count,
    bn_alive=int((tz.merged == 0).sum()),
    hibernate=run_5c(pop, rs, "/tmp/f_ll.vnpkg", True),
    archive=run_5c(pop, rs, "/tmp/f_q.vnpkg", False),
    d5=run_5d(pop, rs, cycles=20),
    score=sr, stage5b=b5, history=hist,
)
print(json.dumps(out, indent=2, default=str))
open("run_100k.json", "w").write(json.dumps(out, indent=2, default=str))
