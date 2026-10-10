"""Separate the noise-block win from the sample/tz wins.

The previous optbench compared (10,1) against (20,2) on a code base that already
had the cached noise block, so the block's own contribution was folded into the
baseline and invisible.  ``run_5a(noise_live=True)`` restores the per-tick draw,
which makes the three changes independently measurable.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

STEPS = 4000
REPS = 3

WORKER = r'''
import os, time, statistics, json
import vn_mobile.core.trial as T
import vn_mobile.core.tzbuild as TB
import stages5a_5d as S
T.SAMPLE_EVERY = int(os.environ["SE"]); T.TZ_EVERY = int(os.environ["TE"])
TB.SAMPLE_EVERY = T.SAMPLE_EVERY
S.SAMPLE_EVERY = T.SAMPLE_EVERY; S.TZ_EVERY = T.TZ_EVERY
NL = os.environ["NL"] == "1"
STEPS = int(os.environ["STEPS"]); REPS = int(os.environ["REPS"])
S.run_5a(steps=200, log_every=0)
xs = []
for _ in range(REPS):
    t0 = time.perf_counter()
    S.run_5a(steps=STEPS, log_every=0, noise_live=NL)
    xs.append((time.perf_counter() - t0) / STEPS * 1e6)
print(json.dumps(round(statistics.median(xs), 1)))
'''


def run(se: int, te: int, nl: bool) -> float:
    env = dict(os.environ, SE=str(se), TE=str(te), NL="1" if nl else "0",
               STEPS=str(STEPS), REPS=str(REPS))
    p = subprocess.run([sys.executable, "-c", WORKER], capture_output=True,
                       text=True, env=env,
                       cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
    if p.returncode != 0:
        raise SystemExit(p.stderr[-800:])
    return float(p.stdout.strip().splitlines()[-1])


def main() -> None:
    base = run(10, 1, True)        # original: sample 10, tz every tick, live
    out = dict(
        baseline_sample10_tz1_livenoise=base,
        plus_noise_block=run(10, 1, False),
        plus_sample20=run(20, 1, False),
        plus_tz2=run(20, 2, False),
        final_all_three=run(20, 2, False),
        target_us_per_step=250.0,
    )
    out["final_all_three"] = out["plus_tz2"]
    out["total_saved_us"] = round(base - out["final_all_three"], 1)
    out["meets_target"] = bool(out["final_all_three"] < 250)
    print(json.dumps(out, indent=2))
    with open("optbench.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
