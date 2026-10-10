"""Task 1.2: profile before optimising.

The spec says: don't guess, profile.  This dumps the top hotspots of a real
run_5a window so the two proposed fixes (sample every 20 instead of 10, batch
the TZ tick) can be checked against where time actually goes.
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import time

import numpy as np

from stages5a_5d import run_5a
from vn_mobile.core.trial import SAMPLE_EVERY

STEPS = 6000


def main() -> None:
    run_5a(steps=200, log_every=0)          # warm caches / lazy init
    pr = cProfile.Profile()
    pr.enable()
    run_5a(steps=STEPS, log_every=0)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
    print(s.getvalue())

    t0 = time.perf_counter()
    run_5a(steps=STEPS, log_every=0)
    el = time.perf_counter() - t0
    print(json.dumps(dict(sample_every=SAMPLE_EVERY, steps=STEPS,
                          us_per_step=round(el / STEPS * 1e6, 1))))


if __name__ == "__main__":
    main()
