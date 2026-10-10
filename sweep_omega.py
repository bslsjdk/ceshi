"""Sweep the Kuramoto frequency spread and report where the field sits.

Stage 3 found omega_std=0.1 locks (R ~ 0.975, frozen) and 1.0 is incoherent
(R ~ 0.04).  This finds the middle.  Records R at three checkpoints, the
range of R inside each 1000-step window, and whether R stops moving.
"""
from __future__ import annotations

import json

import numpy as np

from vn_mobile.core.field import Field
from vn_mobile.core.neuron import build_population


def sweep(n: int = 1000, steps: int = 60_000, seed: int = 0,
          grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)) -> list[dict]:
    out = []
    for w in grid:
        pop = build_population(n=n, seed=seed, omega_std=w)
        f = Field(pop, seed=seed)
        marks, series = {}, []
        for i in range(steps):
            f.tick()
            if (i + 1) % 1000 == 0:
                r = f.measure().sync_r
                series.append(r)
                if (i + 1) in (10_000, 30_000, 60_000):
                    marks[(i + 1)] = r
        arr = np.asarray(series)
        wins = arr[:len(arr) // 10 * 10].reshape(-1, 10)
        rng_max = float((wins.max(1) - wins.min(1)).max())
        frozen = bool(len(arr) > 1 and abs(arr[-1] - arr[-2]) < 1e-4)
        st = f.measure()
        row = {
            "omega_std": w,
            "R_at_10k": round(marks.get(10_000, float("nan")), 4),
            "R_at_30k": round(marks.get(30_000, float("nan")), 4),
            "R_at_60k": round(marks.get(60_000, float("nan")), 4),
            "R_range_1k": round(rng_max, 4),
            "frozen": frozen,
            "e_mean": round(st.energy_mean, 4),
            "e_std": round(st.energy_std, 4),
            "e_max": round(st.energy_max, 4),
            "nan": st.nan,
        }
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def write_report(rows: list[dict], path: str = "sweep_report.md") -> None:
    lines = [
        "# omega_std sweep (stage 3)",
        "",
        "n=1000 degree=8 steps=60000 seed=0, kappa=0.25 sigma_w=0.5, K=0.5",
        "",
        "| omega_std | R@10k | R@30k | R@60k | R range/1k | frozen | "
        "e_mean | e_std | e_max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| {omega_std} | {R_at_10k} | {R_at_30k} | {R_at_60k} | "
                     "{R_range_1k} | {frozen} | {e_mean} | {e_std} | {e_max} |"
                     .format(**r))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    rows = sweep()
    write_report(rows)
    print("wrote sweep_report.md")
