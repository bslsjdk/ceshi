"""Collect every v4 run into one table: forgetting vs plastic parameters."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vnv4 import data as D  # noqa: E402
from vnv4 import protocol as PR  # noqa: E402


def plastic_count(name, V):
    wrap = PR.build(name, V, 0)
    if name == "vn4":
        wrap.freeze(["emb", "pf", "wof"])
    elif name == "vn4f":
        wrap.freeze(["emb", "pf", "wof", "woe"])
    elif name in ("ssmf", "ssmm"):
        wrap.freeze(["emb", "woe"])
    elif name == "tff":
        wrap.freeze(["emb", "out"])
    n = sum(int(wrap.P[k].size) for k in wrap.trainable())
    return wrap.count_params(), n


def main():
    _, _, V = D.load_tasks()
    rows = {}
    for path in sorted(glob.glob(os.path.join(HERE, "v4_*.json"))):
        d = json.load(open(path))
        rows.setdefault(d["model"], []).append(d)
    out = []
    for name, runs in rows.items():
        fr = [r["forgetting_rate"] for r in runs]
        lb = [r["L_B_after_nats"] for r in runs]
        tot, plas = plastic_count(name, V)
        out.append({
            "model": name,
            "seeds": [r["seed"] for r in runs],
            "n_params": tot,
            "plastic_params": plas,
            "plastic_frac": round(plas / tot, 3),
            "FR_mean": round(float(np.mean(fr)), 4),
            "FR_std": round(float(np.std(fr)), 4),
            "FR_all": [round(float(x), 3) for x in fr],
            "L_B_mean": round(float(np.mean(lb)), 4),
        })
    out.sort(key=lambda r: r["plastic_frac"])
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "v4_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
