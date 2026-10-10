"""6A learning-rate sweep: does training help at all, and at what lr?

Untrained the model sits at 6.025 bits, i.e. exactly the uniform baseline
(log2(65) = 6.022), so the readout starts unbiased.  Anything above that after
training is the optimiser doing damage, not the architecture being weak.

Reference points: unigram 4.774 bits, GPT-team 14,770-param model 1.088 bits
(0.754 nats).
"""
from __future__ import annotations

import json

import numpy as np

from train6a import PRESETS, run

STEPS = 20000
VAL_EVERY = 10000


def main() -> None:
    rows = []
    for preset in ("adapter", "head", "head_bias", "all"):
        for lr in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
            r = run(STEPS, 1000, lr, 0, 0, 20, VAL_EVERY, preset)
            last = r["hist"][-1]["val"] if r["hist"] else {}
            rows.append(dict(preset=preset, lr=lr,
                             train_acc=round(r["train_acc"], 4),
                             val_bits=last.get("val_bits"),
                             val_acc=last.get("val_acc"),
                             val_loss=last.get("val_loss")))
            print(json.dumps(rows[-1]), flush=True)
    rows.sort(key=lambda d: (d["val_bits"] is None, d["val_bits"]))
    out = dict(steps=STEPS, unigram_bits=4.774, uniform_bits=round(
        float(np.log2(65)), 4), rows=rows)
    print(json.dumps(out, indent=2))
    with open("sweep6a.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
