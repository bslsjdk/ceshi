"""6A: which parameters can actually be trained, and at what lr?

Untrained sits at 6.02 bits (uniform).  Unigram is 4.774 bits.  GPT-team
14,770-param model: 1.088 bits.  Each cell reports val_bits after training.
"""
from __future__ import annotations
import json
import numpy as np
from train6a import PRESETS, run

STEPS = 20000
VAL_EVERY = 20000   # one validation at the end

def main() -> None:
    rows = []
    # untrained reference
    r = run(200, 1000, 0.0, 0, 0, 20, 0, "head_bias")
    rows.append(dict(preset="none", lr=0.0, val_bits=None, note="untrained"))
    for preset in ("adapter", "head", "head_bias", "all"):
        for lr in (1e-4, 1e-3, 1e-2):
            r = run(STEPS, 1000, lr, 0, 0, 20, VAL_EVERY, preset)
            last = r["hist"][-1]["val"] if r["hist"] else {}
            rows.append(dict(preset=preset, lr=lr,
                             train_acc=round(r["train_acc"], 4),
                             val_bits=last.get("val_bits"),
                             val_acc=last.get("val_acc")))
            print(json.dumps(rows[-1]), flush=True)
    out = dict(steps=STEPS, unigram_bits=4.774, uniform_bits=6.022,
               gpt_team_bits=1.088, rows=rows)
    print(json.dumps(out, indent=2))
    json.dump(out, open("sweep6b.json","w"), indent=2)

if __name__ == "__main__":
    main()
