"""CLI: one model, both phases, emit the forgetting numbers as JSON."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vnv4 import data as D  # noqa: E402
from vnv4 import protocol as PR  # noqa: E402
from vn_mobile.io.adam import Adam  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps-a", type=int, default=1500)
    ap.add_argument("--steps-b", type=int, default=800)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tasks, stoi, V = D.load_tasks()
    sa = {k: D.Sampler(v, a.T, seed=a.seed + i)
          for i, (k, v) in enumerate(
              (("tr", tasks["A"]["train"]), ("va", tasks["A"]["val"]),
               ("btr", tasks["B"]["train"]), ("bva", tasks["B"]["val"])))}

    wrap = PR.build(a.model, V, a.seed)
    n_params = wrap.count_params()

    # phase A: everything trainable
    opt = Adam(lr=a.lr)
    histA = PR.run_phase(wrap, sa["tr"], a.steps_a, opt, a.lr,
                         log_every=max(1, a.steps_a // 10))
    lA0 = D.eval_nll(wrap, sa["va"], a.eval_batches)

    # freeze the anchor for this architecture before phase B
    if a.model == "vn4":
        wrap.freeze(["emb", "pf", "wof"])
    elif a.model == "vn4f":
        wrap.freeze(["emb", "pf", "wof", "woe"])
    elif a.model == "ssmf":
        wrap.freeze(["emb", "woe"])
    elif a.model == "ssmm":
        wrap.freeze(["emb", "woe"])
    elif a.model == "tff":
        wrap.freeze(["emb", "out"])
    lB_before = D.eval_nll(wrap, sa["bva"], a.eval_batches)

    optB = Adam(lr=a.lr)
    histB = PR.run_phase(wrap, sa["btr"], a.steps_b, optB, a.lr,
                         log_every=max(1, a.steps_b // 10))
    lA1 = D.eval_nll(wrap, sa["va"], a.eval_batches)
    lB1 = D.eval_nll(wrap, sa["bva"], a.eval_batches)

    res = {
        "model": a.model, "seed": a.seed, "V": V, "T": a.T,
        "steps_a": a.steps_a, "steps_b": a.steps_b, "lr": a.lr,
        "n_params": n_params,
        "frozen": sorted(wrap.frozen),
        "trainable_in_B": len(wrap.trainable()),
        "L_A_before_nats": lA0, "L_A_after_nats": lA1,
        "L_B_before_nats": lB_before, "L_B_after_nats": lB1,
        "forgetting_rate": PR.forgetting(lA0, lA1),
        "histA": histA, "histB": histB,
    }
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt)


if __name__ == "__main__":
    main()
