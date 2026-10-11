"""v7 experiment C: is big-N worse only because it is under-trained?

The strongest objection to 'more neurons do not help' is that a larger pool
simply needs more optimisation steps.  Train N=1000 for 4x the budget and
see whether it finally beats N=20.
"""
from __future__ import annotations
import json, sys, os, time
import numpy as np
from vnv6.model import ConceptNet
from vnv6.train import streams, train

tr, va, V = streams()
N = int(sys.argv[1]); STEPS = int(sys.argv[2]); SEED = int(sys.argv[3]); OUT = sys.argv[4]
net = ConceptNet(V, N, seed=SEED)
log = []
t0 = time.perf_counter()
hist, _ = train(net, tr, va, steps=STEPS, lr=3e-3, val_every=max(STEPS // 6, 1), log=log)
rec = dict(n=N, seed=SEED, params=net.n_params(), steps=STEPS,
           wall_s=round(time.perf_counter() - t0, 1),
           final=hist[-1], curve=log)
rows = []
if os.path.exists(OUT):
    rows = json.load(open(OUT))
rows.append(rec)
json.dump(rows, open(OUT, 'w'))
print(json.dumps(dict(n=N, steps=STEPS, curve=[(r['step'], r['val_bits']) for r in log])), flush=True)
