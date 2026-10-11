"""v7 experiment A: over-parameterisation scan.

If large N buys nothing over small N, then v6's 'lossless 5x compression'
is just removing redundancy, not extracting structure.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from vnv6.model import ConceptNet
from vnv6.train import streams, train

tr, va, V = streams()
NS = [int(x) for x in sys.argv[1].split(',')]
SEEDS = [int(x) for x in sys.argv[2].split(',')]
STEPS = int(sys.argv[3])
OUT = sys.argv[4]

rows = []
if os.path.exists(OUT):
    rows = json.load(open(OUT))
done = {(r['n'], r['seed']) for r in rows}

for N in NS:
    for seed in SEEDS:
        if (N, seed) in done:
            continue
        net = ConceptNet(V, N, seed=seed)
        t0 = time.perf_counter()
        hist, _ = train(net, tr, va, steps=STEPS, lr=3e-3, val_every=max(STEPS // 2, 1))
        r = dict(n=N, seed=seed, params=net.n_params(),
                 bits=hist[-1]['val_bits'], acc=hist[-1]['val_acc'],
                 wall_s=round(time.perf_counter() - t0, 1))
        rows.append(r)
        json.dump(rows, open(OUT, 'w'))
        print(json.dumps(r), flush=True)
print('DONE', flush=True)
