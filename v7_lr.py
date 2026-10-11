from __future__ import annotations
import json, os, sys, time
import numpy as np
from vnv6.model import ConceptNet
from vnv6.train import streams, train

tr, va, V = streams()
OUT = 'v7_lr.json'
# baseline lr 3e-3 was tuned at N=20; if gradients scale as 1/sqrt(N),
# the matched learning rate is lr * sqrt(N/20).
rows = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {(r['n'], round(r['lr'], 6)) for r in rows}
for N, seed in ((105, 0), (105, 1), (1000, 0), (1000, 1)):
    lr = 3e-3 * float(np.sqrt(N / 20.0))
    key = (N, round(lr, 6))
    if key in done:
        continue
    net = ConceptNet(V, N, seed=seed)
    hist, _ = train(net, tr, va, steps=100000, lr=lr, val_every=50000)
    r = dict(n=N, seed=seed, lr=round(lr, 6), params=net.n_params(),
             bits=hist[-1]['val_bits'], acc=hist[-1]['val_acc'])
    rows.append(r); json.dump(rows, open(OUT, 'w'))
    print(json.dumps(r), flush=True)
print('DONE', flush=True)
