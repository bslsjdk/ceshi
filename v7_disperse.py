"""v7 experiment B: does training make concepts cluster or spread?

v6 observed NN-cosine p50 falling 0.822 (20k) -> 0.782 (100k), i.e. TRAINING
SPREADS concepts apart.  That is the opposite of 'similar concepts live at
the same address', which is the first principle of the whole VN design.
Measure it properly: separate the gate-key half from the weight half, and
report the duplicate-tail counts at several checkpoints.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
from vnv6.compress import cosine_matrix
from vnv6.model import ConceptNet, DE
from vnv6.train import streams, train

tr, va, V = streams()
N = int(sys.argv[1])
CKPT = [int(x) for x in sys.argv[2].split(',')]
OUT = sys.argv[3]

def stats(c: np.ndarray, tag: str) -> dict:
    cos = cosine_matrix(c)
    n = c.shape[0]
    iu = np.triu_indices(n, 1)
    off = cos[iu]
    # nearest-neighbour cosine for every row
    cos2 = cos.copy(); np.fill_diagonal(cos2, -np.inf)
    nn = cos2.max(axis=1)
    return dict(tag=tag, n=n,
                nn_p50=round(float(np.median(nn)), 4),
                nn_mean=round(float(nn.mean()), 4),
                off_mean=round(float(off.mean()), 4),
                off_absmean=round(float(np.abs(off).mean()), 4),
                pairs_ge099=int((off >= 0.99).sum()),
                pairs_ge095=int((off >= 0.95).sum()),
                pairs_ge090=int((off >= 0.90).sum()))

rows = []
net = ConceptNet(V, N, seed=0)
rows.append(stats(net.concepts(), 'init'))
rows.append(stats(net.keys, 'init/keys_only'))
prev = 0
for ck in CKPT:
    train(net, tr, va, steps=ck - prev, lr=3e-3, val_every=ck - prev)
    prev = ck
    rows.append(stats(net.concepts(), f'step{ck}'))
    rows.append(stats(net.keys, f'step{ck}/keys_only'))
    rows.append(stats(net.W.reshape(N, -1), f'step{ck}/weights_only'))
    json.dump(rows, open(OUT, 'w'))
    print(json.dumps(rows[-3]), flush=True)
print('DONE', flush=True)
