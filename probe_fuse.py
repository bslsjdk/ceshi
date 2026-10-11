"""Is level-1 fusion lossless on a *trained* model?

The exactness proof only covers cos == 1.  Real neurons never reach it, so the
question is what threshold makes fusion do anything at all, and what it costs.
This measures the fusion step in isolation -- no clustering, no packing --
by evaluating the network straight after the merge.
"""
from __future__ import annotations

import json
import numpy as np

from vnv6.compress import fuse
from vnv6.model import ConceptNet
from vnv6.train import evaluate, streams

tr, va, V = streams()
net = ConceptNet(V, 1000, seed=0)
d = np.load('v6_base100k.npz')
net.emb, net.keys, net.W = d['emb'], d['keys'], d['W']
net.out, net.bias, net.norm = d['out'], d['bias'], float(d['norm'])
base = evaluate(net, va)
print(json.dumps(dict(hi=None, base_bits=base['val_bits'],
                      base_acc=base['val_acc'])), flush=True)
for hi in (0.99, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8):
    f, groups = fuse(net.concepts(), hi)
    fnet = ConceptNet(V, f.shape[0], seed=0)
    fnet.emb, fnet.out, fnet.bias = net.emb.copy(), net.out.copy(), net.bias.copy()
    fnet.norm = net.norm
    fnet.set_concepts(f)
    ev = evaluate(fnet, va)
    sizes = [len(g) for g in groups if len(g) > 1]
    print(json.dumps(dict(hi=hi, n_after=int(f.shape[0]),
                          n_removed=int(1000 - f.shape[0]),
                          groups=len(sizes), max_group=(max(sizes) if sizes else 0),
                          val_bits=ev['val_bits'], val_acc=ev['val_acc'],
                          delta_bits=round(ev['val_bits'] - base['val_bits'], 4))),
          flush=True)
