"""Full-field vs blockwise readout, same field, same optimiser.

sklearn reached 4.276 val bits on these exact features, so 4.774 (unigram)
is the bar and ~4.3 is what a competent solver extracts.  If the full-field
head lands near that, the readout width was the blocker.
"""
from __future__ import annotations
import json
import numpy as np
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import ALPHA, step_energy, step_phase, step_state, read_output
from vn_mobile.io.charhead import CharHead, softmax, train_char
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split, pool_by_block

text = load_text(); symbols, stoi = build_vocab(text); V = len(symbols)
tr_txt, va_txt = split(text); tr = encode(tr_txt, stoi); va = encode(va_txt, stoi)
N = 1000

def go(full: bool, lr: float, steps: int, seed: int = 0):
    pop = build_population(n=N, degree=8, seed=seed, omega_mix=(0.8, 0.3, 1.0))
    block = block_of(N, V)
    head = CharHead.make(V, DIM, seed=seed, full=full)
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(block == v) for v in range(V)]
    ie = np.zeros(N, np.float32); ih = np.zeros((N, DIM), np.float32)
    for t in range(steps):
        c = int(tr[t % tr.size]); nxt = int(tr[(t + 1) % tr.size])
        ie[:] = 0; ih[:] = 0
        ie[members[c]] = 1.0; ih[members[c]] = 1.0
        step_energy(pop, rng, injection=ie); step_phase(pop); step_state(pop, ih)
        train_char(pop, block, head, nxt, lr, ALPHA, ih,
                   train_head=True, train_bias=False, train_adapters=False,
                   )
    # validate
    loss = 0.0; hit = 0; n = 2000
    for i in range(n):
        c = int(va[i]); nxt = int(va[i + 1])
        ie[:] = 0; ih[:] = 0
        ie[members[c]] = 1.0; ih[members[c]] = 1.0
        step_energy(pop, rng, injection=ie); step_phase(pop); step_state(pop, ih)
        pooled = pool_by_block(read_output(pop), block, V)
        lg = head.logits(pooled); pr = softmax(lg)
        loss += float(-np.log(max(float(pr[nxt]), 1e-9)))
        hit += int(int(np.argmax(lg)) == nxt)
    return dict(full=full, lr=lr, steps=steps,
                val_bits=round(loss / n / np.log(2), 4), val_acc=round(hit / n, 4))

rows = []
for full in (True, False):
    for lr in (1e-3, 1e-2):
        rows.append(go(full, lr, 20000))
        print(json.dumps(rows[-1]), flush=True)
out = dict(uniform_bits=6.022, unigram_bits=4.774, sklearn_bits=4.276, rows=rows)
print(json.dumps(out, indent=2)); json.dump(out, open("cmp_head.json","w"), indent=2)
