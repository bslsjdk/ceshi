"""Does giving the field more history help?

The field only ever sees one character.  A unigram-conditioned field cannot
beat a unigram model by much, which is exactly what we measured: 4.38 bits
against sklearn's 4.276 ceiling on the same features.  So the next lever is
the representation, not the optimiser -- inject a window of recent
characters with exponential decay, which is what "the wave carries the
history" was supposed to mean.
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
STEPS = 100000; LR = 1e-3; VAL = 25000

def run(hist: int, decay: float = 0.6, seed: int = 0):
    pop = build_population(n=N, degree=8, seed=seed, omega_mix=(0.8, 0.3, 1.0))
    block = block_of(N, V); head = CharHead.make(V, DIM, seed=seed, full=True)
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(block == v) for v in range(V)]
    ie = np.zeros(N, np.float32); ih = np.zeros((N, DIM), np.float32)
    hbuf = np.zeros(hist + 1, np.int64)
    hist_rows = []
    def feed(seq, i):
        ie[:] = 0; ih[:] = 0
        for k in range(min(hist, i + 1)):
            w = decay ** k
            c = int(seq[i - k])
            ie[members[c]] += w; ih[members[c]] += w
        ie[:] = np.minimum(ie, 2.0); ih[:] = np.minimum(ih, 2.0)
    for t in range(STEPS):
        feed(tr, t % tr.size)
        nxt = int(tr[(t + 1) % tr.size])
        step_energy(pop, rng, injection=ie); step_phase(pop); step_state(pop, ih)
        train_char(pop, block, head, nxt, LR, ALPHA, ih,
                   train_head=True, train_bias=False, train_adapters=False)
        if (t + 1) % VAL == 0:
            loss = 0.0; hit = 0; lim = 2000
            for i in range(lim):
                feed(va, i)
                step_energy(pop, rng, injection=ie); step_phase(pop); step_state(pop, ih)
                lg = head.logits(pool_by_block(read_output(pop), block, V)); pr = softmax(lg)
                loss += float(-np.log(max(float(pr[va[i + 1]]), 1e-9)))
                hit += int(int(np.argmax(lg)) == int(va[i + 1]))
            hist_rows.append(dict(step=t + 1, val_bits=round(loss/lim/np.log(2), 4),
                                  val_acc=round(hit/lim, 4)))
            print(json.dumps(dict(hist=hist, **hist_rows[-1])), flush=True)
    return hist_rows

rows = []
for h in (1, 2, 4, 8):
    hr = run(h)
    rows.append(dict(hist=h, final=hr[-1], curve=hr))
out = dict(steps=STEPS, lr=LR, uniform_bits=6.022, unigram_bits=4.774,
           sklearn_bits=4.276, gpt_team_bits=1.088, rows=rows)
print(json.dumps(out, indent=2)); json.dump(out, open("ctx6a.json","w"), indent=2)
