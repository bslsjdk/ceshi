"""Is the readout just echoing the injected character?

If argmax(logits) == the character we just injected, the model has learned
identity, not a transition -- which would explain val_bits 9.6 being worse
than the 6.02 uniform floor (confidently wrong beats guessing).
"""
from __future__ import annotations
import json
import numpy as np
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import (ALPHA, neighbour_mean, step_energy,
                                   step_phase, step_state, read_output)
from vn_mobile.io.charhead import CharHead, forward_char, softmax
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split, pool_by_block

text = load_text()
symbols, stoi = build_vocab(text)
V = len(symbols)
tr_txt, va_txt = split(text)
va = encode(va_txt, stoi)
N = 1000
pop = build_population(n=N, degree=8, seed=0, omega_mix=(0.8, 0.3, 1.0))
block = block_of(N, V)
head = CharHead.make(V, DIM, seed=0)
rng = np.random.default_rng(0)
members = [np.flatnonzero(block == v) for v in range(V)]
inj_e = np.zeros(N, np.float32); inj_h = np.zeros((N, DIM), np.float32)

LIM = 2000
same = 0          # argmax == injected char
hit = 0           # argmax == true next char
loss = 0.0
loss_unif = float(np.log(V))
# also: what if we pool h instead of y?
hsame = 0; hhit = 0; hloss = 0.0
from sklearn.metrics import mutual_info_score
prev = None
for i in range(LIM):
    c = int(va[i]); nxt = int(va[i+1])
    inj_e[:] = 0; inj_h[:] = 0
    inj_e[members[c]] = 1.0
    inj_h[members[c]] = 1.0
    step_energy(pop, rng, injection=inj_e)
    step_phase(pop)
    step_state(pop, inj_h)
    pooled, logits, probs = forward_char(pop, block, head)
    a = int(np.argmax(logits))
    same += int(a == c); hit += int(a == nxt)
    loss += float(-np.log(max(float(probs[nxt]), 1e-9)))
    # variant: pool the state h directly (history) instead of y=s*h*e
    ph = pool_by_block(pop.state, block, V)
    lg2 = head.logits(ph); p2 = softmax(lg2)
    a2 = int(np.argmax(lg2))
    hsame += int(a2 == c); hhit += int(a2 == nxt)
    hloss += float(-np.log(max(float(p2[nxt]), 1e-9)))

out = dict(
    vocab=V, limit=LIM, uniform_bits=round(loss_unif/np.log(2), 4),
    y_pooling=dict(identity_rate=round(same/LIM, 4), acc=round(hit/LIM, 4),
                   bits=round(loss/LIM/np.log(2), 4)),
    h_pooling=dict(identity_rate=round(hsame/LIM, 4), acc=round(hhit/LIM, 4),
                   bits=round(hloss/LIM/np.log(2), 4)),
)
print(json.dumps(out, indent=2))
json.dump(out, open("diag_ident.json","w"), indent=2)
