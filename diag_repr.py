"""Is the pooled representation informative, or is our optimiser at fault?

Two independent questions, kept separate:

Q1 signal -- does the field respond differently to different characters at
   all?  Measured as between-character spread of `pooled` versus the
   within-character (noise/tick) spread.  If the ratio is ~1 the field is a
   constant and no readout can help.

Q2 readout -- hand the same features to sklearn's logistic regression, a
   solver we did not write.  If it beats unigram (4.774 bits) and our
   trainer stalls at ~6.02, the bug is in our optimiser.  If sklearn also
   stalls, the representation carries no next-character signal.
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import step_energy, step_phase, step_state, read_output
from vn_mobile.io.vocab import (block_of, build_vocab, encode, load_text,
                                split, pool_by_block)

text = load_text(); symbols, stoi = build_vocab(text); V = len(symbols)
tr_txt, va_txt = split(text); tr = encode(tr_txt, stoi)
N = 1000
pop = build_population(n=N, degree=8, seed=0, omega_mix=(0.8, 0.3, 1.0))
block = block_of(N, V)
rng = np.random.default_rng(0)
members = [np.flatnonzero(block == v) for v in range(V)]
inj_e = np.zeros(N, np.float32); inj_h = np.zeros((N, DIM), np.float32)

M = 12000
X = np.zeros((M, V * DIM), np.float32); y = np.zeros(M, np.int64)
for i in range(M):
    c = int(tr[i % tr.size]); nxt = int(tr[(i + 1) % tr.size])
    inj_e[:] = 0; inj_h[:] = 0
    inj_e[members[c]] = 1.0; inj_h[members[c]] = 1.0
    step_energy(pop, rng, injection=inj_e); step_phase(pop); step_state(pop, inj_h)
    X[i] = pool_by_block(read_output(pop), block, V).ravel(); y[i] = nxt

# ---- Q1: signal-to-noise of the representation ----
mu = X.mean(axis=0)
tot = float(((X - mu) ** 2).sum())
within = 0.0
for v in range(V):
    idx = np.flatnonzero(y == v)
    if idx.size > 1:
        within += float(((X[idx] - X[idx].mean(axis=0)) ** 2).sum())
between = tot - within
snr = between / max(within, 1e-12) * (M - V) / max(V - 1, 1)

# ---- Q2: sklearn readout ----
ntr = int(M * 0.8)
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X[:ntr], y[:ntr])
classes = clf.classes_

def bits(idx):
    p = clf.predict_proba(X[idx])
    col = np.searchsorted(classes, y[idx])
    okc = (col < classes.size) & (classes[np.clip(col, 0, classes.size - 1)] == y[idx])
    lp = -np.log(np.maximum(p[np.arange(idx.size),
                              np.clip(col, 0, classes.size - 1)], 1e-9))
    lp = np.where(okc, lp, np.log(1e9))     # unseen class: worst case
    return float(lp.mean() / np.log(2)), float((clf.predict(X[idx]) == y[idx]).mean()), float(okc.mean())

b_tr, a_tr, c_tr = bits(np.arange(ntr)); b_va, a_va, c_va = bits(np.arange(ntr, M))
out = dict(samples=M, features=V * DIM, uniform_bits=6.022, unigram_bits=4.774,
           repr_snr=round(float(snr), 4),
           sklearn_train_bits=round(b_tr, 4), sklearn_train_acc=round(a_tr, 4),
           sklearn_val_bits=round(b_va, 4), sklearn_val_acc=round(a_va, 4),
           val_class_coverage=round(c_va, 4))
print(json.dumps(out, indent=2)); json.dump(out, open("diag_repr.json", "w"), indent=2)
