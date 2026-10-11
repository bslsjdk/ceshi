"""Training and evaluation loop for the concept-block network.

Everything here is deliberately plain: one Adam over the five parameter
tensors, a fixed character stream, and a validation pass that never touches
the training state.  The only thing worth noting is that ``emb`` receives a
gradient on two rows per step (current and previous character), so its Adam
moments are sparse in practice; that is the normal embedding situation and
needs no special handling here.
"""
from __future__ import annotations

import numpy as np

from vn_mobile.io.adam import Adam
from vn_mobile.io.vocab import build_vocab, encode, load_text, split
from vnv6.model import ConceptNet


def streams(seed: int = 0):
    text = load_text()
    symbols, stoi = build_vocab(text)
    tr_txt, va_txt = split(text)
    return (encode(tr_txt, stoi).astype(np.int64),
            encode(va_txt, stoi).astype(np.int64), len(symbols))


def evaluate(net: ConceptNet, va: np.ndarray, limit: int = 3000,
             offset: int = 0) -> dict:
    loss = 0.0
    hit = 0
    for i in range(limit):
        cur = int(va[offset + i])
        prev = int(va[offset + i - 1])
        tgt = int(va[offset + i + 1])
        lg = net.forward(cur, prev)['logits']
        loss += net.loss(lg, tgt)
        hit += int(int(np.argmax(lg)) == tgt)
    return dict(val_bits=round(loss / limit / np.log(2), 4),
                val_acc=round(hit / limit, 4))


def train(net: ConceptNet, tr: np.ndarray, va: np.ndarray, steps: int,
          lr: float = 3e-3, val_every: int = 10000, clip: float = 5.0,
          train_concepts: bool = True, log: list | None = None):
    """Returns (history, adam).  ``log`` receives one dict per eval."""
    opt = Adam(lr=lr)
    hist = []
    run = np.zeros(val_every, np.float32)
    for t in range(steps):
        i = t % (tr.size - 2)
        cur, prev, tgt = int(tr[i + 1]), int(tr[i]), int(tr[i + 2])
        f = net.forward(cur, prev)
        f['cur'], f['prev'] = cur, prev
        run[t % val_every] = net.loss(f['logits'], tgt)
        grads = net.backward(f, tgt)
        pairs = net.param_pairs(grads, train_concepts=train_concepts)
        opt.clip(pairs, clip)
        opt.step(pairs)
        if (t + 1) % val_every == 0:
            rec = dict(step=t + 1,
                       train_bits=round(float(run.mean()) / np.log(2), 4))
            rec.update(evaluate(net, va))
            hist.append(rec)
            if log is not None:
                log.append(rec)
    return hist, opt


def unigram_bits(tr: np.ndarray, va: np.ndarray, vocab: int) -> float:
    counts = np.bincount(tr, minlength=vocab).astype(np.float64)
    p = (counts + 1.0) / (counts.sum() + vocab)
    loss = 0.0
    n = 3000
    for i in range(n):
        loss += -np.log(p[va[i + 1]])
    return loss / n / np.log(2)
