"""Streaming train / eval over the persistent field (<=200 lines).

One character per tick.  The field is never reset inside a segment: ``psi``
carries over, which is the whole point of "keeps running".  ``segment`` lets a
caller cut the carry-over at length ``hist`` so the long-context experiment can
compare hist=100 against hist=1000 on identical data.
"""
from __future__ import annotations

import numpy as np

from .encode import CharCodec
from .field import ConceptField

LOG2 = float(np.log(2.0))


def _logits_and_loss(field, codec, target):
    aff = codec.affinity(field)          # one [V,N] build per tick, reused
    lg = codec.logits(field, aff)
    loss, grads = codec.grads(field, target, lg, aff)
    return lg, loss, grads


def run(field: ConceptField, codec: CharCodec, ids, adam=None, hist=1000,
        rounds=12, decay_hist=0.9, ctx_decay=0.6, target_offset=1,
        log_every=0, tag=""):
    """Returns (mean_bits, mean_acc, trace).  adam None -> eval only."""
    n = len(ids)
    total, acc, cnt, trace = 0.0, 0, 0, []
    hist_buf = []
    for t in range(n - target_offset):
        tgt = int(ids[t + target_offset])
        if t % hist == 0:
            field.psi[:] = 0
            hist_buf = []
        hist_buf.append(int(ids[t]))
        if len(hist_buf) > 64:
            hist_buf.pop(0)
        field.psi *= np.float32(decay_hist)
        field.set_context(codec.ctx_vector(hist_buf, ctx_decay))
        codec.inject_char(field, int(ids[t]))
        field.settle(max_rounds=rounds)
        lg, loss, grads = _logits_and_loss(field, codec, tgt)
        if adam is not None:
            adam.step({"emb": (codec.emb, grads["emb"]),
                       "bias": (codec.bias, grads["bias"]),
                       "scale": (codec.scale, grads["scale"])})
            codec.renorm()
        total += loss / LOG2
        acc += int(np.argmax(lg) == tgt)
        cnt += 1
        if log_every and cnt % log_every == 0:
            trace.append({"n": cnt, "train_bits": round(total / cnt, 4)})
            print(f"{tag} n={cnt} bits={total/cnt:.4f}", flush=True)
    return total / max(cnt, 1), acc / max(cnt, 1), trace


def evaluate(field, codec, ids, hist=1000, rounds=12, ctx_decay=0.6):
    return run(field, codec, ids, None, hist=hist, rounds=rounds,
               ctx_decay=ctx_decay)
