"""Train/eval for VN v3 (<=200 lines).

loss = CE(next token) + lam * MSE(next_i, emb[target]) + mu * concept sparsity
"""
from __future__ import annotations

import numpy as np

from .predict import Adam, normalise

LOG2 = float(np.log(2.0))


class Codec3:
    def __init__(self, vocab, dc, seed=0, scale=1.0, power=2.0):
        rng = np.random.default_rng(seed + 991)
        self.emb = normalise(rng.normal(size=(vocab, dc))).astype(np.float32)
        self.bias = np.zeros(vocab, np.float32)
        self.scale = np.float32(scale)
        self.power = np.float32(power)
        self.dc = dc
        self.Wout = None

    def renorm(self):
        self.emb = normalise(self.emb).astype(np.float32)

    def pooled(self, field):
        p = field.pooled()
        return p.sum(0)

    def logits(self, field):
        h = self.pooled(field)
        return self.scale * (normalise(self.emb) @ h) + self.bias

    def head_grads(self, field, target, logits):
        p = np.exp(logits - logits.max())
        p /= p.sum()
        loss = float(-np.log(p[target] + 1e-12))
        dz = p.astype(np.float32)
        dz[target] -= 1.0
        en = normalise(self.emb)
        h = self.pooled(field)
        d_h = (self.scale * en.T @ dz).astype(np.float32)      # [Dc]
        d_emb = (self.scale * np.outer(dz, h)).astype(np.float32)
        d_emb = (d_emb - en * (en * d_emb).sum(-1, keepdims=True))
        return loss, {"h": d_h, "emb": d_emb, "bias": dz,
                      "scale": np.float32(float(dz @ (en @ h)))}

    def pooled_bwd(self, field, d_h):
        """d_h [Dc] -> d_act [L,N] and d_context [L,N,Dc]."""
        d_act = np.zeros((field.L, field.n), np.float32)
        d_ctx = np.zeros((field.L, field.n, field.dc), np.float32)
        for l in range(field.L):
            s = float(field.act[l].sum()) + 1e-3
            sl = slice(l * field.dc, (l + 1) * field.dc)
            dh_l = d_h
            d_ctx[l] = (field.act[l][:, None] * dh_l[None, :]) / s
            d_act[l] = (field.context[l] @ dh_l) / s
        return d_act, d_ctx


def mse_term(field, codec, nxt, tgt, lam=0.1):
    t = normalise(codec.emb[tgt].reshape(1, -1)).ravel()
    diff = nxt - t
    denom = float(nxt.size)
    loss = lam * float((diff ** 2).sum()) / denom
    d_nxt = (lam * 2.0 * diff / denom).astype(np.float32)
    return loss, d_nxt


def run(field, F, codec, ids, adam=None, hist=1000, lam=0.1, log_every=0,
        tag=""):
    tot, acc, cnt, trace = 0.0, 0, 0, []
    for t in range(len(ids) - 1):
        tgt = int(ids[t + 1])
        if t % hist == 0:
            field.reset()
        field.seed_from(codec.emb[int(ids[t])])
        field.settle(F, grad=True)
        lg = codec.logits(field)
        loss, hg = codec.head_grads(field, tgt, lg)
        if adam is not None and field._last is not None:
            d_act, d_ctx = codec.pooled_bwd(field, hg["h"])
            nxt = field._last[0][2]
            ml, d_nxt = mse_term(field, codec, nxt, tgt, lam)
            bg = field.backward(F, d_act, d_nxt)
            if bg is not None:
                adam.step({
                    "concept": (field.concept, bg["concept"]),
                    "emb": (codec.emb, hg["emb"]),
                    "bias": (codec.bias, hg["bias"]),
                    "G": (field.G, np.zeros_like(field.G)),
                    "W1": (F.W1, bg["F"]["W1"]), "b1": (F.b1, bg["F"]["b1"]),
                    "W2": (F.W2, bg["F"]["W2"]), "b2": (F.b2, bg["F"]["b2"])})
                field.renorm()
                codec.renorm()
        tot += loss
        acc += int(np.argmax(lg) == tgt)
        cnt += 1
        if log_every and cnt % log_every == 0:
            trace.append({"n": cnt, "bits": round(tot / cnt / LOG2, 4)})
            print(f"{tag} n={cnt} bits={tot/cnt/LOG2:.4f}", flush=True)
    return (tot / max(cnt, 1)) / LOG2, acc / max(cnt, 1), trace


def evaluate(field, F, codec, ids, hist=1000):
    tot, acc, cnt = 0.0, 0, 0
    for t in range(len(ids) - 1):
        tgt = int(ids[t + 1])
        if t % hist == 0:
            field.reset()
        field.seed_from(codec.emb[int(ids[t])])
        field.settle(F)
        lg = codec.logits(field)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        tot += float(-np.log(p[tgt] + 1e-12))
        acc += int(np.argmax(lg) == tgt)
        cnt += 1
    return tot / max(cnt, 1) / LOG2, acc / max(cnt, 1)
