"""Continual-learning protocol: train on A, then on B, then re-measure A.

Forgetting rate is defined as

    FR = (L_A_after - L_A_before) / L_A_before

using mean NLL in nats on the *held-out* tail of corpus A.  L_B is always
reported alongside: a model that forgets nothing because it learned nothing
would be useless, so "no forgetting" is only meaningful next to a real drop
on B.
"""
from __future__ import annotations

import sys
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from vn_mobile.io.adam import Adam  # noqa: E402
import tf_micro  # noqa: E402


class TFWrap:
    """Adapts tf_micro's single-sequence Transformer to the model interface."""

    kind = "tf"

    def __init__(self, V, D=48, L=2, H=2, FF=48, seed=0):
        self.P = tf_micro.make_params(V, D, L, H, FF, seed=seed)
        self.D, self.L, self.H = D, L, H
        self.frozen: set = set()
        self.V = V

    def count_params(self):
        return tf_micro.count_params(self.P)

    def trainable(self):
        return [k for k in self.P if k not in self.frozen]

    def zeros_grad(self):
        return tf_micro.zeros_like_params(self.P)

    def freeze(self, keys):
        self.frozen = set(keys)

    def eval_nll(self, x, y):
        logits, _ = tf_micro.forward(self.P, x, self.D, self.L, self.H)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        return float((-(logits[np.arange(len(y)), y] - m.ravel()
                        - np.log(e.sum(axis=1)))).mean())

    def fwd_bwd(self, G, x, y):
        logits, C = tf_micro.forward(self.P, x, self.D, self.L, self.H)
        return tf_micro.backward(self.P, G, logits, C, y,
                                 self.D, self.L, self.H)


class FWTrain:
    """Uniform train/eval driver for the fast-weight models."""

    def __init__(self, model):
        self.m = model
        self.P = model.P

    @property
    def frozen(self):
        return self.m.frozen

    def count_params(self):
        return self.m.count_params()

    def trainable(self):
        return self.m.trainable()

    def zeros_grad(self):
        return self.m.zeros_grad()

    def freeze(self, keys):
        self.m.freeze(keys)

    def fwd_bwd(self, G, x, y):
        logits, C = self.m.forward(x)
        return self.m.backward(G, logits, C, x, y)

    def eval_nll(self, x, y):
        return self.m.eval_nll(x, y)


def build(model_name: str, V: int, seed: int):
    from vnv4.models import VNv4, SSM
    if model_name in ("vn4", "vn4nf", "vn4f"):
        return FWTrain(VNv4(V, seed=seed))
    if model_name in ("ssm", "ssmf"):
        return FWTrain(SSM(V, seed=seed))
    if model_name == "ssmm":
        return FWTrain(SSM(V, Ne=64, seed=seed))
    if model_name in ("tf", "tff"):
        return TFWrap(V, seed=seed)
    raise ValueError(model_name)


def run_phase(wrap, sampler, steps, opt, lr, log_every=0):
    """Plain SGD-through-Adam loop; returns list of (step, nll)."""
    keys = wrap.trainable()
    hist = []
    for s in range(steps):
        x, y = sampler.next()
        G = wrap.zeros_grad()
        nll = wrap.fwd_bwd(G, x, y)
        pairs = {k: (wrap.P[k], G[k]) for k in keys}
        opt.clip(pairs, limit=5.0)
        opt.step(pairs)
        if log_every and (s % log_every == 0):
            hist.append((s, float(nll)))
        if not np.isfinite(nll):
            hist.append((s, float("inf")))
            break
    return hist


def forgetting(lA0, lA1):
    if not np.isfinite(lA1) or not np.isfinite(lA0) or lA0 <= 0:
        return float("nan")
    return (lA1 - lA0) / lA0
