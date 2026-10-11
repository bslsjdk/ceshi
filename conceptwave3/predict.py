"""Shared predictor F, layer-pair projection, autograd helpers (<=200 lines)."""
from __future__ import annotations

import numpy as np


def normalise(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def norm_bwd(dout, x, eps=1e-8):
    n = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    xn = x / n
    return (dout - xn * (dout * xn).sum(-1, keepdims=True)) / n


class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.clip = float(clip)
        self.t = 0
        self.m, self.v = {}, {}

    def step(self, pairs):
        self.t += 1
        for key, (param, grad) in pairs.items():
            g = np.asarray(grad, np.float32)
            if not np.isfinite(g).all():
                g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            gn = float(np.linalg.norm(g))
            if gn > self.clip:
                g = (g * (self.clip / gn)).astype(np.float32)
            if key not in self.m:
                self.m[key] = np.zeros_like(param, np.float32)
                self.v[key] = np.zeros_like(param, np.float32)
            m, v = self.m[key], self.v[key]
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            mh = m / (1.0 - self.b1 ** self.t)
            vh = v / (1.0 - self.b2 ** self.t)
            param -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)


class Predictor:
    """F: [2*Dc] -> [Dc].  Shared by every neuron of every layer."""

    def __init__(self, dc, hidden=64, seed=0, scale=0.05):
        rng = np.random.default_rng(seed + 31)
        s = scale / np.sqrt(2.0 * dc)
        self.W1 = (rng.normal(size=(2 * dc, hidden)) * s).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = (rng.normal(size=(hidden, dc)) * s * 4.0).astype(np.float32)
        self.b2 = np.zeros(dc, np.float32)
        self.dc, self.hidden = dc, hidden

    def forward(self, h):
        self.h = h
        self.a1 = np.tanh(h @ self.W1 + self.b1)
        self.out = self.a1 @ self.W2 + self.b2
        return self.out

    def backward(self, dout):
        gW2 = self.a1.T @ dout
        gb2 = dout.sum(0)
        dz1 = (dout @ self.W2.T) * (1.0 - self.a1 ** 2)
        gW1 = self.h.T @ dz1
        gb1 = dz1.sum(0)
        dh = dz1 @ self.W1.T
        return {"W1": gW1.astype(np.float32), "b1": gb1.astype(np.float32),
                "W2": gW2.astype(np.float32), "b2": gb2.astype(np.float32),
                "h": dh.astype(np.float32)}
