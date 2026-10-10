"""Concept-wave propagation vs current fixed-weight propagation.

Same data, same readout, same optimiser, same step budget.  The ONLY variable
is the propagation kernel:

  fixed : g_i = sum_j w_ij * e_j * cos(phi_j-phi_i) * d_ij     (dense)
  wave  : m_ij = cos(k_j, P x_i); keep only m_ij > tau
          g_i  = sum_j [m_ij>tau] * m_ij * e_j * cos(phi_j-phi_i)

P is frozen (identity on the key block) so both arms carry the SAME trainable
parameter count; any difference comes from the kernel, not from extra capacity.
tau = -1.0 disables the threshold, which separates "concept similarity" from
"thresholding" as two independent causes.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

F = np.float32
DATA = '/data/workspace/ceshi/tinyshakespeare.txt'
LAM, KAP, ET, ALP, KC, GI = 0.1, 0.25, 0.001, 0.9, 0.5, 0.5


def load_data():
    if not os.path.exists(DATA):
        import urllib.request
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/karpathy/char-rnn/'
            'master/data/tinyshakespeare/input.txt', DATA)
    t = open(DATA, encoding='utf-8').read()
    ch = sorted(set(t))
    st = {c: i for i, c in enumerate(ch)}
    ids = np.array([st[c] for c in t], np.int64)
    k = int(len(ids) * 0.9)
    return ids[:k], ids[k:], len(ch)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            if g is None:
                continue
            g = np.clip(np.nan_to_num(g), -1.0, 1.0)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            self.p[i] -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(F)


class Field:
    def __init__(self, n=1000, ds=8, kk=16, deg=8, rank=16, v=65, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.ds, self.kk, self.deg, self.v = n, ds, kk, deg, v
        self.dst = rng.integers(0, n, size=(n, deg)).astype(np.int64)
        self.src = np.repeat(np.arange(n), deg)
        self.dflat = self.dst.reshape(-1)
        self.w = rng.normal(0, 0.5, size=(n, deg)).astype(F)
        dv = rng.normal(size=(n, 8)).astype(F)
        dv /= (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-8)
        self.dij = (dv[self.dflat] * dv[self.src]).sum(1).astype(F)
        self.e = np.full(n, 0.3, F)
        self.phase = (rng.random(n) * 2 * np.pi).astype(F)
        self.omega = rng.normal(0, 1.0, n).astype(F)
        self.s = np.zeros((n, ds), F)
        self.key = rng.normal(size=(n, kk)).astype(F)
        self.key /= (np.linalg.norm(self.key, axis=1, keepdims=True) + 1e-8)
        self.W = rng.normal(0, 0.3, size=(ds, ds)).astype(F)
        self.bias = np.zeros(ds, F)
        self.block = np.arange(n) % v
        self.cnt = np.bincount(self.block, minlength=v).astype(F)
        self.U = rng.normal(0, 0.05, size=(v, rank)).astype(F)
        self.Wb = rng.normal(0, 0.05, size=(rank, v * ds)).astype(F)
        # frozen concept predictor: identity on the key block
        self.P = np.zeros((kk, ds + kk), F)
        self.P[:, ds:] = np.eye(kk, dtype=F)

    def _contrib(self, mode, tau):
        dphi = self.phase[self.dflat] - self.phase[self.src]
        cosd = np.cos(dphi).astype(F)
        if mode == 'fixed':
            val = self.w.reshape(-1) * self.e[self.dflat] * cosd * self.dij
            return val, np.float32(1.0)
        x = np.concatenate([self.s, self.key], axis=1)
        pred = x @ self.P.T
        pred /= (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        m = (self.key[self.dflat] * pred[self.src]).sum(1).astype(F)
        keep = (m > tau).astype(F)
        return keep * m * self.e[self.dflat] * cosd, keep.mean()

    def step(self, mode, tau, inj, rng):
        val, act = self._contrib(mode, tau)
        g = np.bincount(self.src, weights=val, minlength=self.n).astype(F)
        self.e *= (1.0 - LAM)
        self.e += LAM * 0.3 + KAP * np.tanh(g / np.sqrt(self.deg))
        self.e += GI * inj
        self.e += rng.normal(0, ET, self.n).astype(F)
        dphi = self.phase[self.dflat] - self.phase[self.src]
        cp = np.bincount(self.src, weights=np.sin(dphi),
                         minlength=self.n).astype(F)
        self.phase += self.omega + (KC / self.deg) * cp
        np.mod(self.phase, 2 * np.pi, out=self.phase)
        agg = np.stack([np.bincount(self.src, weights=self.s[self.dflat, c],
                                    minlength=self.n)
                        for c in range(self.ds)], axis=1).astype(F)
        lin = (inj[:, None] + agg / self.deg) @ self.W.T + self.bias
        self.s *= ALP
        self.s += (1.0 - ALP) * np.tanh(lin).astype(F)
        np.putmask(self.s, np.abs(self.s) < 1e-20, F(0.0))
        return float(act)

    def pooled(self):
        p = np.stack([np.bincount(self.block, weights=self.s[:, c],
                                  minlength=self.v)
                      for c in range(self.ds)], axis=1).astype(F)
        return (p / self.cnt[:, None]).reshape(-1)

    def logits(self):
        flat = self.pooled()
        hid = self.Wb @ flat
        return self.U @ hid, hid, flat

    def node_active(self, eps=1e-6):
        return float(np.mean(np.abs(self.e - 0.3) > eps))


def inject(f, c):
    inj = np.zeros(f.n, F)
    inj[f.block == c] = F(1.0)
    return inj


def run(mode, tau, steps, seed=0, lr=1e-3, eval_every=5000, val_len=4000):
    tr, va, v = load_data()
    f = Field(v=v, seed=seed)
    opt = Adam([f.U, f.Wb], lr=lr)
    rng = np.random.default_rng(seed + 1)
    pos, hist = 0, []
    for t in range(1, steps + 1):
        c = int(tr[pos % len(tr)])
        nxt = int(tr[(pos + 1) % len(tr)])
        pos += 1
        f.step(mode, tau, inject(f, c), rng)
        lg, hid, flat = f.logits()
        pr = np.exp(lg - lg.max())
        pr /= pr.sum()
        g = pr.astype(F)
        g[nxt] -= F(1.0)
        opt.step([np.outer(g, hid), np.outer(f.U.T @ g, flat)])
        if t % eval_every == 0:
            hist.append(eval_bits(f, va, val_len, mode, tau, rng))
    return f, hist


def eval_bits(f, va, n, mode, tau, rng):
    ce, acc = 0.0, 0
    for i in range(n):
        c, nxt = int(va[i]), int(va[i + 1])
        f.step(mode, tau, inject(f, c), rng)
        lg, _, _ = f.logits()
        pr = np.exp(lg - lg.max())
        pr /= pr.sum()
        ce += -np.log(max(pr[nxt], 1e-9))
        acc += int(pr.argmax() == nxt)
    return {'step': None, 'bits': float(ce / n / np.log(2)),
            'acc': acc / n, 'edge_active': None,
            'node_active': f.node_active()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()
    arms = [('fixed', -1.0), ('wave', -1.0), ('wave', 0.0),
            ('wave', 0.3), ('wave', 0.5), ('wave', 0.7)]
    out = []
    for mode, tau in arms:
        f, hist = run(mode, tau, a.steps, lr=a.lr)
        last = hist[-1]
        _, act = f._contrib(mode, tau)
        r = {'mode': mode, 'tau': tau, 'bits': last['bits'],
             'acc': last['acc'], 'edge_active': float(act),
             'node_active': last['node_active'],
             'trainable': int(f.U.size + f.Wb.size)}
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open('wave_ab.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
