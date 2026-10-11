"""VN v3: two-layer concept wave with inter-layer projection (<=200 lines).

Per layer l in {0,1} the intra-layer propagation is the v2 prediction-driven
wave:

    h_i    = concat(concept[l,i], context[l,i])
    next_i = F(h_i)                       shared MLP, predicts next concept
    M[j,i] = cos(concept[l,j], next_i)
    w[:,i] = softmax(M[:,i] / tau)
    act[l] = sum_i w[:,i]
    ctx    = (1-a)*ctx + a*(w @ next)/act

Inter-layer coupling is what makes this two-dimensional.  Each layer pools its
field into one summary vector, the pair of summaries is mixed by a learned
[2*Dc, 2*Dc] matrix, and the mixed summary is injected back into both layers'
context.  That is 4*Dc*Dc parameters and is fully differentiable, so gradient
reaches the lower layer through the upper one.

    pooled[l] = sum_i act[l,i] * context[l,i] / (sum act + eps)   [Dc]
    mix       = G @ concat(pooled[0], pooled[1])                  [2*Dc]
    context[l] += beta * mix[l*Dc:(l+1)*Dc]

Nothing is reset between ticks unless the caller asks: the field keeps running,
which is the "continuously operating" property.
"""
from __future__ import annotations

import numpy as np

from .predict import normalise, norm_bwd


class Field3:
    def __init__(self, n, dc, seed=0, alpha=0.5, tau=0.3, rounds=6,
                 nsrc=32, k=4, beta=0.5, layers=2):
        rng = np.random.default_rng(seed + 7)
        self.n, self.dc, self.L = n, dc, layers
        self.concept = normalise(
            rng.normal(size=(layers, n, dc))).astype(np.float32)
        self.context = np.zeros((layers, n, dc), np.float32)
        self.act = np.zeros((layers, n), np.float32)
        self.alpha = np.float32(alpha)
        self.tau = np.float32(tau)
        self.rounds, self.nsrc, self.k = rounds, nsrc, k
        self.beta = np.float32(beta)
        self.ctx_cap = np.float32(1.0)
        g = rng.normal(size=(layers * dc, layers * dc))
        self.G = (g * (0.05 / np.sqrt(layers * dc))).astype(np.float32)
        self.cn = normalise(self.concept)
        self._last = None
        self._mix = None

    def renorm(self):
        self.concept = normalise(self.concept).astype(np.float32)
        self.cn = normalise(self.concept)

    def reset(self):
        self.context[:] = 0
        self.act[:] = 0

    def seed_from(self, vec):
        v = normalise(np.asarray(vec, np.float32).reshape(1, -1)).ravel()
        for l in range(self.L):
            sim = self.cn[l] @ v
            idx = np.argpartition(-sim, min(self.nsrc, self.n - 1))[:self.nsrc]
            self.act[l] = 0.0
            self.act[l, idx] = 1.0
            self.context[l, idx] = v

    def _layer_step(self, l, F):
        act = self.act[l]
        idx = np.flatnonzero(act > 0)
        if idx.size == 0:
            return None
        if idx.size > self.nsrc:
            idx = idx[np.argpartition(-act[idx], self.nsrc - 1)[:self.nsrc]]
        h = np.concatenate([self.concept[l, idx], self.context[l, idx]], axis=1)
        nxt = F.forward(h.astype(np.float32))
        nn = normalise(nxt)
        M = self.cn[l] @ nn.T                                   # [N, A]
        z = M / self.tau
        z -= z.max(0, keepdims=True)
        e = np.exp(z)
        w = (e / e.sum(0, keepdims=True)).astype(np.float32)    # [N, A]
        a = w.sum(1).astype(np.float32)
        af = np.maximum(a, 1e-3).astype(np.float32)
        nc = (w @ nxt) / af[:, None]
        upd = a > 1e-3
        self.context[l, upd] = ((1.0 - self.alpha) * self.context[l, upd]
                                + self.alpha * nc[upd])
        self.act[l] = a
        return (l, idx, nxt, nn, M, w, a, upd)

    def settle(self, F, grad=False):
        steps = []
        for _ in range(self.rounds):
            for l in range(self.L):
                steps.append(self._layer_step(l, F))
            self._inter_layer()
        self._last = steps[-self.L:] if grad else None
        return self.act

    def pooled(self):
        out = np.zeros((self.L, self.dc), np.float32)
        for l in range(self.L):
            s = float(self.act[l].sum()) + 1e-3
            out[l] = (self.act[l] @ self.context[l]) / s
        return out

    def _inter_layer(self):
        """Bounded inter-layer coupling.

        The naive form (context += beta * G @ pooled) is a positive feedback
        loop: pooled grows with context, G @ pooled grows faster, and the field
        reaches inf within ~2k steps (measured).  tanh bounds the injected
        term and a per-row L2 cap bounds the accumulated context.
        """
        p = self.pooled().reshape(-1)
        pre = (self.G @ p).astype(np.float32)
        mix = np.tanh(pre)
        self._mix = mix
        self._pre = pre
        for l in range(self.L):
            sl = slice(l * self.dc, (l + 1) * self.dc)
            self.context[l] += self.beta * mix[sl]
        nr = np.linalg.norm(self.context, axis=2, keepdims=True)
        self.context /= np.maximum(nr / self.ctx_cap, 1.0).astype(np.float32)

    def all_act(self):
        return self.act.reshape(-1)

    def sparsity(self):
        return float((self.all_act() > 1e-6).mean())

    def concept_sim_top(self, vec, k=5):
        v = normalise(np.asarray(vec, np.float32).reshape(1, -1)).ravel()
        flat_a = self.all_act()
        flat_c = self.cn.reshape(-1, self.dc)
        top = np.argsort(-flat_a)[:k]
        sim = flat_c @ v
        return float(sim[top].mean()), [round(float(x), 4) for x in sim[top]]

    def backward(self, F, d_act, d_ctx_out=None, d_nxt_extra=None):
        """Backward through the LAST round only (truncated, same as v2).

        d_act  : [L, N]   upstream gradient on the activation vector
        d_ctx_out : [L, N, Dc] upstream gradient on final context (from readout
                 via the pooled summary); it also carries the inter-layer term
        d_nxt_extra : optional [A, Dc] added to dL/d next_i (the MSE term)
        """
        if self._last is None:
            return None
        d_ctx_out = (np.zeros_like(self.context) if d_ctx_out is None
                     else d_ctx_out)
        d_concept = np.zeros_like(self.concept)
        gF = None
        dG = np.zeros_like(self.G)
        pool = self.pooled().reshape(-1)
        for (l, idx, nxt, nn, M, w, a, upd) in self._last:
            da = d_act[l].astype(np.float64)
            # --- inter-layer: context[l] += beta * (G @ pool)[slice] ---
            sl = slice(l * self.dc, (l + 1) * self.dc)
            d_ctx_l = d_ctx_out[l].astype(np.float64)
            # G receives gradient only through the injection that followed
            # this round's own layer step; use the pooled summary as it was.
            d_mix = self.beta * d_ctx_l.sum(0) * (1.0 - self._mix[sl] ** 2)
            dG[sl, :] += np.outer(d_mix, pool).astype(np.float32)
            # --- context update: ctx = (1-a)*ctx + a*(w @ nxt)/act ---
            d_nc = self.alpha * d_ctx_l[upd]
            denom = np.maximum(a[upd], 1e-3)[:, None]
            # nc[j,:] = sum_i w[j,i]*nxt[i,:] / a[j];  j in upd, i in idx
            d_w_from_ctx = (d_nc @ nxt.T) / denom            # [n_upd, A]
            d_nxt_from_ctx = w[upd, :].T @ (d_nc / denom)    # [A, Dc]
            # --- act = sum_i w[:,i] ---
            d_w = np.repeat(da[:, None], w.shape[1], axis=1)
            d_w[upd, :] += d_w_from_ctx
            d_M = (w * (d_w - (w * d_w).sum(0, keepdims=True))
                   / self.tau).astype(np.float32)
            d_cn = d_M @ nn
            d_nxt = norm_bwd(d_M.T @ self.cn[l], nxt).astype(np.float64)
            d_nxt += d_nxt_from_ctx
            if d_nxt_extra is not None:
                d_nxt = d_nxt + d_nxt_extra
            g = F.backward(d_nxt.astype(np.float32))
            if gF is None:
                gF = {k: v.copy() for k, v in g.items()}
            else:
                for k in gF:
                    gF[k] = gF[k] + g[k]
            d_concept[l] += norm_bwd(d_cn, self.concept[l])
            dh = g.pop("h")
            np.add.at(d_concept, (l, idx), dh[:, :self.dc])
        return {"concept": d_concept.astype(np.float32), "F": gF, "G": dG}
