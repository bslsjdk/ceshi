"""Concept block network: one trainable block per neuron.

concept_i packs the neuron's gate key and its weight block:

    concept_i = [k_i (DE) | vec(W_i) (DO*DE)]      D = DE + DO*DE

forward, for input x built from the current and previous character:

    g_i     = <k_i, x> / sqrt(DE)        linear gate, no softmax
    u_i     = W_i @ x
    pooled  = sum_i g_i * u_i / sqrt(N)
    logits  = Out @ tanh(pooled) + b

The gate is left unnormalised on purpose.  A softmax would couple every
neuron to every other one through the denominator, and then deleting a
duplicate slot would move all the other gates -- "merging two identical
neurons" would no longer be a no-op.  Linear keeps each neuron's
contribution independent, which is what makes exact fusion possible.
"""
from __future__ import annotations

import numpy as np

DE = 16          # embedding / key width
DO = 8           # neuron output width
DCONCEPT = DE + DO * DE


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class ConceptNet:
    def __init__(self, vocab: int, n: int, seed: int = 0,
                 emb_scale: float = 0.3, key_scale: float = 0.1,
                 w_scale: float = 0.05):
        rng = np.random.default_rng(seed)
        self.V, self.N = vocab, n
        self.emb = (rng.normal(0, emb_scale, (vocab, DE))).astype(np.float32)
        self.keys = (rng.normal(0, key_scale, (n, DE))).astype(np.float32)
        self.W = (rng.normal(0, w_scale, (n, DO, DE))).astype(np.float32)
        self.out = (rng.normal(0, 0.1, (vocab, DO))).astype(np.float32)
        self.bias = np.zeros(vocab, np.float32)
        # Fixed pool normalisation, carried across compression so that fusing
        # neurons (which changes N) cannot silently rescale the whole field.
        self.norm = 1.0 / np.sqrt(max(n, 1))

    # ---- concept vector <-> parameters ---------------------------------
    def concepts(self) -> np.ndarray:
        return np.concatenate(
            [self.keys, self.W.reshape(self.N, DO * DE)], axis=1
        ).astype(np.float32)

    def set_concepts(self, c: np.ndarray) -> None:
        assert c.shape == (self.N, DCONCEPT)
        self.keys = np.ascontiguousarray(c[:, :DE], dtype=np.float32)
        self.W = np.ascontiguousarray(
            c[:, DE:].reshape(self.N, DO, DE), dtype=np.float32)

    def n_params(self) -> int:
        return (self.emb.size + self.keys.size + self.W.size
                + self.out.size + self.bias.size)

    # ---- input ---------------------------------------------------------
    def x_of(self, cur: int, prev: int) -> np.ndarray:
        return self.emb[cur] + 0.5 * self.emb[prev]

    # ---- forward -------------------------------------------------------
    def forward(self, cur: int, prev: int):
        x = self.x_of(cur, prev)
        g = (self.keys @ x) * (1.0 / np.sqrt(DE))
        U = (self.W.reshape(-1, DE) @ x).reshape(self.N, DO)
        pooled = (g @ U) * self.norm
        h = np.tanh(pooled)
        logits = self.out @ h + self.bias
        return dict(x=x, g=g, U=U, pooled=pooled, h=h, logits=logits)

    def loss(self, logits: np.ndarray, target: int) -> float:
        pr = _softmax(logits)
        return float(-np.log(max(float(pr[target]), 1e-9)))

    # ---- backward ------------------------------------------------------
    def backward(self, f, target: int, dx_emb=True):
        pr = _softmax(f['logits'])
        dlogits = pr.copy()
        dlogits[target] -= 1.0
        h, g, U, x, W = f['h'], f['g'], f['U'], f['x'], self.W
        d_out = np.outer(dlogits, h)
        d_bias = dlogits.copy()
        d_h = self.out.T @ dlogits
        d_pooled = (d_h * (1.0 - h * h)) * self.norm

        d_U = g[:, None] * d_pooled[None, :]
        d_g = U @ d_pooled
        d_W = (d_U.reshape(-1, 1) @ x.reshape(1, -1)).reshape(self.N, DO, DE)
        d_keys = np.outer(d_g, x) * (1.0 / np.sqrt(DE))

        dx = d_U.reshape(-1) @ W.reshape(-1, DE)
        dx += (d_g @ self.keys) * (1.0 / np.sqrt(DE))
        grads = dict(emb=None, keys=d_keys, W=d_W, out=d_out, bias=d_bias)
        if dx_emb:
            grads['emb'] = np.zeros_like(self.emb)
            grads['emb'][f['cur']] += dx
            grads['emb'][f['prev']] += 0.5 * dx
        return grads

    def param_pairs(self, grads, train_concepts=True):
        pairs = {'out': (self.out, grads['out']),
                 'bias': (self.bias, grads['bias'])}
        if grads['emb'] is not None:
            pairs['emb'] = (self.emb, grads['emb'])
        if train_concepts:
            pairs['keys'] = (self.keys, grads['keys'])
            pairs['W'] = (self.W, grads['W'])
        return pairs
