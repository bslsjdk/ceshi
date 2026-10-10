"""Character <-> concept-space codec (<=200 lines).

The concept vectors are the address structure and are frozen: they decide
*where* something lives, they do not hold knowledge.  What is learned is the
map from a character into concept space (``emb``) and the readout back out of
the settled wave.  At V=65, Dc=32 that is ~2.1k trainable parameters.

Readout is a *coherent* sum, not an energy sum:

    A_v = sum_i affinity(c_i, emb_v) * psi_i        (complex)
    logit_v = scale * |A_v| + bias

Pooling ``|psi|^2`` would throw the phase away, and phase is the only place
interference lives -- an energy readout is structurally unable to see the one
mechanism this architecture adds.  Taking the modulus (rather than the real
part) keeps it invariant to a global phase rotation, which the per-round phase
advance would otherwise turn into readout noise.
"""
from __future__ import annotations

import numpy as np

from .kernels import cosine, normalise


class CharCodec:
    def __init__(self, vocab_size, dc, seed=0, scale=8.0, power=2.0):
        rng = np.random.default_rng(seed + 991)
        self.emb = normalise(rng.normal(size=(vocab_size, dc))).astype(np.float32)
        self.bias = np.zeros(vocab_size, np.float32)
        self.scale = np.float32(scale)
        self.power = np.float32(power)
        self.dc = dc

    def renorm(self):
        self.emb = normalise(self.emb).astype(np.float32)

    def ctx_vector(self, ids_hist, decay=0.6):
        if len(ids_hist) == 0:
            return self.emb.mean(0)
        w = decay ** np.arange(len(ids_hist) - 1, -1, -1, dtype=np.float32)
        return (w[:, None] * self.emb[np.asarray(ids_hist)]).sum(0) / w.sum()

    def affinity(self, field, char_id=None):
        """[N] or [V,N] affinity of nodes to a character's concept."""
        c = getattr(field, "cn", None)
        c = normalise(field.concept) if c is None else c
        if char_id is None:
            e = normalise(self.emb)
            sim = c @ e.T                      # [N, V]
            return np.clip(sim.T, 0.0, None) ** self.power   # [V, N]
        sim = cosine(c, normalise(self.emb[char_id].reshape(1, -1))).ravel()
        return np.clip(sim, 0.0, None) ** self.power         # [N]

    def amp_for(self, field, char_id):
        return self.affinity(field, int(char_id))

    def inject_char(self, field, char_id, phase=0.0, gain=1.0, power=None):
        amp = self.amp_for(field, char_id) if power is None else \
            np.clip(cosine(normalise(field.concept),
                           normalise(self.emb[char_id].reshape(1, -1))).ravel(),
                    0.0, None) ** power
        field.inject(float(gain) * amp, phase=phase)

    def logits(self, field, aff=None):
        aff = self.affinity(field) if aff is None else aff
        acc = aff @ field.psi                         # [V] complex
        return self.scale * np.abs(acc) + self.bias

    def grads(self, field, target, logits, aff=None):
        """Straight-through: the settled wave is treated as an input."""
        aff = self.affinity(field) if aff is None else aff
        acc = aff @ field.psi                         # [V] complex
        mod = np.abs(acc)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        loss = float(-np.log(p[target] + 1e-12))
        dz = p.astype(np.float32)
        dz[target] -= 1.0
        safe = np.where(mod > 1e-8, mod, 1.0)
        # d|A_v| / d aff[v,i]  =  Re(conj(A_v) psi_i) / |A_v|
        coeff = self.scale * dz / safe                # [V]
        d_aff = (coeff[:, None]
                 * (acc.real[:, None] * field.psi.real[None, :]
                    + acc.imag[:, None] * field.psi.imag[None, :]))
        cn = getattr(field, "cn", None)
        cn = normalise(field.concept) if cn is None else cn
        sim = cn @ normalise(self.emb).T                         # [N, V]
        pos = np.clip(sim, 0.0, None)
        d_sim = (self.power * pos ** (self.power - 1.0)).T       # [V, N]
        g_emb = (d_aff * d_sim) @ cn                             # [V, Dc]
        return loss, {"emb": g_emb.astype(np.float32),
                      "bias": dz.astype(np.float32),
                      "scale": np.float32(float(dz @ mod))}
