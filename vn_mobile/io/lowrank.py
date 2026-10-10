"""Low-rank character readout (stage 6A correction).

The dense full-field readout is [V, V*DIM] = 33,800 parameters at V=65 and
DIM=8.  The GPT-team reference is 14,770 parameters *in total*, so one dense
readout is already 2.3x the whole comparison budget before a single field
neuron is counted -- which makes any accuracy comparison structurally unfair.

Factoring it as U [V, R] @ P [R, V*DIM] with R=16 costs

    65*16 + 16*520 = 1,040 + 8,320 = 9,360

and keeps every logit able to see every block, which is the property the
dense version was introduced for (the blockwise [V, DIM] readout measured
5.99 val bits, i.e. no better than uniform).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LowRankHead:
    """Two-factor readout: flat [V*DIM] -> mid [R] -> logits [V]."""

    u: np.ndarray          # float32 [V, R]
    p: np.ndarray          # float32 [R, V*DIM]

    @classmethod
    def make(cls, vocab: int, dim: int, rank: int = 16, seed: int = 0,
             scale: float = 0.01) -> "LowRankHead":
        rng = np.random.default_rng(seed)
        cols = vocab * dim
        return cls(
            u=(rng.normal(0.0, scale, size=(vocab, rank))).astype(np.float32),
            p=(rng.normal(0.0, scale, size=(rank, cols))).astype(np.float32),
        )

    @property
    def n_params(self) -> int:
        return int(self.u.size + self.p.size)

    @property
    def rank(self) -> int:
        return int(self.u.shape[1])

    def logits(self, flat: np.ndarray) -> np.ndarray:
        """float32 [V*DIM] -> float32 [V]."""
        return (self.u @ (self.p @ flat)).astype(np.float32)

    def grad(self, dlogits: np.ndarray, flat: np.ndarray) -> tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """Return (dU, dP, dL/dflat) for one step's gradient."""
        mid = self.p @ flat
        du = np.outer(dlogits, mid).astype(np.float32)
        dmid = (self.u.T @ dlogits).astype(np.float32)
        dp = np.outer(dmid, flat).astype(np.float32)
        dflat = (self.p.T @ dmid).astype(np.float32)
        return du, dp, dflat

    def apply(self, du: np.ndarray, dp: np.ndarray, lr: float) -> None:
        self.u -= (lr * du).astype(np.float32)
        self.p -= (lr * dp).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)
