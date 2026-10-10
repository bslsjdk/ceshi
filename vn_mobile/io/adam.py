"""Adam, in place of plain SGD (stage 6A2).

Why it is needed is a measurement, not a preference: on the same 12,000
(pooled_activity, next_char) pairs a scikit-learn logistic regression reached
4.276 val bits while our hand-rolled SGD stopped at 4.392.  That 0.116 gap is
an optimiser gap, not a representation gap -- the features are identical, so
what is missing is momentum and per-coordinate step sizing.

The update is stored as (param, grad) pairs so the caller keeps ownership of
the arrays: the field, the head and the adapters all live in different
objects and none of them should have to know about the optimiser.  ``t`` is
bumped once per ``step`` call, not once per parameter, which is what the bias
correction assumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Adam:
    """First and second moment estimates, one pair per registered name."""

    lr: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)

    def _moments(self, key: str, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
        if key not in self.m:
            self.m[key] = np.zeros(shape, dtype=np.float32)
            self.v[key] = np.zeros(shape, dtype=np.float32)
        return self.m[key], self.v[key]

    def step(self, pairs: dict) -> None:
        """``pairs`` maps a name to (array, grad); arrays are updated in place."""
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for key, (param, grad) in pairs.items():
            g = np.asarray(grad, dtype=np.float32)
            if not np.all(np.isfinite(g)):
                continue
            m, v = self._moments(key, param.shape)
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            mh = m / c1
            vh = v / c2
            param -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)

    def clip(self, pairs: dict, limit: float = 1.0) -> None:
        """Rescale gradients whose norm exceeds ``limit`` (per-parameter)."""
        for _, (_, grad) in pairs.items():
            g = np.asarray(grad, dtype=np.float32)
            nrm = float(np.sqrt(float((g * g).sum())))
            if np.isfinite(nrm) and nrm > limit:
                grad[...] = (g * (limit / nrm)).astype(np.float32)
