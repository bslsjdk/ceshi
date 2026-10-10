"""Trial zone and blank neurons (stage 5A) plus evaluation (stage 5B).

A BN is NOT a copy of a main VN: it lives in TZ with its own adjacency and only
sees the main field through a shadow projection.

Stage 5A final -- spatial division of labour + heterogeneous horizon.  BNs
watch one address block each and every zone predicts a *different* number of
ticks ahead, so cross-zone score spread exceeds within-zone spread.

Scoring follows the agreed fix: ``acc = 1/(1+50*se)`` (10 was too flat) and the
utilisation gate is the running median of ``|pred|`` rather than a fixed 0.05
that every BN cleared.  Culling is relative (worst 40% of each zone) because an
absolute threshold needs absolute failures, and none exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neuron import DIM, VNPopulation
from .update import step_energy, step_phase

TZ_BUDGET_BYTES = 100 * 1024 * 1024
CULL_WINDOW = 2000      # samples kept for the proportional cull test
CULL_LOW_FRAC = 0.5     # cull when score<0.3 for more than half the window
LOW_THRESHOLD = 0.3
ACC_COEF = 50.0         # 1/(1+10*se) was too flat to separate BNs
CULL_FRAC = 0.40        # relative: worst 40% of each zone, every 2000 steps
BASE_HORIZON = 50       # ticks; zone i uses BASE_HORIZON*(0.5+0.1*i)
SAMPLE_EVERY = 10       # ticks between stored shadow samples


@dataclass
class TrialZone:
    pop: VNPopulation
    shadow: np.ndarray       # int32 [N_bn, K] indices into the main population
    zone_id: np.ndarray      # int32 [N_bn] which main zone this BN shadows
    kappa: np.ndarray        # float32 [N_bn] per-BN energy gain
    sigma_w: np.ndarray      # float32 [N_bn] per-BN weight spread
    score: np.ndarray        # float32 [N_bn]
    util: np.ndarray         # float32 [N_bn] activation frequency
    stab: np.ndarray         # float32 [N_bn] E[x^2] accumulator
    pred_mean: np.ndarray    # float32 [N_bn] E[x] accumulator
    prev_shadow: np.ndarray  # float32 [N_bn] shadow energy one sample ago
    low_count: np.ndarray    # int32 [N_bn] low samples inside the window
    low_ring: np.ndarray     # bool [CULL_WINDOW, N_bn]
    ring_pos: int
    zero_util_steps: np.ndarray  # int32 [N_bn]
    merged: np.ndarray       # int8 [N_bn] 0=trial 1=EZ 2=RZ -1=culled
    hz_bn: np.ndarray        # int32 [N_bn] prediction lag in samples
    cull_events: int = 0     # cumulative relative culls
    cull_rounds: int = 0     # how many 2000-step rounds have run
    reinit_count: int = 0
    lr: float = 3e-2

    def shadow_energy(self, main_energy: np.ndarray) -> np.ndarray:
        return main_energy[self.shadow].mean(axis=1)

    def _predict(self, shadow_vals: np.ndarray) -> np.ndarray:
        p = self.pop
        x = np.repeat(shadow_vals[:, None], DIM, axis=1)
        linear = x @ p.W.T + p.bias
        mid = np.matmul(p.adapter_a, x[:, :, None])[:, :, 0]
        adapted = np.matmul(p.adapter_b, mid[:, :, None])[:, :, 0]
        return (p.scale * np.tanh(linear + adapted).mean(axis=1)).astype(np.float32)

    def forward(self, main_energy: np.ndarray) -> np.ndarray:
        return self._predict(self.shadow_energy(main_energy))

    def train_step(self, prev_energy: np.ndarray, target: np.ndarray) -> float:
        """Adapters-only step on MSE; W, b, scale frozen.  prev -> current."""

        p = self.pop
        x = np.repeat(prev_energy[:, None], DIM, axis=1)
        u = np.matmul(p.adapter_a, x[:, :, None])[:, :, 0]          # [N, R]
        z = x @ p.W.T + p.bias + np.matmul(p.adapter_b, u[:, :, None])[:, :, 0]
        t = np.tanh(z)
        pred = p.scale * t.mean(axis=1)
        err = (pred - target).astype(np.float32)

        # Each BN owns its own adapters, so this is N independent losses, not
        # one batched loss -- dividing by p.n made the gradient ~1e-7 and lr had
        # literally no effect across two orders of magnitude.
        dout = (2.0 * err / DIM)[:, None] * p.scale[:, None]
        dz = dout * (1.0 - t * t)                                   # [N, DIM]
        grad_b = dz[:, :, None] * u[:, None, :]                     # [N, DIM, R]
        du = np.matmul(np.transpose(p.adapter_b, (0, 2, 1)),
                       dz[:, :, None])[:, :, 0]                     # [N, R]
        grad_a = du[:, :, None] * x[:, None, :]                     # [N, R, DIM]

        p.adapter_b -= (self.lr * grad_b).astype(np.float32)
        p.adapter_a -= (self.lr * grad_a).astype(np.float32)
        return float(np.mean(err ** 2))

    def evaluate(self, prev_shadow: np.ndarray, target: np.ndarray) -> None:
        """Stage 5B score + cull-window sample; same prev->current pair as train."""
        pred = self._predict(prev_shadow)
        act = np.abs(pred)
        se = (pred - target) ** 2
        acc = 1.0 / (1.0 + ACC_COEF * se)
        # Per-BN baseline: squared error of a zero predictor for THIS zone.
        base = (target * target).astype(np.float32)
        contrib = np.maximum(0.0, base - se) / (base + 1e-6)
        # Utilisation gate tracks the running median, not a fixed constant that
        # every BN cleared at once (which made util std exactly 0).
        gate = float(np.median(act)) if act.size else 0.0
        self.util = 0.99 * self.util + 0.01 * (act > gate).astype(np.float32)
        # Variance, not magnitude: scoring |pred| punishes a BN for being
        # accurate, because a well-trained BN necessarily has a larger output.
        self.pred_mean = 0.99 * self.pred_mean + 0.01 * pred
        self.stab = 0.99 * self.stab + 0.01 * (pred * pred)
        var = np.maximum(0.0, self.stab - self.pred_mean ** 2)
        stab_score = 1.0 / (1.0 + var)
        # The two penalty terms the spec lists and the first pass silently
        # dropped (both constant 0.0), which is why every score sat at ~0.72.
        # cost = output magnitude; conflict = distance from this zone's
        # consensus prediction, i.e. how much the BN disagrees with its peers.
        cost = np.clip(act, 0.0, 1.0)
        med = np.zeros_like(pred)
        for u in np.unique(self.zone_id):
            m = self.zone_id == u
            med[m] = float(np.median(pred[m])) if m.any() else 0.0
        conflict = np.clip(np.abs(pred - med) / (np.abs(med) + 1e-6), 0.0, 1.0)
        self.score = (0.4 * acc + 0.2 * self.util + 0.2 * contrib
                      + 0.1 * stab_score - 0.05 * cost
                      - 0.05 * conflict).astype(np.float32)

        self._sample_cull_window()

    def _sample_cull_window(self) -> None:
        """Record one low-score sample; the window is the cull evidence."""
        low = self.score < LOW_THRESHOLD
        self.low_count += low.astype(np.int32) - self.low_ring[self.ring_pos]
        self.low_ring[self.ring_pos] = low
        self.ring_pos = (self.ring_pos + 1) % CULL_WINDOW

    def promote_or_cull(self) -> tuple[int, int, int]:
        """Score>0.85 -> RZ, >0.7 -> EZ.  Culling is relative; see cull.py."""
        s = self.score
        self.zero_util_steps = np.where(self.util <= 1e-6,
                                        self.zero_util_steps + 1, 0)
        to_rz = np.flatnonzero((s > 0.85) & (self.merged == 0))
        to_ez = np.flatnonzero((s > 0.7) & (s <= 0.85) & (self.merged == 0))
        self.merged[to_rz] = 2
        self.merged[to_ez] = 1
        return int(to_rz.size), int(to_ez.size), 0

    def zone_stats(self) -> dict:
        """Between-zone vs within-zone score spread (division of labour)."""
        z, alive = self.zone_id, self.merged == 0
        if alive.sum() == 0:
            return {}
        ids = [u for u in np.unique(z[alive])]
        means = np.array([self.score[alive & (z == u)].mean() for u in ids])
        within = np.mean([self.score[alive & (z == u)].std() for u in ids])
        return {"between_zone_std": float(means.std()),
                "within_zone_std": float(within)}
