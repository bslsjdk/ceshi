"""Trial zone and blank neurons (stage 5A) plus evaluation (stage 5B).

A BN is NOT a copy of a main VN: it lives in TZ with its own adjacency and only
sees the main field through a shadow projection.  It is trained adapters-only on
a self-supervised target (predict the energy aggregate of the main neurons it
shadows), which is what gives stage 5B a loss it can actually score.

Stage 2-3 had BN output correlated 1.000 with the main system because mode only
gates diffusion, it does not change the maths.  Independent adjacency + a
distinct learning target is what makes the two distinguishable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neuron import DIM, MODE_TRIAL, RANK, VNPopulation, build_population
from .update import step_energy, step_phase

TZ_BUDGET_BYTES = 100 * 1024 * 1024


@dataclass
class TrialZone:
    pop: VNPopulation
    shadow: np.ndarray      # int32 [N_bn, K] indices into the main population
    score: np.ndarray       # float32 [N_bn]
    util: np.ndarray        # float32 [N_bn] activation frequency
    stab: np.ndarray        # float32 [N_bn] E[x^2] accumulator
    pred_mean: np.ndarray   # float32 [N_bn] E[x] accumulator
    low_score_steps: np.ndarray  # int32 [N_bn]
    zero_util_steps: np.ndarray  # int32 [N_bn]
    merged: np.ndarray      # int8 [N_bn] 0=trial 1=EZ 2=RZ
    lr: float = 5e-3

    def shadow_energy(self, main_energy: np.ndarray) -> np.ndarray:
        return main_energy[self.shadow].mean(axis=1)

    def forward(self, main_energy: np.ndarray) -> np.ndarray:
        """One prediction pass; returns float32 [N_bn] readings."""
        p = self.pop
        x = np.repeat(self.shadow_energy(main_energy)[:, None], DIM, axis=1)
        linear = x @ p.W.T + p.bias
        mid = np.matmul(p.adapter_a, x[:, :, None])[:, :, 0]
        adapted = np.matmul(p.adapter_b, mid[:, :, None])[:, :, 0]
        return (p.scale * np.tanh(linear + adapted).mean(axis=1)).astype(np.float32)

    def train_step(self, main_energy: np.ndarray, target: np.ndarray) -> float:
        """Adapters-only gradient step on MSE.  W, b, scale stay frozen."""
        p = self.pop
        x = np.repeat(self.shadow_energy(main_energy)[:, None], DIM, axis=1)
        u = np.matmul(p.adapter_a, x[:, :, None])[:, :, 0]          # [N, R]
        z = x @ p.W.T + p.bias + np.matmul(p.adapter_b, u[:, :, None])[:, :, 0]
        t = np.tanh(z)
        pred = p.scale * t.mean(axis=1)
        err = (pred - target).astype(np.float32)

        dout = ((2.0 * err / p.n) / DIM)[:, None] * p.scale[:, None]  # mean, not sum
        dz = dout * (1.0 - t * t)                                   # [N, DIM]
        grad_b = dz[:, :, None] * u[:, None, :]                     # [N, DIM, R]
        du = np.matmul(np.transpose(p.adapter_b, (0, 2, 1)),
                       dz[:, :, None])[:, :, 0]                     # [N, R]
        grad_a = du[:, :, None] * x[:, None, :]                     # [N, R, DIM]

        p.adapter_b -= (self.lr * grad_b).astype(np.float32)
        p.adapter_a -= (self.lr * grad_a).astype(np.float32)
        return float(np.mean(err ** 2))

    def evaluate(self, main_energy: np.ndarray, target: np.ndarray,
                 baseline_err: float) -> None:
        """Stage 5B score = 0.4 acc + 0.2 util + 0.2 contrib + 0.1 stab."""
        pred = self.forward(main_energy)
        act = np.abs(pred)
        se = (pred - target) ** 2
        # Per-neuron terms: a single scalar score would make promote/cull
        # all-or-nothing, which is exactly what stage 5B must not be.
        acc = 1.0 / (1.0 + 10.0 * se)
        contrib = np.maximum(0.0, baseline_err - se) / (baseline_err + 1e-9)
        self.util = 0.99 * self.util + 0.01 * (act > 0.05).astype(np.float32)
        # Variance, not magnitude: scoring |pred| punishes a BN for being
        # accurate, because a well-trained BN necessarily has a larger output.
        self.pred_mean = 0.99 * self.pred_mean + 0.01 * pred
        self.stab = 0.99 * self.stab + 0.01 * (pred * pred)
        var = np.maximum(0.0, self.stab - self.pred_mean ** 2)
        stab_score = 1.0 / (1.0 + var)
        self.score = (0.4 * acc + 0.2 * self.util + 0.2 * contrib
                      + 0.1 * stab_score).astype(np.float32)

    def promote_or_cull(self) -> tuple[int, int, int]:
        """Score>0.85 -> RZ, >0.7 -> EZ, <0.3 for 10k steps -> cull."""
        s = self.score
        self.low_score_steps = np.where(s < 0.3, self.low_score_steps + 1, 0)
        self.zero_util_steps = np.where(self.util <= 1e-6,
                                        self.zero_util_steps + 1, 0)
        to_rz = np.flatnonzero((s > 0.85) & (self.merged == 0))
        to_ez = np.flatnonzero((s > 0.7) & (s <= 0.85) & (self.merged == 0))
        cull = np.flatnonzero(((self.low_score_steps > 10000)
                               | (self.zero_util_steps > 50000))
                              & (self.merged == 0))
        self.merged[to_rz] = 2
        self.merged[to_ez] = 1
        self.merged[cull] = -1
        return int(to_rz.size), int(to_ez.size), int(cull.size)


def make_trial_zone(n_bn: int = 1000, n_main: int = 1000, k_shadow: int = 8,
                    degree: int = 6, seed: int = 7) -> TrialZone:
    """BNs get their own small graph so they are not main-VN duplicates."""
    rng = np.random.default_rng(seed)
    pop = build_population(n=n_bn, degree=degree, seed=seed,
                           mode=MODE_TRIAL, health=8)
    pop.energy[:] = rng.uniform(0.0, 0.02, size=n_bn).astype(np.float32)
    pop.region[:] = 2
    shadow = rng.integers(0, n_main, size=(n_bn, k_shadow)).astype(np.int32)
    return TrialZone(
        pop=pop, shadow=shadow,
        score=np.zeros(n_bn, dtype=np.float32),
        util=np.zeros(n_bn, dtype=np.float32),
        stab=np.zeros(n_bn, dtype=np.float32),
        pred_mean=np.zeros(n_bn, dtype=np.float32),
        low_score_steps=np.zeros(n_bn, dtype=np.int32),
        zero_util_steps=np.zeros(n_bn, dtype=np.int32),
        merged=np.zeros(n_bn, dtype=np.int8),
    )


def tick_trial(tz: TrialZone, rng: np.random.Generator) -> None:
    """Keep the TZ wave field alive without touching the main equations."""
    step_energy(tz.pop, rng)
    step_phase(tz.pop)
