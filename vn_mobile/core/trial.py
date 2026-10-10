"""Trial zone and blank neurons (stage 5A) plus evaluation (stage 5B).

A BN is NOT a copy of a main VN: it lives in TZ with its own adjacency and only
sees the main field through a shadow projection.

Stage 5A rerun -- spatial division of labour.  First pass had all BNs watching
one global aggregate, so they scored identically and the 5%-20% merge rate was
impossible: promotion was all-or-nothing.  The main system is now split into
``n_zones`` address blocks, each BN shadows one block, and each BN gets its own
dynamics (omega, kappa, sigma_w).  Culling is proportional (share of a sliding
window below threshold), not an absolute step count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neuron import DIM, MODE_TRIAL, VNPopulation, build_population
from .update import step_energy, step_phase

TZ_BUDGET_BYTES = 100 * 1024 * 1024
CULL_WINDOW = 2000      # samples kept for the proportional cull test
CULL_LOW_FRAC = 0.5     # cull when score<0.3 for more than half the window
LOW_THRESHOLD = 0.3


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
    lr: float = 5e-3

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

        dout = ((2.0 * err / p.n) / DIM)[:, None] * p.scale[:, None]
        dz = dout * (1.0 - t * t)                                   # [N, DIM]
        grad_b = dz[:, :, None] * u[:, None, :]                     # [N, DIM, R]
        du = np.matmul(np.transpose(p.adapter_b, (0, 2, 1)),
                       dz[:, :, None])[:, :, 0]                     # [N, R]
        grad_a = du[:, :, None] * x[:, None, :]                     # [N, R, DIM]

        p.adapter_b -= (self.lr * grad_b).astype(np.float32)
        p.adapter_a -= (self.lr * grad_a).astype(np.float32)
        return float(np.mean(err ** 2))

    def evaluate(self, prev_shadow: np.ndarray, target: np.ndarray,
                 baseline_err: float) -> None:
        """Stage 5B score + cull-window sample; same prev->current pair as train."""
        pred = self._predict(prev_shadow)
        act = np.abs(pred)
        se = (pred - target) ** 2
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

        self._sample_cull_window()

    def _sample_cull_window(self) -> None:
        """Record one low-score sample; the window is the cull evidence."""
        low = self.score < LOW_THRESHOLD
        self.low_count += low.astype(np.int32) - self.low_ring[self.ring_pos]
        self.low_ring[self.ring_pos] = low
        self.ring_pos = (self.ring_pos + 1) % CULL_WINDOW

    def promote_or_cull(self) -> tuple[int, int, int]:
        """Score>0.85 -> RZ, >0.7 -> EZ, low for >50% of window -> cull."""
        s, u = self.score, self.util <= 1e-6
        self.zero_util_steps = np.where(u, self.zero_util_steps + 1, 0)
        need = int(CULL_WINDOW * CULL_LOW_FRAC)
        to_rz = np.flatnonzero((s > 0.85) & (self.merged == 0))
        to_ez = np.flatnonzero((s > 0.7) & (s <= 0.85) & (self.merged == 0))
        cull = np.flatnonzero((self.low_count > need) & (self.merged == 0))
        self.merged[to_rz] = 2
        self.merged[to_ez] = 1
        self.merged[cull] = -1
        return int(to_rz.size), int(to_ez.size), int(cull.size)

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

def zone_of_main(main_addr: np.ndarray, n_zones: int) -> np.ndarray:
    """Split the main system into contiguous address blocks."""
    n = main_addr.shape[0]
    order = np.argsort(main_addr, kind="stable")
    z = np.empty(n, dtype=np.int64)
    z[order] = (np.arange(n) * n_zones) // n
    return z


def make_trial_zone(n_bn: int = 1000, n_main: int = 1000, n_zones: int = 20,
                    k_shadow: int = 8, degree: int = 6, seed: int = 7,
                    main_addr: np.ndarray | None = None) -> TrialZone:
    """BNs get their own graph, their own zone, and their own dynamics."""
    rng = np.random.default_rng(seed)
    if main_addr is None:
        main_addr = np.arange(n_main, dtype=np.int64)
    zm = zone_of_main(np.asarray(main_addr), n_zones)

    per_zone_bn = max(1, n_bn // n_zones)
    zone_id = np.minimum(np.arange(n_bn) // per_zone_bn, n_zones - 1)

    shadow = np.zeros((n_bn, k_shadow), dtype=np.int32)
    for u in range(n_zones):
        members = np.flatnonzero(zm == u)
        pick = np.flatnonzero(zone_id == u)
        if members.size == 0 or pick.size == 0:
            continue
        rep = members.size < k_shadow
        shadow[pick] = [rng.choice(members, k_shadow, replace=rep) for _ in pick]

    pop = build_population(n=n_bn, degree=degree, seed=seed,
                           mode=MODE_TRIAL, health=8)
    pop.energy[:] = rng.uniform(0.0, 0.02, size=n_bn).astype(np.float32)
    pop.region[:] = 2

    # omega is per-neuron already; kappa rides on the zone, sigma_w is folded
    # into the edge weights (wedge = w*d).
    kappa = rng.uniform(0.1, 0.2, size=n_bn).astype(np.float32)
    sigma_w = rng.uniform(0.2, 0.4, size=n_bn).astype(np.float32)
    pop.omega = (rng.normal(0.0, 1.0, size=n_bn)
                 * rng.uniform(0.3, 0.7, size=n_bn)).astype(np.float32)
    base = float(np.abs(pop.weight).mean())
    if base > 0:
        pop.weight = (pop.weight / base) * sigma_w[pop.src]
        pop.weight -= pop.weight.mean()
        pop.wedge = (pop.weight * pop.dir_cos).astype(np.float32)

    return TrialZone(
        pop=pop, shadow=shadow, zone_id=zone_id.astype(np.int32),
        kappa=kappa, sigma_w=sigma_w,
        score=np.zeros(n_bn, dtype=np.float32),
        util=np.zeros(n_bn, dtype=np.float32),
        stab=np.zeros(n_bn, dtype=np.float32),
        pred_mean=np.zeros(n_bn, dtype=np.float32),
        prev_shadow=np.zeros(n_bn, dtype=np.float32),
        low_count=np.zeros(n_bn, dtype=np.int32),
        low_ring=np.zeros((CULL_WINDOW, n_bn), dtype=bool),
        ring_pos=0,
        zero_util_steps=np.zeros(n_bn, dtype=np.int32),
        merged=np.zeros(n_bn, dtype=np.int8),
    )


def tick_trial(tz: TrialZone, rng: np.random.Generator) -> None:
    """Keep the TZ wave field alive; per-BN kappa gives each BN its own gain."""
    step_energy(tz.pop, rng, kappa=tz.kappa)
    step_phase(tz.pop)
