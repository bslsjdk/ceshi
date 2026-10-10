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
from .update import _fused, step_energy, step_phase
from .zstats import zone_means, zone_median_std

TZ_BUDGET_BYTES = 100 * 1024 * 1024
CULL_WINDOW = 2000      # selection-round length, kept for callers
LOW_THRESHOLD = 0.3
ACC_COEF = 50.0         # 1/(1+10*se) was too flat to separate BNs
CULL_FRAC = 0.40        # relative: worst 40% of each zone, every 2000 steps
PROMOTE_FRAC = 0.12     # relative: best 12% of each zone merges, not score>0.7
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
    merged: np.ndarray       # int8 [N_bn] 0=trial 1=EZ 2=RZ -1=culled
    hz_bn: np.ndarray        # int32 [N_bn] prediction lag in samples
    cull_events: int = 0     # cumulative relative culls
    cull_rounds: int = 0     # how many 2000-step rounds have run
    samples: np.ndarray = None   # evaluate calls since the slot was reset
    reinit_count: int = 0
    promote_events: int = 0  # cumulative relative promotions
    promote_round_log: list = None   # (promoted, alive) per round
    last_parts: dict = None   # per-component breakdown of the last score
    reinit_round: np.ndarray = None   # selection round a slot was reset in
    lr: float = 3e-2
    util_norm: str = "zone"   # "zone" (spec) or "global" scale

    def shadow_energy(self, main_energy: np.ndarray) -> np.ndarray:
        return main_energy[self.shadow].mean(axis=1)

    def _predict(self, shadow_vals: np.ndarray) -> np.ndarray:
        p = self.pop
        x = np.repeat(shadow_vals[:, None], DIM, axis=1)
        # Fused W + B@A: one batched matmul, the same trick as step_state.
        z = np.matmul(_fused(p), x[:, :, None])[:, :, 0] + p.bias
        return (p.scale * np.tanh(z).mean(axis=1)).astype(np.float32)

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
        p.WM = None                      # fused matrix is stale after training
        return float(np.mean(err ** 2))

    def evaluate(self, prev_shadow: np.ndarray, target: np.ndarray) -> None:
        if self.samples is not None:
            self.samples += 1        # evaluate calls since the slot was reset
        """Stage 5B score + cull-window sample; same prev->current pair as train."""
        pred = self._predict(prev_shadow)
        act = np.abs(pred)
        se = (pred - target) ** 2
        acc = 1.0 / (1.0 + ACC_COEF * se)
        # Per-BN baseline: squared error of a zero predictor for THIS zone.
        base = (target * target).astype(np.float32)
        contrib = np.maximum(0.0, base - se) / (base + 1e-6)
        # Continuous utilisation, scored against this BN's own zone:
        # sigmoid((|pred| - median_zone) / (std_zone + eps)).  A binary median
        # gate is bimodal by construction -- half of every zone reads 1 and half
        # reads 0 -- so within-zone spread was the gate's artifact.
        med_a, std_a = zone_median_std(act, self.zone_id)
        if self.util_norm == "global":
            # Dividing by the zone's own std rescales every zone to unit width,
            # so sigmoid output has std ~0.19 inside EVERY zone regardless of
            # skill.  A global scale encodes level and leaves spread alone.
            med_a[:] = float(np.median(act)) if act.size else 0.0
            std_a[:] = float(np.std(act)) if act.size else 1.0
        inst = 1.0 / (1.0 + np.exp(-(act - med_a) / (std_a + 1e-6)))
        self.util = 0.99 * self.util + 0.01 * inst.astype(np.float32)
        # Variance, not magnitude: scoring |pred| punishes a BN for being
        # accurate, because a well-trained BN necessarily has a larger output.
        self.pred_mean = 0.99 * self.pred_mean + 0.01 * pred
        self.stab = 0.99 * self.stab + 0.01 * (pred * pred)
        var = np.maximum(0.0, self.stab - self.pred_mean ** 2)
        stab_score = 1.0 / (1.0 + var)
        # The two penalty terms the spec lists and the first pass silently
        # dropped (both constant 0.0).  cost = output magnitude; conflict =
        # distance from this zone's consensus prediction.
        cost = np.clip(act, 0.0, 1.0)
        med, _ = zone_median_std(pred, self.zone_id)
        conflict = np.clip(np.abs(pred - med) / (np.abs(med) + 1e-6), 0.0, 1.0)
        self.score = (0.4 * acc + 0.2 * self.util + 0.2 * contrib
                      + 0.1 * stab_score - 0.05 * cost
                      - 0.05 * conflict).astype(np.float32)
        self.last_parts = dict(acc=acc, util=self.util, contrib=contrib,
                               stab=stab_score, cost=cost, conflict=conflict)


    def promote_relative(self, promote_frac: float = PROMOTE_FRAC
                         ) -> tuple[int, int]:
        """Merge the best ``promote_frac`` of each zone, ranked against peers.

        An absolute 0.7 floor has no discriminating power: an untrained BN
        already scores ~0.668 and a trained one ~0.75, so the distribution
        straddles the threshold and the merge rate pins to ~48%.  Ranking makes
        the rate a parameter instead of a coincidence.
        """
        alive = np.flatnonzero(self.merged == 0)
        if alive.size == 0:
            return 0, 0
        n_rz = n_ez = 0
        for u in np.unique(self.zone_id[alive]):
            idx = np.flatnonzero((self.zone_id == u) & (self.merged == 0))
            k = int(round(idx.size * promote_frac))
            if k <= 0:
                continue
            best = idx[np.argsort(-self.score[idx])[:k]]
            top = best[self.score[best] > 0.85]      # elites go resident
            rest = np.setdiff1d(best, top)
            self.merged[top] = 2
            self.merged[rest] = 1
            n_rz += int(top.size)
            n_ez += int(rest.size)
        self.promote_events += n_rz + n_ez
        if self.promote_round_log is not None:
            self.promote_round_log.append((n_rz + n_ez, int(alive.size)))
        return n_rz, n_ez

    def promote_or_cull(self) -> tuple[int, int, int]:
        """Terminal merge decision; culling is relative, see cull.py."""
        n_rz, n_ez = self.promote_relative()
        return n_rz, n_ez, 0


    def zone_stats(self) -> dict:
        """Zone-level score spread plus the difficulty-ordering check.

        ``between/within > 1`` was dropped: it measures individual skill
        spread against task-difficulty spread, and in a self-supervised
        predictor the former is naturally an order of magnitude larger.
        The replacement asks the direct question -- does a zone that has to
        predict further ahead actually score lower?
        """
        z, alive = self.zone_id, self.merged == 0
        if alive.sum() == 0:
            return {}
        ids = [u for u in np.unique(z[alive])]
        means = np.array([self.score[alive & (z == u)].mean() for u in ids])
        within = np.mean([self.score[alive & (z == u)].std() for u in ids])
        _, hz = zone_means(self.hz_bn, z, alive)
        corr = float("nan")
        if np.std(hz) > 1e-12 and np.std(means) > 1e-12:
            corr = float(np.corrcoef(means, -hz)[0, 1])
        return {"between_zone_std": round(float(means.std()), 4),
                "within_zone_std": round(float(within), 4),
                "horizon_corr": round(corr, 4),
                "n_zones": int(len(ids))}
