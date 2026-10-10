"""Relative culling for the trial zone (stage 5A final).

An absolute score floor only fires if genuinely bad BNs exist, and none do:
every BN can predict its own zone reasonably well.  So culling is relative --
every ``CULL_ROUND_STEPS`` the worst ``CULL_FRAC`` of each zone is reinitialised
in place, which keeps the population at its target size while still applying
selection pressure.
"""
from __future__ import annotations

import numpy as np

from .trial import CULL_FRAC, TrialZone

CULL_ROUND_STEPS = 2000


def reinit(tz: TrialZone, idx: np.ndarray, rng: np.random.Generator) -> None:
    """Wipe a BN slot back to a blank neuron and let it re-learn."""
    if idx.size == 0:
        return
    n = idx.size
    p = tz.pop
    p.adapter_a[idx] = rng.normal(0.0, 0.01, size=(n, p.adapter_a.shape[1],
                                                   p.adapter_a.shape[2])
                                  ).astype(np.float32)
    p.adapter_b[idx] = rng.normal(0.0, 0.01, size=(n, p.adapter_b.shape[1],
                                                   p.adapter_b.shape[2])
                                  ).astype(np.float32)
    p.energy[idx] = rng.uniform(0.0, 0.02, size=n).astype(np.float32)
    tz.score[idx] = 0.0
    tz.util[idx] = 0.0
    tz.stab[idx] = 0.0
    tz.pred_mean[idx] = 0.0
    tz.low_count[idx] = 0
    tz.low_ring[:, idx] = False
    tz.zero_util_steps[idx] = 0
    tz.merged[idx] = 0
    tz.reinit_count += int(n)


def relative_round(tz: TrialZone, rng: np.random.Generator,
                   cull_frac: float = CULL_FRAC) -> dict:
    """Cull the worst ``cull_frac`` of each zone, ranked against its peers."""
    tz.cull_rounds += 1
    alive = np.flatnonzero(tz.merged == 0)
    if alive.size == 0:
        return dict(culled=0, cull_rate=0.0)
    total = 0
    for u in np.unique(tz.zone_id[alive]):
        idx = np.flatnonzero((tz.zone_id == u) & (tz.merged == 0))
        k = int(round(idx.size * cull_frac))
        if k <= 0:
            continue
        worst = idx[np.argsort(tz.score[idx])[:k]]
        reinit(tz, worst, rng)
        total += int(worst.size)
    tz.cull_events += total
    return dict(culled=total,
                cull_rate=round(total / max(1, alive.size), 4))
