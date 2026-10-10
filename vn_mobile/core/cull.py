"""Relative selection for the trial zone (stage 5A final).

An absolute score floor only fires if genuinely bad BNs exist, and none do:
every BN can predict its own zone reasonably well.  So selection is relative --
every ``CULL_ROUND_STEPS`` each zone is ranked once and cut into three disjoint
bands: the best ``PROMOTE_FRAC`` merges (12%), the worst ``CULL_FRAC`` is
reinitialised in place (40%), and the middle simply survives.

Reinitialising in place keeps the population at its target size while still
applying selection pressure.
"""
from __future__ import annotations

import numpy as np

from .trial import CULL_FRAC, PROMOTE_FRAC, TrialZone

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
    p.WM = None
    tz.score[idx] = 0.0
    tz.util[idx] = 0.0
    tz.stab[idx] = 0.0
    tz.pred_mean[idx] = 0.0
    tz.merged[idx] = 0
    tz.reinit_count += int(n)
    if tz.reinit_round is not None:
        tz.reinit_round[idx] = tz.cull_rounds
        tz.samples[idx] = 0


def relative_round(tz: TrialZone, rng: np.random.Generator,
                   cull_frac: float = CULL_FRAC,
                   promote_frac: float = 0.0) -> dict:
    """One selection round; ``promote_frac=0`` makes it cull-only.

    Promotion and culling read the SAME ranking, so the two bands can never
    collide: worst ``cull_frac`` from the bottom, best ``promote_frac`` from the
    top.  With 40% + 12% = 52% <= 100% the bands are always disjoint.
    """
    tz.cull_rounds += 1
    alive = np.flatnonzero(tz.merged == 0)
    if alive.size == 0:
        return dict(culled=0, cull_rate=0.0, promoted=0, promote_rate=0.0)
    culled = promoted = 0
    for u in np.unique(tz.zone_id[alive]):
        idx = np.flatnonzero((tz.zone_id == u) & (tz.merged == 0))
        if idx.size == 0:
            continue
        order = idx[np.argsort(tz.score[idx])]          # ascending by score
        k = int(round(idx.size * cull_frac))
        p = int(round(idx.size * promote_frac))
        if k > 0:
            reinit(tz, order[:k], rng)
            culled += int(k)
        if p > 0:
            best = order[::-1][:p]
            top = best[tz.score[best] > 0.85]           # elites go resident
            rest = np.setdiff1d(best, top)
            tz.merged[top] = 2
            tz.merged[rest] = 1
            promoted += int(best.size)
    tz.cull_events += culled
    tz.promote_events += promoted
    if tz.promote_round_log is not None:
        tz.promote_round_log.append((promoted, int(alive.size)))
    return dict(culled=culled, cull_rate=round(culled / max(1, alive.size), 4),
                promoted=promoted,
                promote_rate=round(promoted / max(1, alive.size), 4))
