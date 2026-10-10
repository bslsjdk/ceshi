"""Trial-zone construction helpers (stage 5A).

Kept out of trial.py so that module stays under the 200-line budget.
"""
from __future__ import annotations

import numpy as np

from .neuron import MODE_TRIAL, build_population
from .trial import BASE_HORIZON, SAMPLE_EVERY, TrialZone
from .update import step_energy, step_phase


def zone_of_main(main_addr: np.ndarray, n_zones: int) -> np.ndarray:
    """Split the main system into contiguous address blocks."""
    n = main_addr.shape[0]
    order = np.argsort(main_addr, kind="stable")
    z = np.empty(n, dtype=np.int64)
    z[order] = (np.arange(n) * n_zones) // n
    return z


def zone_horizon(zone_id: np.ndarray, n_zones: int, lo: float = 0.5,
                 step: float = 0.1) -> np.ndarray:
    """Samples of look-ahead per zone: BASE_HORIZON*(0.5+0.1*i) ticks.

    Zone 0 predicts 25 ticks out, zone 19 predicts 120.  Unequal difficulty is
    what makes cross-zone score spread exceed within-zone spread.
    """
    i = np.asarray(zone_id, dtype=np.float64)
    ticks = BASE_HORIZON * (lo + step * np.minimum(i, n_zones - 1))
    return np.maximum(2, (ticks / SAMPLE_EVERY + 0.5).astype(np.int32))


def make_trial_zone(n_bn: int = 1000, n_main: int = 1000, n_zones: int = 20,
                    k_shadow: int = 8, degree: int = 6, seed: int = 7,
                    main_addr: np.ndarray | None = None,
                    util_norm: str = "zone", hz_lo: float = 0.5,
                    hz_step: float = 0.1) -> TrialZone:
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
        merged=np.zeros(n_bn, dtype=np.int8),
        hz_bn=zone_horizon(zone_id, n_zones, lo=hz_lo, step=hz_step),
        util_norm=util_norm,
        reinit_round=np.zeros(n_bn, dtype=np.int32),
        samples=np.zeros(n_bn, dtype=np.int32),
    )


def tick_trial(tz: TrialZone, rng: np.random.Generator) -> None:
    """Keep the TZ wave field alive; per-BN kappa gives each BN its own gain."""
    step_energy(tz.pop, rng, kappa=tz.kappa)
    step_phase(tz.pop)
