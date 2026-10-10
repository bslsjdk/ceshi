"""Region management: RZ resident, EZ external, FVN / EVN / TVN (stage 4).

Every slot is preallocated and the CSR graph never changes shape.  Loading an
EVN activates a slot, releasing deactivates it.  A C++ port wants exactly this:
no per-tick graph surgery, only an active mask the update kernel skips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neuron import MODE_ACTIVE, MODE_BACKGROUND, MODE_RESIDENT, MODE_SAFE

REGION_RZ, REGION_EZ, REGION_TZ = 0, 1, 2
REGION_NAMES = ("RZ", "EZ", "TZ")
RZ_BUDGET_BYTES = 300 * 1024 * 1024
EZ_BUDGET_BYTES = 500 * 1024 * 1024


@dataclass
class RegionState:
    """Per-slot region bookkeeping layered over a fixed-size population."""
    n: int
    active: np.ndarray          # bool [N]: resident in RZ or loaded in EZ
    region: np.ndarray          # uint8 [N]
    frozen: np.ndarray          # bool [N]: FVN, never mutated
    trigger_at: np.ndarray      # float64 [N]: TVN firing step, -1 = none
    trigger_value: np.ndarray   # float32 [N]
    fired: np.ndarray           # bool [N]
    e_std: float = 0.05
    loaded_count: int = 0
    released_count: int = 0
    overflow_count: int = 0
    triggered_count: int = 0

    @property
    def mutable(self) -> np.ndarray:
        """Slots the update equations are allowed to write to."""
        return self.active & ~self.frozen

    def bytes_rz(self, per_slot: int) -> int:
        k = int(np.count_nonzero(self.active & (self.region == REGION_RZ)))
        return k * per_slot

    def bytes_ez(self, per_slot: int) -> int:
        k = int(np.count_nonzero(self.active & (self.region == REGION_EZ)))
        return k * per_slot

    def update_e_std(self, energy: np.ndarray) -> None:
        live = energy[self.active]
        if live.size == 0:
            return
        v = float(np.std(live))
        if np.isfinite(v) and v > 0.0:
            self.e_std = v

    def overflow_mask(self, energy: np.ndarray, e_target: float) -> np.ndarray:
        thr = e_target + 2.0 * self.e_std
        return self.active & (self.region == REGION_RZ) & (energy > thr)

    def maybe_load(self, pop, e_target: float) -> int:
        """Load inactive slots into EZ when RZ boundary neurons run hot."""
        hot = self.overflow_mask(pop.energy, e_target)
        if not hot.any():
            return 0
        self.overflow_count += 1
        free = np.flatnonzero(~self.active)
        if free.size == 0:
            return 0
        k = int(min(free.size, max(1, int(np.count_nonzero(hot)))))
        pick = free[:k]
        self.active[pick] = True
        self.region[pick] = REGION_EZ
        pop.mode[pick] = MODE_ACTIVE
        pop.energy[pick] = np.float32(e_target * 0.5)
        pop.state[pick] = 0.0
        self.loaded_count += k
        return k

    def maybe_release(self, energy: np.ndarray) -> int:
        """Release EZ slots sitting in the bottom decile of the live field.

        The threshold tracks the empirical P10 rather than a closed form,
        because the steady-state band is set by e_target and kappa and a fixed
        formula cannot know where the lower tail actually lands.
        """
        ez = np.flatnonzero(self.active & (self.region == REGION_EZ))
        if ez.size == 0:
            return 0
        live = energy[self.active]
        thr = float(np.percentile(live, 10)) * 1.5
        cold = ez[energy[ez] < thr]
        if cold.size == 0:
            return 0
        self.active[cold] = False
        self.released_count += int(cold.size)
        return int(cold.size)

    def fire_timers(self, pop, step: int) -> int:
        """TVN: at trigger_at, push energy in and let the wave carry it out."""
        due = ((self.trigger_at >= 0.0) & (self.trigger_at <= step)
               & self.active & ~self.fired)
        if not due.any():
            return 0
        idx = np.flatnonzero(due)
        pop.energy[idx] += self.trigger_value[idx]
        self.fired[idx] = True
        self.triggered_count += int(idx.size)
        return int(idx.size)


def make_regions(n: int, n_rz: int, seed: int = 0,
                 n_fvn: int = 0, n_tvn: int = 0) -> RegionState:
    """RZ holds slots [0, n_rz); the rest are dormant EVN pool slots."""
    rng = np.random.default_rng(seed + 991)
    active = np.zeros(n, dtype=bool)
    active[:n_rz] = True
    region = np.full(n, REGION_RZ, dtype=np.uint8)
    region[n_rz:] = REGION_EZ
    frozen = np.zeros(n, dtype=bool)
    if n_fvn > 0:
        frozen[:n_fvn] = True
    trigger_at = np.full(n, -1.0, dtype=np.float64)
    if n_tvn > 0:
        start = n_fvn
        trigger_at[start:start + n_tvn] = rng.integers(
            500, 5000, size=n_tvn).astype(np.float64)
    return RegionState(
        n=n, active=active, region=region, frozen=frozen,
        trigger_at=trigger_at,
        trigger_value=np.full(n, 0.5, dtype=np.float32),
        fired=np.zeros(n, dtype=bool),
    )
