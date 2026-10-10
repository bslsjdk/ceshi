"""Stage 4 region management: RZ / EZ zones, FVN / EVN neurons, TVN timers.

No new equations are added here.  This module only owns *where* a neuron lives
and *when* it is allowed to change:

    RZ  resident zone, always in memory
    EZ  external zone, loaded on demand, released when it goes quiet
    TZ  trial zone (see trial.py)

    FVN fixed neuron, ``frozen=True``, never mutated
    EVN external neuron, lives in the pool until loaded into EZ
    TVN timed neuron, fires a wave when its clock comes due
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .neuron import MODE_BACKGROUND, MODE_RESIDENT, MODE_SAFE, VNPopulation

REGION_RZ, REGION_EZ, REGION_TZ = 0, 1, 2
REGION_NAMES = ("RZ", "EZ", "TZ")

NO_TRIGGER = -1.0


@dataclass
class RegionManager:
    pop: VNPopulation
    pool: np.ndarray = field(init=False)        # EVN ids not currently loaded
    ez_ids: np.ndarray = field(init=False)      # EVN ids currently in EZ
    trigger_at: np.ndarray = field(init=False)  # float32 [N], NO_TRIGGER = none
    threshold: float = 0.0                      # EZ release threshold
    e_std: float = 0.05                         # sliding estimate
    loads: int = 0
    releases: int = 0
    fired: int = 0

    def __post_init__(self) -> None:
        self.pool = np.empty(0, dtype=np.int64)
        self.ez_ids = np.empty(0, dtype=np.int64)
        self.trigger_at = np.full(self.pop.n, NO_TRIGGER, dtype=np.float32)

    # --- neuron kinds -------------------------------------------------
    def mark_fvn(self, ids: np.ndarray, mode: int = MODE_SAFE) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        self.pop.frozen[ids] = True
        self.pop.mode[ids] = mode
        self.pop.region[ids] = REGION_RZ

    def mark_evn(self, ids: np.ndarray, mode: int = MODE_BACKGROUND) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        self.pop.frozen[ids] = False
        self.pop.mode[ids] = mode
        self.pool = np.union1d(self.pool, ids)

    def set_timer(self, ids: np.ndarray, at: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        self.trigger_at[ids] = np.asarray(at, dtype=np.float32)

    # --- thresholds ---------------------------------------------------
    def update_threshold(self, e_target: float = 0.3, mult: float = 1.5) -> float:
        """Release threshold from the empirical low tail: P10 * mult.

        A fixed ``e_target - 2*e_std`` never fires in steady state because the
        distribution never reaches that low; tracking the realised P10 keeps
        the threshold attached to the distribution that is actually there.
        """
        e = self.pop.energy
        self.e_std = float(e.std())
        self.threshold = float(np.percentile(e, 10) * mult)
        return self.threshold

    def overflow_ids(self, e_target: float = 0.3, k: float = 2.0) -> np.ndarray:
        """RZ neurons whose energy pushed past the band -> ask EZ for help."""
        cut = e_target + k * self.e_std
        in_rz = self.pop.region == REGION_RZ
        return np.flatnonzero(in_rz & (self.pop.energy > cut))

    # --- load / release -----------------------------------------------
    def load(self, count: int) -> int:
        count = int(min(count, self.pool.size))
        if count <= 0:
            return 0
        ids = self.pool[:count]
        self.pool = self.pool[count:]
        self.ez_ids = np.union1d(self.ez_ids, ids)
        self.pop.region[ids] = REGION_EZ
        self.loads += int(ids.size)
        return int(ids.size)

    def release(self) -> int:
        if self.ez_ids.size == 0:
            return 0
        quiet = self.ez_ids[self.pop.energy[self.ez_ids] < self.threshold]
        if quiet.size == 0:
            return 0
        self.ez_ids = np.setdiff1d(self.ez_ids, quiet)
        self.pool = np.union1d(self.pool, quiet)
        self.pop.region[quiet] = REGION_RZ
        self.releases += int(quiet.size)
        return int(quiet.size)

    # --- timers -------------------------------------------------------
    def tick_timers(self, t: float, energy: float = 0.5) -> np.ndarray:
        """Fire every TVN whose clock has come due.  Returns the ids fired."""
        due = np.flatnonzero((self.trigger_at >= 0.0) & (self.trigger_at <= t))
        if due.size == 0:
            return due
        self.trigger_at[due] = NO_TRIGGER
        self.pop.energy[due] += np.float32(energy)
        self.fired += int(due.size)
        return due

    def stats(self) -> dict:
        return {
            "pool": int(self.pool.size),
            "ez_loaded": int(self.ez_ids.size),
            "loads": self.loads,
            "releases": self.releases,
            "timers_fired": self.fired,
            "threshold": round(self.threshold, 4),
            "e_std": round(self.e_std, 4),
            "fvn": int(self.pop.frozen.sum()),
        }


def split_resident(n: int, n_rz: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic split of ids into a resident part and the rest."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n)
    return np.sort(ids[:n_rz]), np.sort(ids[n_rz:])
