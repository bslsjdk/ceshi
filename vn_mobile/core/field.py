"""The continuously running field: drives the update loop and keeps counters.

There is no "start" and "finish": the field always runs, input is a
perturbation, and reads return a snapshot of the current state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .group import BlockLayout, build_blocks
from .neuron import DIM, VNPopulation
from .update import (
    KAPPA,
    K_COUPLING,
    normalise_energy,
    read_output,
    step_energy,
    step_phase,
    step_state,
    sync_order_parameter,
)


@dataclass
class FieldStats:
    step: int = 0
    energy_max: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    total_energy: float = 0.0
    top_frac: float = 0.0        # fraction with e > 0.95
    neg_frac: float = 0.0        # fraction with e < 0
    sync_r: float = 0.0          # Kuramoto order parameter
    state_max: float = 0.0
    nan: bool = False
    rescales: int = 0


@dataclass
class Field:
    pop: VNPopulation
    seed: int = 0
    block_size: int = 64
    norm_every: int = 100
    kappa: float = KAPPA
    k_coupling: float = K_COUPLING
    rng: np.random.Generator = field(init=False)
    layout: BlockLayout = field(init=False)
    stats: FieldStats = field(init=False)
    injection: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.layout = build_blocks(self.pop, self.block_size)
        self.stats = FieldStats()
        self.injection = np.zeros((self.pop.n, DIM), dtype=np.float32)

    def perturb(self, ids: np.ndarray, values: np.ndarray) -> None:
        """Inject an external drive.  Cleared on the next tick."""
        self.injection[:] = 0.0
        if len(ids):
            self.injection[ids] = values.astype(np.float32)

    def tick(self) -> None:
        # Energy reads the current phases, then the phases advance; both use
        # phi(t), matching the stage-2 spec.
        step_energy(self.pop, self.rng, kappa=self.kappa)
        step_phase(self.pop, k=self.k_coupling)
        step_state(self.pop, self.injection)
        s = self.stats
        s.step += 1
        if s.step % self.norm_every == 0:
            if normalise_energy(self.pop):
                s.rescales += 1

    def measure(self) -> FieldStats:
        e = self.pop.energy
        s = self.stats
        s.energy_max = float(e.max())
        s.energy_mean = float(e.mean())
        s.energy_std = float(e.std())
        s.total_energy = float(e.sum())
        s.top_frac = float((e > 0.95).mean())
        s.neg_frac = float((e < 0.0).mean())
        s.sync_r = sync_order_parameter(self.pop)
        s.state_max = float(np.abs(self.pop.state).max())
        s.nan = bool(np.isnan(e).any() or np.isnan(self.pop.state).any()
                     or np.isnan(self.pop.phase).any())
        return s

    def run(self, steps: int, log_every: int = 0, reporter=None) -> FieldStats:
        for i in range(steps):
            self.tick()
            if log_every and (i + 1) % log_every == 0:
                st = self.measure()
                if reporter is not None:
                    reporter(st)
                if st.nan:
                    break
        return self.measure()

    def snapshot(self, top_k: int = 8) -> dict:
        """Current state readout: the "output" of the system."""
        y = read_output(self.pop)
        mag = np.linalg.norm(y, axis=1)
        ids = np.argsort(mag)[::-1][:top_k]
        return {
            "step": self.stats.step,
            "top_ids": ids.tolist(),
            "top_magnitude": mag[ids].tolist(),
            "energy_mean": float(self.pop.energy.mean()),
            "energy_max": float(self.pop.energy.max()),
        }
