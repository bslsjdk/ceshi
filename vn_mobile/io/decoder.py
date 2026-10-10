"""Output decoding: read a snapshot back into something readable.

Stage 1 is intentionally dumb -- it reports which neurons lit up.  Any real
"answer" needs the diffusion / readout work that lands in stage 2.
"""
from __future__ import annotations

import numpy as np

from ..core.neuron import VNPopulation
from ..core.update import read_output


def output_field(pop: VNPopulation) -> np.ndarray:
    return read_output(pop)


def top_neurons(pop: VNPopulation, k: int = 8) -> list[tuple[int, float]]:
    mag = np.linalg.norm(read_output(pop), axis=1)
    ids = np.argsort(mag)[::-1][:k]
    return [(int(i), float(mag[i])) for i in ids]


def energy_histogram(pop: VNPopulation, bins: int = 10) -> np.ndarray:
    counts, _ = np.histogram(pop.energy, bins=bins, range=(0.0, 1.0))
    return counts


def describe(pop: VNPopulation, k: int = 8) -> str:
    top = top_neurons(pop, k)
    ids = ", ".join(f"{i}:{m:.3f}" for i, m in top)
    return (f"e_mean={pop.energy.mean():.4f} e_max={pop.energy.max():.4f} "
            f"top=[{ids}]")
