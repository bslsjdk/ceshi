"""Address blocks: "similar things live at similar addresses".

Stage 1 only provides the block layout and cheap statistics.  Similarity-based
rewiring (route_key top-k / addr adjacency) is stage 2 work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neuron import VNPopulation


@dataclass
class BlockLayout:
    order: np.ndarray      # int64 [N] neuron ids sorted by address
    block_id: np.ndarray   # int64 [N] block index of each neuron
    bounds: np.ndarray     # int64 [B+1] slice bounds into ``order``

    @property
    def n_blocks(self) -> int:
        return int(self.bounds.shape[0] - 1)


def build_blocks(pop: VNPopulation, block_size: int = 64) -> BlockLayout:
    """Sort neurons by address and cut them into fixed-size blocks."""
    order = np.argsort(pop.addr.astype(np.int64), kind="stable")
    n_blocks = int(np.ceil(pop.n / block_size))
    block_id = np.minimum(np.arange(pop.n) // block_size, n_blocks - 1)
    bounds = np.arange(n_blocks + 1, dtype=np.int64) * block_size
    bounds[-1] = pop.n
    return BlockLayout(order=order, block_id=block_id, bounds=bounds)


def block_stats(pop: VNPopulation, layout: BlockLayout) -> dict:
    """Per-block energy / activity summary, used by the decoder and logs."""
    starts, ends = layout.bounds[:-1], layout.bounds[1:]
    energy = pop.energy[layout.order]
    out = {
        "n_blocks": layout.n_blocks,
        "energy_mean": [],
        "energy_max": [],
        "active_frac": [],
    }
    for s, e in zip(starts, ends):
        seg = energy[s:e]
        out["energy_mean"].append(float(seg.mean()))
        out["energy_max"].append(float(seg.max()))
        out["active_frac"].append(float((seg > 0.5).mean()))
    for k in ("energy_mean", "energy_max", "active_frac"):
        out[k] = np.asarray(out[k], dtype=np.float64)
    return out


def hot_blocks(pop: VNPopulation, layout: BlockLayout, top_k: int = 5) -> np.ndarray:
    """Block indices with the highest mean energy."""
    stats = block_stats(pop, layout)
    return np.argsort(stats["energy_mean"])[::-1][:top_k]
