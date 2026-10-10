"""VN population: structure-of-arrays container, CSR adjacency, quantisation.

Layout is deliberately flat (NumPy, no Python objects per neuron) so that the
whole thing can be ported to C++17 / Android NDK later by replacing each array
with a ``float*`` / ``uint8*``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 8          # width of direction / state vectors
RANK = 4         # low-rank adapter rank
TWO_PI = 2.0 * np.pi

MODE_SAFE, MODE_RESIDENT, MODE_BACKGROUND, MODE_ACTIVE = 0, 1, 2, 3
MODE_DEEP, MODE_ARCHIVE, MODE_TRIAL, MODE_RESERVED = 4, 5, 6, 7
MODE_NAMES = ("SAFE", "RESIDENT", "BACKGROUND", "ACTIVE",
              "DEEP", "ARCHIVE", "TRIAL", "RESERVED")

HEALTH_LOW, HEALTH_NORMAL, HEALTH_IMPORTANT, HEALTH_CORE = 7, 15, 23, 31


@dataclass
class VNPopulation:
    """All fields are arrays indexed by neuron id, except the weight matrices."""
    n: int
    addr: np.ndarray        # uint32 [N]
    route_key: np.ndarray   # uint32 [N]
    adapter_ref: np.ndarray  # uint32 [N]
    energy: np.ndarray      # float32 [N]
    phase: np.ndarray       # float32 [N] radians, frozen in stage 1
    mode: np.ndarray        # uint8 [N]
    health: np.ndarray      # uint8 [N]
    direction: np.ndarray   # float32 [N, DIM]
    state: np.ndarray       # float32 [N, DIM]  (h)
    scale: np.ndarray       # float32 [N]       (s)
    ptr: np.ndarray         # int64 [N+1] CSR row pointer
    src: np.ndarray         # int32 [E]
    dst: np.ndarray         # int32 [E]
    weight: np.ndarray      # float32 [E]  w_ij
    dir_cos: np.ndarray     # float32 [E]  d_ij
    kedge: np.ndarray       # float32 [E]  w*d*cos(dphi), phase frozen
    deg: np.ndarray         # float32 [N]
    W: np.ndarray           # float32 [DIM, DIM]
    bias: np.ndarray        # float32 [DIM]
    adapter_a: np.ndarray   # float32 [N, RANK, DIM]
    adapter_b: np.ndarray   # float32 [N, DIM, RANK]

    @property
    def n_edges(self) -> int:
        return int(self.dst.shape[0])

    def nbytes(self) -> int:
        return sum(v.nbytes for v in vars(self).values()
                   if isinstance(v, np.ndarray))


def quantise_u8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a float array onto 8 bits over [lo, hi]."""
    t = (np.clip(x, lo, hi) - lo) / (hi - lo)
    return np.rint(t * 255.0).astype(np.uint8)


def dequantise_u8(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse of :func:`quantise_u8`."""
    return (q.astype(np.float32) / 255.0) * (hi - lo) + lo


def _random_edges(n: int, degree: int, rng: np.random.Generator):
    """Undirected random graph, no self loops, no duplicate edges."""
    off = rng.integers(1, n, size=(n, degree))
    dst = (np.arange(n)[:, None] + off) % n
    src = np.repeat(np.arange(n), degree)
    dst = dst.reshape(-1)
    a = np.concatenate([src, dst]).astype(np.int64)
    b = np.concatenate([dst, src]).astype(np.int64)
    _, idx = np.unique(a * n + b, return_index=True)
    a, b = a[idx], b[idx]
    order = np.argsort(a, kind="stable")
    return a[order].astype(np.int32), b[order].astype(np.int32)


def build_population(n: int = 1000, degree: int = 8, seed: int = 0,
                     mode: int = MODE_BACKGROUND,
                     health: int = 16,
                     weight_sigma: float = 0.1) -> VNPopulation:
    """Create a population with a random degree-``degree`` graph.

    Weight mean is exactly 0 (centred) so no systematic positive feedback can
    build up after ``tanh`` saturation.
    """
    rng = np.random.default_rng(seed)
    src, dst = _random_edges(n, degree, rng)

    weight = rng.normal(0.0, weight_sigma, size=src.shape[0]).astype(np.float32)
    weight -= weight.mean()

    direction = rng.normal(size=(n, DIM)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    du = direction[src]
    dv = direction[dst]
    dir_cos = (du * dv).sum(1) / (
        np.linalg.norm(du, axis=1) * np.linalg.norm(dv, axis=1) + 1e-8)
    dir_cos = dir_cos.astype(np.float32)

    phase = rng.uniform(0.0, TWO_PI, size=n).astype(np.float32)
    kedge = (weight * dir_cos * np.cos(phase[dst] - phase[src])).astype(np.float32)

    deg = np.bincount(src, minlength=n).astype(np.float32)
    ptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)

    return VNPopulation(
        n=n,
        addr=rng.integers(0, 2**31, size=n, dtype=np.int64).astype(np.uint32),
        route_key=rng.integers(0, 2**31, size=n, dtype=np.int64).astype(np.uint32),
        adapter_ref=np.arange(n, dtype=np.int64).astype(np.uint32),
        energy=rng.uniform(0.0, 0.2, size=n).astype(np.float32),
        phase=phase,
        mode=np.full(n, mode, dtype=np.uint8),
        health=np.full(n, health, dtype=np.uint8),
        direction=direction,
        state=np.zeros((n, DIM), dtype=np.float32),
        scale=np.ones(n, dtype=np.float32),
        ptr=ptr, src=src, dst=dst, weight=weight, dir_cos=dir_cos, kedge=kedge,
        deg=deg,
        W=(rng.normal(0.0, 1.0 / np.sqrt(DIM), size=(DIM, DIM))).astype(np.float32),
        bias=np.zeros(DIM, dtype=np.float32),
        adapter_a=rng.normal(0.0, 0.01, size=(n, RANK, DIM)).astype(np.float32),
        adapter_b=rng.normal(0.0, 0.01, size=(n, DIM, RANK)).astype(np.float32),
    )
