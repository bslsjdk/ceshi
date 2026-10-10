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
    phase: np.ndarray       # float32 [N] radians, Kuramoto-driven in stage 2
    omega: np.ndarray       # float32 [N] intrinsic frequency
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
    wedge: np.ndarray       # float32 [E]  w_ij*d_ij, phase-independent part
    deg: np.ndarray         # float32 [N]
    W: np.ndarray           # float32 [DIM, DIM]
    bias: np.ndarray        # float32 [DIM]
    adapter_a: np.ndarray   # float32 [M, RANK, DIM],  M = N or n_zones
    adapter_b: np.ndarray   # float32 [M, DIM, RANK]
    region: np.ndarray      # uint8 [N]   0=RZ 1=EZ 2=TZ (stage 4)
    frozen: np.ndarray      # bool [N]    FVN: never mutated (stage 4)
    zone_of: np.ndarray | None = None  # int32 [N] -> adapter group, or None
    A1: object = None       # cached scipy CSR, unit weights (see update._adjacency)
    Aw: object = None       # cached scipy CSR, wedge weights
    WM: object = None       # cached [N, DIM, DIM] fused W + B@A (see update._fused)

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


def make_omega(n: int, rng: np.random.Generator,
               std: float = 0.3,
               mix: tuple[float, float, float] | None = None) -> np.ndarray:
    """Intrinsic frequencies, optionally layered.

    ``mix`` = (frac_on_std, std, std2): the first ``frac`` of neurons get
    N(0, std) and the rest N(0, std2).  Stage 4 uses 80% @ 0.3 + 20% @ 1.0 so
    the bulk keeps a sync frame while a minority keeps the phase drifting.
    """
    if mix is None:
        return rng.normal(0.0, std, size=n).astype(np.float32)
    frac, s1, s2 = mix
    k = int(round(n * frac))
    out = np.empty(n, dtype=np.float32)
    out[:k] = rng.normal(0.0, s1, size=k)
    out[k:] = rng.normal(0.0, s2, size=n - k)
    return out


def build_population(n: int = 1000, degree: int = 8, seed: int = 0,
                     mode: int = MODE_BACKGROUND,
                     health: int = 16,
                     weight_sigma: float = 0.5,
                     omega_std: float = 0.3,
                     omega_mix: tuple[float, float, float] | None = None,
                     region: int = 0,
                     n_zones: int = 0) -> VNPopulation:
    """Create a population with a random degree-``degree`` graph.

    Weight mean is exactly 0 (centred) so no systematic positive feedback can
    build up after ``tanh`` saturation.

    ``n_zones > 0`` switches the adapters from per-neuron to per-zone: the
    population is cut into that many contiguous address blocks and A/B are
    sized [n_zones, ...] instead of [n, ...].  That is what keeps the
    trainable set inside a parameter budget -- per-neuron A/B costs 64,000
    at n=1000, rank 4, on its own.
    """
    rng = np.random.default_rng(seed)
    src, dst = _random_edges(n, degree, rng)
    zones = (np.arange(n) // max(1, n // n_zones)).astype(
        np.int32) if n_zones > 0 else None
    n_ad = n_zones if n_zones > 0 else n

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
    # cos(phi_j - phi_i) is now dynamic, so only the static w*d factor is
    # precomputed; the phase factor is applied per tick in step_energy.
    wedge = (weight * dir_cos).astype(np.float32)

    deg = np.bincount(src, minlength=n).astype(np.float32)
    ptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)

    return VNPopulation(
        n=n,
        addr=rng.integers(0, 2**31, size=n, dtype=np.int64).astype(np.uint32),
        route_key=rng.integers(0, 2**31, size=n, dtype=np.int64).astype(np.uint32),
        adapter_ref=np.arange(n, dtype=np.int64).astype(np.uint32),
        energy=rng.uniform(0.0, 0.2, size=n).astype(np.float32),
        phase=phase,
        omega=make_omega(n, rng, omega_std, omega_mix),
        mode=np.full(n, mode, dtype=np.uint8),
        health=np.full(n, health, dtype=np.uint8),
        direction=direction,
        state=np.zeros((n, DIM), dtype=np.float32),
        scale=np.ones(n, dtype=np.float32),
        ptr=ptr, src=src, dst=dst, weight=weight, dir_cos=dir_cos, wedge=wedge,
        deg=deg,
        W=(rng.normal(0.0, 1.0 / np.sqrt(DIM), size=(DIM, DIM))).astype(np.float32),
        bias=np.zeros(DIM, dtype=np.float32),
        adapter_a=rng.normal(0.0, 0.01, size=(n_ad, RANK, DIM)).astype(np.float32),
        adapter_b=rng.normal(0.0, 0.01, size=(n_ad, DIM, RANK)).astype(np.float32),
        region=np.full(n, region, dtype=np.uint8),
        frozen=np.zeros(n, dtype=bool),
        zone_of=zones,
    )
