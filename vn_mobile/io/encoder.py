"""Input encoding: turn an external signal into a perturbation on the field."""
from __future__ import annotations

import hashlib

import numpy as np

from ..core.neuron import DIM


def seed_ids(text: str, n: int, k: int = 16) -> np.ndarray:
    """Deterministically pick ``k`` neurons from a string.

    Stage 1 uses a plain hash so the same input always hits the same neurons;
    stage 2 replaces this with route-key similarity lookup.
    """
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(h, "big"))
    return rng.choice(n, size=min(k, n), replace=False)


def encode(text: str, n_neurons: int, k: int = 16,
           amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ids, values[DIM])`` ready for :meth:`Field.perturb`."""
    ids = seed_ids(text, n_neurons, k)
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=DIM).digest()
    raw = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    raw = (raw / 255.0) * 2.0 - 1.0
    values = np.tile(raw, (len(ids), 1)) * amplitude
    return ids, values.astype(np.float32)


def encode_scalar(value: float, ids: np.ndarray) -> np.ndarray:
    """Spread a single scalar over all directions of the given neurons."""
    return np.full((len(ids), DIM), value, dtype=np.float32)
