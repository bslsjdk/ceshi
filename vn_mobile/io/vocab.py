"""Character vocabulary wired onto address blocks (stage 6A).

"Similar things share an address" is the design rule this stage leans on: one
address block per character, so a symbol and the neurons that carry it are the
same thing.  Encoding a character is raising energy in its block; decoding is
pooling each block's output into a logit.

The pool is a fixed partition of the population, not a learned assignment, so
it is reproducible from a seed and needs no extra state.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "tinyshakespeare.txt"
UNK = "<unk>"


def load_text(path: Path | str = DATA) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def build_vocab(text: str) -> tuple[list[str], dict[str, int]]:
    symbols = sorted(set(text))
    return symbols, {c: i for i, c in enumerate(symbols)}


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    """Text -> int32 indices; unknown characters collapse to one bucket."""
    if UNK in stoi:
        return np.array([stoi.get(c, stoi[UNK]) for c in text],
                        dtype=np.int32)
    return np.array([stoi[c] for c in text], dtype=np.int32)


def split(text: str, frac: float = 0.9) -> tuple[str, str]:
    n = int(len(text) * frac)
    return text[:n], text[n:]


def block_of(n_vn: int, vocab: int) -> np.ndarray:
    """Which character block each VN belongs to -> int32 [n_vn].

    Contiguous by index so neighbours in the same block are also near each
    other in the population's own ordering.
    """
    return np.minimum(np.arange(n_vn) * vocab // n_vn, vocab - 1
                      ).astype(np.int32)


def block_members(block: np.ndarray, vocab: int) -> list[np.ndarray]:
    return [np.flatnonzero(block == v) for v in range(vocab)]


def one_hot(idx: np.ndarray, vocab: int) -> np.ndarray:
    """int32 [B] -> float32 [B, V]."""
    out = np.zeros((idx.size, vocab), dtype=np.float32)
    out[np.arange(idx.size), idx] = 1.0
    return out


def pool_by_block(values: np.ndarray, block: np.ndarray,
                  vocab: int) -> np.ndarray:
    """Sum ``values`` inside each character block -> float32 [V, ...]."""
    out = np.zeros((vocab,) + values.shape[1:], dtype=np.float32)
    np.add.at(out, block, values)
    return out
