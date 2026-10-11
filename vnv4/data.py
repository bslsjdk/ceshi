"""Task A / Task B corpora for the continual-learning protocol.

Task A = TinyShakespeare (1.11 MB, already vendored in this repo).
Task B = a real WikiText-format corpus fetched from GitHub (63,966 B).

Why not the full WikiText-2: raw.githubusercontent.com and huggingface.co
are unreachable from this sandbox (read timeouts), and the only wikitext-2
copies reachable through the GitHub API turned out to be a *synthetic
fallback* -- one 90,400-byte sentence repeated.  This file is genuine
Wikipedia prose in wikitext token format, not that fallback.

Both tasks share one character vocabulary so that the two phases are
directly comparable and no model gains capacity from a task-specific
alphabet.
"""
from __future__ import annotations

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load_tasks(val_frac: float = 0.1):
    a = _read(os.path.join(ROOT, "tinyshakespeare.txt"))
    b = _read(os.path.join(ROOT, "wikitext_b.txt"))
    chars = sorted(set(a) | set(b))
    stoi = {c: i for i, c in enumerate(chars)}
    V = len(chars)
    enc = lambda s: np.array([stoi[c] for c in s], dtype=np.int64)
    out = {}
    for name, txt in (("A", a), ("B", b)):
        ids = enc(txt)
        n = int(len(ids) * (1.0 - val_frac))
        out[name] = {"train": ids[:n], "val": ids[n:], "n_chars": len(ids)}
    return out, stoi, V


class Sampler:
    """Draws contiguous (x, y) windows; y is x shifted by one."""

    def __init__(self, ids: np.ndarray, T: int, seed: int = 0):
        self.ids = ids
        self.T = T
        self.rng = np.random.default_rng(seed)

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        hi = len(self.ids) - self.T - 1
        s = int(self.rng.integers(0, hi))
        x = self.ids[s:s + self.T]
        y = self.ids[s + 1:s + 1 + self.T]
        return x, y


def nll(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood in nats over a (T, V) / (T,) pair."""
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return float((-(logits[np.arange(len(y)), y] - m.ravel()
                    - np.log(e.sum(axis=1)))).mean())


def eval_nll(model, sampler, n_batches: int) -> float:
    """Average NLL over fresh windows; models expose eval_nll(x, y)."""
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = sampler.next()
        tot += model.eval_nll(x, y)
        n += 1
    return tot / max(n, 1)
