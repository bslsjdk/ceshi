"""Vectorised per-zone statistics (stage 5A optimisation pass).

Lives in its own module because ``trial`` and ``tzbuild`` import each other;
a helper placed in either one would close the cycle.
"""
from __future__ import annotations

import numpy as np


def zone_median_std(vals: np.ndarray, zone_id: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-zone median and std, broadcast back to each member.

    The old version looped over zones in Python calling ``np.median`` and
    ``np.std``: 40 numpy calls per evaluation, ~75% of the evaluation cost and
    ~18% of the whole step.  The median comes from one lexsort by
    (zone, value) -- groups end up contiguous, so the middle elements are
    addressable -- and the std from two bincounts.  No Python loop.
    """
    v = np.asarray(vals, dtype=np.float64)
    z = np.asarray(zone_id, dtype=np.int64)
    cnt = np.bincount(z)
    nz = int(cnt.size)
    safe = np.maximum(cnt, 1)
    s1 = np.bincount(z, weights=v, minlength=nz)
    s2 = np.bincount(z, weights=v * v, minlength=nz)
    mean = s1 / safe
    std = np.sqrt(np.maximum(0.0, s2 / safe - mean * mean))
    order = np.lexsort((v, z))
    sv = v[order]
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    med = 0.5 * (sv[starts + (cnt - 1) // 2] + sv[starts + cnt // 2])
    return med[z].astype(np.float32), std[z].astype(np.float32)


def zone_means(vals: np.ndarray, zone_id: np.ndarray, mask: np.ndarray
               | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(zone ids present, mean of ``vals`` per zone) for the alive subset."""
    v = np.asarray(vals, dtype=np.float64)
    z = np.asarray(zone_id, dtype=np.int64)
    if mask is not None:
        v, z = v[mask], z[mask]
    ids = np.unique(z)
    return ids, np.array([v[z == u].mean() for u in ids])
