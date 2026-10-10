"""Kernel construction and evaluation for the concept wave (<=200 lines).

Edges are the six directions of the spec, folded into one CSR matrix so a
round is a single sparse matvec:

    (l,j)      same layer, concept neighbour
    (l+1,i)    the same slot one layer up
    (l-1,i)    the same slot one layer down
    (l+1,j)    concept neighbour one layer up
    (l-1,j)    concept neighbour one layer down
    (l,i,t+1)  in-place time evolution (a self loop)

The weight is ``w[dst_layer] * m_ij * g[src_layer]`` where ``m_ij`` is the
cosine between the two concept vectors.  ``m_ij`` is *signed*: an antipodal
neighbour contributes with a minus sign, which is one of the two ways paths
cancel (the other is phase).
"""
from __future__ import annotations

import numpy as np


def normalise(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalise(a) @ normalise(b).T


def _knn(sim: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k strongest neighbours per row, self excluded. -> (rows, dst, val)."""
    n = sim.shape[0]
    k = min(k, n - 1)
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    dst = idx.ravel()
    return rows, dst, sim[rows, dst]


def build_edges(concept, layer, per_layer, k, nlayers):
    """Signed CSR for all six directions.  Returns dict of arrays."""
    n = concept.shape[0]
    sim = cosine(normalise(concept).astype(np.float32),
                 normalise(concept).astype(np.float32)).astype(np.float32)
    rows_l, dst_l, val_l = [], [], []
    same = sim.copy()
    np.fill_diagonal(same, -np.inf)
    r, d, v = _knn(same, k)
    keep = layer[d] == layer[r]
    rows_l.append(r[keep]); dst_l.append(d[keep]); val_l.append(v[keep])
    p = np.arange(n) % per_layer
    li = layer.astype(np.int64)
    for delta in (-1, 1):
        tgt = np.arange(n) + delta * per_layer
        ok = (li + delta >= 0) & (li + delta < nlayers)
        # (l+delta, i) : the same slot one layer away
        rows_l.append(np.arange(n)[ok]); dst_l.append(tgt[ok])
        val_l.append(sim[np.arange(n)[ok], tgt[ok]])
        # (l+delta, j) : concept neighbours one layer away
        rr, vv = r, v
        dd = p[d] + (li[rr] + delta) * per_layer
        good = (li[rr] + delta >= 0) & (li[rr] + delta < nlayers)
        rr, dd, vv = rr[good], dd[good], vv[good]
        rows_l.append(rr); dst_l.append(dd)
        val_l.append(sim[rr, dd])
    rows = np.concatenate(rows_l).astype(np.int64)
    dst = np.concatenate(dst_l).astype(np.int64)
    val = np.concatenate(val_l).astype(np.float32)
    order = np.argsort(rows, kind="stable")
    rows, dst, val = rows[order], dst[order], val[order]
    counts = np.bincount(rows, minlength=n)
    ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return {"ptr": ptr, "dst": dst, "base": val,
            "src_layer": layer[dst], "dst_layer": layer[rows], "n": n}


def coefficients(g, w, edge):
    """Per-edge gain: w[dst_layer] * m_ij * g[src_layer]."""
    return w[edge["dst_layer"]] * edge["base"] * g[edge["src_layer"]]


def matvec(psi, ptr, dst, coeff, n, self_c, idx=None):
    """psi_new = self_c * psi + sum over CSR edges (complex, segment sums)."""
    contrib = coeff * psi[dst]
    if idx is None:
        idx = np.repeat(np.arange(n), np.diff(ptr))
    out = (np.bincount(idx, weights=contrib.real, minlength=n)
           + 1j * np.bincount(idx, weights=contrib.imag, minlength=n))
    return (self_c * psi + out).astype(np.complex64)
