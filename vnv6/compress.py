"""Two-level compression of the concept block.

Level 1 -- fusion.  Neurons whose concepts are the same direction to within
``hi`` are, for this architecture, doing literally the same thing: the gate is
linear and the pool is a sum, so two identical blocks contribute exactly twice
one block.  Folding them into a single neuron holding the sum cannot move a
single logit.  That is exact for cos == 1 and approximate below it, so the
measured delta is reported rather than assumed.

Level 2 -- clustering.  Neurons that are merely similar keep their identity
but stop owning a private block: each cluster owns a base ``B_c`` and each
neuron keeps a small modulation ``m_i`` in a rank-r subspace,

    concept_i ~= B_{c(i)} + P @ m_i

``P`` is a frozen basis, so the trainable cost of the block collapses from
N*D to K*D + N*r.
"""
from __future__ import annotations

import numpy as np

from vnv6.model import DE


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def cosine_matrix(c: np.ndarray) -> np.ndarray:
    return _unit(c.astype(np.float64)) @ _unit(c.astype(np.float64)).T


def fuse(c: np.ndarray, hi: float = 0.99):
    """Union neurons with cos >= hi.

    The merge is NOT a plain sum of the concept vectors.  A neuron's
    contribution is g_i * u_i with g_i linear in its key and u_i linear in its
    block, so the contribution is *bilinear* in the concept: doubling both
    halves makes it four times as large, not twice.  The rule that preserves
    the contribution of a group S is therefore

        key'  = mean_i key_i        (gate stays the same size)
        W'    = sum_i  W_i          (blocks add up)

    which for |S| identical neurons gives g * (|S| u) = |S| * (g u), exactly
    what they contributed separately.  Summing the whole concept instead is a
    real bug and quadruples every merged contribution -- test_fuse_is_exact
    pins the difference down.
    """
    
    n = c.shape[0]
    cos = cosine_matrix(c)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in np.flatnonzero(cos[i, i + 1:] >= hi) + (i + 1):
            ra, rb = find(i), find(int(j))
            if ra != rb:
                parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    order = sorted(groups.values(), key=lambda g: g[0])
    fused = np.stack([np.concatenate([c[g, :DE].mean(axis=0),
                                      c[g, DE:].sum(axis=0)])
                      for g in order]).astype(np.float32)
    return fused, order


def cluster(c: np.ndarray, k: int):
    """Agglomerative average-linkage on cosine distance, cut at exactly k."""
    from scipy.cluster.hierarchy import fcluster, linkage
    if c.shape[0] <= k:
        return np.arange(c.shape[0], dtype=np.int64)
    z = linkage(_unit(c.astype(np.float64)), method='average',
                metric='cosine')
    return fcluster(z, k, criterion='maxclust').astype(np.int64) - 1


def cluster_report(c: np.ndarray, lab: np.ndarray) -> dict:
    """How coherent are the clusters really?  Reported, not assumed."""
    u = _unit(c.astype(np.float64))
    within, sizes = [], []
    for k in np.unique(lab):
        idx = np.flatnonzero(lab == k)
        sizes.append(int(idx.size))
        if idx.size > 1:
            sub = u[idx] @ u[idx].T
            iu = np.triu_indices(idx.size, 1)
            within.append(float(sub[iu].mean()))
    return dict(n_clusters=int(np.unique(lab).size),
                max_cluster=int(max(sizes)), singleton=int((np.array(sizes) == 1).sum()),
                mean_within_cos=(round(float(np.mean(within)), 4) if within else None),
                min_within_cos=(round(float(np.min(within)), 4) if within else None))


def basis(residual: np.ndarray, r: int, mode: str = 'pca') -> np.ndarray:
    """Frozen rank-r basis for the modulation subspace, orthonormal."""
    d = residual.shape[1]
    r = int(min(r, d))
    if mode == 'random':
        rng = np.random.default_rng(12345)
        q, _ = np.linalg.qr(rng.normal(size=(d, r)))
        return q.T.astype(np.float32)[:r]
    u, s, vt = np.linalg.svd(residual.astype(np.float64), full_matrices=False)
    return vt[:r].astype(np.float32)


def factorize(c: np.ndarray, lab: np.ndarray, r: int = 8,
              mode: str = 'pca'):
    """Split concepts into (cluster base, per-neuron modulation, basis)."""
    n, d = c.shape
    k = int(np.unique(lab).size)
    B = np.zeros((k, d), np.float32)
    for j in range(k):
        idx = np.flatnonzero(lab == j)
        B[j] = c[idx].mean(axis=0)
    resid = c - B[lab]
    P = basis(resid, r, mode)
    m = (resid @ P.T).astype(np.float32)
    rec = (B[lab] + m @ P).astype(np.float32)
    err = float(np.linalg.norm(rec - c) / max(np.linalg.norm(c), 1e-12))
    return dict(B=B, m=m, P=P, lab=lab,
                recon_rel_err=round(err, 5),
                resid_var_kept=round(float(((resid @ P.T) ** 2).sum()
                                           / max((resid ** 2).sum(), 1e-12)), 4))
