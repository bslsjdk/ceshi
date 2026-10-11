"""v7: does scale buy anything?  Four experiments, all answering 'no'.

These are short-run versions of the swept experiments (10k-20k steps).  They
pin the *direction* of each effect, not its magnitude; the full numbers live
in v7_nscan_a.json / v7_disperse.json / v7_long.json / v7_lr.json.
"""
from __future__ import annotations

import numpy as np
import pytest

from vnv6.compress import cosine_matrix
from vnv6.model import ConceptNet
from vnv6.train import streams, train

TR, VA, V = streams()


def _bits(n: int, steps: int, lr: float = 3e-3, seed: int = 0) -> float:
    net = ConceptNet(V, n, seed=seed)
    hist, _ = train(net, TR, VA, steps=steps, lr=lr, val_every=steps)
    return float(hist[-1]['val_bits'])


def _nn_p50(c: np.ndarray) -> float:
    cos = cosine_matrix(c)
    np.fill_diagonal(cos, -np.inf)
    return float(np.median(cos.max(axis=1)))


def test_small_pool_is_not_worse_than_large_pool():
    """A: extra neurons buy nothing.  N=20 must not lose to N=200."""
    small = _bits(20, 20000)
    large = _bits(200, 20000)
    assert small <= large + 0.10, f"N=20 {small:.4f} lost to N=200 {large:.4f}"


def test_raising_lr_with_pool_size_hurts():
    """D: the 'big N is just under-trained' defence fails.

    If gradients really scaled as 1/sqrt(N), lr*sqrt(N/20) would match the
    N=20 optimum.  It does not - it is clearly worse.
    """
    base = _bits(105, 10000, lr=3e-3)
    scaled = _bits(105, 10000, lr=3e-3 * float(np.sqrt(105 / 20.0)))
    assert scaled > base, f"lr scaling helped ({scaled:.4f} vs {base:.4f})"


def test_training_diversifies_concepts_after_initial_clustering():
    """B: concepts first cluster, then spread apart.

    Training does not drive similar concepts to the same address - past the
    early peak it pushes them apart.
    """
    net = ConceptNet(V, 200, seed=0)
    init = _nn_p50(net.concepts())
    train(net, TR, VA, steps=10000, lr=3e-3, val_every=10000)
    early = _nn_p50(net.concepts())
    train(net, TR, VA, steps=40000, lr=3e-3, val_every=40000)
    late = _nn_p50(net.concepts())
    assert early > init, f"no initial clustering: {init:.4f} -> {early:.4f}"
    assert late < early, f"no later spreading: {early:.4f} -> {late:.4f}"


def test_no_duplicate_concepts_emerge_so_fusion_has_nothing_to_do():
    """B: the 'merge identical concepts' stage is empty - cos >= 0.99 never occurs."""
    net = ConceptNet(V, 200, seed=0)
    train(net, TR, VA, steps=20000, lr=3e-3, val_every=20000)
    cos = cosine_matrix(net.concepts())
    iu = np.triu_indices(cos.shape[0], 1)
    off = cos[iu]
    assert int((off >= 0.99).sum()) == 0, "duplicate concepts exist after training"
    assert int((off >= 0.95).sum()) == 0, "near-duplicate concepts exist after training"
