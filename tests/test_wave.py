"""Tests for the concept-wave A/B harness."""
from __future__ import annotations

import numpy as np

from wave_ab import Field, inject

F = np.float32


def _f():
    return Field(n=200, v=65, seed=3)


def test_threshold_is_the_only_sparsity_source():
    f = _f()
    rng = np.random.default_rng(0)
    for _ in range(200):
        f.step('wave', -1.0, inject(f, 7), rng)
    _, dense = f._contrib('wave', -1.0)
    _, tau0 = f._contrib('wave', 0.0)
    _, tau7 = f._contrib('wave', 0.7)
    assert dense > 0.999, dense
    assert tau0 < dense, (tau0, dense)
    assert tau7 < tau0, (tau7, tau0)


def test_identical_trainable_params_across_arms():
    f = _f()
    assert f.U.size + f.Wb.size == 65 * 16 + 16 * 65 * 8
    assert f.P.shape == (f.kk, f.ds + f.kk)


def test_no_nan_after_long_run():
    f = _f()
    rng = np.random.default_rng(5)
    for t in range(3000):
        f.step('wave', 0.5, inject(f, t % 65), rng)
    assert np.isfinite(f.e).all() and np.isfinite(f.s).all()
    assert np.isfinite(f.phase).all()
    lg, _, _ = f.logits()
    assert np.isfinite(lg).all()


def test_logits_shape_and_pooled_size():
    f = _f()
    assert f.logits()[0].shape == (65,)
    assert f.pooled().shape == (65 * f.ds,)


def test_fixed_arm_uses_every_edge():
    f = _f()
    _, act = f._contrib('fixed', -1.0)
    assert act > 0.999
