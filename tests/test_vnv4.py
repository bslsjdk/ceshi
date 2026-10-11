"""Correctness tests for VN v4: gradients, freezing, forgetting metric."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from vnv4.models import VNv4, SSM  # noqa: E402
from vnv4 import protocol as PR  # noqa: E402


def _loss(m, x, y):
    return m.eval_nll(x, y)


def _fd(m, key, x, y, idx, eps=1e-5):
    old = m.P[key][idx]
    m.P[key][idx] = old + eps
    lp = _loss(m, x, y)
    m.P[key][idx] = old - eps
    lm = _loss(m, x, y)
    m.P[key][idx] = old
    return (lp - lm) / (2 * eps)


@pytest.mark.parametrize("cls,kw", [(VNv4, {"Nf": 32}), (SSM, {"Ne": 32})])
@pytest.mark.parametrize("T", [6, 14])
def test_analytic_grad_matches_finite_difference(cls, kw, T):
    """Checked in float64: in float32 the FD difference is below the
    resolution of the loss and every parameter looks wrong."""
    rng = np.random.default_rng(0)
    m = cls(V=12, d=16, dh=16, seed=1, **kw)
    for k in m.P:
        m.P[k] = m.P[k].astype(np.float64)
    x = rng.integers(0, 12, T).astype(np.int64)
    y = rng.integers(0, 12, T).astype(np.int64)
    logits, C = m.forward(x)
    G = m.zeros_grad()
    m.backward(G, logits, C, x, y)
    worst = {}
    for key in m.P:
        g = G[key]
        if g.size == 0 or not np.isfinite(g).all():
            continue
        flat = np.abs(g).ravel()
        i = int(flat.argmax())
        idx = np.unravel_index(np.argmax(np.abs(g)), g.shape)
        num = _fd(m, key, x, y, idx)
        ana = float(g[idx])
        denom = max(abs(num), abs(ana), 1e-8)
        worst[key] = abs(num - ana) / denom
    bad = {k: v for k, v in worst.items() if v > 2e-3}
    assert not bad, f"gradient mismatch: {bad}"


def test_freeze_actually_freezes():
    m = VNv4(V=10, d=16, dh=16, Ne=16, Nf=16, seed=2)
    m.freeze(["emb", "pf", "wof"])
    before = {k: v.copy() for k, v in m.P.items()}
    opt = __import__("vn_mobile.io.adam", fromlist=["Adam"]).Adam(lr=1e-2)
    rng = np.random.default_rng(3)
    x = rng.integers(0, 10, 8).astype(np.int64)
    y = rng.integers(0, 10, 8).astype(np.int64)
    G = m.zeros_grad()
    logits, C = m.forward(x)
    m.backward(G, logits, C, x, y)
    PR_run_keys = m.trainable()
    pairs = {k: (m.P[k], G[k]) for k in PR_run_keys}
    opt.step(pairs)
    for k in m.frozen:
        assert np.array_equal(before[k], m.P[k]), f"{k} moved while frozen"
    moved = [k for k in m.trainable() if not np.array_equal(before[k], m.P[k])]
    assert moved, "no trainable parameter moved"


def test_forgetting_rate_definition():
    assert PR.forgetting(2.0, 2.4) == pytest.approx(0.2)
    assert PR.forgetting(2.0, 1.6) == pytest.approx(-0.2)
    assert np.isnan(PR.forgetting(0.0, 1.0))
    assert np.isnan(PR.forgetting(1.0, float("inf")))


def test_state_is_input_dependent_not_fixed_decay():
    """The forget gate must vary with the input, else it is not selective."""
    m = VNv4(V=20, d=16, dh=16, Ne=16, Nf=16, seed=4)
    rng = np.random.default_rng(5)
    x = rng.integers(0, 20, 10).astype(np.int64)
    _, _, _, a, _, _, _, _ = m._encode(x)
    spread = float(a.max(axis=0).max() - a.min(axis=0).min())
    assert spread > 1e-3, f"forget gate is constant across inputs ({spread})"
    assert a.min() > 0.0 and a.max() < 1.0


def test_write_gate_is_signed():
    m = VNv4(V=20, d=16, dh=16, Ne=16, Nf=16, seed=6)
    rng = np.random.default_rng(7)
    x = rng.integers(0, 20, 10).astype(np.int64)
    _, _, _, _, _, _, c, _ = m._encode(x)
    assert c.min() < 0.0 < c.max(), "write direction never changes sign"
