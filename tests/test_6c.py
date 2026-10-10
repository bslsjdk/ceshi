"""Tests for the stage-6A corrections: low-rank readout and truncated BPTT.

The BPTT backward pass is hand-written, so it is pinned in two independent
ways: the transpose used to push gradients back through the neighbour mean is
checked against the definition of a transpose, and the whole backward pass is
checked against a finite difference.  The finite-difference tolerance is loose
because the field runs in float32 -- the gradient check script shows the
relative error falling from 0.30 to 0.00 as the step grows out of the rounding
floor, which is the signature of a correct formula on a noisy estimate.
"""
from __future__ import annotations

import numpy as np
import pytest

from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import neighbour_mean, step_state
from vn_mobile.io.bptt import (backward_window, head_apply, head_backward,
                               neighbour_mean_t, step_state_x)
from vn_mobile.io.charhead import CharHead
from vn_mobile.io.lowrank import LowRankHead
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split

V = len(build_vocab(load_text())[0])


def _pop(n: int = 200, seed: int = 0):
    return build_population(n=n, degree=8, seed=seed,
                            omega_mix=(0.8, 0.3, 1.0))


def test_lowrank_param_count_fits_the_budget():
    """[V,R] + [R,V*DIM] at R=16 must clear 14,770 with the bias included."""
    head = LowRankHead.make(V, DIM, rank=16, seed=0)
    assert head.n_params == V * 16 + 16 * V * DIM == 9360
    assert head.n_params + DIM <= 14_770


def test_lowrank_logits_shape_and_finiteness():
    head = LowRankHead.make(V, DIM, rank=16, seed=1)
    flat = np.random.default_rng(2).normal(size=V * DIM).astype(np.float32)
    logits = head.logits(flat)
    assert logits.shape == (V,)
    assert np.all(np.isfinite(logits))


def test_lowrank_backward_matches_finite_difference():
    """dU/dP against a central difference on one logit.

    The probe is the logit itself, so the incoming gradient is the one-hot for
    that logit -- feeding a softmax gradient here would be comparing a loss
    derivative against a logit derivative.
    """
    head = LowRankHead.make(V, DIM, rank=8, seed=3)
    flat = np.random.default_rng(4).normal(size=V * DIM).astype(np.float32)
    dlogits = np.zeros(V, dtype=np.float32)
    dlogits[7] = 1.0
    du, dp, dflat = head.grad(dlogits, flat)
    assert dflat.shape == (V * DIM,)
    eps = 1e-3
    for tensor, grad, idx in ((head.u, du, (V - 1, 0)), (head.p, dp, (0, 3))):
        old = float(tensor[idx])
        tensor[idx] = old + eps
        hi = float(head.logits(flat)[7])
        tensor[idx] = old - eps
        lo = float(head.logits(flat)[7])
        tensor[idx] = old
        numeric = (hi - lo) / (2 * eps)
        assert abs(float(grad[idx]) - numeric) < 1e-4 + 0.02 * abs(numeric)


def test_neighbour_mean_t_is_the_exact_transpose():
    pop = _pop()
    rng = np.random.default_rng(5)
    u = rng.normal(size=(pop.n, DIM)).astype(np.float32)
    v = rng.normal(size=(pop.n, DIM)).astype(np.float32)
    left = float((v * neighbour_mean_t(pop, u)).sum())
    right = float((neighbour_mean(pop, v) * u).sum())
    assert abs(left - right) < 1e-3 * max(1.0, abs(right))


def test_step_state_x_reproduces_step_state():
    n = 120
    inj = np.zeros((n, DIM), dtype=np.float32)
    inj[:, 0] = 1.0
    a = _pop(n, seed=6)
    b = _pop(n, seed=6)
    step_state(a, inj)
    _, th = step_state_x(b, inj)
    assert np.allclose(a.state, b.state, atol=1e-6)
    assert np.all(np.isfinite(th))


def test_bptt_gradient_matches_finite_difference():
    """End-to-end check on a short window, including the unrolled recurrence."""
    from train6c import forward
    from vn_mobile.io.lowrank import softmax

    text = load_text()
    _, stoi = build_vocab(text)
    tr = encode(split(text)[0], stoi)
    n_vn, T, seed = 200, 4, 0
    pop = _pop(n_vn, seed)
    block = block_of(n_vn, V)
    members = [np.flatnonzero(block == v) for v in range(V)]
    head = LowRankHead.make(V, DIM, rank=16, seed=seed)
    inj_e = np.zeros(n_vn, dtype=np.float32)
    inj_h = np.zeros((n_vn, DIM), dtype=np.float32)

    def replay():
        rng = np.random.default_rng(seed)
        win = forward(pop, head, block, members, tr, 0, T, rng, inj_e, inj_h,
                      V, hist=1, decay=0.6)
        return float(np.sum(win.losses))

    snap = (pop.energy.copy(), pop.phase.copy(), pop.state.copy())
    rng = np.random.default_rng(seed)
    win = forward(pop, head, block, members, tr, 0, T, rng, inj_e, inj_h, V,
                  hist=1, decay=0.6)
    g = backward_window(pop, head, win, block, V, DIM)

    # bias sits inside the recurrence, so this exercises the unrolled path
    idx = 1
    old = float(pop.bias[idx])
    eps = 0.2
    pop.bias[idx] = old + eps
    pop.energy[:], pop.phase[:], pop.state[:] = (s.copy() for s in snap)
    hi = replay()
    pop.bias[idx] = old - eps
    pop.energy[:], pop.phase[:], pop.state[:] = (s.copy() for s in snap)
    lo = replay()
    pop.bias[idx] = old
    numeric = (hi - lo) / (2 * eps)
    # backward_window returns the mean over the window (the objective is the
    # mean loss), while the difference above is on the sum, so scale by T.
    assert abs(float(g["db"][idx]) * T - numeric) < 0.25 * max(1e-6,
                                                              abs(numeric))


def test_head_apply_supports_both_head_types():
    flat = np.random.default_rng(7).normal(size=V * DIM).astype(np.float32)
    dlogits = np.full(V, 0.1, dtype=np.float32)
    for head in (LowRankHead.make(V, DIM, rank=8, seed=8),
                 CharHead.make(V, DIM, seed=8, full=True)):
        g = head_backward(head, dlogits, flat)
        before = (head.u.copy() if isinstance(head, LowRankHead)
                  else head.w.copy())
        head_apply(head, g, 1e-2)
        after = (head.u if isinstance(head, LowRankHead) else head.w)
        assert not np.allclose(before, after)


def test_window_gradients_are_finite_on_a_long_window():
    from train6c import forward
    text = load_text()
    _, stoi = build_vocab(text)
    tr = encode(split(text)[0], stoi)
    n_vn = 200
    pop = _pop(n_vn, seed=9)
    block = block_of(n_vn, V)
    members = [np.flatnonzero(block == v) for v in range(V)]
    head = LowRankHead.make(V, DIM, rank=16, seed=9)
    inj_e = np.zeros(n_vn, dtype=np.float32)
    inj_h = np.zeros((n_vn, DIM), dtype=np.float32)
    rng = np.random.default_rng(9)
    win = forward(pop, head, block, members, tr, 0, 20, rng, inj_e, inj_h, V,
                  hist=1, decay=0.6)
    g = backward_window(pop, head, win, block, V, DIM)
    for v in g["dw"]:
        assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(g["db"])) and np.all(np.isfinite(g["ga"]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
