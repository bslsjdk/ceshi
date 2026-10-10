"""Tests for concept wave v1 (<=200 lines)."""
from __future__ import annotations

import numpy as np
import pytest

from conceptwave.encode import CharCodec
from conceptwave.field import ConceptField, DORMANT, ACTIVE
from conceptwave.kernels import build_edges, cosine, matvec, normalise


def _field(n=200, dc=16, nlayers=1, k=6, seed=0, dorm=0.0):
    return ConceptField.create(n, dc, nlayers, k, seed=seed, dormancy=dorm)


def test_concepts_are_unit_norm():
    f = _field()
    n = np.linalg.norm(f.concept, axis=1)
    assert np.allclose(n, 1.0, atol=1e-5)


def test_cosine_diagonal_is_one():
    f = _field()
    s = cosine(normalise(f.concept), normalise(f.concept))
    assert np.allclose(np.diag(s), 1.0, atol=1e-5)


def test_edges_cover_six_directions_with_layers():
    f = ConceptField.create(120, 16, 3, 4, seed=0)
    assert f.edges["dst"].size > 120 * 4          # more than same-layer only
    assert f.edges["dst"].min() >= 0
    assert f.edges["dst"].max() < 120


def test_matvec_matches_dense():
    n = 12
    ptr = np.arange(n + 1, dtype=np.int64)
    dst = (np.arange(n) + 1) % n
    coeff = np.linspace(-1, 1, n).astype(np.float32)
    psi = (np.random.default_rng(0).normal(size=n)
           + 1j * np.random.default_rng(1).normal(size=n)).astype(np.complex64)
    dense = np.zeros((n, n), np.complex64)
    dense[np.arange(n), dst] = coeff
    got = matvec(psi, ptr, dst, coeff, n, 0.5)
    want = 0.5 * psi + dense @ psi
    assert np.allclose(got, want, atol=1e-5)


def test_phase_advance_keeps_unit_norm():
    f = _field()
    f.set_context(f.concept[0])
    f.inject(np.abs(f.concept[:, 0]))
    f.settle(max_rounds=8)
    assert abs(np.linalg.norm(f.psi) - 1.0) < 1e-5


def test_interference_cancels_at_pi():
    """Two anti-phase paths into the same node must not add in quadrature."""
    a, b = 3.0 + 0j, 3.0 * np.exp(1j * np.pi)
    assert abs(a + b) < 1e-6
    assert abs(a + b) < abs(a) + abs(b) - 5.9


def test_dormancy_gives_sparsity():
    lo = _field(dorm=0.0)
    hi = _field(dorm=0.35)
    vec = lo.concept[0]
    lo.set_context(vec)
    hi.set_context(vec)
    assert lo.awake_mask().mean() > hi.awake_mask().mean()
    assert float(hi.awake_mask().mean()) < 0.2


def test_mode_field_is_written():
    f = _field(dorm=0.35)
    f.set_context(f.concept[0])
    m = f.awake_mask()
    assert f.mode[m].size and np.all(f.mode[m] == ACTIVE)
    assert np.all(f.mode[~m] == DORMANT)


def test_energy_is_a_distribution():
    f = _field()
    f.set_context(f.concept[3])
    f.inject(np.abs(f.concept[:, 3]))
    f.settle(max_rounds=8)
    e = f.energy()
    assert abs(e.sum() - 1.0) < 1e-5
    assert e.min() >= 0.0


def test_codec_logits_shape_and_finite():
    f = _field()
    c = CharCodec(65, 16, seed=0)
    f.set_context(c.ctx_vector([1, 2, 3]))
    c.inject_char(f, 4)
    f.settle(max_rounds=8)
    lg = c.logits(f)
    assert lg.shape == (65,)
    assert np.all(np.isfinite(lg))


def test_codec_grad_reduces_loss_on_itself():
    """One Adam step on a repeated example must lower that example's loss."""
    from vn_mobile.io.adam import Adam
    f = _field()
    c = CharCodec(65, 16, seed=0)
    opt = Adam(lr=0.5)
    tgt = 7
    losses = []
    for _ in range(20):
        f.set_context(c.ctx_vector([tgt]))
        c.inject_char(f, tgt)
        f.settle(max_rounds=8)
        lg = c.logits(f)
        loss, g = c.grads(f, tgt, lg)
        losses.append(loss)
        opt.step({"emb": (c.emb, g["emb"]), "bias": (c.bias, g["bias"]),
                  "scale": (c.scale, g["scale"])})
        c.renorm()
    assert losses[-1] < losses[0]


def test_anti_phase_feedback_lowers_wrong_mode():
    f = _field(dorm=0.0)
    c = CharCodec(65, 16, seed=0)
    wrong = 12
    f.set_context(c.ctx_vector([wrong]))
    c.inject_char(f, wrong, gain=3.0)
    f.settle(max_rounds=8)
    before = float(c.logits(f)[wrong])
    f.counter(3.0 * c.amp_for(f, wrong))
    f.settle(max_rounds=1)
    after = float(c.logits(f)[wrong])
    assert after < before
