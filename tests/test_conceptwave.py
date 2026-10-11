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


def test_no_global_renorm_amplitude_is_bounded_not_fixed():
    """v1.1: a per-node cap replaces the global renorm that erased interference."""
    f = _field()
    f.set_context(f.concept[0])
    f.inject(np.abs(f.concept[:, 0]))
    f.settle(max_rounds=20)
    mag = np.abs(f.psi)
    assert np.isfinite(mag).all()
    assert mag.max() <= f.psi_max + 1e-4          # per-node cap holds
    assert abs(np.linalg.norm(f.psi) - 1.0) > 1e-5  # and is NOT pinned to 1


def test_signed_edges_open_the_second_interference_channel():
    """|cos| top-k keeps the sign; the old top-k by cos only ever saw positives."""
    from conceptwave.field import ConceptField
    pos = {}
    for signed in (False, True):
        f = ConceptField.create(1000, 32, 1, 8, seed=0, dormancy=0.0,
                                signed=signed)
        b = f.edges["base"]
        assert not np.isinf(b).any()
        pos[signed] = float((b > 0).mean())
    assert 0.3 < pos[True] < 0.7          # signed: both channels present
    assert pos[True] < pos[False]         # fewer negatives than v1


def _phase_amp(f, s, sup, phi):
    f.psi[:] = 0
    f.inject(s, phase=0.0)
    f.inject(s, phase=float(phi))
    f.settle(max_rounds=1)
    r1 = float(np.abs(f.psi[sup]).sum())
    f.settle(max_rounds=9)
    return r1, float(np.abs(f.psi[sup]).sum())


def test_interference_survives_the_round_without_global_renorm():
    """v1's global renorm erased phase dependence entirely; v1.1 keeps it."""
    from conceptwave.field import ConceptField
    rng = np.random.default_rng(7)

    def sweep(**kw):
        f = ConceptField.create(1000, 32, 1, 8, seed=0, dormancy=0.0, **kw)
        c = normalise(rng.normal(size=(1, 32)))[0]
        s = np.clip(cosine(f.cn, c.reshape(1, -1)).ravel(), 0, None)
        f.set_context(c)
        return [_phase_amp(f, s, np.argsort(-s)[:60], p)[0]
                for p in (0.0, np.pi / 2, np.pi)]

    a = sweep()                                    # v1.1
    assert a[0] > a[1] > a[2]                      # monotone in relative phase
    assert a[2] < 1e-6                             # anti-phase cancels exactly
    b = sweep(signed=False, nonlinear=False,
              psi_max=0.0, norm_global=True)       # v1
    assert abs(b[0] - b[2]) < 1e-3                 # v1 is phase-blind


def test_eigenvector_lock_is_not_solved_by_v11():
    """Known limitation, measured: both variants still settle to ~one mode.

    The lock metric is context dependent (random context: v1 0.993 / v1.1 0.948;
    concept[0] context: v1 0.897 / v1.1 0.944), so no ordering is asserted --
    only that three fixes did NOT break the dominant-mode convergence.
    """
    from conceptwave.field import ConceptField

    def lock(**kw):
        f = ConceptField.create(1000, 32, 1, 8, seed=0, dormancy=0.0, **kw)
        f.set_context(f.concept[0])
        outs = []
        for s in (1, 2):
            r = np.random.default_rng(s)
            f.psi[:] = (r.normal(size=f.n)
                        + 1j * r.normal(size=f.n)).astype(np.complex64)
            f.settle(max_rounds=30)
            outs.append(f.energy())
        return float(outs[0] @ outs[1]
                     / (np.linalg.norm(outs[0]) * np.linalg.norm(outs[1])))

    assert lock() > 0.85
    assert lock(signed=False, nonlinear=False,
                psi_max=0.0, norm_global=True) > 0.85
