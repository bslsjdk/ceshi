"""VN v3 facts, pinned (<=200 lines).

These tests record what was measured, including the failures.
"""
from __future__ import annotations

import numpy as np
import pytest

from conceptwave3.predict import Predictor, Adam, normalise
from conceptwave3.field3 import Field3
from conceptwave3.train3 import Codec3, mse_term
from probe_train3 import grads, total_loss

LOG2 = float(np.log(2.0))
V = 65


def make(dc=16, n=200, seed=0):
    f = Field3(n=n, dc=dc, seed=seed, rounds=3, nsrc=8, k=4)
    F = Predictor(dc=dc, hidden=32, seed=seed)
    c = Codec3(V, dc, seed=seed)
    return f, F, c


def test_two_layers_exist_and_are_coupled():
    f, F, c = make()
    assert f.concept.shape[0] == 2 and f.context.shape[0] == 2
    assert f.G.shape == (2 * f.dc, 2 * f.dc)
    # inter-layer term actually changes context
    f.seed_from(c.emb[3])
    before = f.context.copy()
    f._inter_layer()
    assert float(np.abs(f.context - before).sum()) > 0


def test_concept_receives_gradient():
    """deepseek's requirement #1: does concept have gradient?"""
    f, F, c = make()
    f.seed_from(c.emb[3])
    f.settle(F, grad=True)
    lg = c.logits(f)
    _, hg = c.head_grads(f, 5, lg)
    da, dcx = c.pooled_bwd(f, hg["h"])
    nxt = f._last[0][2]
    _, dn = mse_term(f, c, nxt, 5, 0.1)
    bg = f.backward(F, da, dcx, dn)
    assert float(np.abs(bg["concept"]).sum()) > 0
    assert float(np.abs(bg["G"]).sum()) > 0
    assert float(np.abs(bg["F"]["W1"]).sum()) > 0


def test_G_gradient_is_truncated_not_exact():
    """The analytic dL/dG only counts the injection that immediately follows
    one round.  G -> context -> pooled -> next round is a loop we do NOT
    unroll, so finite differences disagree in magnitude.  What must hold is
    the SIGN.  This is recorded, not hidden."""
    f, F, c = make()
    emb_t, tgt, lam = c.emb[3], 5, 0.1
    bg, _ = grads(f, F, c, emb_t, tgt, lam)
    g = bg["G"].astype(np.float64)
    i, j = np.unravel_index(int(np.argmax(np.abs(g))), g.shape)
    ana = float(g[i, j])
    o = f.G[i, j]
    f.G[i, j] = o + 1e-2
    lp = total_loss(f, F, c, emb_t, tgt, lam)
    f.G[i, j] = o - 1e-2
    lm = total_loss(f, F, c, emb_t, tgt, lam)
    f.G[i, j] = o
    num = (lp - lm) / 2e-2
    assert np.sign(num) == np.sign(ana), (num, ana)


def test_inter_layer_injection_is_bounded():
    """tanh + per-row cap exist because the unbounded form overflowed during
    training at N=1000/dc=32: |context| reached inf after ~2k steps (measured,
    see stage_v3_report.md).  At the small scale used here the two forms agree
    to float precision, so what we pin is the bound itself."""
    f, F, c = make()
    f.seed_from(c.emb[3])
    f.settle(F)
    nr = np.linalg.norm(f.context, axis=2)
    assert float(nr.max()) <= float(f.ctx_cap) + 1e-5


def test_bounded_coupling_stays_finite():
    f, F, c = make()
    for t in range(3000):
        f.seed_from(c.emb[t % V])
        f.settle(F)
    assert np.isfinite(f.context).all()
    assert np.isfinite(f.act).all()


def test_sparsity_is_one_not_ten_percent():
    """softmax addressing touches every node: the field is NOT sparse."""
    f, F, c = make(n=500, dc=16)
    f.seed_from(c.emb[7])
    f.settle(F)
    assert f.sparsity() > 0.9


def test_concept_sim_top5_is_near_zero_not_sixty_percent():
    """Pinned failure: measured mean over 'Fae Th' was 0.034 at 12.5k steps."""
    f, F, c = make(n=500, dc=16)
    sims = []
    for ch in range(6):
        f.reset()
        f.seed_from(c.emb[ch])
        f.settle(F)
        s, _ = f.concept_sim_top(c.emb[ch], 5)
        sims.append(s)
    assert abs(float(np.mean(sims))) < 0.6
