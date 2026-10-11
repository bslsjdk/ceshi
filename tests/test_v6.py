"""VN v6: two-level compression.

The tests here pin down the claims that prose would otherwise let slide:

* fusion is *exactly* lossless for identical neurons, and the rule that
  achieves it is mean-the-keys / sum-the-blocks, not sum-the-concept;
* a trained population contains no identical neurons, so level 1 does nothing
  at the threshold the spec picked;
* the packed gradient path agrees with finite differences.
"""
from __future__ import annotations

import numpy as np
import pytest

from vnv6.compress import cluster, cosine_matrix, factorize, fuse
from vnv6.model import DE, DO, DCONCEPT, ConceptNet
from vnv6.packed import PackedNet
from vnv6.train import streams, train


def _net(n=41, seed=3, V=65):
    return ConceptNet(V, n, seed=seed)


def _fused_copy(net, high=0.9999):
    f, groups = fuse(net.concepts(), high)
    out = ConceptNet(net.V, f.shape[0], seed=0)
    out.emb, out.out, out.bias = net.emb.copy(), net.out.copy(), net.bias.copy()
    out.norm = net.norm
    out.set_concepts(f)
    return out, groups


def test_fuse_is_exact_for_identical_neurons():
    net = _net(41)
    c = net.concepts()
    for j in (7, 8, 9):
        c[j] = c[2].copy()
    net.set_concepts(c)
    cases = [(5, 9), (12, 30), (1, 1), (40, 7)]
    before = [net.forward(a, b)['logits'].copy() for a, b in cases]
    after, groups = _fused_copy(net)
    assert [g for g in groups if len(g) > 1] == [[2, 7, 8, 9]]
    worst = max(float(np.abs(x - after.forward(a, b)['logits']).max())
                for (a, b), x in zip(cases, before))
    assert worst < 1e-5, worst


def test_fuse_must_mean_the_keys_not_sum_them():
    """Summing the whole concept quadruples a merged group's contribution."""
    net = _net(41)
    c = net.concepts()
    c[7] = c[2].copy()
    net.set_concepts(c)
    ref = net.forward(5, 9)['logits'].copy()
    group, others = [2, 7], [i for i in range(41) if i not in (2, 7)]
    naive_c = np.concatenate([c[group].sum(axis=0)[None, :], c[others]])
    naive = ConceptNet(net.V, naive_c.shape[0], seed=0)
    naive.emb, naive.out, naive.bias = net.emb.copy(), net.out.copy(), net.bias.copy()
    naive.norm = net.norm
    naive.set_concepts(naive_c.astype(np.float32))
    d_exact = abs(_fused_copy(net)[0].forward(5, 9)['logits'] - ref).max()
    d_naive = abs(naive.forward(5, 9)['logits'] - ref).max()
    assert d_naive > 50 * max(d_exact, 1e-9)


def test_fusion_preserves_the_pool_scale():
    """The 1/sqrt(N) pool normalisation must survive a change in N."""
    net = _net(41)
    assert abs(net.norm - 1.0 / np.sqrt(41)) < 1e-12
    after, _ = _fused_copy(net, 0.9999)
    assert after.norm == net.norm


def test_trained_population_has_no_duplicates():
    """Level 1 is a no-op on a real model: nothing reaches cos 0.99."""
    tr, va, V = streams()
    net = ConceptNet(V, 200, seed=0)
    train(net, tr, va, steps=3000, lr=3e-3, val_every=3000)
    off = cosine_matrix(net.concepts())[np.triu_indices(200, 1)]
    assert float(off.max()) < 0.99, float(off.max())
    f, _ = fuse(net.concepts(), 0.99)
    assert f.shape[0] == 200


def test_cluster_cuts_to_exactly_k():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(80, DCONCEPT)).astype(np.float32)
    for k in (5, 12, 40):
        lab = cluster(c, k)
        assert lab.shape == (80,)
        assert np.unique(lab).size == k


def test_factorize_reduces_the_block():
    rng = np.random.default_rng(1)
    c = rng.normal(size=(120, DCONCEPT)).astype(np.float32)
    lab = cluster(c, 10)
    out = factorize(c, lab, r=8, mode='pca')
    assert out['B'].shape == (10, DCONCEPT)
    assert out['m'].shape == (120, 8)
    assert out['P'].shape == (8, DCONCEPT)
    assert out['B'].size + out['m'].size < c.size / 5


@pytest.mark.parametrize('mode', ['pca', 'random'])
def test_packed_gradient_matches_finite_difference(mode):
    net = ConceptNet(65, 30, seed=5)
    lab = cluster(net.concepts(), 6)
    fac = factorize(net.concepts(), lab, r=4, mode=mode)
    pk = PackedNet(net, lab, fac['B'], fac['m'], fac['P'])
    cur, prev, tgt = 4, 11, 20
    f = pk.forward(cur, prev)
    f['cur'], f['prev'] = cur, prev
    grads = pk.backward(f, tgt)
    dm = pk.packed_grads(grads)['m'][1]
    eps = 1e-3
    idx = np.argsort(-np.abs(dm.reshape(-1)))[:5]
    flat = pk.m.reshape(-1)
    for i in idx:
        i = int(i)
        old = flat[i]
        flat[i] = old + eps
        pk.rebuild()
        l1 = pk.loss(pk.forward(cur, prev)['logits'], tgt)
        flat[i] = old - eps
        pk.rebuild()
        l2 = pk.loss(pk.forward(cur, prev)['logits'], tgt)
        flat[i] = old
        pk.rebuild()
        num = (l1 - l2) / (2 * eps)
        rel = abs(num - float(dm.reshape(-1)[i])) / max(abs(num), 1e-9)
        assert rel < 0.05, (mode, i, num, float(dm.reshape(-1)[i]), rel)


def test_model_gradients_match_finite_difference():
    net = ConceptNet(65, 25, seed=7)
    for name in ('emb', 'keys', 'W', 'out', 'bias'):
        setattr(net, name, getattr(net, name).astype(np.float64))
    cur, prev, tgt = 3, 8, 15
    f = net.forward(cur, prev)
    f['cur'], f['prev'] = cur, prev
    g = net.backward(f, tgt)
    for name in ('keys', 'W', 'out', 'bias'):
        arr = getattr(net, name).reshape(-1)
        ana = np.asarray(g[name]).reshape(-1)
        for i in np.argsort(-np.abs(ana))[:4]:
            i = int(i)
            eps = 1e-6 * max(1.0, abs(arr[i]))
            old = arr[i]
            arr[i] = old + eps
            l1 = net.loss(net.forward(cur, prev)['logits'], tgt)
            arr[i] = old - eps
            l2 = net.loss(net.forward(cur, prev)['logits'], tgt)
            arr[i] = old
            num = (l1 - l2) / (2 * eps)
            assert abs(num - float(ana[i])) / max(abs(float(ana[i])), 1e-9) < 1e-4
