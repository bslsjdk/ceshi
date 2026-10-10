"""Minimal numpy char-level causal Transformer, for controlled comparison with VN.

Design notes:
  * single file, no torch available in this sandbox
  * explicit activations cache so backward is verifiable by finite differences
  * layer input of layer i is: h0 (i==0) or h_out of layer i-1
"""
import numpy as np


def make_params(V, D, L, H, FF, seed=0):
    assert D % H == 0
    rng = np.random.default_rng(seed)
    s = 0.08
    P = {}
    P['emb'] = (rng.standard_normal((V, D)) * s).astype(np.float32)
    for i in range(L):
        for n in ('wq', 'wk', 'wv', 'wo'):
            P[f'{n}{i}'] = (rng.standard_normal((D, D)) * s).astype(np.float32)
        P[f'w1{i}'] = (rng.standard_normal((D, FF)) * s).astype(np.float32)
        P[f'b1{i}'] = np.zeros(FF, dtype=np.float32)
        P[f'w2{i}'] = (rng.standard_normal((FF, D)) * s).astype(np.float32)
        P[f'b2{i}'] = np.zeros(D, dtype=np.float32)
    P['out'] = (rng.standard_normal((D, V)) * s).astype(np.float32)
    return P


def count_params(P):
    return sum(int(v.size) for v in P.values())


def zeros_like_params(P):
    return {k: np.zeros_like(v) for k, v in P.items()}


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def forward(P, x, D, L, H, mask=None):
    """x: (T,) int ids -> logits (T,V); C holds everything backward needs."""
    T = len(x)
    hd = D // H
    C = {'x': np.asarray(x), 'layers': []}
    h = P['emb'][x].astype(np.float32)            # (T,D)
    C['h_in'] = []
    if mask is None or mask.shape[0] != T:
        mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    for i in range(L):
        hin = h
        q = hin @ P[f'wq{i}']
        k = hin @ P[f'wk{i}']
        v = hin @ P[f'wv{i}']
        qh = q.reshape(T, H, hd).transpose(1, 0, 2)
        kh = k.reshape(T, H, hd).transpose(1, 0, 2)
        vh = v.reshape(T, H, hd).transpose(1, 0, 2)
        att = _softmax(qh @ kh.transpose(0, 2, 1) / np.sqrt(hd) + mask, axis=-1)
        oh = att @ vh
        opre = oh.transpose(1, 0, 2).reshape(T, D)
        o = opre @ P[f'wo{i}']
        h2 = hin + o
        f1 = h2 @ P[f'w1{i}'] + P[f'b1{i}']
        f = np.maximum(f1, 0)
        h = h2 + (f @ P[f'w2{i}'] + P[f'b2{i}'])
        C['layers'].append((hin, q, k, v, att, opre, o, h2, f1, f))
    C['hL'] = h
    return h @ P['out'], C


def backward(P, G, logits, C, y, D, L, H):
    """Accumulate grads into G (same keys as P). Returns mean NLL in nats."""
    T = len(y)
    hd = D // H
    x = C['x']
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    nll = -(logits[np.arange(T), y] - m.ravel() - np.log(e.sum(axis=1)))
    loss = float(nll.mean())
    dz = p.copy()
    dz[np.arange(T), y] -= 1.0
    dz /= T
    G['out'] += C['hL'].T @ dz
    dh = dz @ P['out'].T
    for i in range(L - 1, -1, -1):
        hin, q, k, v, att, opre, o, h2, f1, f = C['layers'][i]
        # ---- feed forward
        dff = dh @ P[f'w2{i}'].T
        G[f'w2{i}'] += f.T @ dh
        G[f'b2{i}'] += dh.sum(axis=0)
        dpre = dff * (f1 > 0)
        G[f'w1{i}'] += h2.T @ dpre
        G[f'b1{i}'] += dpre.sum(axis=0)
        dh2 = dpre @ P[f'w1{i}'].T + dh
        # ---- attention output proj
        G[f'wo{i}'] += opre.T @ dh2
        doh = (dh2 @ P[f'wo{i}'].T).reshape(T, H, hd).transpose(1, 0, 2)
        dvh = att.transpose(0, 2, 1) @ doh
        datt = doh @ v.reshape(T, H, hd).transpose(1, 0, 2).transpose(0, 2, 1)
        datt = datt * (1.0 / np.sqrt(hd))
        s = datt * att
        datt = s - att * s.sum(axis=-1, keepdims=True)
        qh = q.reshape(T, H, hd).transpose(1, 0, 2)
        kh = k.reshape(T, H, hd).transpose(1, 0, 2)
        dqh = datt @ kh
        dkh = datt.transpose(0, 2, 1) @ qh
        dq = dqh.transpose(1, 0, 2).reshape(T, D)
        dk = dkh.transpose(1, 0, 2).reshape(T, D)
        dv = dvh.transpose(1, 0, 2).reshape(T, D)
        G[f'wq{i}'] += hin.T @ dq
        G[f'wk{i}'] += hin.T @ dk
        G[f'wv{i}'] += hin.T @ dv
        dh = dq @ P[f'wq{i}'].T + dk @ P[f'wk{i}'].T + dv @ P[f'wv{i}'].T + dh2
    np.add.at(G['emb'], x, dh)
    return loss
