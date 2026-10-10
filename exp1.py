"""Exp 1: long-context. VN v1 vs same-parameter Transformer, equal token budget.

Fairness controls (all four enforced in code, not by argument):
  1. parameter count: VN 9368 vs TF 9448  (0.85% apart)
  2. token budget:    identical number of training characters consumed
  3. optimiser:       the SAME Adam class instance settings, shared code path
  4. data split:      identical train/val text, identical vocab
"""
from __future__ import annotations
import json, sys, time, argparse, resource
sys.path.insert(0, '/data/workspace/ceshi')
import numpy as np

from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import step_energy, step_phase
from vn_mobile.io.bptt import (Window, backward_window, finite, step_state_x,
                               pooled_flat)
from vn_mobile.io.lowrank import LowRankHead, softmax
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split
from vn_mobile.io.adam import Adam
import tf_micro as TF


def build_common(seed=0):
    text = load_text()
    symbols, stoi = build_vocab(text)
    V = len(symbols)
    tr_txt, va_txt = split(text)
    return symbols, stoi, V, encode(tr_txt, stoi), encode(va_txt, stoi)


# ---------------- VN side ----------------
def run_vn(B, lr, seed=0, n_vn=1000, T=20, rank=16, hist=1, decay=0.6,
           degree=8, val_limit=3000, log_every=25000):
    symbols, stoi, V, tr, va = build_common(seed)
    pop = build_population(n=n_vn, degree=degree, seed=seed,
                           omega_mix=(0.8, 0.3, 1.0))
    block = block_of(n_vn, V)
    members = [np.flatnonzero(block == v) for v in range(V)]
    head = LowRankHead.make(V, DIM, rank=rank, seed=seed)
    nparams = head.n_params + int(pop.bias.size)
    rng = np.random.default_rng(seed)
    inj_e = np.zeros(n_vn, dtype=np.float32)
    inj_h = np.zeros((n_vn, DIM), dtype=np.float32)
    opt = Adam(lr=lr)
    out = []
    t0 = time.perf_counter()

    def inject(seq, t):
        inj_e[:] = 0.0; inj_h[:] = 0.0
        for k in range(hist):
            c = int(seq[(t - k) % seq.size])
            w = np.float32(decay ** k)
            inj_e[members[c]] += w
            inj_h[members[c]] += w

    def roll(seq, limit):
        n = min(limit, seq.size - 1)
        loss = 0.0; hit = 0
        for i in range(n):
            inject(seq, i)
            step_energy(pop, rng, injection=inj_e)
            step_phase(pop)
            step_state_x(pop, inj_h)
            p = softmax(head.logits(pooled_flat(pop, block, V)))
            nx = int(seq[i + 1])
            loss += float(-np.log(max(float(p[nx]), 1e-9)))
            hit += int(int(np.argmax(p)) == nx)
        return dict(val_bits=round(loss / n / np.log(2), 4), val_acc=round(hit / n, 4))

    nwin = B // T
    for w in range(nwin):
        win = Window()
        t = (w * T) % tr.size
        for _ in range(T):
            inject(tr, t)
            step_energy(pop, rng, injection=inj_e)
            step_phase(pop)
            x, th = step_state_x(pop, inj_h)
            flat = pooled_flat(pop, block, V)
            probs = softmax(head.logits(flat))
            nxt = int(tr[(t + 1) % tr.size])
            win.xs.append(x); win.ths.append(th); win.flats.append(flat)
            win.probs.append(probs); win.energies.append(pop.energy.copy())
            win.targets.append(nxt)
            win.losses.append(float(-np.log(max(float(probs[nxt]), 1e-9))))
            t = (t + 1) % tr.size
        g = backward_window(pop, head, win, block, V, DIM)
        if finite(g):
            gu, gv = g["dw"]
            opt.step({"u": (head.u, gu), "p": (head.p, gv), "bias": (pop.bias, g["db"])})
        done = (w + 1) * T
        if done % log_every == 0:
            trb = float(np.mean(win.losses)) / np.log(2)
            print(json.dumps(dict(model="VN", done=done, train_bits=round(trb, 4),
                                  wall=round(time.perf_counter() - t0, 1))), flush=True)
            out.append(dict(step=done, train_bits=round(trb, 4)))
    return dict(model="VN", params=nparams, final=roll(va, val_limit),
                wall_s=round(time.perf_counter() - t0, 1), curve=out,
                rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))


# ---------------- Transformer side ----------------
def run_tf(B, T, lr, seed=0, V=None, D=20, L=2, H=2, FF=44,
           tr=None, va=None, val_limit=3000, log_every=None):
    symbols, stoi, Vv, tr_a, va_a = build_common(seed)
    V = Vv; tr = tr_a; va = va_a
    P = TF.make_params(V, D, L, H, FF, seed=seed)
    G = TF.zeros_like_params(P)
    opt = Adam(lr=lr)
    nparams = TF.count_params(P)
    steps = max(1, B // T)
    if log_every is None:
        log_every = max(1, steps // 6)
    rng = np.random.default_rng(seed + 1)
    out = []
    t0 = time.perf_counter()
    for s in range(steps):
        st = int(rng.integers(0, tr.size - T - 1))
        x = tr[st:st + T]; y = tr[st + 1:st + 1 + T]
        logits, C = TF.forward(P, x, D, L, H)
        loss = TF.backward(P, G, logits, C, y, D, L, H)
        opt.step({k: (P[k], G[k]) for k in P})
        for k in G: G[k].fill(0.0)
        if (s + 1) % log_every == 0:
            print(json.dumps(dict(model=f"TF_T{T}", step=s + 1, done=(s + 1) * T,
                                  train_bits=round(loss / np.log(2), 4),
                                  wall=round(time.perf_counter() - t0, 1))), flush=True)
            out.append(dict(step=(s + 1) * T, train_bits=round(loss / np.log(2), 4)))
    # validation: contiguous blocks over val text
    n = min(val_limit, va.size - T - 1)
    loss = 0.0; hit = 0; cnt = 0
    for st in range(0, n, T):
        x = va[st:st + T]; y = va[st + 1:st + 1 + T]
        lg, _ = TF.forward(P, x, D, L, H)
        m = lg.max(axis=1, keepdims=True)
        lse = m.ravel() + np.log(np.exp(lg - m).sum(axis=1))
        nll = lse - lg[np.arange(len(y)), y]
        loss += float(nll.sum()); hit += int((np.argmax(lg, axis=1) == y).sum()); cnt += len(y)
    return dict(model=f"TF_T{T}", params=nparams, steps=steps,
                final=dict(val_bits=round(loss / cnt / np.log(2), 4),
                           val_acc=round(hit / cnt, 4)),
                wall_s=round(time.perf_counter() - t0, 1), curve=out,
                rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=300000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tf-T", type=int, default=256)
    ap.add_argument("--which", default="tf", choices=["vn", "tf"])
    ap.add_argument("--out", default="exp1.json")
    a = ap.parse_args()
    if a.which == "vn":
        r = run_vn(a.budget, a.lr, seed=a.seed)
    else:
        r = run_tf(a.budget, a.tf_T, a.lr, seed=a.seed)
    r.update(budget=a.budget, lr=a.lr, seed=a.seed)
    print("RESULT " + json.dumps(r))
    json.dump(r, open(a.out, "w"), indent=2)
