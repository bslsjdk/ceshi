"""Stage 6A2 driver: Adam optimiser + zone-shared adapters + truncated BPTT.

Changes relative to 6C, each pinned to a measurement:

* Adam replaces SGD: the 0.116 bit gap to a scikit-learn logistic regression
  on identical features is an optimiser gap, not a representation gap.
* Adapters become per-zone (20 x A[4,8]/B[8,4] = 1,280) so the trainable set
  fits the 14,770 budget -- per-neuron A/B is 64,000 on its own.  It also
  gives BPTT somewhere inside the field to put a gradient; in 6C the only
  parameter on the time path was the 8-element bias.
* BPTT T=20, low-rank readout R=16 and the persistent field all stay.
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import step_energy, step_phase
from vn_mobile.io.adam import Adam
from vn_mobile.io.bptt import (Window, backward_window, finite,
                               head_backward, pooled_flat, step_state_x)
from vn_mobile.io.lowrank import LowRankHead, softmax
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split

PRESETS = {
    "readout": dict(bias=False, adapters=False),
    "readout_bias": dict(bias=True, adapters=False),
    "all": dict(bias=True, adapters=True),
}

def count_params(pop, head, preset: dict) -> dict:
    r = head.n_params
    b = int(pop.bias.size) if preset["bias"] else 0
    a = int(pop.adapter_a.size + pop.adapter_b.size) if preset["adapters"] else 0
    return dict(readout=r, bias=b, adapters=a, total=r + b + a)

def adam_pairs(pop, head, g, preset: dict) -> dict:
    """(array, grad) map handed to Adam; arrays stay owned by their object."""
    p = {"head_u": (head.u, g["dw"][0]), "head_p": (head.p, g["dw"][1])}
    if preset["bias"]:
        p["bias"] = (pop.bias, g["db"])
    if preset["adapters"]:
        p["adapter_b"] = (pop.adapter_b, g["gb"])
        p["adapter_a"] = (pop.adapter_a, g["ga"])
    return p

def inject(members, seq, t, hist: int, decay: float, inj_e, inj_h) -> None:
    inj_e[:] = 0.0
    inj_h[:] = 0.0
    for k in range(hist):
        c = int(seq[(t - k) % seq.size])
        w = np.float32(decay ** k)
        inj_e[members[c]] += w
        inj_h[members[c]] += w

def forward(pop, head, block, members, seq, t, T, rng, inj_e, inj_h, vocab,
            hist: int, decay: float) -> Window:
    win = Window()
    for _ in range(T):
        inject(members, seq, t, hist, decay, inj_e, inj_h)
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        x, th = step_state_x(pop, inj_h)
        flat = pooled_flat(pop, block, vocab)
        probs = softmax(head.logits(flat))
        nxt = int(seq[(t + 1) % seq.size])
        win.xs.append(x)
        win.ths.append(th)
        win.flats.append(flat)
        win.probs.append(probs)
        win.energies.append(pop.energy.copy())
        win.targets.append(nxt)
        win.losses.append(float(-np.log(max(float(probs[nxt]), 1e-9))))
        t = (t + 1) % seq.size
    return win

def roll(pop, head, block, members, seq, rng, inj_e, inj_h, vocab: int,
         hist: int, decay: float, limit: int) -> dict:
    """Forward-only validation pass; no gradients, no updates."""
    n = min(limit, seq.size - 1)
    loss, hit = 0.0, 0
    for i in range(n):
        inject(members, seq, i, hist, decay, inj_e, inj_h)
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        step_state_x(pop, inj_h)
        probs = softmax(head.logits(pooled_flat(pop, block, vocab)))
        nxt = int(seq[i + 1])
        loss += float(-np.log(max(float(probs[nxt]), 1e-9)))
        hit += int(int(np.argmax(probs)) == nxt)
    return dict(val_loss=round(loss / n, 4), val_acc=round(hit / n, 4),
                val_bits=round(loss / n / np.log(2), 4))

def generate(pop, head, block, members, seed, rng, inj, ih, vocab, hist,
             decay: float, length: int) -> str:
    out = [int(c) for c in seed]
    for _ in range(length):
        cur = np.array(out, dtype=np.int32)
        inject(members, cur, cur.size - 1, hist, decay, inj, ih)
        step_energy(pop, rng, injection=inj)
        step_phase(pop)
        step_state_x(pop, ih)
        probs = softmax(head.logits(pooled_flat(pop, block, vocab)))
        out.append(int(rng.choice(probs.size, p=probs)))
    return "".join(chr(int(v)) for v in out)

def run(steps, n_vn, lr, seed, T, rank, preset, hist, decay, zones, val_every,
        log_every, gen_len):
    text = load_text()
    symbols, stoi = build_vocab(text)
    V = len(symbols)
    tr_txt, va_txt = split(text)
    tr = encode(tr_txt, stoi)
    va = encode(va_txt, stoi)
    pop = build_population(n=n_vn, degree=8, seed=seed,
                           omega_mix=(0.8, 0.3, 1.0), n_zones=zones)
    block = block_of(n_vn, V)
    members = [np.flatnonzero(block == v) for v in range(V)]
    head = LowRankHead.make(V, DIM, rank=rank, seed=seed)
    params = count_params(pop, head, PRESETS[preset])
    rng = np.random.default_rng(seed)
    opt = Adam(lr=lr)
    inj_e = np.zeros(n_vn, dtype=np.float32)
    inj_h = np.zeros((n_vn, DIM), dtype=np.float32)
    hist_out, losses, t0 = [], [], time.perf_counter()
    for w in range(steps // T):
        t = (w * T) % tr.size
        win = forward(pop, head, block, members, tr, t, T, rng, inj_e, inj_h,
                      V, hist, decay)
        losses.extend(win.losses)
        g = backward_window(pop, head, win, block, V, DIM)
        if finite(g):
            pairs = adam_pairs(pop, head, g, PRESETS[preset])
            opt.step(pairs)
            if PRESETS[preset]["adapters"]:
                pop.WM = None        # fused W+B@A cache is now stale
        done = (w + 1) * T
        if val_every and done % val_every == 0:
            hist_out.append(dict(
                step=done,
                train_bits=round(float(np.mean(losses[-val_every:]))
                                 / np.log(2), 4),
                val=roll(pop, head, block, members, va, rng, inj_e, inj_h,
                         V, hist, decay, 2000),
                wall_s=round(time.perf_counter() - t0, 1)))
            print(json.dumps(hist_out[-1]), flush=True)
        elif log_every and done % log_every == 0:
            print(json.dumps(dict(
                step=done,
                train_bits=round(float(np.mean(losses[-log_every:]))
                                 / np.log(2), 4),
                wall_s=round(time.perf_counter() - t0, 1))), flush=True)
    final = roll(pop, head, block, members, va, rng, inj_e, inj_h, V, hist,
                 decay, 3000)
    sample = ""
    if gen_len:
        sample = generate(pop, head, block, members, va[:8], rng, inj_e, inj_h,
                          V, hist, decay, gen_len)
        with open("samples6d.txt", "w") as fh:
            fh.write(sample)
    return dict(params=params, hist_cfg=hist_out, final=final, sample=sample)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--n-vn", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--zones", type=int, default=20)
    ap.add_argument("--preset", default="all", choices=sorted(PRESETS))
    ap.add_argument("--hist", type=int, default=1)
    ap.add_argument("--decay", type=float, default=0.6)
    ap.add_argument("--val-every", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--gen-len", type=int, default=0)
    ap.add_argument("--out", default="train6d.json")
    a = ap.parse_args()
    r = run(a.steps, a.n_vn, a.lr, a.seed, a.T, a.rank, a.preset, a.hist,
            a.decay, a.zones, a.val_every, a.log_every, a.gen_len)
    out = dict(steps=a.steps, T=a.T, lr=a.lr, seed=a.seed, rank=a.rank,
               zones=a.zones, preset=a.preset, hist=a.hist, params=r["params"],
               hist_cfg=r["hist_cfg"], final=r["final"],
               rss_mb=round(resource.getrusage(
                   resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1))
    print(json.dumps(out, indent=2))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

if __name__ == "__main__":
    main()
