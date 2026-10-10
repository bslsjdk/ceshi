"""Stage 6A: character-level training on TinyShakespeare.

Sequence memory is the field itself -- injecting a character raises its
address block and the wave carries the history -- so there is no separate
hidden state to size or reset.

Baseline to compare against is the GPT-team 14,770-parameter model at 80.71%
accuracy / 0.754 cross-entropy.  A unigram model on this split sits at 4.774
bits, so anything below that is real structure, not marginals.
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from stages5a_5d import _R_live
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import (ALPHA, LAMBDA, neighbour_mean,
                                   step_energy, step_phase, step_state)
from vn_mobile.io.charhead import CharHead, forward_char, softmax, train_char
from vn_mobile.io.vocab import (block_of, build_vocab, encode, load_text,
                                split)


def make_field(n_vn: int, seed: int):
    pop = build_population(n=n_vn, degree=8, seed=seed,
                           omega_mix=(0.8, 0.3, 1.0))
    return pop, block_of(n_vn, len(build_vocab(load_text())[0]))


PRESETS = {
    # spec: adapters only
    "adapter": dict(train_head=False, train_bias=False, train_adapters=True),
    "head": dict(train_head=True, train_bias=False, train_adapters=False),
    "head_bias": dict(train_head=True, train_bias=True, train_adapters=False),
    "all": dict(train_head=True, train_bias=True, train_adapters=True),
}


def run(steps: int, n_vn: int, lr: float, seed: int, log_every: int,
        gen_len: int, val_every: int, preset: str = "head_bias"):
    text = load_text()
    symbols, stoi = build_vocab(text)
    V = len(symbols)
    tr_txt, va_txt = split(text)
    tr = encode(tr_txt, stoi)
    va = encode(va_txt, stoi)

    pop = build_population(n=n_vn, degree=8, seed=seed,
                           omega_mix=(0.8, 0.3, 1.0))
    block = block_of(n_vn, V)
    head = CharHead.make(V, DIM, seed=seed, full=True)
    rng = np.random.default_rng(seed)
    inj_e = np.zeros(n_vn, dtype=np.float32)
    inj_h = np.zeros((n_vn, DIM), dtype=np.float32)
    members = [np.flatnonzero(block == v) for v in range(V)]

    hist, losses, hits, n = [], [], 0, 0
    t0 = time.perf_counter()
    for t in range(steps):
        c = tr[t % tr.size]
        nxt = tr[(t + 1) % tr.size]
        inj_e[:] = 0.0
        inj_h[:] = 0.0
        inj_e[members[c]] = 1.0
        inj_h[members[c]] = 1.0
        x = inj_h + neighbour_mean(pop, pop.state)
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        step_state(pop, inj_h)
        loss, ok = train_char(pop, block, head, int(nxt), lr, ALPHA,
                               x, **PRESETS[preset])
        losses.append(loss)
        hits += int(ok)
        n += 1
        if val_every and (t + 1) % val_every == 0:
            hist.append(dict(step=t + 1,
                             train_loss=round(float(np.mean(losses[-val_every:])), 4),
                             train_acc=round(hits / n, 4),
                             val=validate(pop, block, head, va, rng, inj_e,
                                          inj_h, members, n_vn, 2000),
                             wall_s=round(time.perf_counter() - t0, 1)))
            print(json.dumps(hist[-1]), flush=True)
        elif log_every and (t + 1) % log_every == 0:
            print(json.dumps(dict(step=t + 1,
                                  train_loss=round(float(np.mean(losses[-log_every:])), 4),
                                  train_acc=round(hits / n, 4),
                                  wall_s=round(time.perf_counter() - t0, 1))),
                  flush=True)
    sample = generate(pop, block, head, va[:8], rng, inj_e, inj_h, members,
                      n_vn, gen_len)
    return dict(pop=pop, head=head, hist=hist, sample=sample,
                train_acc=hits / n, symbols=symbols)


def validate(pop, block, head, seq, rng, inj_e, inj_h, members, n_vn,
             limit: int) -> dict:
    n = min(limit, seq.size - 1)
    loss, hit = 0.0, 0
    for i in range(n):
        inj_e[:] = 0.0
        inj_h[:] = 0.0
        inj_e[members[seq[i]]] = 1.0
        inj_h[members[seq[i]]] = 1.0
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        step_state(pop, inj_h)
        _, _, probs = forward_char(pop, block, head)
        loss += float(-np.log(max(float(probs[seq[i + 1]]), 1e-9)))
        hit += int(int(np.argmax(probs)) == int(seq[i + 1]))
    return dict(val_loss=round(loss / n, 4), val_acc=round(hit / n, 4),
                val_bits=round(loss / n / np.log(2), 4))


def generate(pop, block, head, seed_seq, rng, inj_e, inj_h, members, n_vn,
             length: int, temperature: float = 1.0) -> str:
    out = list(seed_seq)
    for _ in range(length):
        c = int(out[-1])
        inj_e[:] = 0.0
        inj_h[:] = 0.0
        inj_e[members[c]] = 1.0
        inj_h[members[c]] = 1.0
        step_energy(pop, rng, injection=inj_e)
        step_phase(pop)
        step_state(pop, inj_h)
        _, logits, _ = forward_char(pop, block, head)
        p = softmax(logits / temperature)
        out.append(int(rng.choice(p.size, p=p)))
    return "".join(chr(int(v)) for v in out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--n-vn", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10000)
    ap.add_argument("--val-every", type=int, default=50000)
    ap.add_argument("--gen-len", type=int, default=500)
    ap.add_argument("--preset", default="head_bias", choices=sorted(PRESETS))
    ap.add_argument("--out", default="train6a.json")
    a = ap.parse_args()
    r = run(a.steps, a.n_vn, a.lr, a.seed, a.log_every, a.gen_len,
            a.val_every, a.preset)
    text = load_text()
    out = dict(steps=a.steps, n_vn=a.n_vn, lr=a.lr, seed=a.seed,
               preset=a.preset,
               vocab=len(build_vocab(text)[0]),
               train_acc=round(r["train_acc"], 4), hist=r["hist"],
               rss_mb=round(resource.getrusage(
                   resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
               sample=r["sample"])
    print(json.dumps({k: v for k, v in out.items() if k != "sample"},
                     indent=2))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    with open("samples.txt", "w") as fh:
        fh.write(r["sample"])


if __name__ == "__main__":
    main()
