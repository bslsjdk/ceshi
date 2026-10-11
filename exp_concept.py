"""Concept wave v1 experiments (<=200 lines).

  exp 1  concept wave vs energy diffusion: sparsity + val bits
  exp 2  interference = error correction: phase sweep + wrong-path decay
  exp 3  long context: hist=100 vs hist=1000
  exp 4  external feedback: anti-phase "wrong" signal recovers the mode

Baseline A (energy diffusion v1) is not re-run: exp1_vn.json was produced by
exp1.py with the same seed / budget / val slice, and that run is the number
quoted in the report.
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from conceptwave.encode import CharCodec
from conceptwave.field import ConceptField
from conceptwave.kernels import cosine, normalise
from conceptwave.train import run
from vn_mobile.io.adam import Adam
from vn_mobile.io.vocab import build_vocab, encode, load_text, split

LOG2 = float(np.log(2.0))


def common(seed=0):
    text = load_text()
    symbols, stoi = build_vocab(text)
    tr_txt, va_txt = split(text)
    return len(symbols), encode(tr_txt, stoi), encode(va_txt, stoi)


VARIANTS = {
    "v1": dict(signed=False, nonlinear=False, psi_max=0.0, norm_global=True),
    "v11": dict(signed=True, nonlinear=True, psi_max=10.0, norm_global=False),
}


def make(n=1000, dc=32, nlayers=1, k=8, seed=0, dorm=0.3):
    f = ConceptField.create(n, dc, nlayers, k, seed=seed, dormancy=dorm)
    c = CharCodec(n, dc, seed=seed) if False else None
    return f


# ---------------- exp 1 ----------------
def exp1(budget=300000, lr=1e-3, seed=0, val_limit=3000, n=1000, dc=32,
         k=8, dorm=0.3, rounds=12, variant="v11"):
    V, tr, va = common(seed)
    f = ConceptField.create(n, dc, 1, k, seed=seed, dormancy=dorm,
                            **VARIANTS[variant])
    codec = CharCodec(V, dc, seed=seed)
    opt = Adam(lr=lr)
    t0 = time.perf_counter()
    trb, acc, curve = run(f, codec, tr[:budget], opt, hist=1000, rounds=rounds,
                          log_every=25000, tag="cw")
    fb, facc, _ = run(f, codec, va[:val_limit], None, hist=1000, rounds=rounds)
    fa = float(np.mean([_frac(f, codec, int(t)) for t in va[:200]]))
    dom = float(np.mean([_dominance(f, codec, int(t)) for t in va[:100]]))
    return dict(exp=1, model="concept_wave", variant=variant,
                concept_sim_top5=round(dom, 4),
                params=int(codec.emb.size + codec.bias.size + 1),
                val_bits=round(float(fb), 4), val_acc=round(float(facc), 4),
                train_bits_last=curve[-1]["train_bits"] if curve else None,
                frac_active=round(fa, 4), rounds=rounds, dormancy=dorm,
                curve=curve,
                wall_s=round(time.perf_counter() - t0, 1),
                rss_mb=round(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 1024, 1))


def _dominance(f, codec, tok):
    """cos(concept, char) of the 5 most energetic nodes after settling."""
    f.set_context(codec.ctx_vector([tok]))
    codec.inject_char(f, tok)
    f.settle(max_rounds=12)
    e = f.energy()
    top = np.argsort(-e)[:5]
    return float(cosine(f.cn[top], normalise(codec.emb[tok].reshape(1, -1))).mean())


def _frac(f, codec, tok):
    f.set_context(codec.ctx_vector([tok]))
    codec.inject_char(f, tok)
    f.settle(max_rounds=12)
    return f.frac_active()


# ---------------- exp 2 ----------------
def exp2(seed=0, n=1000, dc=32, k=8, rounds=10, variant="v11"):
    f = ConceptField.create(n, dc, 1, k, seed=seed, dormancy=0.0,
                            **VARIANTS[variant])
    rng = np.random.default_rng(seed + 7)
    c1 = normalise(rng.normal(size=(1, dc)))[0]
    c2 = normalise(rng.normal(size=(1, dc)))[0]
    s1 = cosine(normalise(f.concept), c1.reshape(1, -1)).ravel()
    s2 = cosine(normalise(f.concept), c2.reshape(1, -1)).ravel()
    overlap = np.argsort(-np.minimum(s1, s2))[:50]
    f.set_context((c1 + c2) / 2)
    sweep = {}
    for phi in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        f.psi[:] = 0
        f.inject(np.clip(s1, 0, None) ** 2, phase=0.0)
        f.inject(np.clip(s2, 0, None) ** 2, phase=float(phi))
        nrm = np.linalg.norm(f.psi)
        if nrm > 0:
            f.psi /= nrm
        e0 = float(np.abs(f.psi[overlap]).sum())
        f.settle(max_rounds=rounds)
        e1 = float(np.abs(f.psi[overlap]).sum())
        sweep[f"{phi:.3f}"] = dict(before=round(e0, 5), after=round(e1, 5),
                                   ratio=round(e1 / e0, 4) if e0 else None)
    # wrong-path decay: anti-phase error injected on top of a settled mode
    f.psi[:] = 0
    f.inject(np.clip(s1, 0, None) ** 2, phase=0.0)
    f.settle(max_rounds=12)
    err_support = np.argsort(-s2)[:50]
    f.inject(1.0 * np.clip(s2, 0, None) ** 2, phase=float(np.pi))
    nrm = np.linalg.norm(f.psi)
    if nrm > 0:
        f.psi /= nrm
    traj = []
    for i in range(rounds):
        e_err = float(np.abs(f.psi[err_support]).sum())
        traj.append(round(e_err, 5))
        f.settle(max_rounds=1)
    first, last = traj[0], traj[-1]
    fp = float((f.edges["base"] > 0).mean())
    return dict(exp=2, variant=variant, frac_positive=round(fp, 4),
                phase_sweep=sweep,
                wrong_path_traj=traj,
                wrong_path_decay=round(1 - last / first, 4) if first else None)


# ---------------- exp 3 ----------------
def exp3(budget=100000, lr=1e-3, seed=0, val_limit=3000, rounds=12,
         variant="v11"):
    V, tr, va = common(seed)
    out = {}
    for hist in (100, 1000):
        f = ConceptField.create(1000, 32, 1, 8, seed=seed, dormancy=0.3,
                                **VARIANTS[variant])
        codec = CharCodec(V, 32, seed=seed)
        opt = Adam(lr=lr)
        run(f, codec, tr[:budget], opt, hist=hist, rounds=rounds)
        b, a, _ = run(f, codec, va[:val_limit], None, hist=hist, rounds=rounds)
        out[f"hist{hist}"] = dict(val_bits=round(float(b), 4),
                                  val_acc=round(float(a), 4))
    return dict(exp=3, variant=variant, **out)


# ---------------- exp 4 ----------------
def exp4(seed=0, n=1000, dc=32, k=8, rounds=100, variant="v11"):
    V, tr, va = common(seed)
    f = ConceptField.create(n, dc, 1, k, seed=seed, dormancy=0.3,
                            **VARIANTS[variant])
    codec = CharCodec(V, dc, seed=seed)
    correct, wrong = 5, 40
    f.set_context(codec.ctx_vector([correct]))
    codec.inject_char(f, correct)
    f.settle(max_rounds=12)
    base = int(np.argmax(codec.logits(f)))
    codec.inject_char(f, wrong, gain=3.0)      # a confident wrong answer
    f.settle(max_rounds=12)
    after_wrong = int(np.argmax(codec.logits(f)))
    recovered, steps = None, None
    for i in range(rounds):
        f.counter(1.0 * codec.amp_for(f, wrong))   # feedback "wrong"
        f.settle(max_rounds=1)
        if int(np.argmax(codec.logits(f))) == correct:
            recovered, steps = True, i + 1
            break
    return dict(exp=4, variant=variant, correct=correct, wrong=wrong,
                argmax_settled=base, argmax_after_wrong=after_wrong,
                recovered=bool(recovered), rounds_to_recover=steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="1")
    ap.add_argument("--budget", type=int, default=300000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="v11", choices=["v1", "v11"])
    ap.add_argument("--out", default="exp_concept.json")
    a = ap.parse_args()
    if a.exp == "1":
        r = exp1(budget=a.budget, lr=a.lr, seed=a.seed, variant=a.variant)
    elif a.exp == "2":
        r = exp2(seed=a.seed, variant=a.variant)
    elif a.exp == "3":
        r = exp3(budget=a.budget, lr=a.lr, seed=a.seed, variant=a.variant)
    else:
        r = exp4(seed=a.seed, variant=a.variant)
    r.update(seed=a.seed)
    print("RESULT " + json.dumps(r))
    with open(f"{a.out}", "w") as fh:
        json.dump(r, fh, indent=2)
