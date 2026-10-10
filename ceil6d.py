"""Re-measure the "sklearn ceiling" on features from the *trained* field (6A2).

The 4.276 figure that motivated switching to Adam was measured on 12,000
(pooled_activity, next_char) pairs collected from one particular field state.
Training moves the field, so that number is a moving target: if the ceiling
has dropped below our val bits then the remaining gap is still the optimiser,
and if it has risen above them then the representation is the binding
constraint and no optimiser change will close it.

Both numbers come from the same run so they are comparable: the field is
trained, then frozen, then the pairs are collected and handed to a logistic
regression that never sees the field again.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import step_energy, step_phase
from vn_mobile.io.adam import Adam
from vn_mobile.io.bptt import (Window, backward_window, finite,
                               pooled_flat, step_state_x)
from vn_mobile.io.lowrank import LowRankHead, softmax
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split


def collect(pop, head, block, members, seq, V, limit, rng, inj_e, inj_h,
            train: bool, opt=None, T=20, hist=1, decay=0.6):
    """Run the field over ``limit`` chars, optionally training, gathering pairs."""
    feats, targets = [], []
    n = min(limit, seq.size - 1)
    for w in range(0, n - 1, T):
        win = Window()
        for k in range(T):
            i = w + k
            inj_e[:] = 0.0
            inj_h[:] = 0.0
            for m in range(hist):
                c = int(seq[(i - m) % seq.size])
                inj_e[members[c]] += np.float32(decay ** m)
                inj_h[members[c]] += np.float32(decay ** m)
            step_energy(pop, rng, injection=inj_e)
            step_phase(pop)
            x, th = step_state_x(pop, inj_h)
            flat = pooled_flat(pop, block, V)
            probs = softmax(head.logits(flat))
            nxt = int(seq[i + 1])
            win.xs.append(x)
            win.ths.append(th)
            win.flats.append(flat)
            win.probs.append(probs)
            win.energies.append(pop.energy.copy())
            win.targets.append(nxt)
            win.losses.append(float(-np.log(max(float(probs[nxt]), 1e-9))))
        if train and opt is not None:
            g = backward_window(pop, head, win, block, V, DIM)
            if finite(g):
                opt.step({"head_u": (head.u, g["dw"][0]),
                          "head_p": (head.p, g["dw"][1]),
                          "bias": (pop.bias, g["db"]),
                          "adapter_b": (pop.adapter_b, g["gb"]),
                          "adapter_a": (pop.adapter_a, g["ga"])})
                pop.WM = None
        feats.extend(win.flats)
        targets.extend(win.targets)
    return np.asarray(feats, dtype=np.float32), np.asarray(targets)


def bits(model, X, y):
    """NLL in bits over rows; an unseen class costs log2(#seen classes).

    The two halves of the validation split do not contain the same character
    set, so ``predict_proba`` does not have a column for every target -- a
    class the fit never saw has to be charged the chance rate instead of being
    silently dropped, which would flatter the ceiling.
    """
    cls = {int(c): i for i, c in enumerate(model.classes_)}
    p = np.clip(model.predict_proba(X), 1e-9, 1.0)
    nll, hit = 0.0, 0
    for r in range(y.size):
        j = cls.get(int(y[r]), -1)
        if j < 0:
            nll += float(np.log2(p.shape[1]))
            continue
        nll += float(-np.log2(p[r, j]))
        hit += int(int(model.classes_[int(np.argmax(p[r]))]) == int(y[r]))
    return float(nll / y.size), float(hit / y.size)


def main() -> None:
    text = load_text()
    symbols, stoi = build_vocab(text)
    V = len(symbols)
    tr_txt, va_txt = split(text)
    tr, va = encode(tr_txt, stoi), encode(va_txt, stoi)
    out = []
    for lr in (3e-4,):
        pop = build_population(n=1000, degree=8, seed=0,
                               omega_mix=(0.8, 0.3, 1.0), n_zones=20)
        block = block_of(1000, V)
        members = [np.flatnonzero(block == v) for v in range(V)]
        head = LowRankHead.make(V, DIM, rank=16, seed=0)
        rng = np.random.default_rng(0)
        opt = Adam(lr=lr)
        inj_e = np.zeros(1000, dtype=np.float32)
        inj_h = np.zeros((1000, DIM), dtype=np.float32)
        collect(pop, head, block, members, tr, V, 100000, rng, inj_e, inj_h,
                train=True, opt=opt)
        Xva, yva = collect(pop, head, block, members, va, V, 12000, rng,
                           inj_e, inj_h, train=False)
        half = Xva.shape[0] // 2
        mu = Xva[:half].mean(0)
        sd = Xva[:half].std(0) + 1e-6
        model = LogisticRegression(max_iter=1000, C=1.0).fit(
            ((Xva[:half] - mu) / sd), yva[:half])
        Xtr_s = (Xva[:half] - mu) / sd
        Xva_s = (Xva[half:] - mu) / sd
        b_tr, a_tr = bits(model, Xtr_s, yva[:half])
        b_va, a_va = bits(model, Xva_s, yva[half:])
        del Xtr_s, Xva_s
        field_b = float(np.mean([-np.log(max(float(
            softmax(head.logits(f))[t]), 1e-9))
            for f, t in zip(Xva, yva)]) / np.log(2))
        out.append(dict(lr=lr, sklearn_val_bits=round(b_va, 4),
                        sklearn_val_acc=round(a_va, 4),
                        sklearn_fit_bits=round(b_tr, 4),
                        field_val_bits=round(field_b, 4),
                        gap=round(field_b - b_va, 4)))
    print(json.dumps(out, indent=2))
    with open("ceil6d.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
