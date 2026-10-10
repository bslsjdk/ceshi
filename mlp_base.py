"""Task 4 control: a plain MLP with an explicit context window.

The VN field is claimed to carry state.  The cheapest honest control is a
feed-forward net that is handed the same context explicitly, sized so the
trainable parameter count matches the VN run.  If the field beats this at
equal parameters, the state is doing something; if not, the field is an
expensive way to concatenate the last k characters.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from vn_mobile.io.adam import Adam
from vn_mobile.io.vocab import build_vocab, encode, load_text, split


def make(k: int, V: int, h: int, seed: int):
    r = np.random.default_rng(seed)
    s = 0.08
    return dict(w1=(r.normal(0, s, (k * V, h))).astype(np.float32),
                b1=np.zeros(h, dtype=np.float32),
                w2=(r.normal(0, s, (h, V))).astype(np.float32),
                b2=np.zeros(V, dtype=np.float32))


def fwd(p, x, k, V):
    z = x @ p["w1"] + p["b1"]
    a = np.tanh(z)
    return a, a @ p["w2"] + p["b2"]


def train(k, V, h, tr, va, steps, lr, seed):
    p = make(k, V, h, seed)
    opt = Adam(lr=lr)
    rng = np.random.default_rng(seed)
    one = np.eye(V, dtype=np.float32)
    t0 = time.perf_counter()
    hist = []
    for s in range(steps):
        i = int(rng.integers(0, tr.size - k - 1))
        x = one[tr[i:i + k]].reshape(-1)
        y = int(tr[i + k])
        a, lg = fwd(p, x, k, V)
        lg = lg - lg.max()
        pr = np.exp(lg); pr /= pr.sum()
        dl = pr.copy(); dl[y] -= 1.0
        g = dict(w2=np.outer(a, dl), b2=dl,
                 w1=np.outer(x, ((1 - a * a) * (dl @ p["w2"].T))),
                 b1=(1 - a * a) * (dl @ p["w2"].T))
        opt.step({n: (p[n], g[n]) for n in p})
        if (s + 1) % 5000 == 0:
            hist.append(dict(step=s + 1, val=roll(p, va, k, V, 3000),
                             wall_s=round(time.perf_counter() - t0, 1)))
            print(json.dumps(hist[-1]), flush=True)
    return dict(k=k, hidden=h, params=int(k * V * h + h + h * V + V),
                hist=hist, final=roll(p, va, k, V, 3000))


def roll(p, seq, k, V, limit):
    one = np.eye(V, dtype=np.float32)
    n = min(limit, seq.size - k - 1)
    loss, hit = 0.0, 0
    for i in range(n):
        _, lg = fwd(p, one[seq[i:i + k]].reshape(-1), k, V)
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        y = int(seq[i + k])
        loss += float(-np.log(max(float(pr[y]), 1e-9)))
        hit += int(int(np.argmax(pr)) == y)
    return dict(val_loss=round(loss / n, 4), val_acc=round(hit / n, 4),
                val_bits=round(loss / n / np.log(2), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=81)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mlp.json")
    a = ap.parse_args()
    text = load_text(); sym, stoi = build_vocab(text); V = len(sym)
    tr = encode(split(text)[0], stoi); va = encode(split(text)[1], stoi)
    r = train(a.k, V, a.hidden, tr, va, a.steps, a.lr, a.seed)
    print(json.dumps(r, indent=2))
    json.dump(r, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
