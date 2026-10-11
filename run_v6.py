"""VN v6 driver.  Subcommands: baseline, cosdist, sweep, scratch."""
from __future__ import annotations

import argparse, json, time
import numpy as np

from vnv6.compress import cosine_matrix
from vnv6.model import ConceptNet
from vnv6.run import pipeline
from vnv6.train import evaluate, streams, train, unigram_bits


def save(net: ConceptNet, path: str) -> None:
    np.savez(path, emb=net.emb, keys=net.keys, W=net.W, out=net.out,
             bias=net.bias, norm=np.float32(net.norm))


def load(path: str, V: int, N: int) -> ConceptNet:
    d = np.load(path)
    net = ConceptNet(V, N, seed=0)
    net.emb, net.keys, net.W = d['emb'], d['keys'], d['W']
    net.out, net.bias, net.norm = d['out'], d['bias'], float(d['norm'])
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--r', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='v6_out.json')
    ap.add_argument('--state', default='v6_state.npz')
    ap.add_argument('--ks', default='10,25,50,100,200')
    ap.add_argument('--ns', default='50,105,200')
    ap.add_argument('--ftsteps', type=int, default=50000)
    ap.add_argument('--mode', default='pca')
    a = ap.parse_args()
    tr, va, V = streams()

    if a.cmd == 'baselines':
        print(json.dumps(dict(vocab=V, unigram_bits=round(unigram_bits(tr, va, V), 4),
                              uniform_bits=round(float(np.log2(V)), 4))), flush=True)
    elif a.cmd == 'baseline':
        net = ConceptNet(V, a.n, seed=a.seed)
        t0 = time.perf_counter()
        hist, _ = train(net, tr, va, steps=a.steps, lr=a.lr, val_every=10000)
        save(net, a.state)
        print(json.dumps(dict(cmd='baseline', n=a.n, params=net.n_params(),
                              wall_s=round(time.perf_counter() - t0, 1),
                              hist=hist, final=evaluate(net, va))), flush=True)
    elif a.cmd == 'cosdist':
        net = load(a.state, V, a.n)
        c = net.concepts()
        cos = cosine_matrix(c)
        iu = np.triu_indices(c.shape[0], 1)
        off = cos[iu]
        out = dict(n=c.shape[0], max_cos=round(float(off.max()), 4),
                   mean_cos=round(float(off.mean()), 4),
                   mean_abs_cos=round(float(np.abs(off).mean()), 4))
        for t in (0.99, 0.95, 0.9, 0.8, 0.7, 0.5):
            out[f'pairs_ge_{t}'] = int((off >= t).sum())
        out['nearest_neighbour_hist'] = np.histogram(
            cos.max(axis=1) if False else np.sort(off)[:0], bins=1)[0].tolist()
        nn = np.clip(cos - np.eye(c.shape[0]) * 2, -1, 1).max(axis=1)
        out['nn_cos'] = dict(p50=round(float(np.percentile(nn, 50)), 4),
                             p90=round(float(np.percentile(nn, 90)), 4),
                             p99=round(float(np.percentile(nn, 99)), 4),
                             max=round(float(nn.max()), 4))
        print(json.dumps(out), flush=True)
    elif a.cmd == 'sweep':
        net = load(a.state, V, a.n)
        base_bits = evaluate(net, va)
        rows = []
        for k in [int(x) for x in a.ks.split(',')]:
            t0 = time.perf_counter()
            rep, pn = pipeline(net, tr, va, k=k, r=a.r, mode=a.mode,
                               steps=a.ftsteps, lr=a.lr)
            rows.append(dict(k=k, r=a.r, mode=a.mode, base_bits=base_bits,
                             params=rep['params'], fuse=rep['fuse'],
                             cluster=rep['cluster'], pack=rep['pack'],
                             finetune=rep['finetune'],
                             wall_s=round(time.perf_counter() - t0, 1)))
            print(json.dumps(rows[-1]), flush=True)
        json.dump(rows, open(a.out, 'w'), indent=1, default=str)
    elif a.cmd == 'scratch':
        rows = []
        for n in [int(x) for x in a.ns.split(',')]:
            net = ConceptNet(V, n, seed=a.seed)
            t0 = time.perf_counter()
            hist, _ = train(net, tr, va, steps=a.steps, lr=a.lr,
                            val_every=max(1, a.steps // 5))
            rows.append(dict(n=n, params=net.n_params(),
                             block_params=n * 144,
                             wall_s=round(time.perf_counter() - t0, 1),
                             hist=hist, final=evaluate(net, va)))
            print(json.dumps(rows[-1]), flush=True)
        json.dump(rows, open(a.out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
