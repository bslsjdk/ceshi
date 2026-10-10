"""6A with the full-field readout, trained longer.

blockwise (logit v sees only block v) -> 8.50 bits, worse than uniform.
full-field (logit v sees the whole pattern) -> 4.50 bits at 20k steps.
sklearn on the same features reaches 4.276, so that is the ceiling this
representation supports.  Question here is how close a plain SGD gets.
"""
from __future__ import annotations
import argparse, json, resource, time
import numpy as np
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.core.update import ALPHA, step_energy, step_phase, step_state, read_output
from vn_mobile.io.charhead import CharHead, softmax, train_char
from vn_mobile.io.vocab import block_of, build_vocab, encode, load_text, split, pool_by_block

def run(steps, n_vn, lr, seed, val_every, gen_len, train_bias=False,
        train_adapters=False):
    text = load_text(); symbols, stoi = build_vocab(text); V = len(symbols)
    tr_txt, va_txt = split(text); tr = encode(tr_txt, stoi); va = encode(va_txt, stoi)
    pop = build_population(n=n_vn, degree=8, seed=seed, omega_mix=(0.8, 0.3, 1.0))
    block = block_of(n_vn, V)
    head = CharHead.make(V, DIM, seed=seed, full=True)
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(block == v) for v in range(V)]
    ie = np.zeros(n_vn, np.float32); ih = np.zeros((n_vn, DIM), np.float32)
    hist = []; losses = []

    def validate(limit=2000):
        loss = 0.0; hit = 0
        for i in range(limit):
            c = int(va[i]); nxt = int(va[i + 1])
            ie[:] = 0; ih[:] = 0
            ie[members[c]] = 1.0; ih[members[c]] = 1.0
            step_energy(pop, rng, injection=ie); step_phase(pop); step_state(pop, ih)
            lg = head.logits(pool_by_block(read_output(pop), block, V))
            pr = softmax(lg)
            loss += float(-np.log(max(float(pr[nxt]), 1e-9)))
            hit += int(int(np.argmax(lg)) == nxt)
        return dict(val_bits=round(loss / limit / np.log(2), 4),
                    val_acc=round(hit / limit, 4))

    t0 = time.perf_counter()
    for t in range(steps):
        c = int(tr[t % tr.size]); nxt = int(tr[(t + 1) % tr.size])
        ie[:] = 0; ih[:] = 0
        ie[members[c]] = 1.0; ih[members[c]] = 1.0
        step_energy(pop, rng, injection=ie); step_phase(pop); step_state(pop, ih)
        loss, ok = train_char(pop, block, head, nxt, lr, ALPHA, ih,
                              train_head=True, train_bias=train_bias,
                              train_adapters=train_adapters)
        losses.append(loss)
        if (t + 1) % val_every == 0:
            hist.append(dict(step=t + 1,
                             train_bits=round(float(np.mean(losses[-val_every:]))/np.log(2), 4),
                             **validate(), wall_s=round(time.perf_counter()-t0, 1)))
            print(json.dumps(hist[-1]), flush=True)
    # generate
    out = [int(x) for x in va[:8]]
    for _ in range(gen_len):
        c = int(out[-1]); ie[:] = 0; ih[:] = 0
        ie[members[c]] = 1.0; ih[members[c]] = 1.0
        step_energy(pop, rng, injection=ie); step_phase(pop); step_state(pop, ih)
        lg = head.logits(pool_by_block(read_output(pop), block, V))
        out.append(int(rng.choice(lg.size, p=softmax(lg))))
    sample = "".join(chr(int(v)) for v in out)
    return dict(lr=lr, hist=hist, sample=sample,
                rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1),
                head_params=int(head.w.size))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--n-vn", type=int, default=1000)
    ap.add_argument("--lrs", default="0.0003,0.001,0.003")
    ap.add_argument("--val-every", type=int, default=25000)
    ap.add_argument("--gen-len", type=int, default=500)
    ap.add_argument("--out", default="train6a_full.json")
    a = ap.parse_args()
    rows = []
    for lr in [float(x) for x in a.lrs.split(",")]:
        r = run(a.steps, a.n_vn, lr, 0, a.val_every, a.gen_len)
        rows.append(dict(lr=lr, hist=r["hist"], head_params=r["head_params"],
                         rss_mb=r["rss_mb"], sample=r["sample"]))
        print(json.dumps(dict(lr=lr, final=r["hist"][-1])), flush=True)
    out = dict(steps=a.steps, n_vn=a.n_vn, uniform_bits=6.022,
               unigram_bits=4.774, sklearn_bits=4.276, gpt_team_bits=1.088,
               rows=rows)
    json.dump(out, open(a.out, "w"), indent=2)
    with open("samples.txt", "w") as fh:
        best = min(rows, key=lambda r: r["hist"][-1]["val_bits"])
        fh.write(best["sample"])
    print("BEST", best["lr"], best["hist"][-1])
