"""Stage 2: does a perturbation actually travel as a wave?

Method
------
1. Run the field to steady state and record the baseline energy per neuron.
2. Disturb a seed neuron and record, per tick, the mean |e - e_base| grouped by
   graph distance (BFS layer) from the seed.
3. A wave is present only if the disturbance front moves outward: far layers
   light up later than near ones, and the lit radius grows with time.

This also checks whether an input going through the documented encoder path
(text -> h layer) can move the energy field at all.
"""
from __future__ import annotations

import argparse

import numpy as np

from vn_mobile.core.field import Field
from vn_mobile.core.neuron import DIM, build_population
from vn_mobile.io.encoder import encode


def bfs_layers(pop, seed: int, max_layers: int = 8) -> np.ndarray:
    """Graph distance from ``seed`` for every neuron; -1 if unreachable."""
    dist = np.full(pop.n, -1, dtype=np.int32)
    dist[seed] = 0
    frontier = np.array([seed], dtype=np.int32)
    for d in range(1, max_layers + 1):
        nxt = []
        for u in frontier:
            nxt.append(pop.dst[pop.ptr[u]:pop.ptr[u + 1]])
        if not nxt:
            break
        cand = np.unique(np.concatenate(nxt))
        cand = cand[dist[cand] < 0]
        if cand.size == 0:
            break
        dist[cand] = d
        frontier = cand
    return dist


def run_diffusion(n: int = 1000, degree: int = 8, seed: int = 0,
                  sigma: float = 4.0, k_coupling: float = 0.2,
                  warmup: int = 5000, observe: int = 200,
                  amp: float = 0.5, max_layers: int = 6) -> dict:
    pop = build_population(n=n, degree=degree, seed=seed, weight_sigma=sigma)
    field = Field(pop, seed=seed, k_coupling=k_coupling)
    field.run(warmup)

    base = pop.energy.copy()
    dist = bfs_layers(pop, 0, max_layers)
    layers = [np.where(dist == d)[0] for d in range(max_layers + 1)]

    # --- A: disturb the energy field directly ---
    pop.energy[0] += amp
    trace = []
    for t in range(observe):
        field.tick()
        row = [float(np.abs(pop.energy[l] - base[l]).mean()) if l.size else 0.0
               for l in layers]
        trace.append(row)
    trace = np.array(trace)

    # --- B: does the documented text->h path move the energy field? ---
    # Needs a control run: the field never stops moving, so comparing against
    # its own baseline measures drift, not causation.  Two fields built from
    # the same seed are identical until one of them is perturbed, so their
    # divergence is the causal effect of the input.
    pop_a = build_population(n=n, degree=degree, seed=seed, weight_sigma=sigma)
    pop_b = build_population(n=n, degree=degree, seed=seed, weight_sigma=sigma)
    fa = Field(pop_a, seed=seed, k_coupling=k_coupling)
    fb = Field(pop_b, seed=seed, k_coupling=k_coupling)
    fa.norm_every = fb.norm_every = 10 ** 9   # no rescaling, keep them parallel
    fa.run(warmup)
    fb.run(warmup)

    ids, values = encode("hello wave", n)
    fb.perturb(ids, values)
    h_before = float(np.abs(pop_b.state).max())
    for _ in range(200):
        fb.perturb(ids, values)   # sustained input, as a real query would be
        fa.tick()
        fb.tick()
    h_after = float(np.abs(pop_b.state).max())
    e_causal = float(np.abs(pop_b.energy - pop_a.energy).max())
    e_drift = float(np.abs(pop_a.energy - base).max())

    return {
        "trace": trace,
        "layers": [int(l.size) for l in layers],
        "encoder_e_shift": e_causal,
        "encoder_e_drift": e_drift,
        "encoder_h_before": h_before,
        "encoder_h_after": h_after,
    }


def summarise(res: dict, thresh: float = 0.01) -> None:
    trace = res["trace"]
    print("layer sizes:", res["layers"])
    print("\nper-layer |e - baseline| (rows = ticks)")
    print("%6s " % "tick" + " ".join("%8d" % d for d in range(trace.shape[1])))
    for t in sorted(set(min(v, trace.shape[0]-1) for v in [0,1,2,5,10,20,50,100,149])):
        print("%6d " % t + " ".join("%8.4f" % v for v in trace[t]))

    print("\nfront arrival (first tick a layer exceeds %.3f):" % thresh)
    for d in range(trace.shape[1]):
        hit = np.where(trace[:, d] > thresh)[0]
        print("  layer %d: %s" % (d, "never" if hit.size == 0 else "tick %d" % hit[0]))

    print("\nencoder path (text -> h layer):")
    print("  |h| before=%.4e after=%.4e" % (res["encoder_h_before"], res["encoder_h_after"]))
    print("  causal |e(in) - e(control)|  = %.4e" % res["encoder_e_shift"])
    print("  natural drift |e(control)-base| = %.4e" % res["encoder_e_drift"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VN-Mobile wave diffusion probe")
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--k", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--observe", type=int, default=200)
    ap.add_argument("--amp", type=float, default=0.5)
    args = ap.parse_args()
    summarise(run_diffusion(sigma=args.sigma, k_coupling=args.k,
                            warmup=args.warmup, observe=args.observe,
                            amp=args.amp))
