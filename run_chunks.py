"""Run the field in resumable chunks (sandbox has a per-command time cap).

State is checkpointed to ``ckpt.npz`` so successive invocations continue the
same run instead of restarting it.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from vn_mobile.core.field import Field
from vn_mobile.core.neuron import build_population


def load(args) -> Field:
    pop = build_population(args.n, args.degree, args.seed,
                           weight_sigma=args.sigma)
    f = Field(pop, seed=args.seed, kappa=args.kappa)
    try:
        z = np.load("ckpt.npz", allow_pickle=True)
        f.pop.energy[:] = z["energy"]
        f.pop.state[:] = z["state"]
        if "phase" in z:
            f.pop.phase[:] = z["phase"]
        f.stats.step = int(z["step"])
        f.stats.rescales = int(z["rescales"])
        f.rng.bit_generator.state = z["rng_state"].item()
        print(f"resumed from step {f.stats.step}")
    except FileNotFoundError:
        f.pop.state[:] = np.random.default_rng(args.seed + 1).normal(
            0.0, args.init_state, size=f.pop.state.shape).astype(np.float32)
        print("fresh start")
    return f


def save(f: Field) -> None:
    np.savez("ckpt.npz", energy=f.pop.energy, state=f.pop.state,
             phase=f.pop.phase,
             step=f.stats.step, rescales=f.stats.rescales,
             rng_state=np.array(f.rng.bit_generator.state, dtype=object))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--init-state", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()

    f = load(args)
    t0 = time.perf_counter()
    st = f.run(args.chunk, log_every=0)
    dt = time.perf_counter() - t0
    save(f)
    e = f.pop.energy
    print(f"step={st.step} us/step={dt/args.chunk*1e6:.1f} "
          f"e_max={st.energy_max:.4f} e_mean={st.energy_mean:.5f} "
          f"e_std={st.energy_std:.4f} |e|={np.abs(e).mean():.4f} "
          f"neg%={100*(e<0).mean():.1f} top%={100*st.top_frac:.3f} "
          f"h_max={st.state_max:.4f} nan={st.nan} rescales={st.rescales}")


if __name__ == "__main__":
    main()
