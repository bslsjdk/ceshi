"""Entry point: build a field, run it, report stability and memory."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from vn_mobile.core.field import Field
from vn_mobile.core.update import KAPPA, K_COUPLING
from vn_mobile.core.neuron import build_population
from vn_mobile.io.decoder import describe
from vn_mobile.io.encoder import encode


def build(n, degree, seed, sigma, kappa, k_coupling) -> Field:
    pop = build_population(n=n, degree=degree, seed=seed, weight_sigma=sigma)
    return Field(pop, seed=seed, kappa=kappa, k_coupling=k_coupling)


def memory_report(field: Field) -> dict:
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        rss = float("nan")
    return {"arrays_mb": field.pop.nbytes() / (1024.0 * 1024.0), "rss_mb": rss}


def main() -> None:
    ap = argparse.ArgumentParser(description="VN-Mobile stage 1 driver")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--k", type=float, default=K_COUPLING, dest="k_coupling")
    ap.add_argument("--init-state", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--log-every", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=1_000)
    ap.add_argument("--prompt", type=str, default="")
    args = ap.parse_args()

    field = build(args.n, args.degree, args.seed, args.sigma, args.kappa,
                  args.k_coupling)
    if args.init_state:
        field.pop.state[:] = np.random.default_rng(args.seed + 1).normal(
            0.0, args.init_state, size=field.pop.state.shape).astype(np.float32)
    print(f"n={args.n} degree={args.degree} edges={field.pop.n_edges} "
          f"sigma={args.sigma} kappa={args.kappa} K={args.k_coupling}")

    if args.prompt:
        ids, values = encode(args.prompt, args.n)
        field.perturb(ids, values)
        print(f"injected {len(ids)} neurons for {args.prompt!r}")

    t0 = time.perf_counter()
    field.run(args.warmup)
    t1 = time.perf_counter()
    print(f"warmup {args.warmup} steps in {t1 - t0:.3f}s -> "
          f"{(t1 - t0) / args.warmup * 1e6:.1f} us/step")

    def report(st) -> None:
        print(f"step={st.step:>9} e_max={st.energy_max:.4f} "
              f"e_mean={st.energy_mean:.4f} e_std={st.energy_std:.4f} "
              f"total={st.total_energy:.2f} top_frac={st.top_frac:.2e} "
              f"h_max={st.state_max:.4f} neg={st.neg_frac:.3f} "
              f"R={st.sync_r:.4f} rescales={st.rescales}")

    t0 = time.perf_counter()
    st = field.run(args.steps, log_every=args.log_every, reporter=report)
    elapsed = time.perf_counter() - t0

    mem = memory_report(field)
    print(json.dumps({
        "steps": st.step,
        "elapsed_s": round(elapsed, 2),
        "us_per_step": round(elapsed / max(st.step, 1) * 1e6, 1),
        "energy_max": st.energy_max,
        "energy_mean": st.energy_mean,
        "energy_std": st.energy_std,
        "top_frac": st.top_frac,
        "nan": st.nan,
        "rescales": st.rescales,
        "energy_neg_frac": st.neg_frac,
        "sync_R": st.sync_r,
        "arrays_mb": round(mem["arrays_mb"], 3),
        "rss_mb": round(mem["rss_mb"], 2),
    }, indent=2))
    print("readout:", describe(field.pop))


if __name__ == "__main__":
    main()
