"""One-tick update equations for the virtual neuron field.

Energy (stage 2: added the lambda*e_target restoring term):
    g_i   = sum_j w_ij * e_j * cos(phi_j - phi_i) * d_ij
    e_i'  = (1-lam)*e_i + lam*e_target + kappa*tanh(g_i/sqrt(deg_i)) + eta

Phase (stage 2, Kuramoto):
    phi_i'= phi_i + omega_i + (K/deg_i) * sum_j sin(phi_j - phi_i)

State:
    x_i   = I_i + (1/deg_i) * sum_j h_j
    h_i'  = alpha * h_i + (1-alpha) * tanh(W x_i + b + B A x_i)
    y_i   = s_i * h_i * e_i

Without the restoring term the energy map has a transcritical bifurcation at
sigma = lam/kappa: below it the only fixed point is e*=0 (dead), above it the
map runs away (explodes).  lam*e_target removes the bifurcation and puts the
fixed point at e* ~ e_target for the whole parameter space.
"""
from __future__ import annotations

import numpy as np

from .neuron import TWO_PI, VNPopulation

LAMBDA = 0.1
KAPPA = 0.25          # stage 3: 5x stage-2 value, widens the fluctuation band
E_TARGET = 0.3
ETA_STD = 0.001
ALPHA = 0.9
K_COUPLING = 0.5
GAMMA_I = 0.5         # stage 3: input reaches the wave field directly
# Safety net only.  With the restoring term the mean sits at e_target, so a
# cap at 0.3 would fire constantly and mask what we are trying to measure;
# 1.0 lets the field run freely and only catches genuine runaway.
ENERGY_CAP = 1.0


def _adjacency(pop: VNPopulation):
    """Cached CSR adjacency: unit-weighted and wedge-weighted copies.

    A sparse matvec beats the old per-edge path by 3-4x because the old one
    materialised an [E] (or [E, DIM]) temporary every single tick before
    reducing it; the matvec reads the values where they already live.
    """
    if pop.A1 is None:
        try:
            import scipy.sparse as sp
        except ImportError:                      # pragma: no cover
            return None, None
        n = pop.n
        pop.A1 = sp.csr_matrix(
            (np.ones(pop.src.size, dtype=np.float32), (pop.src, pop.dst)),
            shape=(n, n))
        pop.Aw = sp.csr_matrix(
            (pop.wedge.astype(np.float32), (pop.src, pop.dst)), shape=(n, n))
    return pop.A1, pop.Aw


def _fused(pop: VNPopulation) -> np.ndarray:
    """W + B_i @ A_i as one [N, DIM, DIM] stack.

    The state update used two batched matmuls plus a dense Wx per tick;
    because W and the adapters are frozen between training steps they can be
    collapsed into a single per-neuron 8x8 matrix first -- 48 us -> 19 us.
    Callers must drop the cache (``pop.WM = None``) after training.
    """
    if pop.WM is None:
        pop.WM = np.matmul(pop.adapter_b, pop.adapter_a) + pop.W[None]
    return pop.WM


def _aggregate_dense(pop: VNPopulation, values: np.ndarray) -> np.ndarray:
    """Numpy-only fallback, used when scipy is unavailable."""
    flat = values[pop.dst]
    if flat.ndim == 1:
        return np.bincount(pop.src, weights=flat,
                           minlength=pop.n).astype(np.float32)
    if pop.deg.min() > 0:
        starts = pop.ptr[:-1]
        return np.stack([np.add.reduceat(flat[:, c], starts)
                         for c in range(flat.shape[1])],
                        axis=1).astype(np.float32)
    out = np.empty((pop.n, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.bincount(pop.src, weights=flat[:, c], minlength=pop.n)
    return out


def aggregate(pop: VNPopulation, values: np.ndarray) -> np.ndarray:
    """Sum ``values`` over each neuron's neighbours -> float32 [N, ...]."""
    A1, _ = _adjacency(pop)
    if A1 is None:                               # pragma: no cover
        return _aggregate_dense(pop, values)
    return A1 @ values


def neighbour_mean(pop: VNPopulation, values: np.ndarray) -> np.ndarray:
    agg = aggregate(pop, values)
    if agg.ndim == 1:
        return agg / pop.deg
    return agg / pop.deg[:, None]


def step_energy(pop: VNPopulation, rng: np.random.Generator,
                lam: float = LAMBDA, kappa: float = KAPPA,
                eta_std: float = ETA_STD,
                e_target: float = E_TARGET,
                injection: np.ndarray | None = None,
                gamma_i: float = GAMMA_I) -> None:
    """Advance the energy field one tick (in place).

    cos(phi_j - phi_i) is recomputed every tick because the phase now moves.
    """
    c = np.cos(pop.phase)
    s = np.sin(pop.phase)
    _, Aw = _adjacency(pop)
    if Aw is None:                               # pragma: no cover
        dphi = pop.phase[pop.dst] - pop.phase[pop.src]
        g = np.bincount(pop.src, weights=pop.wedge * np.cos(dphi)
                        * pop.energy[pop.dst],
                        minlength=pop.n).astype(np.float32)
    else:
        # cos(a-b) = cos a cos b + sin a sin b, so the two neighbour sums are
        # plain sparse matvecs on [N].  That replaces a trig pass over all E
        # edges with one over N nodes -- 94 us -> 22 us at N=1000.
        g = (c * (Aw @ (c * pop.energy))
             + s * (Aw @ (s * pop.energy))).astype(np.float32)
    drive = kappa * np.tanh(g / np.sqrt(pop.deg))
    pop.energy *= (1.0 - lam)
    pop.energy += lam * e_target + drive
    if injection is not None:
        pop.energy += (gamma_i * injection).astype(np.float32)
    if eta_std > 0.0:
        pop.energy += rng.normal(0.0, eta_std, size=pop.n).astype(np.float32)


def step_phase(pop: VNPopulation, k: float = K_COUPLING) -> None:
    """Kuramoto phase update, wrapped into [0, 2*pi)."""
    c = np.cos(pop.phase)
    s = np.sin(pop.phase)
    A1, _ = _adjacency(pop)
    if A1 is None:                               # pragma: no cover
        dphi = pop.phase[pop.dst] - pop.phase[pop.src]
        couple = np.bincount(pop.src, weights=np.sin(dphi),
                             minlength=pop.n).astype(np.float32)
    else:
        # sin(a-b) = sin a cos b - cos a sin b; two matvecs, no [E] temporaries.
        couple = (c * (A1 @ s) - s * (A1 @ c)).astype(np.float32)
    pop.phase += pop.omega + (k / pop.deg) * couple
    np.mod(pop.phase, TWO_PI, out=pop.phase)


def sync_order_parameter(pop: VNPopulation) -> float:
    """Kuramoto order parameter R = |mean(exp(i*phi))| in [0, 1].

    R~0 means phases are spread out (incoherent); R~1 means they locked.
    """
    c = float(np.cos(pop.phase).mean())
    s = float(np.sin(pop.phase).mean())
    return float(np.hypot(c, s))


def step_state(pop: VNPopulation, injection: np.ndarray,
               alpha: float = ALPHA) -> None:
    """Advance the state field one tick (in place)."""
    x = injection + neighbour_mean(pop, pop.state)
    # One batched matmul against the fused matrix instead of Wx + B(Ax).
    linear = np.matmul(_fused(pop), x[:, :, None])[:, :, 0] + pop.bias
    pop.state *= alpha
    pop.state += (1.0 - alpha) * np.tanh(linear).astype(np.float32)
    # With no input the state layer decays geometrically towards zero; once it
    # reaches denormals NumPy arithmetic gets ~5x slower for no numerical gain.
    np.putmask(pop.state, np.abs(pop.state) < 1e-20, np.float32(0.0))


def read_output(pop: VNPopulation) -> np.ndarray:
    """y_i = s_i * h_i * e_i -> float32 [N, DIM]."""
    return pop.scale[:, None] * pop.state * pop.energy[:, None]


def normalise_energy(pop: VNPopulation, mean_cap: float = ENERGY_CAP) -> bool:
    """Global conservation constraint, applied at most every ``every`` ticks.

    Returns True when a rescale actually happened.  This is not a clip: the
    shape of the field is preserved, only the total is scaled back.
    """
    total = float(pop.energy.sum())
    limit = mean_cap * pop.n
    if total > limit:
        pop.energy *= np.float32(limit / total)
        return True
    return False
