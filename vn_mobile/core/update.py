"""One-tick update equations for the virtual neuron field.

Energy:
    g_i   = sum_j w_ij * e_j * cos(phi_j - phi_i) * d_ij
    e_i'  = (1-lam) * e_i + kappa * tanh(g_i / sqrt(deg_i)) + eta

State:
    x_i   = I_i + (1/deg_i) * sum_j h_j
    h_i'  = alpha * h_i + (1-alpha) * tanh(W x_i + b + B A x_i)
    y_i   = s_i * h_i * e_i
"""
from __future__ import annotations

import numpy as np

from .neuron import VNPopulation

LAMBDA = 0.05
KAPPA = 0.3
ETA_STD = 0.001
ALPHA = 0.9
ENERGY_CAP = 0.3      # target mean energy used by the global constraint


def aggregate(pop: VNPopulation, values: np.ndarray) -> np.ndarray:
    """Sum ``values`` over each neuron's neighbours -> float32 [N, ...].

    ``src`` is stored CSR-sorted, so a segmented reduce beats a scatter-add;
    it falls back to bincount when some neuron has no neighbours (where
    reduceat would silently return a wrong element).
    """
    flat = values[pop.dst]
    if flat.ndim == 1:
        out = np.bincount(pop.src, weights=flat, minlength=pop.n)
        return out.astype(np.float32)
    if pop.deg.min() > 0:
        starts = pop.ptr[:-1]
        out = np.stack([np.add.reduceat(flat[:, c], starts)
                        for c in range(flat.shape[1])], axis=1)
        return out.astype(np.float32)
    out = np.empty((pop.n, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.bincount(pop.src, weights=flat[:, c], minlength=pop.n)
    return out


def neighbour_mean(pop: VNPopulation, values: np.ndarray) -> np.ndarray:
    agg = aggregate(pop, values)
    if agg.ndim == 1:
        return agg / pop.deg
    return agg / pop.deg[:, None]


def step_energy(pop: VNPopulation, rng: np.random.Generator,
                lam: float = LAMBDA, kappa: float = KAPPA,
                eta_std: float = ETA_STD) -> None:
    """Advance the energy field one tick (in place)."""
    g = np.bincount(pop.src, weights=pop.kedge * pop.energy[pop.dst],
                    minlength=pop.n).astype(np.float32)
    drive = kappa * np.tanh(g / np.sqrt(pop.deg))
    pop.energy *= (1.0 - lam)
    pop.energy += drive
    if eta_std > 0.0:
        pop.energy += rng.normal(0.0, eta_std, size=pop.n).astype(np.float32)


def step_state(pop: VNPopulation, injection: np.ndarray,
               alpha: float = ALPHA) -> None:
    """Advance the state field one tick (in place)."""
    x = injection + neighbour_mean(pop, pop.state)
    linear = x @ pop.W.T + pop.bias
    # B @ (A @ x) as two batched matmuls: einsum on three operands is ~19x
    # slower here because it cannot hit a BLAS gemm path.
    mid = np.matmul(pop.adapter_a, x[:, :, None])[:, :, 0]
    adapted = np.matmul(pop.adapter_b, mid[:, :, None])[:, :, 0]
    pop.state *= alpha
    pop.state += (1.0 - alpha) * np.tanh(linear + adapted).astype(np.float32)
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
