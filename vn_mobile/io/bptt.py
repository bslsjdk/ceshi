"""Truncated BPTT over the state recurrence (stage 6A correction).

Why this exists: with the gradient cut at the current tick the field can only
be trained to react, never to remember.  The measurement that showed it is
val bits getting *worse* as more history is injected -- 4.382 at hist 1 to
10.136 at hist 8 -- because nothing in the objective rewards integrating.

Which paths actually need unrolling is a measurement, not a guess.  The
energy and phase maps carry no trainable parameter (w_ij is frozen, the
injection is not learned), and x_i = I_i + mean_j h_j contains no e, so the
state recurrence is the only place a parameter sits:

    x_t      = I_t + M h_t
    z_t      = F x_t + b
    h_{t+1}  = alpha h_t + (1-alpha) tanh(z_t)

Carrying G_{m+1} = dL/dh_{m+1} backwards gives

    dz_m = (1-alpha) * G_{m+1} * (1 - tanh(z_m)^2)
    G_m  = dL_m/dh_m + alpha * G_{m+1} + M^T (F^T dz_m)

and dz_m is also the accumulator for the bias and adapter gradients.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.update import ALPHA, _fused, neighbour_mean
from .lowrank import LowRankHead, softmax
from .vocab import pool_by_block


def neighbour_mean_t(pop, values: np.ndarray) -> np.ndarray:
    """M^T v, exact transpose of ``neighbour_mean`` -> float32 [N, DIM].

    (M v)_i = (1/deg_i) sum over edges with src=i of v_dst, so
    (M^T v)_j = sum over edges with dst=j of v_src / deg_src.
    """
    w = (values / pop.deg[:, None]).astype(np.float32)[pop.src]   # [E, DIM]
    out = np.empty(values.shape, dtype=np.float32)
    for c in range(values.shape[1]):
        out[:, c] = np.bincount(pop.dst, weights=w[:, c], minlength=pop.n)
    return out


def step_state_x(pop, injection: np.ndarray, alpha: float = ALPHA) -> tuple[
        np.ndarray, np.ndarray]:
    """One state tick that also returns (x_t, tanh(z_t)) for the backward pass."""
    x = injection + neighbour_mean(pop, pop.state)
    linear = np.matmul(_fused(pop), x[:, :, None])[:, :, 0] + pop.bias
    th = np.tanh(linear).astype(np.float32)
    pop.state *= alpha
    pop.state += (1.0 - alpha) * th
    np.putmask(pop.state, np.abs(pop.state) < 1e-20, np.float32(0.0))
    return x, th


@dataclass
class Window:
    """Everything the backward pass needs, one entry per unrolled tick."""

    xs: list = field(default_factory=list)        # [N, DIM] input at tick
    ths: list = field(default_factory=list)       # [N, DIM] tanh(z)
    flats: list = field(default_factory=list)     # [V*DIM] pooled activity
    probs: list = field(default_factory=list)     # [V] next-char distribution
    energies: list = field(default_factory=list)  # [N] e at output time
    targets: list = field(default_factory=list)   # int next char
    losses: list = field(default_factory=list)    # float CE per tick


def head_backward(head, dlogits: np.ndarray, flat: np.ndarray):
    """Gradient w.r.t. readout weights plus dL/dflat, for either head type."""
    if isinstance(head, LowRankHead):
        return head.grad(dlogits, flat)
    dflat = (dlogits @ head.w).astype(np.float32)
    dw = np.outer(dlogits, flat).astype(np.float32)
    return dw, dflat


def head_apply(head, g, lr: float) -> None:
    if isinstance(head, LowRankHead):
        head.apply(g[0], g[1], lr)
        return
    w = g[0] if isinstance(g, tuple) else g
    head.w -= (lr * w).astype(np.float32)


def output_grad(pop, block: np.ndarray, vocab: int, dim: int,
                dflat: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """dL/dh at the tick that produced ``dflat`` -> float32 [N, DIM]."""
    dpooled = dflat.reshape(vocab, dim)
    dy = dpooled[block]
    return (dy * pop.scale[:, None] * energy[:, None]).astype(np.float32)


def backward_window(pop, head, win: Window, block: np.ndarray, vocab: int,
                    dim: int, alpha: float = ALPHA, clip: float = 1.0) -> dict:
    """Walk the window backwards accumulating parameter gradients."""
    n, d = pop.state.shape
    g_h = np.zeros((n, d), dtype=np.float32)      # G_{m+1}
    db = np.zeros(d, dtype=np.float32)
    gb = np.zeros_like(pop.adapter_b)
    ga = np.zeros_like(pop.adapter_a)
    dw = None
    ft = np.transpose(_fused(pop), (0, 2, 1))     # F^T

    for m in range(len(win.targets) - 1, -1, -1):
        probs = win.probs[m]
        dlogits = probs.copy()
        dlogits[win.targets[m]] -= 1.0
        g = head_backward(head, dlogits, win.flats[m])
        payload = g[:-1]        # drop dflat, keep only weight gradients
        dw = payload if dw is None else _acc(dw, payload)
        g_h = g_h + output_grad(pop, block, vocab, dim, g[-1],
                                win.energies[m])
        dz = ((1.0 - alpha) * g_h * (1.0 - win.ths[m] ** 2)).astype(np.float32)
        nrm = float(np.sqrt(float((dz * dz).sum())))
        if np.isfinite(nrm) and nrm > clip:
            dz = (dz * (clip / nrm)).astype(np.float32)
        db += dz.sum(axis=0)
        df = dz[:, :, None] * win.xs[m][:, None, :]
        gb += np.matmul(df, np.transpose(pop.adapter_a, (0, 2, 1)))
        ga += np.matmul(np.transpose(pop.adapter_b, (0, 2, 1)), df)
        dx = np.matmul(ft, dz[:, :, None])[:, :, 0]
        g_h = (alpha * g_h + neighbour_mean_t(pop, dx)).astype(np.float32)
    # The objective is the mean loss over the window, not the sum, so the
    # accumulated gradients are divided by T.  Summing instead makes the step
    # grow with T: at T=20 and lr=3e-3 the bias overflowed to NaN by 4k steps.
    n = max(1, len(win.targets))
    if isinstance(dw, tuple):
        dw = tuple(x / n for x in dw)
    else:
        dw = dw / n
    return dict(dw=dw, db=db / n, gb=gb / n, ga=ga / n)


def finite(g: dict) -> bool:
    """False if any accumulated gradient went non-finite -> caller skips step."""
    for v in (g["db"], g["gb"], g["ga"]):
        if not np.all(np.isfinite(v)):
            return False
    for v in (g["dw"] if isinstance(g["dw"], tuple) else (g["dw"],)):
        if not np.all(np.isfinite(v)):
            return False
    return True


def _acc(a, b):
    if isinstance(a, tuple):
        return tuple(np.add(x, y) for x, y in zip(a, b))
    return np.add(a, b)


def pooled_flat(pop, block: np.ndarray, vocab: int) -> np.ndarray:
    from ..core.update import read_output
    return pool_by_block(read_output(pop), block, vocab).ravel()
