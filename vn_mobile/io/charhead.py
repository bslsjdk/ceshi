"""Logit readout and the adapters-only training step (stage 6A).

The main field is the memory: injecting a character raises energy in that
character's address block, one tick mixes it with what the field already
holds, and each block's output pools into one logit.  So the next-character
distribution is read straight out of where the wave has accumulated.

Gradients are truncated to the current tick.  The field is a running state,
not an unrolled graph, so backpropagation through time is not available; what
is trained is how the adapters shape this tick's response.  That is a
deliberate approximation and it is the reason the loss floor is what it is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.update import _fused
from .vocab import pool_by_block


@dataclass
class CharHead:
    """Readout from pooled block activity to vocab logits.

    Two widths are supported and chosen by shape, because they are not the
    same model.  ``[V, DIM]`` gives logit v only block v's own activity, so
    a character's score cannot see what the rest of the field is doing --
    measured at 5.99 val bits, i.e. no better than uniform.  ``[V, V*DIM]``
    lets every logit see the whole pattern, which is what "read the next
    character out of the stable mode of the field" actually requires.
    """

    w: np.ndarray          # float32 [V, V*DIM] (full) or [V, DIM] (blockwise)

    @classmethod
    def make(cls, vocab: int, dim: int, seed: int = 0, scale: float = 0.01,
             full: bool = True) -> "CharHead":
        rng = np.random.default_rng(seed)
        cols = vocab * dim if full else dim
        return cls(w=(rng.normal(0.0, scale, size=(vocab, cols))
                      ).astype(np.float32))

    @property
    def full(self) -> bool:
        return self.w.ndim == 2 and self.w.shape[1] % self.w.shape[0] == 0

    def logits(self, pooled: np.ndarray) -> np.ndarray:
        """float32 [V, DIM] -> float32 [V]."""
        flat = pooled.ravel()
        if self.w.shape[1] == flat.size:
            return (self.w @ flat).astype(np.float32)
        return (pooled * self.w).sum(axis=1).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


def cross_entropy(probs: np.ndarray, target: int, eps: float = 1e-9) -> float:
    return float(-np.log(max(float(probs[target]), eps)))


def forward_char(pop, block: np.ndarray, head: CharHead) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pooled activity -> (pooled, logits, probs).  Keeps x and t for grads."""
    from ..core.update import read_output
    out = read_output(pop)                       # [N, DIM]
    pooled = pool_by_block(out, block, head.w.shape[0])
    logits = head.logits(pooled)
    return pooled, logits, softmax(logits)


def train_char(pop, block: np.ndarray, head: CharHead, target: int,
               lr: float, alpha: float, x: np.ndarray,
               train_head: bool = True, train_bias: bool = False,
               train_adapters: bool = True,
               clip: float = 1.0) -> tuple[float, bool]:
    """One step on ``target``.  Returns (loss, predicted_ok).

    ``x`` is the same input the last forward pass used (injection plus
    neighbour mean), so the gradient sees the true pre-activation.

    Which parameters move is a first-class choice, not a constant: measured
    sensitivity to +0.01 on every entry is head_w 3.4e-4, bias 3e-5..1.5e-4,
    adapter_a/b 5e-7..1.6e-5, scale 1e-6..9e-6.  A low-rank adapter on a
    frozen W is a ~1e-5 perturbation, so "adapters only" is structurally
    unable to produce a usable gradient signal on its own.
    """
    from ..core.update import read_output
    out = read_output(pop)
    pooled = pool_by_block(out, block, head.w.shape[0])
    logits = head.logits(pooled)
    probs = softmax(logits)
    loss = cross_entropy(probs, target)
    ok = bool(int(np.argmax(logits)) == target)

    dlogits = probs.copy()
    dlogits[target] -= 1.0                       # dL/dlogit  [V]
    flat = pooled.ravel()
    if head.w.shape[1] == flat.size:
        # full-field readout: every logit sees every block
        dpooled = (dlogits @ head.w).reshape(pooled.shape)
        if train_head:
            head.w -= (lr * dlogits[:, None] * flat[None, :]).astype(np.float32)
    else:
        dpooled = dlogits[:, None] * head.w       # [V, DIM]
        if train_head:
            head.w -= (lr * dpooled * pooled).astype(np.float32)

    dout = dpooled[block]                        # [N, DIM]
    dstate = (dout * pop.scale[:, None]
              * pop.energy[:, None]).astype(np.float32)
    t = np.tanh(np.matmul(_fused(pop), x[:, :, None])[:, :, 0] + pop.bias)
    dz = ((1.0 - alpha) * dstate * (1.0 - t * t)).astype(np.float32)
    # Unclipped, lr=1e-1 overflows to NaN within 10k steps.  One global
    # factor keeps the direction and bounds the step.
    nrm = float(np.sqrt(float((dz * dz).sum())))
    if not np.isfinite(nrm) or nrm <= 0.0:
        return loss, ok
    if nrm > clip:
        dz = (dz * (clip / nrm)).astype(np.float32)

    if train_bias:
        # bias is shared [DIM]; the chain rule sums the per-VN signal.
        pop.bias -= (lr * dz.sum(axis=0)).astype(np.float32)
    if train_adapters:
        dfused = dz[:, :, None] * x[:, None, :]      # [N, DIM, DIM]
        grad_b = np.matmul(dfused, np.transpose(pop.adapter_a, (0, 2, 1)))
        grad_a = np.matmul(np.transpose(pop.adapter_b, (0, 2, 1)), dfused)
        gb = np.clip(grad_b, -clip, clip)
        ga = np.clip(grad_a, -clip, clip)
        pop.adapter_b -= (lr * gb).astype(np.float32)
        pop.adapter_a -= (lr * ga).astype(np.float32)
        pop.WM = None                                # fused cache is stale
    return loss, ok
