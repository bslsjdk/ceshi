"""The persistent concept wave field (<=200 lines).

State carried across ticks: ``psi`` (complex amplitude) and ``mode``.  Nothing
else.  A call to :meth:`settle` runs

    propagate -> interfere -> decay

until the pattern stops moving, which is what the spec calls the stable mode.
Dormancy is re-evaluated every round from the *context concept* so the kernel
itself is input dependent -- without that the iteration would converge to the
same eigenvector for every input and carry no information at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kernels import build_edges, coefficients, cosine, matvec, normalise

ACTIVE, IDLE, DORMANT = 2, 1, 0


@dataclass
class ConceptField:
    concept: np.ndarray          # [N, Dc] float32 (storage float16, see encode)
    layer: np.ndarray            # [N] uint8
    psi: np.ndarray              # [N] complex64
    mode: np.ndarray             # [N] uint8
    edges: dict = field(default_factory=dict)
    w: np.ndarray | None = None  # [L] destination layer weight
    g: np.ndarray | None = None  # [L] source layer gain
    self_c: float = 0.5
    decay: float = 0.9
    nonlinear: bool = True       # tanh amplitude squash (breaks eigen-convergence)
    psi_max: float = 10.0        # per-node cap; 0 disables (no global renorm)
    norm_global: bool = False    # v1 behaviour: renorm whole vector each round
    dormancy: float = 0.0
    omega: np.ndarray | None = None   # [N] per-round phase advance
    ctx: np.ndarray | None = None     # current context concept
    rounds_used: int = 0

    @property
    def n(self) -> int:
        return self.concept.shape[0]

    @classmethod
    def create(cls, n, dc, nlayers, k, seed, dormancy=0.0, sigma_w=0.3,
               signed=True, nonlinear=True, psi_max=10.0, norm_global=False):
        rng = np.random.default_rng(seed)
        per = n // nlayers
        concept = normalise(rng.normal(size=(n, dc)).astype(np.float32))
        layer = np.repeat(np.arange(nlayers), per).astype(np.uint8)
        if layer.shape[0] < n:
            layer = np.concatenate(
                [layer, np.full(n - layer.shape[0], nlayers - 1, np.uint8)])
        edges = build_edges(concept, layer, per, k, nlayers, signed=signed)
        omega = (rng.normal(scale=sigma_w, size=n)).astype(np.float32)
        obj = cls(concept=concept, layer=layer,
                   psi=np.zeros(n, np.complex64),
                   mode=np.full(n, IDLE, np.uint8), edges=edges,
                   w=np.ones(nlayers, np.float32),
                   g=np.ones(nlayers, np.float32),
                   dormancy=dormancy, omega=omega, nonlinear=nonlinear,
                   psi_max=psi_max, norm_global=norm_global,
                   ctx=normalise(concept.mean(0, keepdims=True))[0])
        obj.cn = normalise(concept).astype(np.float32)
        obj.prepare()
        return obj

    def set_context(self, vec):
        self.ctx = normalise(np.asarray(vec, np.float32).reshape(1, -1))[0]

    def awake_mask(self):
        """mode from concept match -- the sparsity source."""
        sim = cosine(self.cn, normalise(self.ctx.reshape(1, -1))).ravel()
        mask = sim > self.dormancy
        self.mode[:] = np.where(mask, ACTIVE, DORMANT)
        return mask

    def inject(self, amp, phase=0.0):
        """In-phase (phase=0) or anti-phase (phase=pi) seeding."""
        amp = np.asarray(amp, np.float32)
        self.psi += (amp * np.exp(1j * phase)).astype(np.complex64)

    def _unit(self):
        mag = np.abs(self.psi)
        return self.psi / np.where(mag > 0, mag, 1.0)

    def reinforce(self, amp):
        """Feedback "correct": inject along each node's own phase."""
        amp = np.asarray(amp, np.float32)
        self.psi += (amp * self._unit()).astype(np.complex64)

    def counter(self, amp):
        """Feedback "wrong": inject against each node's own phase.

        Plain ``inject(..., phase=pi)`` does *not* work as feedback: the
        readout energy is ``|psi|^2``, which is blind to a global sign flip,
        so flipping every phase by pi leaves the energy untouched.  What
        damps a mode is cancelling it along its own local phase direction.
        """
        amp = np.asarray(amp, np.float32)
        mag = np.abs(self.psi)
        self.psi -= (np.minimum(amp, mag) * self._unit()).astype(np.complex64)

    def prepare(self):
        """Cache the pieces that never change while w/g/omega are fixed."""
        self._coeff = coefficients(self.g, self.w, self.edges)
        self._idx = np.repeat(np.arange(self.n),
                              np.diff(self.edges["ptr"]))
        self._rot = np.exp(1j * self.omega).astype(np.complex64)

    def _round(self, mask):
        src = self.psi * mask
        out = matvec(src, self.edges["ptr"], self.edges["dst"], self._coeff,
                     self.n, self.self_c, self._idx)
        out *= mask
        out *= self._rot                                      # phase advance
        out *= self.decay
        if self.nonlinear:
            mag = np.abs(out)
            safe = np.where(mag > 1e-12, mag, 1.0)
            out = out * (np.tanh(mag) / safe)      # phase kept, amplitude sqashed
        if self.psi_max > 0:                       # per-node cap, not global
            mag = np.abs(out)
            over = mag > self.psi_max
            if over.any():
                out = out.copy()
                out[over] *= (self.psi_max / mag[over]).astype(np.complex64)
        if self.norm_global:                       # v1 only: destroys interference
            nrm = np.linalg.norm(out)
            if nrm > 0:
                out /= nrm
        delta = float(np.abs(out - self.psi).sum())
        self.psi = out.astype(np.complex64)
        return delta

    def settle(self, max_rounds=12, tol=1e-3):
        for i in range(max_rounds):
            mask = self.awake_mask()
            if not mask.any():
                self.psi[:] = 0
                self.rounds_used = i + 1
                return
            if self._round(mask) < tol:
                self.rounds_used = i + 1
                return
        self.rounds_used = max_rounds

    def energy(self):
        e = np.abs(self.psi) ** 2
        s = e.sum()
        return e / s if s > 0 else e

    def pooled(self):
        return self.energy() @ self.concept

    def frac_active(self, thresh=1e-3):
        e = self.energy()
        return float((e > thresh).mean())
