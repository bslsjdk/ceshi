"""The compressed network: cluster bases + per-neuron modulation.

    concept_i = B[label_i] + m_i @ P

Only ``m`` (and emb/out/bias) is trainable once packed; ``B`` is either frozen
or trained depending on the variant, and ``P`` is a frozen basis produced by
the packing step.  Gradient bookkeeping is the only thing this class adds:

    dC  = [dkeys | flat(dW)]          [N, D]
    dm  = dC @ P.T                    [N, r]
    dB  = scatter-add of dC by label  [K, D]

so the optimiser never sees the full N*D block again.
"""
from __future__ import annotations

import numpy as np

from vnv6.model import DE, DO, DCONCEPT, ConceptNet


class PackedNet(ConceptNet):
    def __init__(self, base: ConceptNet, lab: np.ndarray, B: np.ndarray,
                 m: np.ndarray, P: np.ndarray):
        n = lab.size
        super().__init__(base.V, n, seed=0)
        self.emb = base.emb
        self.out = base.out
        self.bias = base.bias
        self.norm = base.norm
        self.lab = np.asarray(lab, np.int64)
        self.B = B.astype(np.float32)
        self.m = m.astype(np.float32)
        self.P = P.astype(np.float32)
        self.k, self.r = self.B.shape[0], self.m.shape[1]
        self.rebuild()

    def rebuild(self) -> None:
        c = self.B[self.lab] + self.m @ self.P
        self.keys = np.ascontiguousarray(c[:, :DE], dtype=np.float32)
        self.W = np.ascontiguousarray(
            c[:, DE:].reshape(self.lab.size, DO, DE), dtype=np.float32)

    def block_params(self) -> int:
        """Trainable parameters standing in for the N*D block."""
        return int(self.B.size + self.m.size)

    def frozen_params(self) -> int:
        return int(self.P.size)

    def packed_grads(self, grads, train_B: bool = False):
        dC = np.concatenate([grads['keys'],
                             np.asarray(grads['W']).reshape(self.N, DO * DE)],
                            axis=1).astype(np.float32)
        dm = dC @ self.P.T
        out = {'m': (self.m, dm)}
        if train_B:
            dB = np.zeros_like(self.B)
            np.add.at(dB, self.lab, dC)
            out['B'] = (self.B, dB)
        return out

    def step_pairs(self, grads, train_B: bool = False):
        pairs = {'out': (self.out, grads['out']),
                 'bias': (self.bias, grads['bias'])}
        if grads['emb'] is not None:
            pairs['emb'] = (self.emb, grads['emb'])
        pairs.update(self.packed_grads(grads, train_B=train_B))
        return pairs
