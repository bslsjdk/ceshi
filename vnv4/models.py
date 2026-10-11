"""VN v4 and a selective-SSM control, both hand-written numpy.

VN v4 is the fast-weights design:

    EVN state   S_t = a_t * S_{t-1} + (b_t * w_t) (x) x_t
    FVN branch  a_f = softmax(P_f x_t / sqrt(d)) ; logits_f = a_f W_f
    readout     logits = logits_f + a_e W_e,  a_e = <S_t, x_t> / sqrt(d)

a_t is an input-dependent forget gate, b_t an input-dependent write gate and
w_t a signed write direction -- all three come from the input, which is what
makes the state a *fast weight* rather than a read-only cache.  The FVN
branch contains no recurrence at all; it is pure associative knowledge and
is the part that gets frozen between the two phases.

The SSM control is the same recurrence with the FVN branch removed (and a
larger node count so the parameter totals stay close).  Comparing the two
isolates the frozen anchor from the recurrence itself.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class FastWeight:
    """Base: input-gated recurrent state + associative readout."""

    kind = "fastweight"

    def __init__(self, V, d=48, dh=48, Ne=64, Nf=0, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.08
        self.V, self.d, self.dh, self.Ne, self.Nf = V, d, dh, Ne, Nf
        g = lambda *sh: (rng.standard_normal(sh) * s).astype(np.float32)
        self.P = {"emb": g(V, d)}
        # input projection shared by every gate
        self.P["w1"] = g(d, dh)
        self.P["w2"] = g(dh, 3 * Ne)
        self.P["woe"] = g(Ne, V)
        if Nf > 0:
            self.P["pf"] = g(Nf, d)
            self.P["wof"] = g(Nf, V)
        self.frozen: set = set()
        self.r = float(np.sqrt(d))

    # ---------------- params ----------------
    def count_params(self):
        return sum(int(v.size) for v in self.P.values())

    def trainable(self):
        return [k for k in self.P if k not in self.frozen]

    def zeros_grad(self):
        return {k: np.zeros_like(v) for k, v in self.P.items()}

    def freeze(self, keys):
        self.frozen = set(keys)

    # ---------------- forward ----------------
    def _encode(self, x):
        emb = self.P["emb"][x]                       # (T,d)
        h = np.tanh(emb @ self.P["w1"])              # (T,dh)
        g = h @ self.P["w2"]                         # (T,3Ne)
        Ne, T = self.Ne, len(x)
        dt = emb.dtype
        a = _sigmoid(g[:, :Ne]).astype(dt)
        b = _sigmoid(g[:, Ne:2 * Ne]).astype(dt)
        w = np.tanh(g[:, 2 * Ne:]).astype(dt)
        c = (b * w).astype(dt)                       # (T,Ne) signed write
        S = np.zeros((Ne, self.d), dtype=dt)
        Ss = np.empty((T, Ne, self.d), dtype=dt)
        for t in range(T):
            S = a[t][:, None] * S + c[t][:, None] * emb[t][None, :]
            Ss[t] = S
        return emb, h, g, a, b, w, c, Ss

    def forward(self, x):
        emb, h, g, a, b, w, c, Ss = self._encode(x)
        C = dict(emb=emb, h=h, a=a, b=b, w=w, c=c, Ss=Ss)
        ae = np.einsum("tnd,td->tn", Ss, emb) / self.r        # (T,Ne)
        C["ae"] = ae
        logits = ae @ self.P["woe"]
        if self.Nf > 0:
            sf = (emb @ self.P["pf"].T) / self.r              # (T,Nf)
            m = sf.max(axis=1, keepdims=True)
            e = np.exp(sf - m)
            af = (e / e.sum(axis=1, keepdims=True)).astype(emb.dtype)
            C["af"] = af
            logits = logits + af @ self.P["wof"]
        return logits, C

    def eval_nll(self, x, y):
        logits, _ = self.forward(x)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        return float((-(logits[np.arange(len(y)), y] - m.ravel()
                        - np.log(e.sum(axis=1)))).mean())

    # ---------------- backward ----------------
    def backward(self, G, logits, C, x, y):
        T, d, Ne = len(x), self.d, self.Ne
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        nll = (-(logits[np.arange(T), y] - m.ravel()
                 - np.log(e.sum(axis=1)))).mean()
        dz = e / e.sum(axis=1, keepdims=True)
        dz[np.arange(T), y] -= 1.0
        dz = (dz / T).astype(np.float32)

        emb, h, a, b, w, c, Ss = (C["emb"], C["h"], C["a"], C["b"],
                                  C["w"], C["c"], C["Ss"])
        demb = np.zeros_like(emb)
        G["woe"] += C["ae"].T @ dz
        dae = (dz @ self.P["woe"].T) / self.r                 # (T,Ne)

        if self.Nf > 0:
            G["wof"] += C["af"].T @ dz
            daf = dz @ self.P["wof"].T
            saf = C["af"]
            dsf = saf * (daf - (daf * saf).sum(axis=1, keepdims=True))
            dsf = (dsf / self.r).astype(np.float32)
            G["pf"] += dsf.T @ emb
            demb += dsf @ self.P["pf"]

        dSs = dae[:, :, None] * emb[:, None, :]               # (T,Ne,d)
        # dae already carries the 1/r from a_e = <S, x> / r
        demb += np.einsum("tn,tnd->td", dae, Ss)

        dg = np.zeros_like(C["h"] @ self.P["w2"])
        dS = np.zeros((Ne, d), dtype=np.float32)
        for t in range(T - 1, -1, -1):
            dSt = dSs[t] + dS
            prev = Ss[t - 1] if t > 0 else 0.0
            dg[t, :Ne] += (dSt * prev).sum(axis=1) * a[t] * (1.0 - a[t])
            dc = (dSt * emb[t]).sum(axis=1)
            dg[t, Ne:2 * Ne] += dc * w[t] * b[t] * (1.0 - b[t])
            dg[t, 2 * Ne:] += dc * b[t] * (1.0 - w[t] ** 2)
            demb[t] += (c[t][:, None] * dSt).sum(axis=0)
            dS = a[t][:, None] * dSt

        G["w2"] += h.T @ dg
        dh = dg @ self.P["w2"].T * (1.0 - h ** 2)
        G["w1"] += emb.T @ dh
        demb += dh @ self.P["w1"].T
        np.add.at(G["emb"], x, demb)
        return float(nll)


class VNv4(FastWeight):
    """Fast weights + frozen FVN anchor (Nf > 0)."""

    kind = "vn4"

    def __init__(self, V, d=48, dh=48, Ne=64, Nf=64, seed=0):
        super().__init__(V, d=d, dh=dh, Ne=Ne, Nf=Nf, seed=seed)


class SSM(FastWeight):
    """Same recurrence, no FVN branch -> selective SSM / Mamba-like."""

    kind = "ssm"

    def __init__(self, V, d=48, dh=48, Ne=96, seed=0):
        super().__init__(V, d=d, dh=dh, Ne=Ne, Nf=0, seed=seed)
