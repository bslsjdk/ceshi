"""Storage-width experiments for the edge array (task 1).

The population carries five [E] arrays; the hot loop reads two of them,
and one of those two is derivable from the CSR pointer.  This module only
answers "does narrowing the stored width change the model's answer", so it
writes the narrowed value back as float32: the point is the rounding, not
a dtype that scipy cannot multiply with.
"""
from __future__ import annotations
import numpy as np


def narrow(w: np.ndarray, bits: int) -> np.ndarray:
    """Round ``w`` to ``bits`` of storage and return it as float32."""
    w = np.asarray(w, dtype=np.float32)
    if bits >= 32:
        return w
    if bits == 16:
        return w.astype(np.float16).astype(np.float32)
    scale = float(np.abs(w).max()) or 1.0
    q = np.rint(w / scale * 127.0).astype(np.int8)
    return (q.astype(np.float32) / 127.0 * scale).astype(np.float32)


def error(w: np.ndarray, bits: int) -> dict:
    """Relative RMSE of the narrowing, for the ledger."""
    v = narrow(w, bits)
    d = np.abs(v - w)
    den = float(np.sqrt((w ** 2).mean())) or 1e-9
    return dict(bits=bits, max_abs=round(float(d.max()), 6),
                rel_rmse=round(float(np.sqrt((d ** 2).mean()) / den), 6))
