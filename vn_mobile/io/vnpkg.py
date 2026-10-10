""".vnpkg container (stage 5C): custom format, per-section zstd, checksummed.

Deliberately NOT safetensors/GGUF/ONNX - this system carries state the standard
formats have no vocabulary for (phase, timers, region membership, trial scores).
Layout is columnar so a C++ port can mmap sections independently.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
import zstandard as zstd

MAGIC = b"VNPKG1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<6sHIIQ")   # magic, version, n_sections, flags, crc


@dataclass
class Snapshot:
    """Everything needed to put a running field back exactly."""
    energy: np.ndarray
    phase: np.ndarray
    state: np.ndarray
    adapters_a: np.ndarray
    adapters_b: np.ndarray
    region: np.ndarray
    trigger_at: np.ndarray
    meta: dict


def _crc(data: bytes) -> int:
    return int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16)


def _pack(name: str, arr: np.ndarray, level: int = 3) -> bytes:
    raw = np.ascontiguousarray(arr).tobytes()
    comp = zstd.ZstdCompressor(level=level).compress(raw)
    return (struct.Struct("<I").pack(len(name)) + name.encode()
            + struct.Struct("<QQ").pack(len(raw), len(comp)) + comp)


def _unpack(buf: bytes, off: int) -> tuple[str, np.ndarray, int]:
    (nl,) = struct.Struct("<I").unpack_from(buf, off)
    off += 4
    name = buf[off:off + nl].decode()
    off += nl
    raw_len, comp_len = struct.Struct("<QQ").unpack_from(buf, off)
    off += 16
    raw = zstd.ZstdDecompressor().decompress(buf[off:off + comp_len])
    off += comp_len
    return name, np.frombuffer(raw, dtype=np.uint8), off


def save(path: str, snap: Snapshot, level: int = 3,
         lossless: bool = True) -> int:
    """Write a snapshot.  Returns the file size in bytes."""
    body = b""
    if lossless:
        e, p, st = snap.energy, snap.phase, snap.state
    else:
        # Archive mode: 8-bit quantisation.  Random float32 is incompressible
        # (measured 1.13x); u8 gives 5.14x with ~2e-3 error on energy.
        from ..core.neuron import TWO_PI, quantise_u8
        # Adapters are 86% of the payload and are N(0, 0.01): quantising them
        # is what actually gets the archive under 3x.
        e = quantise_u8(snap.energy, 0.0, 1.0)
        p = quantise_u8(snap.phase, 0.0, float(TWO_PI))
        st = quantise_u8(snap.state, -1.0, 1.0)
        aa = quantise_u8(snap.adapters_a, -0.05, 0.05)
        ab = quantise_u8(snap.adapters_b, -0.05, 0.05)
    body += _pack("energy", e, level)
    body += _pack("phase", p, level)
    body += _pack("state", st, level)
    body += _pack("adapter_a", aa if not lossless else snap.adapters_a, level)
    body += _pack("adapter_b", ab if not lossless else snap.adapters_b, level)
    body += _pack("region", snap.region, level)
    body += _pack("trigger_at", snap.trigger_at, level)
    body += _pack("meta", np.frombuffer(repr(snap.meta).encode(),
                                        dtype=np.uint8), level)
    header = HEADER_STRUCT.pack(MAGIC, VERSION, 8, 0, _crc(body))
    with open(path, "wb") as fh:
        fh.write(header + body)
    return len(header) + len(body)


def load(path: str) -> Snapshot:
    """Read a snapshot back and verify its checksum."""
    with open(path, "rb") as fh:
        buf = fh.read()
    magic, version, n_sec, _flags, crc = HEADER_STRUCT.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    body = buf[HEADER_STRUCT.size:]
    if _crc(body) != crc:
        raise ValueError("checksum mismatch - snapshot is corrupt")
    out: dict[str, np.ndarray] = {}
    off = 0
    for _ in range(n_sec):
        name, data, off = _unpack(body, off)
        out[name] = data
    meta_raw = out.pop("meta").tobytes()
    meta = eval(meta_raw.decode(), {"__builtins__": {}}, {})
    dim = int(meta.get("dim", 8))
    rank = int(meta.get("rank", 4))

    def view(name: str, dtype: np.dtype, shape: tuple) -> np.ndarray:
        return out[name].view(dtype).reshape(shape)

    n = len(view("energy", np.uint8 if not meta.get("lossless", True)
                 else np.float32, (-1,)))
    if meta.get("lossless", True):
        energy = view("energy", np.float32, (n,))
        phase = view("phase", np.float32, (n,))
        state = view("state", np.float32, (n, dim))
    else:
        # Archive mode is quantised: restore to float32 with the documented
        # error (~2e-3 on energy).  Not byte-identical by design.
        from ..core.neuron import TWO_PI, dequantise_u8
        energy = dequantise_u8(view("energy", np.uint8, (n,)), 0.0, 1.0)
        phase = dequantise_u8(view("phase", np.uint8, (n,)), 0.0,
                              float(TWO_PI))
        state = dequantise_u8(view("state", np.uint8, (n, dim)), -1.0, 1.0)
    return Snapshot(
        energy=energy,
        phase=phase,
        state=state,
        adapters_a=(view("adapter_a", np.float32, (n, rank, dim))
                    if meta.get("lossless", True) else
                    dequantise_u8(view("adapter_a", np.uint8, (n, rank, dim)),
                                  -0.05, 0.05)),
        adapters_b=(view("adapter_b", np.float32, (n, dim, rank))
                    if meta.get("lossless", True) else
                    dequantise_u8(view("adapter_b", np.uint8, (n, dim, rank)),
                                  -0.05, 0.05)),
        region=view("region", np.uint8, (n,)),
        trigger_at=view("trigger_at", np.float32, (n,)),
        meta=meta,
    )


def hibernate(path: str, pop, rs, level: int = 3,
              lossless: bool = True) -> int:
    """Freeze and serialise; returns bytes written."""
    return save(path, Snapshot(
        energy=pop.energy.copy(), phase=pop.phase.copy(),
        state=pop.state.copy(), adapters_a=pop.adapter_a.copy(),
        adapters_b=pop.adapter_b.copy(), region=pop.region.copy(),
        trigger_at=rs.trigger_at.copy(),
        meta={"n": pop.n, "dim": int(pop.state.shape[1]),
              "rank": int(pop.adapter_a.shape[1]),
              "lossless": lossless},
    ), level=level, lossless=lossless)


def wake(path: str, pop, rs) -> None:
    """Restore in place; raises if the snapshot does not match the layout."""
    snap = load(path)
    if snap.energy.shape[0] != pop.n:
        raise ValueError("snapshot size does not match population")
    pop.energy[:] = snap.energy
    pop.phase[:] = snap.phase
    pop.state[:] = snap.state.reshape(pop.state.shape)
    pop.adapter_a[:] = snap.adapters_a.reshape(pop.adapter_a.shape)
    pop.adapter_b[:] = snap.adapters_b.reshape(pop.adapter_b.shape)
    pop.region[:] = snap.region
    rs.trigger_at[:] = snap.trigger_at
