"""Ternary operators stored as presence and sign bitplanes.

Each row carries its nonzero count. Pairing with a {-1, +1} query returns
`k - 2 * disagree`, using the packed presence and sign planes. Float products
use their corresponding registered kernels.

CPU, OpenMP and Torch device implementations register through rexgraph.compute.
TernaryOperator.to returns a resident DeviceTernary; matvec performs one product
with the selected backend. Sparse boundary incidence retains its own CSR format.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rexgraph import compute
from rexgraph.core import _ternary

__all__ = ["TernaryOperator", "DeviceTernary", "pack", "matvec", "arity",
           "backends_for"]


@dataclass(frozen=True, slots=True)
class TernaryOperator:
    """Bitplanes of a {-1,0,1} matrix. `P` presence, `S` sign, both (rows, words)."""

    P: np.ndarray
    S: np.ndarray
    shape: tuple[int, int]
    K: np.ndarray = field(default=None)          # per row arity, the product needs it

    @property
    def nbytes(self) -> int:
        return int(self.P.nbytes + self.S.nbytes)

    def dense(self) -> np.ndarray:
        """The {-1,0,1} array this was packed from."""
        return _ternary.unpack(self.P, self.S, self.shape[1])

    def arity(self) -> np.ndarray:
        """Per row support size, which is what a share of 1/(k-1) derives from."""
        return self.K if self.K is not None else _ternary.arity(self.P)


    def to(self, device: str = "cuda") -> "DeviceTernary":
        """Ship the planes once and keep them there.

        Not an optimisation detail. The planes are the whole operator, so a lane that
        re sends them per product pays the transfer every time and lands at 96
        Gentry/s where the resident form reaches 599. Anything doing more than one
        product against the same operator should hold this.
        """
        import torch
        return DeviceTernary(
            torch.from_numpy(self.P.view(np.int64)).to(device),
            torch.from_numpy(self.S.view(np.int64)).to(device),
            torch.from_numpy(np.ascontiguousarray(self.arity())).to(device),
            device, self.shape)


@dataclass(frozen=True, slots=True)
class DeviceTernary:
    """An operator already resident on a device. `TernaryOperator.to()` builds it."""

    P: object
    S: object
    K: object
    device: str
    shape: tuple[int, int]

    def matvec(self, x):
        """Product against a +-1 vector, with only the vector crossing the bus."""
        import torch
        v = _pm1_vector(x, self.shape[1])
        X = torch.from_numpy(
            _ternary.pack_vector(v.astype(np.int8)).view(np.int64)).to(self.device)
        out = _compiled(self.device)(self.P, self.S, X, self.K, *_masks(self.device))
        return out.to(torch.int64).cpu().numpy()


def _pm1_vector(x, length):
    """Validate before int8 conversion, which can otherwise round or wrap values."""
    v = np.asarray(x)
    if v.ndim != 1 or v.shape[0] != length:
        raise ValueError(f"vector of length {length} required, got {v.shape}")
    if v.dtype.kind not in "iuf" or not np.isin(v, (-1, 1)).all():
        raise ValueError("the packed integer product requires a vector in {-1,+1}")
    return v


def pack(arr) -> TernaryOperator:
    """Pack a 2-D array whose entries are all in {-1, 0, 1}.

    Refuses anything else rather than rounding it: a value outside the set is a caller
    error about what the operator IS, and truncating it would return a different
    operator that still looks correct.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D operator, got shape {a.shape}")
    u = np.unique(a)
    if u.size and not np.isin(u, (-1, 0, 1)).all():
        bad = u[~np.isin(u, (-1, 0, 1))]
        raise ValueError(f"not ternary: {bad[:4]} outside {{-1,0,1}}")
    P, S, nc = _ternary.pack(a.astype(np.int8))
    return TernaryOperator(P, S, (int(a.shape[0]), int(nc)), _ternary.arity(P))


def arity(op: TernaryOperator) -> np.ndarray:
    return op.arity()


def matvec(op: TernaryOperator, x, *, prefer: str | None = None):
    """Product with `x`, routed to the best backend that implements it.

    A +-1 integer vector takes the popcount path and the answer is exact: a difference
    of counts, with nothing rounded. Anything else takes the float path, which visits
    the support only.
    """
    v = np.asarray(x)
    if v.ndim != 1 or v.shape[0] != op.shape[1]:
        raise ValueError(f"vector of length {op.shape[1]} required, got {v.shape}")
    if np.issubdtype(v.dtype, np.integer) and np.isin(np.unique(v), (-1, 1)).all():
        return compute.dispatch("ternary_matvec_pm1", op, v, prefer=prefer)
    return compute.dispatch("ternary_matvec_f64", op, v, prefer=prefer)


def backends_for(name: str = "ternary_matvec_pm1") -> list[str]:
    """Which backends currently implement an op, for callers that want to see the lane."""
    for entry in compute.ops():
        if entry["name"] == name:
            return entry["backends"]
    return []


#### the compiled CPU kernel, serial and threaded
def _pm1_cpu(op: TernaryOperator, v, threads: int = 1):
    v = _pm1_vector(v, op.shape[1])
    return _ternary.matvec_pm1(op.P, op.S, _ternary.pack_vector(v.astype(np.int8)),
                               op.arity(), threads)


def _pm1_openmp(op: TernaryOperator, v):
    return _pm1_cpu(op, v, compute.effective_threads())


def _f64_cpu(op: TernaryOperator, v, threads: int = 1):
    return _ternary.matvec_f64(op.P, op.S, np.ascontiguousarray(v, dtype=np.float64), threads)


def _f64_openmp(op: TernaryOperator, v):
    return _f64_cpu(op, v, compute.effective_threads())


compute.register_op("ternary_matvec_pm1", "cpu", _pm1_cpu)
compute.register_op("ternary_matvec_pm1", "openmp", _pm1_openmp)
compute.register_op("ternary_matvec_f64", "cpu", _f64_cpu)
compute.register_op("ternary_matvec_f64", "openmp", _f64_openmp)


#### the device lane: torch covers NVIDIA and ROCm alike, which is what compute calls cuda
#
# Bit counting is done by SWAR rather than by a table lookup, so the whole product is
# arithmetic on the packed words and never gathers. That matters more than it looks: a
# gather cannot fuse, and fusion is worth 11.65x here. The kernel is compiled once and
# cached, and falls back to eager if compilation is unavailable.
_MASKS: dict = {}
_COMPILED: dict = {}


def _masks(dev):
    import torch
    if dev not in _MASKS:
        _MASKS[dev] = tuple(torch.tensor(v, dtype=torch.int64, device=dev) for v in
                            (0x5555555555555555, 0x3333333333333333,
                             0x0F0F0F0F0F0F0F0F, 0x0101010101010101))
    return _MASKS[dev]


def _pm1_device(P, S, X, K, M1, M2, M4, H1):
    """k - 2*popcount(P & (S ^ X)), with the popcount done by SWAR."""
    x = P & (S ^ X.unsqueeze(0))
    x = x - ((x >> 1) & M1)
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    return K - 2 * ((x * H1) >> 56).sum(1)


def _compiled(dev):
    """Compile once per device. Eager is a ninth of the speed, so this is not a tuning
    detail: an uncompiled device lane is barely worth having."""
    if dev not in _COMPILED:
        import torch
        try:
            _COMPILED[dev] = torch.compile(_pm1_device, dynamic=False)
        except Exception:
            _COMPILED[dev] = _pm1_device
    return _COMPILED[dev]


def _pm1_cuda(op: TernaryOperator, v):
    import torch
    v = _pm1_vector(v, op.shape[1])
    dev = "cuda"
    P = torch.from_numpy(op.P.view(np.int64)).to(dev)
    S = torch.from_numpy(op.S.view(np.int64)).to(dev)
    X = torch.from_numpy(_ternary.pack_vector(v.astype(np.int8)).view(np.int64)).to(dev)
    K = torch.from_numpy(np.ascontiguousarray(op.arity())).to(dev)
    out = _compiled(dev)(P, S, X, K, *_masks(dev))
    return out.to(torch.int64).cpu().numpy()


def _f64_cuda(op: TernaryOperator, v):
    """Dense on the device: the planes expand there, so the 16x saving is on residency
    and transfer rather than on the multiply."""
    import torch
    dev = "cuda"
    dense = torch.from_numpy(op.dense().astype(np.float64)).to(dev)
    vv = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float64)).to(dev)
    return (dense @ vv).cpu().numpy().astype(np.float64)


def _register_device_lane() -> bool:
    """Register the device lane where torch can actually reach a device."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    compute.register_op("ternary_matvec_pm1", "cuda", _pm1_cuda)
    compute.register_op("ternary_matvec_f64", "cuda", _f64_cuda)
    return True


_DEVICE_LANE = _register_device_lane()
