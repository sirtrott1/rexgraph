"""rexgraph.hip_ternary: the native device lane for a packed ternary operator.

The portable device lane goes through torch, where the product is a chain of
elementwise expressions. Fused it reaches about 70% of what the memory system can
stream; unfused, a ninth of that, since each expression round-trips the whole operator.
Neither reaches the popcount the hardware already has, because torch exposes none.

This lane is the same arithmetic compiled once by hipcc: one pass over the planes, one
__popcll per word, the vector in LDS where it fits.

    MEASURED, gfx1151 (Radeon 8060S), planes 1074 MB, block 256:

        torch, eager                52.9 Gentry/s
        torch, compiled            599.1
        HIP                        854.6      213.6 GB/s, 98% of the 217.9 GB/s
                                              this device streams

It is OPTIONAL. The shared object is built only where hipcc exists, and absent it the
cpu, openmp and cuda lanes are unchanged, so nothing here is required to run rexgraph.

Residency is the caller's decision and is explicit. The planes ARE the operator, so a
lane that re-sends them per product spends its time on the bus rather than on the
arithmetic. `resident()` uploads once and returns a handle.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from rexgraph import compute
from rexgraph.core import _ternary

__all__ = ["available", "library_path", "resident", "ResidentTernary"]

_LIB = None
_TRIED = False
_DEVICE_OK = None


def library_path() -> Path:
    """Where the built object lives, next to the kernel it came from."""
    return Path(__file__).with_name("core") / "lib_ternary_hip.so"


def _load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    path = os.environ.get("REXGRAPH_TERNARY_HIP") or str(library_path())
    if not os.path.exists(path):
        return None
    # Import torch BEFORE this object, where torch exists. Both link libamdhip64, torch
    # from its bundled _rocm_sdk_core and this from the system ROCm, and the loader
    # binds one SONAME for the process. Whichever lands first wins, and if it is this
    # one then torch's later import dies on `undefined symbol: hsa_ext_image_create_v2`.
    # Letting torch go first costs an import that a rexgraph process almost always
    # makes anyway, and it means the two lanes coexist whatever order the caller uses.
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    lib.ternary_alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.ternary_upload.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ternary_download.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ternary_free.argtypes = [ctypes.c_void_p]
    lib.ternary_pm1_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
    lib.ternary_f64_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    lib.tower_launch.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 3
    for n in ("ternary_alloc", "ternary_upload", "ternary_download",
              "ternary_free", "ternary_pm1_launch", "ternary_f64_launch",
              "tower_launch"):
        getattr(lib, n).restype = ctypes.c_int
    _LIB = lib
    return _LIB


def available() -> bool:
    """Whether the compiled kernel can allocate on a device visible to this process.

    Loading ``libamdhip64`` proves that the runtime exists, not that this process can
    reach a GPU. Containers commonly expose the host library without ``/dev/kfd`` or
    a render node; registering the lane there makes automatic dispatch choose a
    backend whose first allocation fails with ``hipErrorNoDevice``. Probe one byte
    once, release it immediately, and cache the answer used by the registry.
    """
    global _DEVICE_OK
    if _DEVICE_OK is not None:
        return _DEVICE_OK
    lib = _load()
    if lib is None:
        _DEVICE_OK = False
        return False
    ptr = ctypes.c_void_p()
    try:
        rc = int(lib.ternary_alloc(ctypes.byref(ptr), 1))
    except Exception:
        _DEVICE_OK = False
        return False
    if rc != 0:
        _DEVICE_OK = False
        return False
    try:
        _DEVICE_OK = int(lib.ternary_free(ptr)) == 0
    except Exception:
        _DEVICE_OK = False
    return _DEVICE_OK


class ResidentTernary:
    """Planes held on the device. Only the vector crosses the bus per product."""

    __slots__ = ("_lib", "_d", "_out_host", "_outf_host", "shape", "nw", "block")

    def __init__(self, op, block: int = 256):
        lib = _load()
        if lib is None:
            raise RuntimeError("the HIP ternary kernel is not built on this machine")
        self._lib = lib
        self.shape = op.shape
        self.nw = int(op.P.shape[1])
        self.block = int(block)
        self._d = {}
        for name, arr in (("P", op.P), ("S", op.S),
                          ("K", np.ascontiguousarray(op.arity(), dtype=np.int64))):
            self._d[name] = self._up(np.ascontiguousarray(arr))
        self._d["X"] = self._alloc(self.nw * 8)
        self._d["O"] = self._alloc(self.shape[0] * 8)
        self._d["V"] = self._alloc(self.nw * 64 * 8)     # padded to whole words
        self._d["F"] = self._alloc(self.shape[0] * 8)
        self._out_host = np.empty(self.shape[0], dtype=np.int64)
        self._outf_host = np.empty(self.shape[0], dtype=np.float64)

    def _alloc(self, nbytes):
        p = ctypes.c_void_p()
        rc = self._lib.ternary_alloc(ctypes.byref(p), nbytes)
        if rc != 0:
            raise RuntimeError(f"device allocation failed, hipError {rc}")
        return p

    def _up(self, arr):
        p = self._alloc(arr.nbytes)
        rc = self._lib.ternary_upload(p, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)
        if rc != 0:
            raise RuntimeError(f"upload failed, hipError {rc}")
        return p

    def matvec(self, x) -> np.ndarray:
        """Exact integer product against a +-1 vector."""
        v = np.asarray(x)
        if v.ndim != 1 or v.shape[0] != self.shape[1]:
            raise ValueError(f"vector of length {self.shape[1]} required, got {v.shape}")
        packed = np.ascontiguousarray(_ternary.pack_vector(v.astype(np.int8)))
        rc = self._lib.ternary_upload(self._d["X"],
                                      packed.ctypes.data_as(ctypes.c_void_p), packed.nbytes)
        if rc != 0:
            raise RuntimeError(f"vector upload failed, hipError {rc}")
        rc = self._lib.ternary_pm1_launch(self._d["P"], self._d["S"], self._d["X"],
                                          self._d["K"], self._d["O"],
                                          self.shape[0], self.nw, self.block)
        if rc != 0:
            raise RuntimeError(f"kernel launch failed, hipError {rc}")
        rc = self._lib.ternary_download(
            self._out_host.ctypes.data_as(ctypes.c_void_p), self._d["O"],
            self._out_host.nbytes)
        if rc != 0:
            raise RuntimeError(f"download failed, hipError {rc}")
        return self._out_host.copy()

    def matvec_f64(self, v) -> np.ndarray:
        """Product against a general float vector.

        The vector is padded to whole words so the kernel can address a full 64 entries
        at the end of the last one, and the padding is zero. Rows are blocked four to a
        workgroup there, since every row reads all of v and that traffic, not the
        planes, is what bounds this product.
        """
        x = np.asarray(v)
        if x.ndim != 1 or x.shape[0] != self.shape[1]:
            raise ValueError(f"vector of length {self.shape[1]} required, got {x.shape}")
        padded = np.zeros(self.nw * 64, dtype=np.float64)
        padded[:x.shape[0]] = x
        rc = self._lib.ternary_upload(self._d["V"],
                                      padded.ctypes.data_as(ctypes.c_void_p), padded.nbytes)
        if rc != 0:
            raise RuntimeError(f"vector upload failed, hipError {rc}")
        rc = self._lib.ternary_f64_launch(self._d["P"], self._d["S"], self._d["V"],
                                          self._d["F"], self.shape[0], self.nw, self.block)
        if rc != 0:
            raise RuntimeError(f"kernel launch failed, hipError {rc}")
        rc = self._lib.ternary_download(
            self._outf_host.ctypes.data_as(ctypes.c_void_p), self._d["F"],
            self._outf_host.nbytes)
        if rc != 0:
            raise RuntimeError(f"download failed, hipError {rc}")
        return self._outf_host.copy()

    def close(self) -> None:
        """Release the device memory. Idempotent."""
        if self._d:
            for p in self._d.values():
                self._lib.ternary_free(p)
            self._d = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def resident(op, block: int = 256) -> ResidentTernary:
    """Upload an operator once and keep it there."""
    return ResidentTernary(op, block)


def _pm1_hip(op, v):
    """The one-shot form. Uploads the planes, so it is for a single product only:
    anything repeated should hold a `resident()` handle instead."""
    with ResidentTernary(op) as r:
        return r.matvec(v)


def _f64_hip(op, v):
    with ResidentTernary(op) as r:
        return r.matvec_f64(v)


if available():
    compute.register_op("ternary_matvec_pm1", "hip", _pm1_hip)
    compute.register_op("ternary_matvec_f64", "hip", _f64_hip)
    compute.register_backend("hip", available=available, kind="gpu",
                             description="native HIP ternary kernel")


#### the channel tower on the device
def channel_tower(bp, bi, nV, w=None, block: int = 256):
    """The four channel diagonals at any arity, on the device.

    Same shape as the CPU kernel and for the same reason: with the incidence transposed
    the accumulation is a loop over VERTICES, one thread each, so nothing needs an
    atomic. C is unweighted and F is not, so the vertex mass is carried twice, and a
    witness joins the positive mass rather than taking the head rule.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("the HIP kernels are not built on this machine")
    from rexgraph.core._channel_tower import transpose_incidence
    bp = np.ascontiguousarray(bp, dtype=np.int32)
    bi = np.ascontiguousarray(bi, dtype=np.int32)
    nE = int(bp.shape[0] - 1)
    wv = (np.ones(nE, np.float64) if w is None
          else np.ascontiguousarray(w, dtype=np.float64))
    vptr, owner, is_head = transpose_incidence(bp, bi, int(nV))

    ptrs = {}

    def up(arr):
        p = ctypes.c_void_p()
        if lib.ternary_alloc(ctypes.byref(p), max(arr.nbytes, 1)) != 0:
            raise RuntimeError("device allocation failed")
        if arr.nbytes and lib.ternary_upload(
                p, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes) != 0:
            raise RuntimeError("upload failed")
        return p

    def blank(nbytes):
        p = ctypes.c_void_p()
        if lib.ternary_alloc(ctypes.byref(p), max(nbytes, 1)) != 0:
            raise RuntimeError("device allocation failed")
        return p

    try:
        for k, a in (("bp", bp), ("bi", bi), ("ow", owner), ("ih", is_head),
                     ("w", wv), ("vp", np.ascontiguousarray(vptr, np.int64))):
            ptrs[k] = up(a)
        for k in ("nw", "pw", "nu", "pu"):
            ptrs[k] = blank(int(nV) * 8)
        for k in ("T", "G", "F", "C"):
            ptrs[k] = blank(nE * 8)
        rc = lib.tower_launch(ptrs["bp"], ptrs["bi"], ptrs["ow"], ptrs["ih"],
                              ptrs["w"], ptrs["vp"],
                              ptrs["nw"], ptrs["pw"], ptrs["nu"], ptrs["pu"],
                              ptrs["T"], ptrs["G"], ptrs["F"], ptrs["C"],
                              int(nV), nE, int(block))
        if rc != 0:
            raise RuntimeError(f"tower launch failed, hipError {rc}")
        out = []
        for k in ("T", "G", "F", "C"):
            host = np.empty(nE, dtype=np.float64)
            if lib.ternary_download(host.ctypes.data_as(ctypes.c_void_p),
                                    ptrs[k], host.nbytes) != 0:
                raise RuntimeError("download failed")
            out.append(host)
        return tuple(out)
    finally:
        for p in ptrs.values():
            lib.ternary_free(p)


def _tower_hip(bp, bi, nV, w):
    return channel_tower(bp, bi, nV, w)


if available():
    compute.register_op("channel_tower", "hip", _tower_hip)
