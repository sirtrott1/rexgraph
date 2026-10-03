"""Native HIP products for packed ternary operators.

The optional hipcc library reads presence and sign bitplanes and uses popcount
for integer query products. Float query products use the separate native kernel.
resident uploads an operator once and returns a device handle for repeated use.
Missing native capabilities are reported by the public availability checks.
"""
from __future__ import annotations

import ctypes
import os
import operator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import numpy as np

from rexgraph import compute
from rexgraph.core import _ternary

__all__ = ["available", "tensor_available", "library_path", "resident", "ResidentTernary"]

_LIB = None
_TRIED = False
_DEVICE_OK = None
_DEVICE_KEY = None
_PROBE_LOCK = RLock()


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
    # Import Torch first so its libamdhip64 is selected before loading this library.
    # The dynamic loader binds one object per SONAME in the process.
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    signatures = {
        "ternary_alloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        "ternary_upload": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
        "ternary_download": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
        "ternary_free": [ctypes.c_void_p],
        "ternary_pm1_launch": [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3,
        "ternary_f64_launch": [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3,
        "tower_launch": [ctypes.c_void_p] * 14 + [ctypes.c_int] * 3,
    }
    try:
        for name, args in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
    except AttributeError:
        return None                  # a loadable, incompatible library is not a backend
    # Bind tower_launch_coef separately from the canonical ABI.
    # Libraries without this symbol support canonical columns only.
    try:
        lib.tower_launch_coef.argtypes = [ctypes.c_void_p] * 16 + [ctypes.c_int] * 3
        lib.tower_launch_coef.restype = ctypes.c_int
    except AttributeError:
        pass
    for name, args in {
        "ternary_f32_batch_async": [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p],
        "ternary_f64_batch_async": [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p],
        "ternary_get_device": [ctypes.POINTER(ctypes.c_int)],
        "ternary_set_device": [ctypes.c_int],
    }.items():
        try:
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        except AttributeError:
            pass
    _LIB = lib
    return _LIB


def available() -> bool:
    """Whether the compiled kernel executes correctly on a visible device.

    Loading ``libamdhip64`` proves that the runtime exists, not that this process can
    reach a GPU. Containers commonly expose the host library without ``/dev/kfd`` or
    a render node; registering the lane there makes automatic dispatch choose a
    backend whose first allocation fails with ``hipErrorNoDevice``. Allocation alone
    also accepts a binary built for the wrong gfx target. Probe a one entry exact
    product once, release its memory, and cache the answer used by the registry.
    """
    with _PROBE_LOCK:
        return _probe_available()


def _probe_available():
    global _DEVICE_OK, _DEVICE_KEY
    lib = _load()
    if lib is None:
        _DEVICE_OK = False
        return False
    try:
        key = _device_id(lib)
    except RuntimeError:
        return False
    if _DEVICE_OK is not None and key == _DEVICE_KEY:
        return _DEVICE_OK
    _DEVICE_KEY = key
    ptr = ctypes.c_void_p()
    allocated = False
    _DEVICE_OK = False
    try:
        if lib.ternary_alloc(ctypes.byref(ptr), 40) != 0:
            return False
        allocated = True
        # Presence=1, sign=0, positive vector=0, arity=1, followed by the output.
        host = np.array([1, 0, 0, 1, 0], dtype=np.uint64)
        if lib.ternary_upload(ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes) != 0:
            return False
        args = [ctypes.c_void_p(ptr.value + i * 8) for i in range(5)]
        if lib.ternary_pm1_launch(*args, 1, 1, 256) != 0:
            return False
        result = np.empty(1, dtype=np.int64)
        if lib.ternary_download(result.ctypes.data_as(ctypes.c_void_p), args[-1], 8) != 0:
            return False
        _DEVICE_OK = bool(result[0] == 1)
    except Exception:
        _DEVICE_OK = False
    finally:
        if allocated:
            try:
                if lib.ternary_free(ptr) != 0:
                    _DEVICE_OK = False
            except Exception:
                _DEVICE_OK = False
    return _DEVICE_OK


def tensor_available():
    """Whether stream aware float32/float64 tensor entry points are usable."""
    lib = _load()
    return (lib is not None and all(hasattr(lib, name) for name in
            ("ternary_f32_batch_async", "ternary_f64_batch_async")) and available())


def _device_id(lib):
    if not all(hasattr(lib, name) for name in ("ternary_get_device", "ternary_set_device")):
        return None                     # older single device ABI
    device = ctypes.c_int()
    rc = lib.ternary_get_device(ctypes.byref(device))
    if rc != 0:
        raise RuntimeError(f"HIP device query failed, hipError {rc}")
    return device.value


@contextmanager
def _on_device(lib, device):
    before = _device_id(lib) if device is not None else None
    if before != device:
        rc = lib.ternary_set_device(device)
        if rc != 0:
            raise RuntimeError(f"HIP device selection failed, hipError {rc}")
    try:
        yield
    finally:
        if before != device:
            rc = lib.ternary_set_device(before)
            if rc != 0:
                raise RuntimeError(f"HIP device restore failed, hipError {rc}")


def _count(value, name):
    try:
        n = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a nonnegative int32 integer") from None
    if isinstance(value, (bool, np.bool_)) or not 0 <= n <= np.iinfo(np.int32).max:
        raise ValueError(f"{name} must be a nonnegative int32 integer")
    return n


def _block(value, *, reduction=True):
    n = _count(value, "block")
    if reduction and (n < 4 or n > 256 or n & (n - 1)):
        raise ValueError("ternary block must be a power of two from 4 to 256")
    if not reduction and not 1 <= n <= 256:
        raise ValueError("tower block must be from 1 to 256")
    return n


def _alloc(lib, nbytes):
    ptr = ctypes.c_void_p()
    rc = lib.ternary_alloc(ctypes.byref(ptr), max(nbytes, 1))
    if rc != 0:
        raise RuntimeError(f"device allocation failed, hipError {rc}")
    return ptr


def _upload(lib, arr):
    ptr = _alloc(lib, arr.nbytes)
    try:
        if arr.nbytes:
            rc = lib.ternary_upload(ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)
            if rc != 0:
                raise RuntimeError(f"upload failed, hipError {rc}")
    except BaseException:
        lib.ternary_free(ptr)
        raise
    return ptr


def _operator_planes(op):
    if len(op.shape) != 2:
        raise ValueError("a two-dimensional ternary shape is required")
    rows = _count(op.shape[0], "rows")
    cols = operator.index(op.shape[1])
    if isinstance(op.shape[1], (bool, np.bool_)) or cols < 0:
        raise ValueError("columns must be a nonnegative integer")
    nw = _count((cols + 63) // 64, "packed words")
    P, S = np.asarray(op.P), np.asarray(op.S)
    for plane in (P, S):
        if plane.dtype != np.dtype(np.uint64) or plane.shape != (rows, nw):
            raise ValueError("native uint64 bitplanes must match the ternary shape")
    if nw and cols % 64 and np.any(P[:, -1] >> np.uint64(cols % 64)):
        raise ValueError("presence bits outside the declared columns are not allowed")
    P, S = np.ascontiguousarray(P), np.ascontiguousarray(S)
    arity = _ternary.arity(P)
    supplied = np.asarray(op.arity())
    if supplied.shape != (rows,) or supplied.dtype.kind not in "iu" or not np.array_equal(supplied, arity):
        raise ValueError("row arity must equal the bitplane support count")
    return (rows, cols), nw, P, S, arity


class ResidentTernary:
    """Planes held on the device. Only the vector crosses the bus per product.

    Calls and close are serialized because the handle reuses its input/output buffers.
    Independent handles can be used by independent workers. Calls return host arrays
    synchronously; this is a numerical lane, without Torch autograd or stream interop.
    """

    __slots__ = ("_lib", "_d", "_out_host", "_outf_host", "shape", "nw", "block",
                 "_lock", "_closed", "_device")

    def __init__(self, op, block: int = 256):
        self._lock, self._d, self._closed = RLock(), {}, False
        self.block = _block(block)
        self.shape, self.nw, P, S, K = _operator_planes(op)
        lib = _load()
        if lib is None:
            raise RuntimeError("the HIP ternary kernel is not built on this machine")
        self._lib = lib
        self._device = _device_id(lib)
        self._out_host = np.empty(self.shape[0], dtype=np.int64)
        self._outf_host = np.empty(self.shape[0], dtype=np.float64)
        if not self.shape[0] or not self.nw:
            return
        try:
            for name, arr in (("P", P), ("S", S), ("K", K)):
                self._d[name] = _upload(lib, arr)
            self._d["X"] = _alloc(lib, self.nw * 8)
            self._d["O"] = _alloc(lib, self.shape[0] * 8)
            self._d["V"] = _alloc(lib, self.nw * 64 * 8)  # padded to whole words
            self._d["F"] = _alloc(lib, self.shape[0] * 8)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("the resident ternary operator is closed")

    def matvec(self, x) -> np.ndarray:
        """Exact integer product against a +-1 vector."""
        with self._lock:
            self._ensure_open()
            with _on_device(self._lib, self._device):
                return self._matvec(x)

    def _matvec(self, x):
        from rexgraph.ternary import _pm1_vector
        v = _pm1_vector(x, self.shape[1])
        if not self.shape[0] or not self.nw:
            return np.zeros(self.shape[0], dtype=np.int64)
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
        with self._lock:
            self._ensure_open()
            with _on_device(self._lib, self._device):
                return self._matvec_f64(v)

    def _matvec_f64(self, v):
        x = np.asarray(v)
        if x.ndim != 1 or x.shape[0] != self.shape[1]:
            raise ValueError(f"vector of length {self.shape[1]} required, got {x.shape}")
        if x.dtype.kind not in "iufb":
            raise ValueError("a real numeric vector is required")
        if not self.shape[0] or not self.nw:
            return np.zeros(self.shape[0], dtype=np.float64)
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
        with self._lock:
            if self._closed:
                return
            errors = []
            with _on_device(self._lib, getattr(self, "_device", None)):
                self._closed = True
                owned, self._d = self._d, {}
                for p in owned.values():
                    try:
                        rc = self._lib.ternary_free(p)
                        if rc != 0:
                            errors.append(f"hipError {rc}")
                    except Exception as exc:
                        errors.append(str(exc))
            if errors:
                raise RuntimeError(f"device release failed: {', '.join(errors)}")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def resident(op, block: int = 256) -> ResidentTernary:
    """Upload an operator once and keep it there."""
    return ResidentTernary(op, block)


def _pm1_hip(op, v):
    """The one shot form. Uploads the planes, so it is for a single product only:
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
def channel_tower(bp, bi, nV, w=None, block: int = 256, coefficients=None):
    """The four channel diagonals at any arity, on the device.

    Same shape as the CPU kernel and for the same reason: with the incidence transposed
    the accumulation is a loop over VERTICES, one thread each, so nothing needs an
    atomic. C is unweighted and F is not, so the vertex mass is carried twice, and a
    witness joins the positive mass rather than taking the head rule.
    As in the CPU diagonal kernel, signed weights enter through their magnitudes;
    orientation is read from B1, and the caller's weights are not modified.

    `coefficients` is the column entry of every incidence, in CSR order, for a complex
    that DECLARES a head or a share; the device then splits the vertex mass by the sign
    of the entry rather than by the head bit. Without it the column is derived from the
    arity, which is the canonical reading and costs the device nothing extra.
    """
    from rexgraph.core._channel_tower import transpose_incidence
    from rexgraph.graph import _as_exact_i32_vector
    block = _block(block, reduction=False)
    nV = _count(nV, "vertices")
    bp = _as_exact_i32_vector(bp, context="boundary_ptr")
    bi = _as_exact_i32_vector(bi, context="boundary_idx")
    if not bp.size or bp[0] != 0 or np.any(bp[1:] < bp[:-1]) or bp[-1] != bi.size:
        raise ValueError("invalid incidence pointers: start at zero, nondecreasing, terminal equals index count")
    if bi.size and (np.any(bi < 0) or np.any(bi >= nV)):
        raise ValueError("incidence indices must lie in the declared vertex domain")
    nE = int(bp.shape[0] - 1)
    _count(nE, "relations")
    if w is not None and np.iscomplexobj(w):
        raise ValueError("one finite real weight per relation is required")
    wv = (np.ones(nE, np.float64) if w is None
          else np.abs(np.ascontiguousarray(w, dtype=np.float64)))
    if wv.shape != (nE,) or not np.isfinite(wv).all():
        raise ValueError("one finite real weight per relation is required")
    declared = coefficients is not None
    coef = source = None
    if declared:
        if np.iscomplexobj(coefficients):
            raise ValueError("one finite real coefficient per incidence is required")
        coef = np.ascontiguousarray(coefficients, dtype=np.float64)
        if coef.shape != bi.shape or not np.isfinite(coef).all():
            raise ValueError("one finite real coefficient per incidence is required")
    if not nE or not bi.size:
        return tuple(np.zeros(nE, dtype=np.float64) for _ in range(4))
    lib = _load()
    if lib is None:
        raise RuntimeError("the HIP kernels are not built on this machine")
    if declared and not hasattr(lib, "tower_launch_coef"):
        raise RuntimeError(
            "the HIP kernels on this machine predate declared columns: rebuild "
            "lib_ternary_hip.so, or read this complex on the compiled CPU lanes")
    if declared:
        vptr, owner, is_head, source = transpose_incidence(bp, bi, int(nV), 1, True)
    else:
        vptr, owner, is_head = transpose_incidence(bp, bi, int(nV))

    ptrs = {}

    try:
        for k, a in (("bp", bp), ("bi", bi), ("ow", owner), ("ih", is_head),
                     ("w", wv), ("vp", np.ascontiguousarray(vptr, np.int64))):
            ptrs[k] = _upload(lib, a)
        if declared:
            ptrs["cf"] = _upload(lib, coef)
            ptrs["sr"] = _upload(lib, np.ascontiguousarray(source, np.int64))
        for k in ("nw", "pw", "nu", "pu"):
            ptrs[k] = _alloc(lib, int(nV) * 8)
        for k in ("T", "G", "F", "C"):
            ptrs[k] = _alloc(lib, nE * 8)
        if declared:
            rc = lib.tower_launch_coef(
                ptrs["bp"], ptrs["bi"], ptrs["ow"], ptrs["ih"], ptrs["w"], ptrs["vp"],
                ptrs["cf"], ptrs["sr"],
                ptrs["nw"], ptrs["pw"], ptrs["nu"], ptrs["pu"],
                ptrs["T"], ptrs["G"], ptrs["F"], ptrs["C"], int(nV), nE, int(block))
        else:
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


def _tower_hip(bp, bi, nV, w, coefficients=None):
    return channel_tower(bp, bi, nV, w, coefficients=coefficients)


if available():
    compute.register_op("channel_tower", "hip", _tower_hip)
