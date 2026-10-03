"""Differentiable fixed ternary maps using Core's packed CPU or native HIP kernels.

The operator is fixed. Forward applies A; backward applies its true transpose.
No rounding surrogate or straight through gradient is used for the bitplanes.
HIP tensor calls run on Torch's active device/stream and keep input/output on device.
"""
from __future__ import annotations

import ctypes
import math

import numpy as np

from rexgraph.core import _ternary
from rexgraph.ternary import TernaryOperator, pack

try:
    import torch
    from torch import nn
    _Module = nn.Module
except ImportError:
    torch = None
    class _Module:
        def __init__(self, *args, **kwargs):
            raise ImportError("PackedTernaryLinear requires PyTorch")


def _plane_tensor(array, device=None):
    # Fixed operators still need version counters to invalidate their adjoints.
    # Model construction/device movement inside inference_mode must not turn
    # these buffers into inference tensors, which have no mutation counter.
    with torch.inference_mode(False):
        return torch.from_numpy(array).to(device=device)


def _launch_batches(lib, P, S, vectors, output, rows, words, block, stream):
    """Bind the ABI with explicit offsets and chunks below the native grid-y limit."""
    name = "ternary_f64_batch_async" if vectors.dtype == torch.float64 else "ternary_f32_batch_async"
    fn = getattr(lib, name, None)
    if fn is None:
        raise RuntimeError("HIP tensor entry points are absent: rebuild lib_ternary_hip.so")
    for start in range(0, vectors.shape[0], 65535):
        count = min(65535, vectors.shape[0] - start)
        rc = fn(ctypes.c_void_p(P.data_ptr()), ctypes.c_void_p(S.data_ptr()),
                ctypes.c_void_p(vectors.data_ptr() + start * words * 64 * vectors.element_size()),
                ctypes.c_void_p(output.data_ptr() + start * rows * output.element_size()),
                rows, words, count, block, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"HIP tensor launch failed, hipError {rc}")


def _product(x, P, S, rows, cols, backend, block, threads):
    size = math.prod(x.shape[:-1])
    shape = (*x.shape[:-1], rows)
    if rows == 0 or cols == 0 or size == 0:
        return x.new_zeros(shape)
    words = P.shape[1]
    if backend == "cpu":
        if x.device.type != "cpu":
            raise ValueError("the packed CPU backend requires CPU tensors")
        p = P.numpy().view(np.uint64)
        s = S.numpy().view(np.uint64)
        vectors = x.detach().reshape(size, cols).contiguous().numpy()
        result = np.empty((size, rows), dtype=np.float64)
        for i, v in enumerate(vectors):
            result[i] = _ternary.matvec_f64(p, s, np.ascontiguousarray(v, dtype=np.float64), threads)
        return torch.from_numpy(result).to(dtype=x.dtype).reshape(shape)
    if x.device.type != "cuda" or not getattr(torch.version, "hip", None):
        raise ValueError("the packed HIP backend requires ROCm device tensors")
    from rexgraph import hip_ternary as H
    lib = H._load()
    if lib is None:
        raise RuntimeError("HIP tensor kernels are unavailable; CPU fallback is disabled")
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        vectors = x.reshape(size, cols).contiguous()
        if cols != words * 64:
            padded = x.new_zeros((size, words * 64))
            padded[:, :cols] = vectors
            vectors = padded
        result = x.new_empty((size, rows))
        # Buffers can have been allocated on another stream, or die immediately
        # after this call. Register lifetime before launch, so a later chunk's
        # error cannot release inputs still used by an already queued chunk.
        for tensor in (P, S, vectors, result):
            tensor.record_stream(stream)
        _launch_batches(lib, P, S, vectors, result, rows, words, block, stream.cuda_stream)
        return result.reshape(shape)


if torch is not None:
    class _PackedProduct(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, P, S, PT, ST, rows, cols, backend, block, threads):
            ctx.save_for_backward(P, S, PT, ST)
            ctx.spec = rows, cols, backend, block, threads
            return _product(x, P, S, rows, cols, backend, block, threads)

        @staticmethod
        def backward(ctx, grad):
            P, S, PT, ST = ctx.saved_tensors
            rows, cols, backend, block, threads = ctx.spec
            # Calling the same Function on the transpose also records the correct
            # derivative for double backward; it does not detach the incoming grad.
            dx = _PackedProduct.apply(grad, PT, ST, P, S, cols, rows, backend, block, threads)
            return (dx,) + (None,) * 9


class PackedTernaryLinear(_Module):
    """Fixed signed map on the last feature axis, with exact transpose gradients.

    ``operator`` is a packed Core operator or a matrix in {-1,0,1}. Bitplanes are
    buffers, not trainable weights. Compose with learnable transforms or biases.
    CPU uses the compiled packed float64 accumulator and casts to the input dtype.
    HIP supports native float32/float64, batching and asynchronous Torch streams.
    Reduced precision inputs accumulate in float32 and return their original dtype.
    Explicit backends never transfer the input to a different device or fall back.
    """
    def __init__(self, operator, *, backend="auto", block=256, threads=1):
        super().__init__()
        from rexgraph import hip_ternary as H
        op = operator if isinstance(operator, TernaryOperator) else pack(operator)
        shape, _, P, S, _ = H._operator_planes(op)
        if backend not in {"auto", "cpu", "hip"}:
            raise ValueError("packed backend must be auto, cpu or hip")
        self.backend, self.block = backend, H._block(block)
        self.threads = H._count(threads, "threads")
        if not self.threads:
            raise ValueError("threads must be positive")
        self.out_features, self.in_features = shape
        self.register_buffer("P", _plane_tensor(P.view(np.int64).copy()))
        self.register_buffer("S", _plane_tensor(S.view(np.int64).copy()))
        self.register_buffer("PT", _plane_tensor(np.empty(0, dtype=np.int64)), persistent=False)
        self.register_buffer("ST", _plane_tensor(np.empty(0, dtype=np.int64)), persistent=False)
        self._rebuild_transpose()

    def _rebuild_transpose(self):
        with torch.inference_mode(False):
            # load_state_dict(assign=True) may supply inference tensors as well.
            if torch.is_inference(self.P):
                self.P = self.P.clone()
            if torch.is_inference(self.S):
                self.S = self.S.clone()
        if self.P.dtype != torch.int64 or self.S.dtype != torch.int64:
            raise ValueError("packed bitplane buffers must have dtype int64")
        P = self.P.detach().cpu().numpy().view(np.uint64)
        S = self.S.detach().cpu().numpy().view(np.uint64)
        op = TernaryOperator(P, S, (self.out_features, self.in_features))
        from rexgraph.hip_ternary import _operator_planes
        _operator_planes(op)
        # Unpack at most 64 rows at once. A full dense transpose would erase the
        # memory benefit for operators with billions of fixed ternary entries.
        PT = np.zeros((self.in_features, (self.out_features + 63) // 64), dtype=np.uint64)
        ST = np.zeros_like(PT)
        for start in range(0, self.out_features, 64):
            tile = _ternary.unpack(P[start:start + 64], S[start:start + 64], self.in_features)
            p, s, _ = _ternary.pack(np.ascontiguousarray(tile.T))
            PT[:, start // 64] = p[:, 0]
            ST[:, start // 64] = s[:, 0]
        self.PT = _plane_tensor(PT.view(np.int64), self.P.device)
        self.ST = _plane_tensor(ST.view(np.int64), self.P.device)
        self._transpose_version = self._buffer_versions()

    def _buffer_versions(self):
        return tuple((id(t), t._version) for t in (self.P, self.S, self.PT, self.ST))

    def _apply(self, fn, *args, **kwargs):
        valid = self._buffer_versions() == self._transpose_version
        with torch.inference_mode(False):
            result = super()._apply(fn, *args, **kwargs)
        if valid:
            self._transpose_version = self._buffer_versions()
        else:
            self._rebuild_transpose()
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in (prefix + "P", prefix + "S"):
            if key in state_dict and state_dict[key].dtype != torch.int64:
                raise RuntimeError("packed state requires int64 bitplanes")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._rebuild_transpose()  # never restore stale/independently supplied gradients

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"last feature axis must have length {self.in_features}")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("packed training input must be a real floating tensor")
        if x.device != self.P.device or x.device != self.S.device:
            raise ValueError("packed operator and input must be on the same device")
        if self._buffer_versions() != self._transpose_version:
            self._rebuild_transpose()
        backend = ("cpu" if x.device.type == "cpu" else "hip") if self.backend == "auto" else self.backend
        dtype = x.dtype
        if dtype not in (torch.float32, torch.float64):
            x = x.float()
        return _PackedProduct.apply(x, self.P, self.S, self.PT, self.ST,
                                    self.out_features, self.in_features,
                                    backend, self.block, self.threads).to(dtype=dtype)
