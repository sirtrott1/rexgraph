"""Canonical closed native values, shared by semantic state and exact results.

No executable object/pickle loading, JSON semantics, numeric inference, or repr
fallback. Maps have canonical key order; list/tuple order and value kinds remain.
"""
from __future__ import annotations

from fractions import Fraction
from math import prod
from numbers import Integral
import struct

import numpy as np

from ._binary import Cursor, integer, uint
from .exact_array import ExactArray
from .value import Absent, Approx, ExactTime, TimeRange

__all__ = ["pack_value", "unpack_value", "pack_spec", "unpack_spec"]
_MAGIC = b"RGVL\x01"
_DEPTH = 64


def _blob(raw):
    return uint(len(raw)) + raw


def _padding_width(array):
    if array.dtype.kind in "fc":
        real = array.real.dtype
        info = np.finfo(real)
        # x87 extended precision occupies ten bytes inside a wider NumPy
        # storage slot. The remaining bytes are allocator padding, not data.
        if info.nmant == 63 and info.iexp == 15 and real.itemsize > 10:
            return real.itemsize
    return 0


def _array_bytes(array):
    raw = array.tobytes()
    width = _padding_width(array)
    if width:
        padded = bytearray(raw)
        slots = np.frombuffer(padded, np.uint8).reshape(-1, width)
        slots[:, 10:] = 0
        raw = bytes(padded)
    return raw


def _encode(value, depth):
    if depth > _DEPTH:
        raise ValueError("native value nesting is too deep")
    if value is Absent:
        return b"A"
    if value is None:
        return b"N"
    if isinstance(value, (bool, np.bool_)):
        return b"T" if value else b"F"
    if isinstance(value, Integral):
        return b"I" + integer(int(value))
    if isinstance(value, Fraction):
        return b"Q" + _blob(ExactArray.from_values(value).to_bytes())
    if isinstance(value, ExactArray):
        return b"E" + _blob(value.to_bytes())
    if isinstance(value, Approx):
        return b"P" + struct.pack("<d", value.value) + _encode(value.source, depth+1)
    if isinstance(value, ExactTime):
        return b"Z" + _encode(value.seconds, depth+1)
    if isinstance(value, TimeRange):
        return b"R" + _encode(value.start, depth+1) + _encode(value.end, depth+1)
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("native real values must be finite")
        return b"D" + struct.pack("<d", value)
    if isinstance(value, str):
        return b"S" + _blob(value.encode("utf-8"))
    if isinstance(value, bytes):
        return b"B" + _blob(value)
    if isinstance(value, np.generic):
        return b"G" + _encode(np.asarray(value), depth+1)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return b"O" + _blob(ExactArray.from_values(value).to_bytes())
        if value.dtype.fields is not None or value.dtype.kind not in "biufcSU":
            raise TypeError("unsupported native tensor dtype")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError("native tensors require finite values")
        dtype = value.dtype.newbyteorder("<")
        array = np.ascontiguousarray(value.astype(dtype, copy=False))
        shape = uint(value.ndim) + b"".join(uint(int(n)) for n in value.shape)
        return b"Y" + _blob(dtype.str.encode("ascii")) + shape + _blob(_array_bytes(array))
    if isinstance(value, (tuple, list)):
        return (b"L" if isinstance(value, list) else b"U") + uint(len(value)) + b"".join(_encode(v, depth+1) for v in value)
    if isinstance(value, dict):
        pairs = sorted((_encode(k, depth+1), _encode(v, depth+1)) for k, v in value.items())
        return b"M" + uint(len(pairs)) + b"".join(k + v for k, v in pairs)
    raise TypeError(f"unsupported native value type {type(value).__name__}")


def _decode(c, depth):
    if depth > _DEPTH:
        raise ValueError("native value nesting is too deep")
    tag = c.take(1)
    if tag in (b"A", b"N", b"T", b"F"):
        return {b"A": Absent, b"N": None, b"T": True, b"F": False}[tag]
    if tag == b"I":
        return c.integer()
    if tag in (b"Q", b"E", b"O"):
        exact = ExactArray.from_bytes(c.take(c.uint()))
        if tag == b"Q":
            if exact.shape != () or not exact.presence.item() or exact.kind.item() != 1:
                raise ValueError("invalid rational scalar")
            return exact.values().item()
        return exact if tag == b"E" else exact.values()
    if tag == b"P":
        return Approx(struct.unpack("<d", c.take(8))[0], _decode(c, depth+1))
    if tag == b"Z":
        value = _decode(c, depth+1)
        if not isinstance(value, Fraction):
            raise ValueError("timestamp requires a rational carrier")
        return ExactTime(value)
    if tag == b"R":
        return TimeRange(_decode(c, depth+1), _decode(c, depth+1))
    if tag == b"D":
        value = struct.unpack("<d", c.take(8))[0]
        if not np.isfinite(value):
            raise ValueError("native real values must be finite")
        return value
    if tag in (b"S", b"B"):
        raw = c.take(c.uint())
        return raw.decode("utf-8") if tag == b"S" else raw
    if tag == b"G":
        array = _decode(c, depth+1)
        if not isinstance(array, np.ndarray) or array.shape != ():
            raise ValueError("invalid NumPy scalar")
        return array[()]
    if tag == b"Y":
        text = c.take(c.uint()).decode("ascii")
        dtype = np.dtype(text)
        if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufcSU" or dtype.str != dtype.newbyteorder("<").str or dtype.str != text:
            raise ValueError("unsupported or noncanonical native tensor dtype")
        ndim = c.uint()
        if ndim > 64:
            raise ValueError("native tensor has too many dimensions")
        shape = tuple(c.uint() for _ in range(ndim))
        raw = c.take(c.uint())
        if prod(shape) * dtype.itemsize != len(raw):
            raise ValueError("native tensor shape does not match its bytes")
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
        if dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError("native tensor requires finite values")
        if dtype.kind == "b" and any(byte > 1 for byte in raw):
            raise ValueError("noncanonical boolean tensor")
        if _padding_width(array) and _array_bytes(array) != raw:
            raise ValueError("noncanonical extended-float tensor padding")
        return array
    if tag in (b"L", b"U", b"M"):
        count = c.uint()
        if count > c.remaining // (2 if tag == b"M" else 1):
            raise ValueError("truncated native collection")
        if tag != b"M":
            values = [_decode(c, depth+1) for _ in range(count)]
            return values if tag == b"L" else tuple(values)
        result, prior = {}, None
        for _ in range(count):
            start = c.pos
            key = _decode(c, depth+1)
            raw = c.data[start:c.pos]
            if prior is not None and raw <= prior:
                raise ValueError("noncanonical native map key order")
            try:
                if key in result:
                    raise ValueError("duplicate native map key")
                result[key] = _decode(c, depth+1)
            except TypeError as exc:
                raise ValueError("native map key must be hashable") from exc
            prior = raw
        return result
    raise ValueError(f"unknown native value tag {tag!r}")


def pack_value(value) -> bytes:
    return _MAGIC + _encode(value, 0)


def unpack_value(data: bytes):
    c = Cursor(data)
    if c.take(len(_MAGIC)) != _MAGIC:
        raise ValueError("unsupported native value format")
    value = _decode(c, 0)
    c.finish()
    return value


def pack_spec(spec, tensors, *, native=False):
    """Component schema adapter; native writes also declare their exact tensor set."""
    if native:
        spec = dict(spec, tensor_names=sorted(tensors))
        raw = pack_value(spec)
    else:
        import json
        raw = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return np.frombuffer(raw, np.uint8).copy()


def unpack_spec(tensors):
    raw = np.asarray(tensors["spec"])
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("invalid component schema tensor")
    data = raw.tobytes()
    if data.startswith(_MAGIC):
        spec = unpack_value(data)
        if not isinstance(spec, dict) or set(spec.get("tensor_names", [])) != set(tensors) - {"spec"}:
            raise ValueError("unclaimed or missing component tensor")
        return spec
    import json
    return json.loads(data.decode("utf-8"))
