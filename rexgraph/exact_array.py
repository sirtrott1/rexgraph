"""One immutable rational/presence carrier, with canonical bigint escape."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod
from numbers import Integral

import numpy as np

from ._binary import Cursor, integer, uint
from .value import Absent, Approx, NumberRule, convert_number

__all__ = ["ExactArray"]
_MIN, _MAX = -(1 << 63), (1 << 63) - 1
_MAGIC = b"RGXA\x01"


def _freeze(value, dtype):
    a = np.asarray(value)
    if a.dtype != np.dtype(dtype):
        raise TypeError(f"exact tensor requires dtype {np.dtype(dtype)}")
    return np.frombuffer(a.tobytes(order="C"), dtype=dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class ExactArray:
    numerator: np.ndarray
    denominator: np.ndarray
    presence: np.ndarray
    kind: np.ndarray
    bigint: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self):
        for name, dtype in [("numerator", "<i8"), ("denominator", "<i8"), ("presence", "bool"), ("kind", "u1")]:
            object.__setattr__(self, name, _freeze(getattr(self, name), dtype))
        shape = self.numerator.shape
        if any(a.shape != shape for a in (self.denominator, self.presence, self.kind)):
            raise ValueError("exact array tensors must have identical shapes")
        if np.any(self.denominator <= 0) or np.any(self.kind > 1):
            raise ValueError("invalid exact denominator or numeric kind")
        if np.any(np.gcd(self.numerator, self.denominator) != 1):
            raise ValueError("exact coefficients must be reduced")
        if np.any((~self.presence) & ((self.numerator != 0) | (self.denominator != 1) | (self.kind != 0))):
            raise ValueError("absent slots must have canonical zero/one placeholders")
        if np.any((self.kind == 0) & (self.denominator != 1)):
            raise ValueError("integer slots must have denominator one")
        entries = tuple(tuple(entry) for entry in self.bigint)
        prior = -1
        for entry in entries:
            if len(entry) != 3 or any(type(x) is not int for x in entry):
                raise ValueError("invalid bigint escape entry")
            i, n, d = entry
            if not prior < i < self.size or d <= 0 or Fraction(n, d).as_integer_ratio() != (n, d):
                raise ValueError("noncanonical bigint escape")
            if _MIN <= n <= _MAX and d <= _MAX:
                raise ValueError("bigint escape used for fixed-width coefficient")
            if not self.presence.flat[i] or self.numerator.flat[i] != 0 or self.denominator.flat[i] != 1:
                raise ValueError("bigint slot has invalid placeholder")
            if self.kind.flat[i] == 0 and d != 1:
                raise ValueError("integer bigint slot requires denominator one")
            prior = i
        object.__setattr__(self, "bigint", entries)

    @property
    def shape(self):
        return self.numerator.shape

    @property
    def size(self):
        return self.numerator.size

    @classmethod
    def absent(cls, shape):
        """An absent exact carrier without building one Python object per slot."""
        return cls(np.zeros(shape, "<i8"), np.ones(shape, "<i8"),
                   np.zeros(shape, bool), np.zeros(shape, np.uint8))

    @classmethod
    def from_values(cls, values, *, rule=NumberRule.EXACT):
        NumberRule(rule)
        if isinstance(values, np.ndarray) and values.dtype.kind in "iu" and (not values.size or int(values.max()) <= _MAX):
            return cls(np.asarray(values, dtype="<i8"), np.ones(values.shape, "<i8"),
                       np.ones(values.shape, bool), np.zeros(values.shape, np.uint8))
        a = np.asarray(values, dtype=object)
        num, den = np.zeros(a.shape, "<i8"), np.ones(a.shape, "<i8")
        presence, kind = np.ones(a.shape, bool), np.zeros(a.shape, np.uint8)
        big = []
        for i, value in enumerate(a.flat):
            if value is Absent:
                presence.flat[i] = False
                continue
            converted = convert_number(value, rule, context=f"exact array slot {i}")
            if isinstance(converted, Approx):
                raise TypeError("an Approx cannot enter an exact array")
            n, d = converted.as_integer_ratio()
            kind.flat[i] = not (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)))
            if _MIN <= n <= _MAX and d <= _MAX:
                num.flat[i], den.flat[i] = n, d
            else:
                big.append((i, n, d))
        return cls(num, den, presence, kind, tuple(big))

    def values(self):
        result = np.empty(self.shape, dtype=object)
        big = {i: (n, d) for i, n, d in self.bigint}
        for i in range(self.size):
            if not self.presence.flat[i]:
                result.flat[i] = Absent
            else:
                n, d = big.get(i, (int(self.numerator.flat[i]), int(self.denominator.flat[i])))
                result.flat[i] = Fraction(n, d) if self.kind.flat[i] else n
        return result

    def _big_bytes(self):
        return uint(len(self.bigint)) + b"".join(uint(i) + integer(n) + integer(d) for i, n, d in self.bigint)

    @staticmethod
    def _read_big(cursor):
        count = cursor.uint()
        if count > cursor.remaining // 5:
            raise ValueError("truncated bigint escape table")
        return tuple((cursor.uint(), cursor.integer(), cursor.integer()) for _ in range(count))

    def to_tensors(self):
        return {"numerator": self.numerator, "denominator": self.denominator,
                "presence": self.presence, "kind": self.kind,
                "bigint": np.frombuffer(self._big_bytes(), np.uint8)}

    @classmethod
    def from_tensors(cls, tensors):
        if set(tensors) != {"numerator", "denominator", "presence", "kind", "bigint"}:
            raise ValueError("unknown or missing exact array tensors")
        b = np.asarray(tensors["bigint"])
        if b.dtype != np.uint8 or b.ndim != 1:
            raise ValueError("invalid bigint tensor")
        cursor = Cursor(b.tobytes())
        big = cls._read_big(cursor)
        cursor.finish()
        return cls(*(tensors[n] for n in ("numerator", "denominator", "presence", "kind")), big)

    def to_bytes(self):
        shape = uint(len(self.shape)) + b"".join(uint(int(n)) for n in self.shape)
        return (_MAGIC + shape + self.numerator.tobytes() + self.denominator.tobytes()
                + np.packbits(self.presence.ravel(), bitorder="little").tobytes()
                + np.packbits(self.kind.ravel(), bitorder="little").tobytes() + self._big_bytes())

    @classmethod
    def from_bytes(cls, data):
        c = Cursor(data)
        if c.take(len(_MAGIC)) != _MAGIC:
            raise ValueError("unsupported exact array format")
        ndim = c.uint()
        if ndim > 64:
            raise ValueError("exact array has too many dimensions")
        shape = tuple(c.uint() for _ in range(ndim))
        size = prod(shape)
        if size > c.remaining // 16:
            raise ValueError("truncated exact array")
        num = np.frombuffer(c.take(size*8), "<i8").reshape(shape)
        den = np.frombuffer(c.take(size*8), "<i8").reshape(shape)
        bit_arrays = []
        for dtype in (bool, np.uint8):
            packed = c.take((size+7)//8)
            if size % 8 and packed[-1] >> (size % 8):
                raise ValueError("noncanonical bitmap padding")
            bit_arrays.append(np.unpackbits(np.frombuffer(packed, np.uint8), bitorder="little", count=size).astype(dtype).reshape(shape))
        big = cls._read_big(c)
        c.finish()
        return cls(num, den, *bit_arrays, big)

    def __eq__(self, other):
        return isinstance(other, ExactArray) and self.shape == other.shape and self.bigint == other.bigint and all(
            np.array_equal(getattr(self, name), getattr(other, name)) for name in ("numerator", "denominator", "presence", "kind"))
