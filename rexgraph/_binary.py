"""Canonical integer framing shared by native value and exact array codecs."""
from __future__ import annotations


def uint(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ValueError("unsigned frame requires a nonnegative integer")
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def integer(value: int) -> bytes:
    if type(value) is not int:
        raise TypeError("integer frame requires an integer")
    size = max(1, (value.bit_length() + 8) // 8)
    raw = value.to_bytes(size, "little", signed=True)
    while len(raw) > 1 and ((raw[-1] == 0 and raw[-2] < 128)
                           or (raw[-1] == 255 and raw[-2] >= 128)):
        raw = raw[:-1]
    return uint(len(raw)) + raw


class Cursor:
    def __init__(self, data: bytes):
        if not isinstance(data, bytes):
            raise TypeError("native record must be bytes")
        self.data = data
        self.pos = 0

    @property
    def remaining(self):
        return len(self.data) - self.pos

    def take(self, size: int) -> bytes:
        if size < 0 or size > self.remaining:
            raise ValueError("truncated native record")
        start = self.pos
        self.pos += size
        return self.data[start:self.pos]

    def uint(self) -> int:
        start = self.pos
        value = shift = 0
        for _ in range(10):
            byte = self.take(1)[0]
            value |= (byte & 127) << shift
            if byte < 128:
                if value > (1 << 63) - 1 or self.data[start:self.pos] != uint(value):
                    raise ValueError("noncanonical or oversized unsigned frame")
                return value
            shift += 7
        raise ValueError("oversized unsigned frame")

    def integer(self) -> int:
        size = self.uint()
        raw = self.take(size)
        if not raw:
            raise ValueError("empty integer frame")
        value = int.from_bytes(raw, "little", signed=True)
        if integer(value) != uint(size) + raw:
            raise ValueError("noncanonical integer frame")
        return value

    def finish(self):
        if self.remaining:
            raise ValueError("trailing bytes in native record")
