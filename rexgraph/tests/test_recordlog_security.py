"""The optional native compatibility scanner validates before pointer arithmetic."""
import struct

import pytest


@pytest.mark.parametrize("width", [-1, 0, 4097, 2**31-1])
def test_invalid_scalar_width_is_refused(width):
    codec = pytest.importorskip("rexgraph.core._recordlog")
    with pytest.raises(ValueError, match="width"):
        codec.read_frames(b"", 0, width)


@pytest.mark.parametrize("start", [-1, 2])
def test_invalid_start_offset_is_refused(start):
    codec = pytest.importorskip("rexgraph.core._recordlog")
    with pytest.raises(ValueError, match="offset"):
        codec.read_frames(b"x", start, 20)


@pytest.mark.parametrize("raw", [b"\x63", b"\x02"+struct.pack("<i", -1),
                                b"\x02"+struct.pack("<i", 1)+b"\xff\x00",
                                b"\x02"+struct.pack("<i", 1)+b"r\x01"])
def test_invalid_protocol_fields_are_never_sanitized(raw):
    codec = pytest.importorskip("rexgraph.core._recordlog")
    with pytest.raises(ValueError):
        codec.read_frames(raw, 0, 20)


def test_native_scanner_reports_exact_complete_prefix_boundaries():
    codec = pytest.importorskip("rexgraph.core._recordlog")
    frame = b"\x02"+struct.pack("<i", 1)+b"r\x00"
    entries, valid_end = codec.read_frames(frame+frame[:-1], 0, 20, return_offsets=True)
    assert valid_end == len(frame) and len(entries) == 1
    assert entries[0][0] == valid_end and entries[0][1][:2] == (2, "r")
