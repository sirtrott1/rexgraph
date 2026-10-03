"""Bounded, complete, single frame compressed record payloads."""
import zlib

import pytest
from rcdb.core import decompress_blob


@pytest.mark.parametrize("suffix", [b"trailing", zlib.compress(b"second")])
def test_zlib_refuses_trailing_data_or_a_second_frame(suffix):
    with pytest.raises(ValueError, match="trailing"):
        decompress_blob(b"RXZ1z" + zlib.compress(b"first") + suffix)


def test_zlib_and_legacy_raw_payloads_obey_the_output_limit():
    with pytest.raises(ValueError, match="limit"):
        decompress_blob(b"RXZ1z" + zlib.compress(b"x"*1000), max_output_bytes=100)
    with pytest.raises(ValueError, match="limit"):
        decompress_blob(b"x"*101, max_output_bytes=100)
    with pytest.raises(ValueError, match="incomplete"):
        decompress_blob(b"RXZ1z" + zlib.compress(b"first")[:-1])


@pytest.mark.parametrize("declared", [True, False])
def test_zstd_size_complete_frame_and_output_limits(declared):
    zstd = pytest.importorskip("zstandard")
    compressor = zstd.ZstdCompressor(write_content_size=declared)
    payload = compressor.compress(b"first")
    assert decompress_blob(b"RXZ1s"+payload, max_output_bytes=100) == b"first"
    for body in (payload[:-1], payload+b"trailing", payload+payload):
        with pytest.raises(ValueError):
            decompress_blob(b"RXZ1s"+body, max_output_bytes=100)
    with pytest.raises(ValueError, match="limit"):
        decompress_blob(b"RXZ1s"+compressor.compress(b"x"*1000), max_output_bytes=100)
