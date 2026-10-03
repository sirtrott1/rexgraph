"""Store identity and explicit codec choices are sealed together, independently of installs."""
from dataclasses import replace
import hashlib

import pytest

from rexgraph._binary import uint
from rcdb import core
from rcdb.header import BlobCodecSpec, CodecRef, HEADER_MAGIC, StoreHeader
from rcdb.store_identity import StoreIdentity
from rexgraph.value_codec import pack_value


def sample():
    return StoreHeader(StoreIdentity("1"*32, "local"), BlobCodecSpec("zlib"),
                       (CodecRef("rexgraph.safetensors", 1), CodecRef("declared.provenance", 3)))


def test_header_roundtrip_seals_exact_ordered_configuration_and_owns_fields():
    header = sample()
    restored = StoreHeader.from_bytes(header.to_bytes())
    assert restored == header and restored.digest == header.digest
    assert restored.compression.level == 6
    declaration = header.as_record()
    declaration["identity"]["id"] = "2"*32
    declaration["record_codecs"][0]["name"] = "other"
    assert header.identity.id == "1"*32 and header.record_codecs[0].name == "rexgraph.safetensors"
    assert replace(header, record_codecs=header.record_codecs[::-1]).digest != header.digest


@pytest.mark.parametrize("field", ["header_version", "engine_version", "journal_version", "envelope_version"])
@pytest.mark.parametrize("value", [True, 2])
def test_unsupported_version_is_not_valid_after_reframing(field, value):
    record = sample().as_record()
    record[field] = value
    body = pack_value(record)
    raw = HEADER_MAGIC+uint(len(body))+hashlib.sha256(b"rexgraph-store-header\x00"+body).digest()+body
    with pytest.raises(ValueError, match="unsupported"):
        StoreHeader.from_bytes(raw)


@pytest.mark.parametrize("mutation", ["unknown", "identity", "codec_fields", "codec_defaults", "duplicates", "list"])
def test_closed_header_fields_and_native_positions_refuse_ambiguity(mutation):
    record = sample().as_record()
    if mutation == "unknown":
        record["unknown"] = 1
    elif mutation == "identity":
        record["identity"]["path"] = "/not-an-identity"
    elif mutation == "codec_fields":
        record["compression"]["unknown"] = 1
    elif mutation == "codec_defaults":
        record["compression"]["level"] = None
    elif mutation == "duplicates":
        record["record_codecs"] *= 2
    else:
        record["record_codecs"] = list(record["record_codecs"])
    with pytest.raises(ValueError):
        StoreHeader.from_record(record)


def test_corrupt_short_or_trailing_header_bytes_refuse():
    raw = sample().to_bytes()
    for bad in (raw[:-1], raw+b"extra", raw[:-1]+bytes([raw[-1]^1])):
        with pytest.raises(ValueError):
            StoreHeader.from_bytes(bad)


def test_codec_reference_requires_explicit_ownership_before_provider_resolution():
    header = sample()
    header.check_record_codec("rexgraph.safetensors", 1)
    with pytest.raises(ValueError, match="not declared"):
        header.check_record_codec("rexgraph.safetensors", 2)
    # A declaration is data, not an import instruction.
    unknown = StoreHeader(header.identity, record_codecs=(CodecRef("never.import.this.module", 1),))
    assert StoreHeader.from_bytes(unknown.to_bytes()) == unknown


@pytest.mark.parametrize("spec", [BlobCodecSpec(), BlobCodecSpec("zlib", 0), BlobCodecSpec("zlib", 6)])
def test_explicit_codec_never_asks_the_ambient_selector(spec, monkeypatch):
    def forbidden():
        raise AssertionError("ambient optional-install codec choice was consulted")
    monkeypatch.setattr(core, "_codec", forbidden)
    raw = b"native-payload"*100
    assert spec.decode(spec.encode(raw)) == raw


def test_identity_codec_preserves_literal_bytes_even_if_they_resemble_legacy_magic():
    codec = BlobCodecSpec()
    raw = b"RXZ1sopaque-native-bytes"
    assert codec.decode(codec.encode(raw)) == raw


def test_explicit_zstd_roundtrips_and_refuses_a_zlib_frame(monkeypatch):
    pytest.importorskip("zstandard")
    monkeypatch.setattr(core, "_codec", lambda: pytest.fail("ambient codec choice"))
    raw = b"native-payload"*100
    codec = BlobCodecSpec("zstd")
    assert codec.level == 3 and codec.decode(codec.encode(raw)) == raw
    with pytest.raises(ValueError, match="store header"):
        codec.decode(BlobCodecSpec("zlib").encode(raw))
    with pytest.raises(ValueError, match="store header"):
        BlobCodecSpec("zlib").decode(codec.encode(raw))


@pytest.mark.parametrize("kwargs", [{"name": "other"}, {"version": True}, {"name": "none", "level": 1},
                                  {"name": "zlib", "level": -1}, {"name": "zlib", "level": 10},
                                  {"name": "zlib", "level": True}, {"name": "zstd", "level": .5}])
def test_invalid_codec_configuration_refuses(kwargs):
    with pytest.raises(ValueError):
        BlobCodecSpec(**kwargs)


def test_encoded_blob_limit_also_covers_compression_overhead(monkeypatch):
    import rcdb.header as module
    monkeypatch.setattr(module, "_RECORD_PAYLOAD_LIMIT", 64)
    with pytest.raises(ValueError, match="encoded.*limit"):
        BlobCodecSpec("zlib").encode(bytes(range(64)))
    with pytest.raises(ValueError, match="bounded"):
        BlobCodecSpec().encode(b"x"*65)


def test_literal_codec_reference_fields_are_bounded_and_exact():
    for name, version in (("", 1), ("x\x00y", 1), ("x"*257, 1), ("x", True), ("x", 0)):
        with pytest.raises(ValueError):
            CodecRef(name, version)
