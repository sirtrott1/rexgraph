"""Complete legacy corruption is refused; only explicit recovery reads a torn prefix."""
import struct

import pytest

from rcdb import ComplexRecord, index


@pytest.mark.parametrize("native", [False, True])
def test_unknown_legacy_operation_is_refused_in_both_readers(native, tmp_path, monkeypatch):
    if native and index._read_frames is None:
        pytest.skip("compiled legacy record reader is not installed")
    if not native:
        monkeypatch.setattr(index, "_read_frames", None)
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {"nV": 2}))
    raw = bytearray(path.read_bytes())
    raw[len(index.LOG_MAGIC)] = 99
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="operation"):
        list(index.log_read(path))


def _layout(raw):
    """Locate protocol fields in a valid fixture for independent byte mutations."""
    at = len(index.LOG_MAGIC)
    fields = {"operation": at, "id_length": at+1, "id": at+5}
    at += 5+struct.unpack_from("<i", raw, at+1)[0]
    fields["presence"] = at
    at += 1+8*len(index.MEASURES)
    fields["offset_count"] = at
    count = struct.unpack_from("<i", raw, at)[0]
    fields["offsets"] = at+4
    at += 4+8*count
    fields["blob_length"] = at
    size = struct.unpack_from("<i", raw, at)[0]
    fields["blob"] = at+4
    at += 4+size
    fields["leaf_count"] = at
    leaves = struct.unpack_from("<i", raw, at)[0]
    at += 4
    fields["leaves"] = []
    for _ in range(leaves):
        length = struct.unpack_from("<i", raw, at+2)[0]
        fields["leaves"].append((at, at+6, at+6+length))
        at += 10+length
    fields["extra_count"] = at
    return fields


@pytest.fixture(params=[False, True], ids=["python", "native"])
def reader(request, monkeypatch):
    if request.param and index._read_frames is None:
        pytest.skip("compiled legacy record reader is not installed")
    if not request.param:
        monkeypatch.setattr(index, "_read_frames", None)
    return index.log_read


@pytest.mark.parametrize("field", ["id_length", "offset_count", "blob_length", "leaf_count", "extra_count"])
def test_negative_legacy_lengths_refuse(reader, field, tmp_path):
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {}), extra=[1, 2])
    raw = bytearray(path.read_bytes())
    struct.pack_into("<i", raw, _layout(raw)[field], -1)
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        list(reader(path))


@pytest.mark.parametrize("field", ["id", "blob"])
def test_invalid_legacy_utf8_refuses(reader, field, tmp_path):
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {}))
    raw = bytearray(path.read_bytes())
    raw[_layout(raw)[field]] = 255
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        list(reader(path))


@pytest.mark.parametrize("flag", [0, 3, 255])
def test_invalid_put_presence_refuses(reader, flag, tmp_path):
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {}))
    raw = bytearray(path.read_bytes())
    raw[_layout(raw)["presence"]] = flag
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="presence"):
        list(reader(path))


@pytest.mark.parametrize("offset", [0, 1, -1])
def test_invalid_string_offsets_refuse(reader, offset, tmp_path):
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {}))
    raw = bytearray(path.read_bytes())
    fields = _layout(raw)
    count = struct.unpack_from("<i", raw, fields["offset_count"])[0]
    position = offset if offset >= 0 else count-1
    struct.pack_into("<q", raw, fields["offsets"]+8*position, -1)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="offset"):
        list(reader(path))


@pytest.mark.parametrize("mutation", ["scope", "kind", "path_length", "path_flag", "value_count"])
def test_invalid_leaf_descriptors_refuse(reader, mutation, tmp_path):
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {}, meta={"a": [1, "two"]}))
    raw = bytearray(path.read_bytes())
    fields = _layout(raw)
    leaf, flags, values = fields["leaves"][-1]
    if mutation == "scope":
        raw[leaf] = 99
    elif mutation == "kind":
        raw[leaf+1] = 99
    elif mutation == "path_flag":
        raw[flags] = 2
    else:
        struct.pack_into("<i", raw, leaf+2 if mutation == "path_length" else values, -1)
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        list(reader(path))


def test_impossible_term_count_is_not_an_empty_signature(reader, tmp_path):
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {}))
    raw = bytearray(path.read_bytes())
    raw[_layout(raw)["blob"]] = ord("9")
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="term"):
        list(reader(path))


def test_torn_tail_requires_explicit_recovery(reader, tmp_path):
    from rcdb.journal import TornJournalError
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {}))
    boundary = path.stat().st_size
    index.log_append(path, "put", "s", ComplexRecord("s", {}))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(TornJournalError) as failure:
        list(reader(path))
    assert failure.value.valid_end == boundary
    assert [entry[1] for entry in reader(path, allow_torn_tail=True)] == ["r"]


@pytest.mark.parametrize("cursor", [-1, True, 3.5, 8, 10**9])
def test_invalid_resume_cursor_refuses(reader, cursor, tmp_path):
    path = tmp_path / "log"
    index.log_append(path, "put", "r", ComplexRecord("r", {}))
    with pytest.raises(ValueError, match="cursor"):
        list(reader(path, cursor))


def test_unknown_legacy_file_format_refuses(reader, tmp_path):
    path = tmp_path / "log"
    path.write_bytes(b"unknown complete log format")
    with pytest.raises(ValueError, match="format"):
        list(reader(path))
