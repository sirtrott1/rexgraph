"""JSON compatibility never turns corruption into an empty or shorter store."""
import json
import struct

import pytest

from rcdb import ComplexRecord, FileStore, RexStore, ObjectStore
from rcdb.legacy import line_changes, rex_json_entries
from rcdb.journal import TornJournalError
from rexgraph import RexGraph


def entry(layout):
    record = ComplexRecord("r", {}, created=1.0, tx_from=1.0)
    if layout == "line":
        return {"op": "put", "id": "r", "record": record.to_dict()}
    return {"op": "put", **record.to_dict(), "blob_off": 0, "blob_len": 1}


def encoded(layout, raw):
    return raw+b"\n" if layout == "line" else struct.pack("<I", len(raw))+raw


@pytest.mark.parametrize("layout", ["line", "rex-json"])
@pytest.mark.parametrize("mutation", ["operation", "version", "identity", "unknown", "duplicate", "nonfinite", "syntax"])
def test_complete_malformed_json_changes_refuse(layout, mutation, tmp_path):
    change = entry(layout)
    record = change["record"] if layout == "line" else change
    if mutation == "operation":
        change["op"] = "other"
    elif mutation == "version":
        record["version"] = True
    elif mutation == "identity":
        change["id"] = "" if layout == "rex-json" else "other"
    elif mutation == "unknown":
        change["unknown"] = 1
    elif mutation == "nonfinite":
        record["created"] = float("nan")
    raw = json.dumps(change).encode()
    if mutation == "duplicate":
        raw = raw[:-1]+b',"op":"delete"}'
    elif mutation == "syntax":
        raw = b'{"op": invalid}'
    path = tmp_path / "log"
    path.write_bytes(encoded(layout, raw))
    reader = line_changes if layout == "line" else rex_json_entries
    with pytest.raises(ValueError):
        list(reader(path))


@pytest.mark.parametrize("layout", ["line", "rex-json"])
def test_incomplete_physical_json_tail_requires_explicit_recovery(layout, tmp_path):
    path = tmp_path / "log"
    whole = encoded(layout, json.dumps(entry(layout)).encode())
    path.write_bytes(whole+whole[:-3])
    reader = line_changes if layout == "line" else rex_json_entries
    with pytest.raises(TornJournalError) as failure:
        list(reader(path))
    assert failure.value.valid_end == len(whole)
    assert len(list(reader(path, allow_torn_tail=True))) == 1


@pytest.mark.parametrize("kind", ["file", "rex"])
def test_backend_open_does_not_skip_malformed_legacy_json(kind, tmp_path):
    if kind == "file":
        path = tmp_path / "index.log"
        path.write_bytes(b'{"op":"other","id":"r"}\n')
        constructor = FileStore
    else:
        path = tmp_path / "records.log"
        raw = b'{"op":"other","id":"r"}'
        path.write_bytes(struct.pack("<I", len(raw))+raw)
        constructor = RexStore
    with pytest.raises(ValueError, match="operation"):
        constructor(str(tmp_path))


@pytest.mark.parametrize("mutation", ["syntax", "opcode", "address", "gap", "duplicate"])
def test_legacy_object_segments_are_not_skipped_or_aliased(mutation, tmp_path):
    pytest.importorskip("fsspec")
    uri = f"file://{tmp_path}"
    store = ObjectStore(uri, read_only=False)
    store.put("r", RexGraph.from_graph([0], [1]), analytics=False)
    path = tmp_path / "journal" / "000000000001.json"
    store.close()
    raw = path.read_bytes()
    if mutation == "gap":
        path.rename(path.with_name("000000000002.json"))
    elif mutation == "duplicate":
        path.with_name("1.json").write_bytes(raw)
    elif mutation == "syntax":
        path.write_bytes(b'{"op":')
    else:
        change = json.loads(raw)
        if mutation == "opcode":
            change["op"] = "other"
        else:
            change["id"] = "other"
        path.write_text(json.dumps(change))
    with pytest.raises(ValueError):
        ObjectStore(uri)


def test_corrupt_object_base_snapshot_cannot_be_replaced_with_a_later_tail(tmp_path):
    pytest.importorskip("fsspec")
    uri = f"file://{tmp_path}"
    store = ObjectStore(uri, read_only=False)
    value = RexGraph.from_graph([0], [1])
    store.put("base", value, analytics=False)
    store.compact()
    store.put("tail", value, analytics=False)
    store.close()
    path = tmp_path / "index.json"
    raw = path.read_bytes()[:-1]
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        ObjectStore(uri)
    assert path.read_bytes() == raw


def test_stale_object_handle_cannot_overwrite_a_published_sequence(tmp_path):
    from rcdb import PublicationUncertainError
    pytest.importorskip("fsspec")
    uri = f"file://{tmp_path}"
    first, stale = ObjectStore(uri, read_only=False), ObjectStore(uri, read_only=False)
    value = RexGraph.from_graph([0], [1])
    first.put("r", value, analytics=False)
    path = tmp_path / "journal" / "000000000001.json"
    before = path.read_bytes()
    with pytest.raises(PublicationUncertainError, match="sequence already exists"):
        stale.put("s", value, analytics=False)
    assert path.read_bytes() == before
    first.close()
    stale.close()
    reopened = ObjectStore(uri)
    assert reopened.get("r").nE == 1 and reopened.get("s") is None
    reopened.close()
