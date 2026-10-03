"""The same declared metadata survives every store, replay and compaction."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import Absent, Approx, ExactArray, ExactTime, RexGraph
from rexgraph.value_codec import pack_value
from rcdb import FileStore, LocalStore, MemoryStore, ObjectStore, RexStore, SQLStore
from rcdb.core import ComplexRecord
from rcdb import index
from rcdb.envelope import dumps_mapping, loads_mapping


def metadata():
    return {"rational": Q(1, 7), "absent": Absent, "integer": 2**1000+1,
            "array": np.array([[1, 2], [3, 4]], np.int16), "scalar": np.float32(.5),
            "values": [1, .5, Q(2, 7), Absent, "00123"],
            "tuple": (Q(3, 7), []), "keys": {1: "integer", "1": "string"},
            "exact_array": ExactArray.from_values([Q(1, 7), Absent, 2**1000]),
            "approximate": Approx(.1, "sensor"), "time": ExactTime(Q(123, 7)),
            "native_name": {"rexgraph.record-value": {"user": "data"}}}


def opened(kind, tmp_path):
    if kind == "memory":
        return MemoryStore()
    if kind == "local":
        return LocalStore(tmp_path / "local")
    if kind == "file":
        return FileStore(str(tmp_path / "files"), read_only=False)
    if kind == "rex":
        return RexStore(str(tmp_path / "rex"), read_only=False)
    if kind == "sql":
        pytest.importorskip("sqlalchemy")
        return SQLStore(f"sqlite:///{tmp_path / 'sql.db'}")
    pytest.importorskip("fsspec")
    return ObjectStore(f"file://{tmp_path / 'objects'}", read_only=False)


@pytest.mark.parametrize("kind", ["memory", "file", "rex", "sql", "object"])
def test_all_stores_preserve_closed_metadata_after_reopen(kind, tmp_path):
    store = opened(kind, tmp_path)
    original = metadata()
    expected = pack_value(original)
    try:
        for record_id in ["first", "second"]:
            store.put(record_id, RexGraph.from_graph([0], [1]), meta=original, analytics=False)
        original["array"][0, 0] = 777
        assert pack_value(store.read_record("first").record.meta) == expected
        if kind != "memory":
            store.close()
            store = opened(kind, tmp_path)
        for record_id in ["first", "second"]:
            assert pack_value(store.read_record(record_id).record.meta) == expected
    finally:
        store.close()


@pytest.mark.parametrize("kind", ["memory", "file", "rex", "sql", "object"])
@pytest.mark.parametrize("bad", [object(), float("nan"), float("inf")])
def test_unsupported_metadata_is_refused_before_publication(kind, bad, tmp_path):
    store = opened(kind, tmp_path)
    try:
        store.put("record", RexGraph.from_graph([0], [1]), meta={"old": True}, analytics=False)
        with pytest.raises((TypeError, ValueError), match="unsupported|finite"):
            store.put("record", RexGraph.from_graph([0, 1], [1, 2]), meta={"invalid": bad}, analytics=False)
        assert store.read_record("record").record.version == 1
        assert store.read_record("record").record.meta == {"old": True}
    finally:
        store.close()


def test_native_index_shared_and_bridge_values_preserve_metadata(tmp_path):
    records = [(name, ComplexRecord(name, {"nV": 2, "tags": ["a"]}, meta=metadata())) for name in ["a", "b"]]
    built = index.build(records)
    path = tmp_path / "index.safetensors"
    index.write(path, built)
    restored = index.read(path)
    for row in range(2):
        assert pack_value(index.record_at(restored, row).meta) == pack_value(metadata())


@pytest.mark.parametrize("kind", ["memory", "file", "rex", "sql", "object"])
def test_public_record_results_never_mutate_stored_metadata(kind, tmp_path):
    store = opened(kind, tmp_path)
    try:
        expected = pack_value(metadata())
        returned = store.put("record", RexGraph.from_graph([0], [1]), meta=metadata(), analytics=False)
        for record in [returned, store.get_record("record"), store.history("record")[0],
                       store.list()[0], store.query()[0]]:
            record.meta["array"] = record.meta["array"].copy()
            record.meta["array"][0, 0] = 777
            record.meta["values"][0] = 777
            record.meta["new"] = "caller-owned"
            record.signature["nE"] = 777
            record.tx_to = 0
        current = store.get_record("record")
        assert pack_value(current.meta) == expected
        assert current.signature["nE"] == 1 and current.tx_to is None
        if kind != "memory":
            store.close()
            store = opened(kind, tmp_path)
            assert pack_value(store.get_record("record").meta) == expected
            assert store.get("record").nE == 1
    finally:
        store.close()


@pytest.mark.parametrize("meta", [metadata(), {"vertex_labels": ["a", "b"]},
                                  {"rexgraph.record-value": {"user": "data"}},
                                  {"number": 2**60+1}])
def test_http_record_envelope_preserves_values_and_plain_json_fields(meta):
    import json
    original = ComplexRecord("record", {"nV": 2}, meta=meta)
    response = json.loads(json.dumps(original.to_wire_dict(), allow_nan=False))
    restored = ComplexRecord.from_dict(response)
    assert response["signature"] == {"nV": 2}
    assert pack_value(restored.meta) == pack_value(meta)
    if meta == {"vertex_labels": ["a", "b"]}:
        assert response["meta"] == meta and "record_values_version" not in response


@pytest.mark.parametrize("compiled", [False, True])
def test_native_log_python_and_compiled_readers_agree(tmp_path, monkeypatch, compiled):
    if not compiled:
        monkeypatch.setattr(index, "_read_frames", None)
    path = tmp_path / "records.log"
    original = ComplexRecord("fixture", {}, meta=metadata())
    index.log_append(path, "put", original.id, original)
    entries = list(index.log_read(path))
    assert len(entries) == 1
    assert pack_value(entries[0][2].meta) == pack_value(metadata())


def test_json_adapter_envelope_detects_tampering_and_preserves_legacy_keys():
    original = metadata()
    encoded = dumps_mapping(original)
    assert pack_value(loads_mapping(encoded)) == pack_value(original)
    assert loads_mapping('{"rexgraph.record-value": {"user": "data"}}') == {"rexgraph.record-value": {"user": "data"}}
    record = ComplexRecord("fixture", {}, meta=original).to_storage_dict()
    record["meta"]["rexgraph.record-value"]["digest"] = "0"*64
    with pytest.raises(ValueError, match="digest mismatch"):
        ComplexRecord.from_dict(record)


@pytest.mark.parametrize("kind", ["memory", "local", "file", "rex", "sql", "object"])
def test_logical_state_hash_preserves_full_native_metadata_across_providers(kind, tmp_path):
    from contextlib import closing
    with closing(opened(kind, tmp_path)) as store, closing(MemoryStore()) as reference:
        for target in (store, reference):
            target.put("r", RexGraph.from_graph([0], [1]), meta=metadata(),
                       analytics=False, _tx_time=10.0)
        assert store.state_digest() == reference.state_digest()
        assert store.state_manifest()["version"] == 2


@pytest.mark.parametrize("left,right", [
    (Q(1, 2), .5), (2**53+1, float(2**53+1)), ((1, 2), [1, 2]),
    (np.array([1, 2], np.int16), np.array([1, 2], np.int32)),
    (np.array([1, 2], np.int16), np.array([[1, 2]], np.int16)),
])
def test_logical_state_hash_distinguishes_exact_metadata_types_and_array_shapes(left, right):
    from contextlib import closing
    digests = []
    for value in (left, right):
        with closing(MemoryStore()) as store:
            store.put("r", RexGraph.from_graph([0], [1]), meta={"value": value},
                       analytics=False, _tx_time=10.0)
            digests.append(store.state_digest())
    assert digests[0] != digests[1]


@pytest.mark.parametrize("kind", ["memory", "local", "sql"])
@pytest.mark.parametrize("failure", ["mode", "fields", "signer"])
def test_failed_security_configuration_preserves_the_entire_previous_policy(kind, failure, tmp_path):
    from contextlib import closing
    class BrokenFields:
        def __iter__(self):
            raise ValueError("failed fields")
    class BrokenSigner:
        def verifier(self):
            raise ValueError("failed signer")
    invalid = {"mode": {"signature_mode": "unknown"},
               "fields": {"metadata_fields": BrokenFields()},
               "signer": {"transition_signer": BrokenSigner()}}[failure]
    with closing(opened(kind, tmp_path)) as store:
        store.configure_security(require_commits=True, metadata_fields=["kept"], signature_mode="structural")
        before = store.transfer_policy_digest()
        with pytest.raises(ValueError): store.configure_security(key_id="other", **invalid)
        assert store.transfer_policy_digest() == before
        with pytest.raises(PermissionError, match="mutation commits"):
            store.put("r", RexGraph.from_graph([0], [1]), analytics=False)


@pytest.mark.parametrize("kind", ["memory", "local", "sql"])
@pytest.mark.parametrize("fields,expected", [([], {}), (["kept"], {"kept": Q(1, 7)})])
def test_explicit_metadata_projection_applies_in_public_signature_mode(kind, fields, expected, tmp_path):
    from contextlib import closing
    with closing(opened(kind, tmp_path)) as store:
        store.configure_security(metadata_fields=fields)
        store.put("r", RexGraph.from_graph([0], [1]), meta={"kept": Q(1, 7), "drop": "private"},
                  analytics=False)
        assert store.get_record("r").meta == expected
