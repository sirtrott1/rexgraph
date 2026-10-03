"""Copies expose checked identities through the one transfer seam."""
from contextlib import closing
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib

import numpy as np
import pytest

from rcdb import CopyReceipt, FileStore, LocalStore, MemoryStore, ObjectStore, RexStore, SQLStore, copy_record, migrate
from rcdb.transfer import RECEIPT_MAGIC
from rexgraph import RexGraph, TemporalRex
from rexgraph.object_identity import object_digest
from rexgraph.value_codec import pack_value


def receipt():
    return CopyReceipt("1"*32, "literal/id@1", 2**53+1, "2"*64,
                       "3"*32, "other/\u03b1@2", 2**53+2, "4"*64)


def graph(n=2):
    return RexGraph.from_graph(list(range(n-1)), list(range(1, n)), w_E=[Fraction(2, 7)]*(n-1))


def test_copy_receipt_is_owned_closed_and_binary_exact():
    source = receipt()
    assert CopyReceipt.from_bytes(source.to_bytes()) == source
    record = source.as_record()
    record["source_version"] = 0
    assert source.source_version == 2**53+1
    with pytest.raises(FrozenInstanceError):
        source.source_version = 1
    assert replace(source, destination_version=1).digest != source.digest
    assert replace(source, destination_store_id="5"*32).digest != source.digest
    assert source.digest == source.to_bytes()[5:37].hex()


@pytest.mark.parametrize("damage", ["extra", "missing", "format-bool", "format-future", "version-bool",
                                   "version-float", "version-zero", "version-overflow", "owner", "digest", "id"])
def test_receipt_rejects_unknown_or_ambiguous_native_declarations(damage):
    record = receipt().as_record()
    if damage == "extra": record["provider"] = "untrusted.module"
    elif damage == "missing": record.pop("source_digest")
    elif damage == "format-bool": record["receipt_version"] = True
    elif damage == "format-future": record["receipt_version"] = 2
    elif damage == "version-bool": record["source_version"] = True
    elif damage == "version-float": record["source_version"] = 1.0
    elif damage == "version-zero": record["destination_version"] = 0
    elif damage == "version-overflow": record["destination_version"] = 2**63
    elif damage == "owner": record["source_store_id"] = "G"*32
    elif damage == "digest": record["destination_digest"] = "4"*63
    else: record["source_record_id"] = ""
    with pytest.raises(ValueError):
        CopyReceipt.from_record(record)
    body = pack_value(record)
    raw = RECEIPT_MAGIC+hashlib.sha256(b"rexgraph-copy-receipt\x00"+body).digest()+body
    with pytest.raises(ValueError):
        CopyReceipt.from_bytes(raw)


@pytest.mark.parametrize("damage", ["type", "short", "truncated", "hash", "trailing", "domain"])
def test_receipt_frame_checks_type_size_domain_and_complete_body(damage):
    raw = receipt().to_bytes()
    if damage == "type": raw = 64
    elif damage == "short": raw = RECEIPT_MAGIC
    elif damage == "truncated": raw = raw[:-1]
    elif damage == "hash": raw = raw[:5]+b"0"*32+raw[37:]
    elif damage == "trailing":
        body = raw[37:]+b"extra"
        raw = RECEIPT_MAGIC+hashlib.sha256(b"rexgraph-copy-receipt\x00"+body).digest()+body
    else: raw = RECEIPT_MAGIC+hashlib.sha256(raw[37:]).digest()+raw[37:]
    with pytest.raises(ValueError):
        CopyReceipt.from_bytes(raw)


@pytest.fixture(params=["memory", "local", "file", "rex", "sql", "legacy-sql", "object"])
def source(request, tmp_path):
    kind = request.param
    if kind == "memory": value = MemoryStore()
    elif kind == "local": value = LocalStore(tmp_path / kind)
    elif kind in {"file", "rex"}: value = (FileStore if kind == "file" else RexStore)(str(tmp_path / kind), read_only=False)
    elif kind in {"sql", "legacy-sql"}:
        pytest.importorskip("sqlalchemy")
        value = SQLStore(f"sqlite:///{tmp_path / (kind+'.sqlite')}", native=False if kind == "legacy-sql" else None)
    else:
        pytest.importorskip("fsspec")
        value = ObjectStore(f"file://{tmp_path / 'objects'}", read_only=False)
    with closing(value):
        yield value


def test_copy_receipt_names_actual_historical_source_and_new_destination_address(source):
    old = source.put("literal/id@1", graph(), meta={"exact": Fraction(1, 7)}, tags=["old"],
                     valid_from=1.0, valid_to=100.0, analytics=False)
    source.put(old.id, graph(3), tags=["new"], analytics=False)
    with closing(MemoryStore()) as target:
        target.put(old.id, graph(4), analytics=False)
        copied = copy_record(source, target, old, return_receipt=True)
        assert isinstance(copied, CopyReceipt)
        assert (copied.source_store_id, copied.source_record_id, copied.source_version) == (source.store_id, old.id, 1)
        assert (copied.destination_store_id, copied.destination_record_id, copied.destination_version) == (target.store_id, old.id, 2)
        assert copied.source_digest == object_digest(source.get_version(old.id, 1))
        assert copied.destination_digest == object_digest(target.get_version(old.id, 2))
        assert copied.source_digest == copied.destination_digest
        row = target.get_record(old.id)
        assert row.meta["exact"] == Fraction(1, 7)
        assert row.signature["tags"] == ["old"]
        assert (row.valid_from, row.valid_to) == (1.0, 100.0)
        assert CopyReceipt.from_bytes(copied.to_bytes()) == copied


def test_migration_report_maps_each_successful_version_through_copy_record(source, monkeypatch):
    import rcdb.core as core
    for value in (graph(), graph(3)):
        source.put("r", value, analytics=False)
    seen = []
    original = core.copy_record
    def copied(src, dst, record, **kwargs):
        seen.append((record.id, record.version))
        return original(src, dst, record, **kwargs)
    monkeypatch.setattr(core, "copy_record", copied)
    with closing(MemoryStore()) as target:
        target.put("r", graph(4), analytics=False)
        report = migrate(source, target)
        assert (report["records"], report["versions"]) == (1, 2)
        assert seen == [("r", 1), ("r", 2)]
        mappings = [CopyReceipt.from_record(value) for value in report["receipts"]]
        assert [(value.source_version, value.destination_version) for value in mappings] == [(1, 2), (2, 3)]
        assert all(value.source_store_id == source.store_id and value.destination_store_id == target.store_id for value in mappings)
        assert all(value.source_digest == value.destination_digest for value in mappings)
        assert tuple(mappings) == tuple(CopyReceipt.from_record(v) for v in report["receipts"])
        pack_value(report)  # The report carries native declarations, not executable Python objects.


def test_copy_receipt_keeps_temporal_core_identity():
    value = TemporalRex([(np.array([0]), np.array([1])), (np.array([0, 1]), np.array([1, 2]))])
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        record = source.put("time", value, analytics=False)
        copied = copy_record(source, target, record, return_receipt=True)
        assert copied.source_digest == copied.destination_digest == object_digest(value)


def test_copy_receipt_keeps_fresh_destination_governance():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        record = source.commit_mutation("r", graph(), analytics=False, tx_time=10.0)
        target.configure_security(require_commits=True)
        copied = copy_record(source, target, record, return_receipt=True)
        assert copied.source_digest == copied.destination_digest == object_digest(graph())
        assert target.verify_commits("r")
        assert source.commit_history("r")[0].link.digest != target.commit_history("r")[0].link.digest


def test_default_copy_return_and_same_store_migration_refusal_are_explicit():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        record = source.put("r", graph(), analytics=False)
        assert copy_record(source, target, record).version == 1
        with pytest.raises(TypeError, match="return_receipt"):
            copy_record(source, target, record, return_receipt="yes")
        cursor = source.change_cursor
        with pytest.raises(ValueError, match="same logical store"):
            migrate(source, source)
        assert source.change_cursor == cursor and source.next_version("r") == 2


def test_agent_uses_the_rcdb_receipt_class_without_a_second_transfer_contract():
    agent_rcdb = pytest.importorskip("agent.rcdb")
    assert agent_rcdb.CopyReceipt is CopyReceipt
