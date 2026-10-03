"""Exact version snapshots use canonical addresses and the verified decoder."""
from contextlib import closing
from fractions import Fraction

import numpy as np
import pytest

from rcdb import (ComplexRecord, LocalStore, MemoryStore, NativeObjectStore, RCStore,
                  PublicationUncertainError, SQLStore, copy_record, record_packet, serialize_complex,
                  structural_signature)
from rexgraph import RexGraph
from rexgraph.object_identity import object_digest
from rexgraph.value_codec import pack_value


KINDS = ("memory", "local", "sql", "object-file", "object-memory")


def opened(kind, path):
    if kind == "memory": return MemoryStore()
    if kind == "local": return LocalStore(path)
    if kind == "sql":
        pytest.importorskip("sqlalchemy")
        return SQLStore(f"sqlite:///{path}.sqlite")
    pytest.importorskip("fsspec")
    return NativeObjectStore(f"{'file' if kind == 'object-file' else 'memory'}://{path}")


def forbid_history(monkeypatch, store):
    def failed(*args, **kwargs):
        pytest.fail("native exact-version snapshot cloned history")
    monkeypatch.setattr(store, "history", failed)


@pytest.mark.parametrize("kind", KINDS)
def test_retained_exact_values_none_deletion_and_revival_without_history(kind, tmp_path, monkeypatch):
    values = ({"q": Fraction(1, 7), "nested": [2**90, (None, Fraction(3, 11))]}, None,
              {"q": Fraction(5, 7)})
    with closing(opened(kind, tmp_path/"store")) as store:
        for i, value in enumerate(values, 1):
            store.put_record("literal/r@1", value, tx_time=float(i),
                             meta={"q": Fraction(i, 7), "nested": {"kept": [i]}})
        store.delete("literal/r@1", tx_time=4.)
        expected = store.history("literal/r@1")
        forbid_history(monkeypatch, store)
        assert store.read_record("literal/r@1") is None
        for version, value in enumerate(values, 1):
            snapshot = store.read_record("literal/r@1", version=version)
            assert pack_value(snapshot.value) == pack_value(value)
            assert pack_value(snapshot.record.to_dict()) == pack_value(expected[version-1].to_dict())
            snapshot.record.meta["nested"]["kept"].append("outside")
            snapshot.record.version = 999
            if value is not None: snapshot.value["q"] = "outside"
            again = store.read_record("literal/r@1", version=np.int64(version))
            assert pack_value(again.value) == pack_value(value)
            assert again.record.meta["nested"]["kept"] == [version]
            assert again.state_digest == record_packet(store, "literal/r@1", version=version).state_digest
        assert store.read_record("literal/r@1", version=4) is None
        revived = store.put_record("literal/r@1", Fraction(9, 7), tx_time=5.)
        assert revived.version == 4
        assert store.read_record("literal/r@1", version=2).value is None
        assert store.read_record("literal/r@1", version=4).value == Fraction(9, 7)
        assert store.read_record("literal/r@1", version=2**100) is None


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_version_uses_literal_id_even_when_it_looks_like_an_alias(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path/"store")) as store:
        store.put_record("r", Fraction(1, 7), tx_time=1.)
        store.put_record("r", Fraction(2, 7), tx_time=2.)
        assert store.read_record("r@1").record.id == "r"
        assert store.read_record("r@1", version=1) is None
        store.put_record("r@1", Fraction(3, 7), tx_time=3.)
        store.delete("r@1", tx_time=4.)
        forbid_history(monkeypatch, store)
        assert store.read_record("r@1") is None
        snapshot = store.read_record("r@1", version=1)
        assert snapshot.record.id == "r@1" and snapshot.value == Fraction(3, 7)
        assert store.read_record("r", version=1).value == Fraction(1, 7)
        assert store.read_record("r@1", version=2) is None


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_exact_selectors_are_refused_before_lookup(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path/"store")) as store:
        store.put_record("r", None)
        forbid_history(monkeypatch, store)
        for version in (True, False, 0, -1, 1.0, "1", Fraction(1)):
            with pytest.raises((TypeError, ValueError)):
                store.read_record("r", version=version)
        for selector in ({"as_of": 1.}, {"valid_at": Fraction(1, 7)},
                         {"as_of": 1., "valid_at": 1.}):
            with pytest.raises(ValueError): store.read_record("r", version=1, **selector)
        for rid in (None, "", 1):
            with pytest.raises((TypeError, ValueError)): store.read_record(rid, version=1)
        assert store.read_record("missing", version=1) is None
        assert store.read_record("r", version=np.int64(1)).value is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fault", ("missing", "corrupt"))
def test_exact_version_still_checks_selected_physical_payload(kind, fault, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path/"store")) as store:
        store.put_record("r", None, tx_time=1.)
        store.put_record("r", Fraction(2, 7), tx_time=2.)
        forbid_history(monkeypatch, store)
        if kind == "memory":
            if fault == "missing": store._blobs.pop(("r", 1))
            else: store._blobs[("r", 1)] = b"corrupt"
        elif kind == "sql":
            from sqlalchemy import update
            with store.engine.begin() as conn:
                conn.execute(update(store.table).where(store.table.c.id == "r",
                             store.table.c.version == 1).values(blob=None if fault == "missing" else b"corrupt"))
        else:
            original = store._read_blob
            digest = store._state.change_for("r", 1).blob_digest
            def read_blob(address):
                if address != digest: return original(address)
                if fault == "missing": raise FileNotFoundError("removed payload")
                return b"corrupt"
            monkeypatch.setattr(store, "_read_blob", read_blob)
        assert store.read_record("r", version=3) is None
        with pytest.raises(ValueError, match="payload|content|digest"):
            store.read_record("r", version=1)
        assert store.read_record("r", version=2).value == Fraction(2, 7)


@pytest.mark.parametrize("kind", ("local", "sql", "object-file", "object-memory"))
def test_peer_updates_and_reopen_refresh_retained_exact_addresses(kind, tmp_path, monkeypatch):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as first, closing(opened(kind, path)) as second:
        first.put_record("r", Fraction(1, 7), tx_time=1.)
        forbid_history(monkeypatch, second)
        assert second.read_record("r", version=1).record.tx_to is None
        first.put_record("r", Fraction(2, 7), tx_time=2.)
        assert second.read_record("r", version=1).record.tx_to == 2.
        first.delete("r", tx_time=3.)
        assert first.read_record("r", version=2).record.tx_to == 3.
        first.put_record("r", None, tx_time=4.)
        assert second.read_record("r", version=3).value is None
    with closing(opened(kind, path)) as reopened:
        forbid_history(monkeypatch, reopened)
        assert reopened.read_record("r", version=1).value == Fraction(1, 7)
        assert reopened.read_record("r", version=2).record.tx_to == 3.
        assert reopened.read_record("r", version=3).value is None


def test_long_lineage_reads_only_selected_address_and_bounds_metadata_copies(monkeypatch):
    with closing(MemoryStore()) as store:
        for i in range(1024): store.put_record("r", Fraction(i, 7), tx_time=float(i))
        reads = []
        class UnscannableRows(list):
            def __iter__(self): pytest.fail("exact-version snapshot scanned retained history")
            def __reversed__(self): pytest.fail("exact-version snapshot scanned retained history")
            def __getitem__(self, index): reads.append(index); return super().__getitem__(index)
        store._state._rows["r"] = UnscannableRows(store._state._rows["r"])
        forbid_history(monkeypatch, store)
        original, copies = ComplexRecord.detached, []
        def detached(row): copies.append(row.version); return original(row)
        monkeypatch.setattr(ComplexRecord, "detached", detached)
        for version in (1, 512, 1024):
            reads.clear(); copies.clear()
            snapshot = store.read_record("r", version=version)
            assert snapshot.value == Fraction(version-1, 7)
            assert reads == [version-1, version-1]
            assert len(copies) == 3 and set(copies) == {version}
        reads.clear(); copies.clear()
        assert store.read_record("r", version=1025) is None
        assert not reads and not copies


def test_injected_memory_compatibility_rows_keep_sparse_versions():
    with closing(MemoryStore()) as store:
        value = RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)])
        version = 2**53+9
        row = ComplexRecord("r@1", structural_signature(value, analytics=False),
                            version=version, meta={"q": Fraction(2, 7)})
        store._recs["r@1"] = [row]
        store._blobs[(row.id, version)] = serialize_complex(value)
        snapshot = store.read_record("r@1", version=version)
        assert snapshot.record.version == version and snapshot.record.meta["q"] == Fraction(2, 7)
        assert snapshot.state_digest == object_digest(value)
        assert store.read_record("r@1", version=1) is None


def test_headerless_sql_retains_history_selection(tmp_path):
    pytest.importorskip("sqlalchemy")
    with closing(SQLStore(f"sqlite:///{tmp_path/'legacy.sqlite'}", native=False)) as store:
        value = RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)])
        store.put("r@1", value, analytics=False, _tx_time=1.)
        store.put("r@1", value, analytics=False, _tx_time=2.)
        assert store.header is None
        assert store.read_record("r@1", version=1).record.tx_to == 2.
        assert store.read_record("r@1", version=2).state_digest == object_digest(value)
        assert store.read_record("r@1", version=3) is None


def test_sql_rollback_does_not_leave_an_exact_version_address(tmp_path):
    with closing(opened("sql", tmp_path/"store")) as store:
        store.put_record("r", Fraction(1, 7), tx_time=1.)
        with pytest.raises(RuntimeError, match="abort"):
            with store.write_scope():
                store.put_record("r", Fraction(2, 7), tx_time=2.)
                assert store.read_record("r", version=2).value == Fraction(2, 7)
                raise RuntimeError("abort")
        assert store.read_record("r", version=2) is None
        assert store.read_record("r", version=1).record.tx_to is None
        assert store.put_record("r", None, tx_time=3.).version == 2
        assert store.read_record("r", version=2).value is None


@pytest.mark.parametrize("kind", KINDS)
def test_exact_snapshots_respect_uncertain_publication_and_closed_handles(kind, tmp_path):
    store = opened(kind, tmp_path/"store")
    try:
        store.put_record("r", None)
        store._publication_uncertain = True
        for rid in ("r", "missing"):
            with pytest.raises(PublicationUncertainError): store.read_record(rid, version=1)
    finally:
        store._publication_uncertain = False
        store.close()
    for rid in ("r", "missing"):
        with pytest.raises(RuntimeError if kind == "sql" else ValueError, match="closed"):
            store.read_record(rid, version=1)


@pytest.mark.parametrize("kind", KINDS)
def test_metadata_and_decode_remain_in_one_provider_read_scope(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path/"store")) as store:
        store.put_record("r", Fraction(1, 7))
        selected, decoded, contexts = store._record_at_version, store.get_version, []
        def context():
            assert store._transaction_lock._is_owned()
            if kind == "sql":
                assert store._sql_connection is not None
                return store._sql_connection
            if kind != "memory": assert store._scope_depth > 0
            return store._state
        def select(*args): contexts.append(context()); return selected(*args)
        def decode(*args):
            assert context() is contexts[-1]
            return decoded(*args)
        monkeypatch.setattr(store, "_record_at_version", select)
        monkeypatch.setattr(store, "get_version", decode)
        assert store.read_record("r", version=1).value == Fraction(1, 7)
        assert len(contexts) == 1


def test_default_provider_hook_preserves_noncontiguous_compatibility_history():
    value = RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)])
    class CompatibilityStore(RCStore):
        def history(self, rid):
            return [ComplexRecord(rid, structural_signature(value, analytics=False), version=v)
                    for v in (3, 9)] if rid == "r@1" else []
        def get_version(self, rid, version): return value
    store = CompatibilityStore()
    assert store.read_record("r@1", version=9).record.version == 9
    assert store.read_record("r@1", version=3).state_digest == object_digest(value)
    assert store.read_record("r@1", version=1) is None


@pytest.mark.parametrize("kind", KINDS)
def test_packet_copy_and_rcql_share_the_exact_version_snapshot(kind, tmp_path, monkeypatch):
    from rcql import Executor, parse
    with closing(opened(kind, tmp_path/"store")) as source, closing(MemoryStore()) as target:
        value = RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)])
        row = source.put("r", value, analytics=False, _tx_time=1., meta={"q": Fraction(2, 7)})
        source.put("r", RexGraph.from_graph([0, 1], [1, 2]), analytics=False, _tx_time=2.)
        forbid_history(monkeypatch, source)
        packet = record_packet(source, "r", version=1)
        receipt = copy_record(source, target, row, destination_id="copied", return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest == packet.state_digest == object_digest(value)
        result = Executor(sources={"db": source}).execute(parse(
            'FROM RCDB_VERSION($db,"r",1) RETURN RANK(1), 1 / 7'))
        assert result.values == (1, Fraction(1, 7))
        assert target.read_record("copied", version=1).record.meta["q"] == Fraction(2, 7)
