"""Selection work stays bounded without changing bitemporal/store semantics."""
from contextlib import closing
from fractions import Fraction
import math

import pytest

from rcdb import ComplexRecord, LocalStore, MemoryStore, NativeObjectStore, RCStore, SQLStore
from rcdb.core import _matches
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


def populate(store):
    for index in range(8):
        record_id = f"literal/{index}@1"
        for version, timestamp in enumerate((10., 10., 20.), 1):
            store.put_record(record_id, {"q": Fraction(index+version, 7)}, tx_time=timestamp,
                valid_from=float(version), valid_to=float(version+3),
                meta={"vertex_labels": ["shared", f"term-{version}"], "exact": Fraction(2, 7)},
                tags=["even" if index % 2 == 0 else "odd", f"v{version}"])
        if index % 3 == 0:
            store.delete(record_id, tx_time=25.)
        if index % 6 == 0:
            store.put_record(record_id, None, tx_time=30., valid_from=6., valid_to=9.,
                             meta={"vertex_labels": ["revived"]}, tags=["revived"])


SELECTORS = ({}, {"as_of": 9.}, {"as_of": 10.}, {"as_of": 15.},
             {"as_of": 20.}, {"as_of": 25.}, {"as_of": 30.},
             {"valid_at": 1.}, {"valid_at": 2.}, {"valid_at": 6.},
             {"as_of": 10., "valid_at": 1.}, {"as_of": 15., "valid_at": 4.},
             {"as_of": 25., "valid_at": 6.}, {"include_history": True})
PREDICATES = ({}, {"tags_any": ["even"]}, {"tags_all": ["odd", "v2"]},
              {"labels_any": ["TERM-1"]}, {"labels_all": ["shared", "term-3"]},
              {"record_type": "NativeValue"}, {"min_nE": 0}, {"tags_any": ["absent"]})


@pytest.mark.parametrize("kind", KINDS)
def test_pages_queries_and_point_reads_match_existing_time_selection(kind, tmp_path):
    with closing(opened(kind, tmp_path / "store")) as store:
        populate(store)
        history = {record_id: store.history(record_id) for record_id in
                   (f"literal/{index}@1" for index in range(8))}
        for selector in SELECTORS:
            if selector.get("include_history"):
                reference = [row for rows in history.values() for row in rows]
            else:
                reference = [row for rows in history.values() if (row := RCStore._select_version(
                    rows, selector.get("as_of"), selector.get("valid_at"))) is not None]
                for record_id, rows in history.items():
                    expected = RCStore._select_version(rows, selector.get("as_of"), selector.get("valid_at"))
                    actual = store.get_record(record_id, **selector)
                    assert (None if actual is None else actual.to_dict()) == (
                        None if expected is None else expected.to_dict())
            reference.sort(key=lambda row: (-row.tx_from, row.id, -row.version))
            for limit, offset in ((0, 0), (1, 0), (3, 2), (1000, 0), (1, 1000)):
                actual = store.list(limit=limit, offset=offset, **selector)
                assert pack_value([r.to_dict() for r in actual]) == pack_value(
                    [r.to_dict() for r in reference[offset:offset+limit]])
            for predicate in PREDICATES:
                expected = [r for r in reference if _matches(r.signature, predicate, r.meta,
                            is_complex=r.is_complex, record_type=r.object_type)][:3]
                actual = store.query(limit=3, **selector, **predicate)
                assert pack_value([r.to_dict() for r in actual]) == pack_value([r.to_dict() for r in expected])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("selector", [{}, {"as_of": 32.}, {"valid_at": 32.}, {"as_of": 32., "valid_at": 32.}])
def test_point_selection_detaches_only_the_selected_row(kind, selector, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path / "store")) as store:
        for i in range(64):
            store.put_record("r", Fraction(i, 7), tx_time=float(i), valid_from=float(i), valid_to=float(i+1))
        store.get_record("r")  # Certify SQL before measuring metadata selection.
        original, calls = ComplexRecord.detached, []
        def detached(record):
            calls.append((record.id, record.version))
            return original(record)
        monkeypatch.setattr(ComplexRecord, "detached", detached)
        row = store.get_record("r", **selector)
        assert row.version == (64 if not selector else 33)
        assert 1 <= len(calls) <= 2, calls
        row.meta["outside"] = True
        assert "outside" not in store.get_record("r", **selector).meta


@pytest.mark.parametrize("kind", KINDS)
def test_small_collections_detach_only_returned_page(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path / "store")) as store:
        for i in range(96):
            store.put_record(f"r{i:03d}", None, tags=["kept"], tx_time=float(i))
        store.list(limit=1)
        original, calls = ComplexRecord.detached, []
        def detached(record): calls.append(record.id); return original(record)
        monkeypatch.setattr(ComplexRecord, "detached", detached)
        for operation in (lambda: store.list(limit=3, offset=5),
                          lambda: store.query(limit=3, tags_all=["kept"]),
                          lambda: store.query(limit=3, include_history=True)):
            calls.clear(); result = operation()
            assert len(result) == 3 and 3 <= len(calls) <= 6, calls
            result[0].signature["tags"].append("outside")
            assert "outside" not in store.get_record(result[0].id).signature["tags"]


@pytest.mark.parametrize("kind", KINDS)
def test_literal_aliases_tombstones_and_empty_bounds_keep_validation(kind, tmp_path):
    with closing(opened(kind, tmp_path / "store")) as store:
        for i in range(16): store.put_record("r", i, tx_time=float(i))
        assert store.get_record("r@8").version == 8
        store.put_record("r@8", None, tx_time=30.)
        assert store.get_record("r@8").version == 1
        store.delete("r@8", tx_time=31.)
        assert store.get_record("r@8") is None
        assert store.get_record("r@8", as_of=30.5).version == 1
        for invalid in ({"limit": True}, {"offset": -1}, {"as_of": math.nan},
                        {"include_history": 1}, {"include_history": True, "as_of": 1.}):
            with pytest.raises((TypeError, ValueError)): store.list(**invalid)
        with pytest.raises(TypeError): store.query(limit=0, unknown=True)
        with pytest.raises((TypeError, ValueError)): store.query(limit=0, include_history=True, as_of=1.)


def test_transaction_selection_probes_logarithmically_without_cloning_history():
    with closing(MemoryStore()) as store:
        for i in range(1024): store.put_record("r", i, tx_time=float(i))
        state, original = store._state, store._state._rows["r"]
        class CountedRows(list):
            probes = 0
            def __getitem__(self, key):
                self.probes += 1
                return super().__getitem__(key)
            def __iter__(self): pytest.fail("transaction point lookup scanned the lineage")
        rows = CountedRows(original); state._rows["r"] = rows
        assert state.selected("r", as_of=511.5).version == 512
        assert rows.probes <= 14


@pytest.mark.parametrize("kind", KINDS)
def test_zero_pages_do_not_evaluate_record_predicates(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path / "store")) as store:
        store.put_record("r", None)
        store.list(limit=1)
        def fail(*a, **kw): pytest.fail("zero page evaluated metadata predicates")
        monkeypatch.setattr("rcdb.core._matches", fail)
        assert store.query(limit=0, tags_any=["kept"]) == []


@pytest.mark.parametrize("kind", KINDS)
def test_exact_rcql_and_system_inspection_use_the_same_selected_native_versions(kind, tmp_path):
    rcql = pytest.importorskip("rcql")
    pytest.importorskip("system")
    from rexgraph import RexGraph
    from system.inspection import source_row
    with closing(opened(kind, tmp_path / "store")) as store:
        store.put("g", RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)]), analytics=False)
        result = rcql.Executor(sources={"db": store}).execute(rcql.parse(
            'FROM RCDB_VERSION(RCDB("db"), "g", 1) RETURN 1 / 7, RANK(1)'))
        assert result.values == (Fraction(1, 7), 1)
        store.put_record("exact", Fraction(2, 7))
        assert store.query(limit=1, record_type="NativeValue")[0].id == "exact"
        row = source_row("db", store)
        assert row["accessible"] and row["default_query"]
        result = rcql.Executor(sources={"db": store}).execute(rcql.parse(row["default_query"]))
        assert result.values and store.get("g").edge_metric_exact == [Fraction(1, 7)]


@pytest.mark.parametrize("kind", KINDS)
def test_rich_query_comparisons_cannot_mutate_borrowed_compound_metadata(kind, tmp_path):
    with closing(opened(kind, tmp_path / "store")) as store:
        store.put_record("r", None, signature={"object_type": "NativeValue", "source": {"q": Fraction(1, 7)}})
        class Comparison:
            def __eq__(self, other):
                other["q"] = Fraction(2, 7)
                return True
        result = store.query(limit=1, source=Comparison())
        assert len(result) == 1
        assert store.get_record("r").signature["source"]["q"] == Fraction(1, 7)
        assert store.read_record("r").value is None
