"""Disposable native postings preserve the complete version selection contract."""
from contextlib import closing
from fractions import Fraction
import hashlib

import numpy as np
import pytest

from rcdb import ComplexRecord, LocalStore, MemoryStore, NativeObjectStore, RCStore, SQLStore
from rcdb.core import _matches, serialize_complex
from rcdb.engine import bind_record
from rexgraph import RexGraph
from rexgraph.value_codec import pack_value


KINDS = ("memory", "local", "sql", "object-file", "object-memory")
SELECTORS = ({}, {"as_of": 9.}, {"as_of": 10.}, {"as_of": 15.},
             {"as_of": 20.}, {"as_of": 25.}, {"as_of": 30.},
             {"valid_at": 1.}, {"valid_at": 2.}, {"valid_at": 6.},
             {"as_of": 10., "valid_at": 1.}, {"as_of": 25., "valid_at": 6.},
             {"include_history": True})
PREDICATES = ({"labels_any": ["NEEDLE"]}, {"labels_all": ["shared", "needle"]},
              {"tags_any": ["needle", "missing"]}, {"tags_all": ["shared", "needle"]},
              {"source": "needle"}, {"record_type": "NativeValue"},
              {"labels_any": ["needle"], "tags_all": ["shared", "needle"], "source": "needle"},
              {"labels_any": ["needle"], "tags_any": ["old"]},
              {"labels_any": []}, {"labels_all": []}, {"tags_any": []}, {"tags_all": []},
              {"labels_any": "n"}, {"tags_any": "n"}, {"source": None},
              {"labels_any": ["absent"]}, {"tags_all": ["absent"]})


def opened(kind, path):
    if kind == "memory":
        return MemoryStore()
    if kind == "local":
        return LocalStore(path)
    if kind == "sql":
        pytest.importorskip("sqlalchemy")
        return SQLStore(f"sqlite:///{path}.sqlite")
    pytest.importorskip("fsspec")
    return NativeObjectStore(f"{'file' if kind == 'object-file' else 'memory'}://{path}")


def state_of(store):
    return store._native.state if isinstance(store, SQLStore) else store._state


def reference(store, predicate, selector, limit=100):
    rows = store.list(limit=10000, include_history=True)
    if not selector.get("include_history"):
        lineages = {}
        for row in rows:
            lineages.setdefault(row.id, []).append(row)
        rows = [row for versions in lineages.values()
                if (row := RCStore._select_version(versions, selector.get("as_of"),
                                                   selector.get("valid_at"))) is not None]
    rows = [row for row in rows if _matches(row.signature, predicate, row.meta,
                                          is_complex=row.is_complex, record_type=row.object_type)]
    return sorted(rows, key=lambda r: (-r.tx_from, r.id, -r.version))[:limit]


def same_rows(actual, expected):
    assert pack_value([r.to_dict() for r in actual]) == pack_value([r.to_dict() for r in expected])


def populate(store):
    for i in range(12):
        for version, tick in enumerate((10., 10., 20.), 1):
            term = "needle" if (i+version) % 3 == 0 else "other"
            store.put_record(f"literal/{i}@1", Fraction(i+version, 7), tx_time=tick,
                valid_from=float(version), valid_to=float(version+3),
                signature={"object_type": "NativeValue", "source": term,
                           "labels_sample": ["sample"], "tags": ["shared", term, f"v{version}"]},
                meta={"vertex_labels": ["shared", term]}, tags=["shared", term, f"v{version}"])


@pytest.mark.parametrize("kind", KINDS)
def test_queries_remain_equal_after_index_warming_deletion_revival_and_equal_ticks(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        populate(store)
        for phase in range(3):
            if phase == 1:
                for i in range(0, 12, 3):
                    assert store.delete(f"literal/{i}@1", tx_time=25.)
            if phase == 2:
                for i in range(0, 12, 6):
                    store.put_record(f"literal/{i}@1", None, tx_time=30., valid_from=6., valid_to=9.,
                        signature={"object_type": "NativeValue", "source": "needle", "tags": ["shared", "needle"]},
                        meta={"vertex_labels": ["shared", "needle"]}, tags=["shared", "needle"])
            for selector in SELECTORS:
                for predicate in PREDICATES:
                    expected = reference(store, predicate, selector, limit=3)
                    same_rows(store.query(limit=3, **selector, **predicate), expected)
                    # SQLite uses its own checked projections for public queries;
                    # verify the shared state index there as well.
                    with store.read_transaction():
                        same_rows(state_of(store).select_records(limit=3, predicate=predicate, **selector), expected)


@pytest.mark.parametrize("kind", ("memory", "local", "object-file", "object-memory"))
def test_warm_selective_queries_visit_only_matching_lineages_and_addresses(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path/"store")) as store:
        graph = RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)])
        for i in range(128):
            for version in range(2):
                term = "needle" if i == 17 else "other"
                kwargs = dict(tx_time=float(10+version*10), valid_from=float(version+1),
                              valid_to=float(version+2), meta={"vertex_labels":
                              [f"term-{n}" for n in range(30)]+[term]}, tags=["shared", term])
                if i == 17:
                    kwargs["_tx_time"] = kwargs.pop("tx_time")
                    store.put(f"r{i:03d}", graph, analytics=False, **kwargs)
                else:
                    store.put_record(f"r{i:03d}", None, **kwargs)
        store.put_record("source", None, tx_time=20., signature={"object_type": "NativeValue", "source": "needle"})
        predicates = ({"labels_any": ["NEEDLE"]}, {"labels_all": ["term-29", "needle"]},
                      {"tags_any": ["needle"]}, {"tags_all": ["shared", "needle"]},
                      {"source": "needle"}, {"record_type": "RexGraph"},
                      {"labels_any": ["needle"], "tags_all": ["shared"], "record_type": "RexGraph"},
                      {"labels_all": ["absent"]}, {"tags_any": []})
        selectors = ({}, {"as_of": 15.}, {"valid_at": 1.5}, {"include_history": True})
        for selector in selectors:
            for predicate in predicates:
                expected = reference(store, predicate, selector)
                store.query(**predicate, **selector)  # Build only the requested projections.
                calls, probes = [], []
                original = store._state._selected_row
                def match(*a, **kw):
                    calls.append(1)
                    return _matches(*a, **kw)
                def selected(*a):
                    probes.append(1)
                    return original(*a)
                with monkeypatch.context() as patch:
                    patch.setattr("rcdb.core._matches", match)
                    patch.setattr(store._state, "_selected_row", selected)
                    actual = store.query(**predicate, **selector)
                same_rows(actual, expected)
                assert len(calls) <= 2 and len(probes) <= 1, (predicate, selector, calls, probes)
        assert set(store._state._predicate_indexes) == {(f, h) for f in
                ("labels", "tags", "source", "record_type") for h in (False, True)}


@pytest.mark.parametrize("kind", ("local", "sql", "object-file", "object-memory"))
def test_reopened_and_peer_updated_handles_rebuild_exact_indexes(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as first:
        populate(first)
        first.query(labels_any=["needle"])
        with closing(opened(kind, path)) as second:
            second.query(tags_any=["needle"])
            first.put_record("peer", Fraction(1, 7), tx_time=30.,
                             meta={"vertex_labels": ["needle"]}, tags=["needle"])
            assert "peer" in {r.id for r in second.query(labels_any=["needle"])}
            second.delete("peer", tx_time=31.)
            assert "peer" not in {r.id for r in first.query(labels_any=["needle"])}
            assert "peer" in {r.id for r in first.query(labels_any=["needle"], as_of=30.5)}
    with closing(opened(kind, path)) as third:
        for selector in SELECTORS:
            for predicate in PREDICATES:
                same_rows(third.query(**predicate, **selector), reference(third, predicate, selector))


@pytest.mark.parametrize("key", ("labels_any", "labels_all", "tags_any", "tags_all", "source", "record_type"))
@pytest.mark.parametrize("selector", ({}, {"include_history": True}, {"as_of": 1.}))
def test_unusual_closed_metadata_and_predicates_keep_residual_semantics(key, selector):
    with closing(MemoryStore()) as store:
        specimens = (
            ({"vertex_labels": ["A", 7, Fraction(1, 7)]}, ["A", "b"], "A"),
            ({"vertex_labels": []}, "Ab", None),
            ({"vertex_labels": "Ab"}, {"A": 1}, {"exact": Fraction(1, 7)}),
            ({"vertex_labels": ["other"]}, [1, True, Fraction(1, 7)], 7),
            ({"vertex_labels": np.array(["A", "b"])}, None, np.array("A")),
            ({"vertex_labels": 42}, [["unhashable"]], b"A"),
        )
        payload = serialize_complex(RexGraph.from_graph([0], [1]))
        for i, (meta, tags, source) in enumerate(specimens):
            # Exercise the checked engine grammar, including metadata that public
            # vocabulary extraction would refuse before proposing a publication.
            record = ComplexRecord(str(i), {"source": source, "tags": tags, "labels_sample": ["sample"]},
                                   meta=meta, created=1., tx_from=1., valid_from=1.)
            record, raw = bind_record(record, payload, store_id=store.header.identity.id)
            store._state.apply(store._state.prepare_put(record, blob_digest=hashlib.sha256(raw).hexdigest()))
        values = ([], ["A"], ["A", "b"], "A", [7], [Fraction(1, 7)],
                  [["unhashable"]], 42, None) if "_" in key and key != "record_type" else (
                      "A", "NativeValue", None, 7, {"exact": Fraction(1, 7)})
        for value in values:
            predicate = {key: value}
            for extra in ({}, {"labels_any": ["A", "sample"]}):
                predicate.update(extra)
                expected = reference(store, predicate, selector)
                same_rows(store.query(**predicate, **selector), expected)


@pytest.mark.parametrize("key", ("labels_any", "tags_any", "source"))
def test_custom_query_extensions_use_owned_candidates_and_no_index(key):
    with closing(MemoryStore()) as store:
        store.put_record("r", None, meta={"vertex_labels": ["needle"]},
                         signature={"object_type": "NativeValue", "source": {"q": Fraction(1, 7)}, "tags": ["needle"]},
                         tags=["needle"])
        class Rich:
            def __iter__(self):
                return iter(["needle"])
            def __eq__(self, other):
                other["q"] = Fraction(2, 7)
                return True
        assert [r.id for r in store.query(**{key: Rich()})] == ["r"]
        assert store._state._predicate_indexes == {}
        assert store.get_record("r").signature["source"] == {"q": Fraction(1, 7)}
        assert store.read_record("r").value is None


def test_sql_rollback_discards_provisional_predicate_indexes(tmp_path):
    with closing(opened("sql", tmp_path/"store")) as store:
        store.put_record("r", None, tx_time=1., meta={"vertex_labels": ["old"]})
        with store.read_transaction():
            assert state_of(store).select_records(predicate={"labels_any": ["old"]})
        with pytest.raises(RuntimeError, match="abort"):
            with store.write_scope():
                store.put_record("r", None, tx_time=2., meta={"vertex_labels": ["new"]})
                assert state_of(store).select_records(predicate={"labels_any": ["new"]})
                raise RuntimeError("abort")
        with store.read_transaction():
            assert state_of(store).select_records(predicate={"labels_any": ["old"]})
            assert not state_of(store).select_records(predicate={"labels_any": ["new"]})
        assert store.get_record("r").version == 1


def test_unused_fields_empty_pages_and_refused_publication_do_not_build_or_change_indexes():
    with closing(MemoryStore()) as store:
        store.put_record("r", None, tx_time=2., meta={"vertex_labels": ["old"]})
        assert store._state._predicate_indexes == {}
        store.list()
        store.query(limit=0, labels_any=["old"])
        store.query(tags_all=[])
        assert store._state._predicate_indexes == {}
        assert store.query(labels_any=["old"])
        assert set(store._state._predicate_indexes) == {("labels", False)}
        with pytest.raises(ValueError):
            store.put_record("r", None, tx_time=1., meta={"vertex_labels": ["new"]})
        assert store.query(labels_any=["old"])
        assert not store.query(labels_any=["new"])
        store.delete("r", tx_time=3.)
        index = store._state._predicate_indexes[("labels", False)]
        assert not index.postings and not index.unknown


@pytest.mark.parametrize("kind", KINDS)
def test_postings_do_not_combine_terms_from_different_versions_or_select_older_matches(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        for tick, valid, label, tag in ((10., 1., "needle", "old"),
                                       (10., 2., "other", "needle"),
                                       (20., 3., "needle", "needle")):
            store.put_record("r@2", Fraction(1, 7), tx_time=tick, valid_from=valid, valid_to=valid+1,
                             meta={"vertex_labels": [label]}, tags=[tag])
        for selector in ({"as_of": 10.}, {"as_of": 15.}, {"valid_at": 1.5},
                         {"valid_at": 2.5}, {"as_of": 20., "valid_at": 1.5},
                         {"include_history": True}, {}):
            for predicate in ({"labels_any": ["needle"]},
                              {"labels_any": ["needle"], "tags_any": ["needle"]}):
                same_rows(store.query(**predicate, **selector), reference(store, predicate, selector))
        assert not store.query(labels_any=["needle"], as_of=10.)
        assert not store.query(labels_any=["needle"], tags_any=["needle"], valid_at=1.5)
        assert [r.version for r in store.query(labels_any=["needle"], include_history=True)] == [3, 1]


def test_indexed_candidates_still_run_the_complete_structural_matcher():
    with closing(MemoryStore()) as store:
        store.put_record("value", None, tags=["needle"])
        store.put("graph", RexGraph.from_graph([0], [1]), analytics=False, tags=["needle"])
        assert [r.id for r in store.query(tags_any=["needle"], min_nE=1)] == ["graph"]
        assert not store.query(tags_any=["needle"], min_nE=2)
        with pytest.raises(TypeError):
            store.query(labels_any=[], unknown=None)


def test_publication_updates_only_built_indexes_for_the_changed_addresses(monkeypatch):
    import rcdb.engine as engine
    with closing(MemoryStore()) as store:
        for i in range(128):
            store.put_record(str(i), None, tx_time=1., meta={"vertex_labels": ["old"]}, tags=["old"])
        for history in (False, True):
            store.query(labels_any=["old"], tags_any=["old"], source="", record_type="NativeValue",
                        include_history=history)
        original, calls = engine._predicate_terms, []
        def terms(family, row):
            calls.append((family, row.id, row.version))
            return original(family, row)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_predicate_terms", terms)
            store.put_record("17", None, tx_time=2., meta={"vertex_labels": ["new"]}, tags=["new"])
            assert len(calls) == 12 and {c[1] for c in calls} == {"17"}
            calls.clear()
            store.delete("17", tx_time=3.)
            assert len(calls) == 4 and {c[2] for c in calls} == {2}
            calls.clear()
            assert not store.delete("17", tx_time=4.)
            assert not calls
        assert not store.query(labels_any=["new"])
        assert [r.version for r in store.query(labels_any=["new"], include_history=True)] == [2]
