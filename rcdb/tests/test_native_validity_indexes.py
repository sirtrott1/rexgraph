"""Validity coverage accelerates retained selection without changing its meaning."""
from contextlib import closing
from fractions import Fraction
import math
import random

import numpy as np
import pytest

from rcdb import ComplexRecord, LocalStore, MemoryStore, NativeObjectStore, RCStore, SQLStore
from rcdb._validity import _Boundary, _ValidityIndex
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


def state_of(store):
    return store._native.state if isinstance(store, SQLStore) else store._state


def same_row(actual, expected):
    assert pack_value(None if actual is None else actual.to_dict()) == pack_value(
        None if expected is None else expected.to_dict())


def covering(rows, point):
    return RCStore._select_version(rows, None, point)


def interval_rows(seed, count=96):
    rng = random.Random(seed)
    for i in range(count):
        start = float(rng.randrange(-32, 32))/4
        end = None if i % 11 == 0 else start+float(rng.randrange(1, 24))/4
        yield ComplexRecord("r", {}, version=i+1, tx_from=float(i),
                            valid_from=start, valid_to=end)


@pytest.mark.parametrize("seed", range(12))
def test_overlapping_nested_disjoint_open_and_out_of_order_ranges_match_retained_selection(seed):
    index, rows = _ValidityIndex(), []
    points = [Fraction(i, 8) for i in range(-80, 113)]
    for row in interval_rows(seed):
        rows.append(row)
        index.add(row)
        # Probe inclusive starts, excluded ends and both sides of every boundary.
        probes = points+[row.valid_from, math.nextafter(row.valid_from, -math.inf)]
        if row.valid_to is not None:
            probes += [row.valid_to, math.nextafter(row.valid_to, -math.inf)]
        for point in probes:
            expected = covering(rows, point)
            assert index.version_at(point) == (None if expected is None else expected.version)


@pytest.mark.parametrize("kind", KINDS)
def test_point_list_query_and_owned_payloads_follow_the_same_validity_coverage(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        fixture = list(interval_rows(7, count=24))
        for i, row in enumerate(fixture):
            store.put_record("literal/r@1", Fraction(i, 7), tx_time=float(i//3),
                             valid_from=row.valid_from, valid_to=row.valid_to,
                             meta={"vertex_labels": ["even" if i % 2 == 0 else "odd"]},
                             tags=["even" if i % 2 == 0 else "odd"])
            if i % 5 == 0:
                store.delete("literal/r@1", tx_time=float(i//3))
            if i % 8 != 7:
                continue
            rows = store.history("literal/r@1")
            for point in (Fraction(-33, 4), Fraction(-1, 7), 0., 1, 3.5, 7., 8., 100.):
                expected = covering(rows, point)
                actual = store.get_record("literal/r@1", valid_at=point)
                same_row(actual, expected)
                same_row(next(iter(store.list(limit=1, valid_at=point)), None), expected)
                for tag in ("even", "odd"):
                    result = store.query(tags_any=[tag], valid_at=point)
                    same_row(next(iter(result), None), expected if expected is not None
                             and tag in expected.signature.get("tags", []) else None)
                if expected is not None:
                    assert store.read_record("literal/r@1", valid_at=point).value == Fraction(expected.version-1, 7)
                    actual.meta["outside"] = True
                    assert "outside" not in store.get_record(actual.id, valid_at=point).meta
            # A combined selector uses the transaction selected version even when
            # a different retained version would match the validity coordinate.
            for tick in (0., 2., 5., 7.):
                for point in (-1., 0., 3., 100.):
                    same_row(store.get_record("literal/r@1", as_of=tick, valid_at=point),
                             RCStore._select_version(rows, tick, point))


@pytest.mark.parametrize("kind", KINDS)
def test_validity_only_retains_deleted_ranges_and_revival_overlays_only_its_own_interval(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        store.put_record("r", Fraction(1, 7), tx_time=1., valid_from=0., valid_to=10., tags=["old"])
        assert store.get_record("r", valid_at=1.).version == 1
        store.delete("r", tx_time=2.)
        assert store.get_record("r") is None
        assert store.get_record("r", valid_at=1.).version == 1
        store.put_record("r", Fraction(2, 7), tx_time=3., valid_from=4., valid_to=6., tags=["new"])
        for point, version in ((-1., None), (0., 1), (3.999, 1), (4., 2), (5.999, 2), (6., 1), (10., None)):
            actual = store.get_record("r", valid_at=point)
            assert (None if actual is None else actual.version) == version
        assert not store.query(tags_any=["old"], valid_at=5.)
        assert store.query(tags_any=["old"], valid_at=6.)[0].version == 1
        assert store.get_record("r", as_of=2.5, valid_at=1.) is None


@pytest.mark.parametrize("kind", ("local", "sql", "object-file", "object-memory"))
def test_peer_changes_reopen_and_checked_replay_keep_the_same_coverage(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as first:
        first.put_record("r", None, tx_time=1., valid_from=0., valid_to=10.)
        assert first.get_record("r", valid_at=5.).version == 1
        with closing(opened(kind, path)) as second:
            assert second.get_record("r", valid_at=5.).version == 1
            first.put_record("r", None, tx_time=2., valid_from=4., valid_to=6.)
            assert second.get_record("r", valid_at=5.).version == 2
            second.delete("r", tx_time=3.)
            assert first.get_record("r", valid_at=5.).version == 2
        rows = first.history("r")
    with closing(opened(kind, path)) as third:
        for point in (-1., 0., 4., 5., 6., 10.):
            same_row(third.get_record("r", valid_at=point), covering(rows, point))


@pytest.mark.parametrize("kind", KINDS)
def test_only_requested_lineages_build_indexes_and_unused_selectors_keep_their_paths(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        for rid in ("r", "r@2", "unused"):
            store.put_record(rid, None, tx_time=1., valid_from=0., valid_to=10.)
        store.get_record("r")
        store.get_record("r", as_of=1., valid_at=5.)
        store.query(limit=0, valid_at=5., labels_any=["any"])
        assert not state_of(store)._validity_indexes
        assert store.get_record("r@1").id == "r"
        with pytest.raises(ValueError): store.get_record("r@1", valid_at=5.)
        store.get_record("r")  # SQL discards cached state after a failed scope.
        assert not state_of(store)._validity_indexes
        assert store.get_record("r@2", valid_at=5.).id == "r@2"
        assert set(state_of(store)._validity_indexes) == {"r@2"}
        store.get_record("missing", valid_at=5.)
        assert set(state_of(store)._validity_indexes) == {"r@2"}
        store.put_record("r@2", None, tx_time=2., valid_from=3., valid_to=4.)
        assert store.get_record("r@2", valid_at=3.5).version == 2
        assert set(state_of(store)._validity_indexes) == {"r@2"}


@pytest.mark.parametrize("order", ("ascending", "descending", "shuffled"))
def test_large_boundary_maps_keep_logarithmic_reads_after_range_overwrites(order, monkeypatch):
    positions = list(range(4096))
    if order == "descending": positions.reverse()
    if order == "shuffled": random.Random(19).shuffle(positions)
    index, rows = _ValidityIndex(), []
    for version, i in enumerate(positions, 1):
        row = ComplexRecord("r", {}, version=version, valid_from=float(2*i), valid_to=float(2*i+1))
        index.add(row)
        rows.append(row)
    def check(points):
        descriptor, probes = _Boundary.start, []
        def start(node):
            probes.append(1)
            return descriptor.__get__(node, _Boundary)
        with monkeypatch.context() as patch:
            patch.setattr(_Boundary, "start", property(start))
            for point in points:
                probes.clear()
                expected = covering(rows, point)
                assert index.version_at(point) == (None if expected is None else expected.version)
                assert len(probes) <= 20, (order, point, len(probes))
    check((-1., 0., 0.5, 1., 4095., 8191., 8192.))
    # A single broad replacement can erase many boundaries, while keeping
    # remaining disjoint coverage and subsequent bounded point selection.
    for start, end in ((2048., 6144.), (100., 110.), (-1., None), (4., 5.)):
        row = ComplexRecord("r", {}, version=len(rows)+1, valid_from=start, valid_to=end)
        index.add(row)
        rows.append(row)
        check((-2., -1., 0., 4., 4.5, 5., 100., 110., 2048., 6144., 8191.))


def test_warm_long_lineage_lookup_never_iterates_history_and_detaches_only_result(monkeypatch):
    with closing(MemoryStore()) as store:
        for i in range(1024):
            store.put_record("r", Fraction(i, 7), tx_time=float(i),
                             valid_from=float(2*i), valid_to=float(2*i+1))
        assert store.get_record("r", valid_at=0.5).version == 1
        assert store.query(record_type="NativeValue", valid_at=0.5)[0].version == 1
        original_rows = store._state._rows["r"]
        class UnscannableRows(list):
            def __iter__(self): pytest.fail("warm validity read iterated history")
            def __reversed__(self): pytest.fail("warm validity read scanned retained versions")
        store._state._rows["r"] = UnscannableRows(original_rows)
        original, clones = ComplexRecord.detached, []
        def detached(row): clones.append(row.version); return original(row)
        monkeypatch.setattr(ComplexRecord, "detached", detached)
        assert store.get_record("r", valid_at=0.5).version == 1
        assert 1 <= len(clones) <= 2 and set(clones) == {1}
        assert not store.get_record("r", valid_at=1.)
        assert store.list(valid_at=0.5)[0].version == 1
        assert store.query(record_type="NativeValue", valid_at=0.5)[0].version == 1


@pytest.mark.parametrize("point", (np.float64(0.5), np.int64(0), np.float32(0.5)))
def test_other_supported_real_selector_types_keep_existing_fallback_semantics(point):
    with closing(MemoryStore()) as store:
        store.put_record("r", None, tx_time=1., valid_from=0., valid_to=1.)
        same_row(store.get_record("r", valid_at=point), covering(store.history("r"), point))
        assert not store._state._validity_indexes


def test_fraction_coordinates_are_not_rounded_to_binary64_by_the_index():
    with closing(MemoryStore()) as store:
        store.put_record("r", None, tx_time=1., valid_from=0.1, valid_to=0.2)
        assert store.get_record("r", valid_at=Fraction(1, 10)) is None
        assert store.get_record("r", valid_at=Fraction.from_float(0.1)).version == 1
        assert store.get_record("r", valid_at=Fraction(1, 5)).version == 1
        assert store.get_record("r", valid_at=Fraction.from_float(0.2)) is None


def test_custom_real_comparison_extensions_retain_reverse_scan_behavior():
    class CustomTime(float):
        def __ge__(self, other): return True
        def __lt__(self, other): return True
    with closing(MemoryStore()) as store:
        store.put_record("r", None, tx_time=1., valid_from=0., valid_to=1.)
        store.put_record("r", None, tx_time=2., valid_from=2., valid_to=3.)
        point = CustomTime(0.5)
        expected = covering(store.history("r"), point)
        assert expected.version == 2
        same_row(store.get_record("r", valid_at=point), expected)
        assert not store._state._validity_indexes


def test_sql_rollback_discards_provisional_validity_map(tmp_path):
    with closing(opened("sql", tmp_path/"store")) as store:
        store.put_record("r", None, tx_time=1., valid_from=0., valid_to=10.)
        assert store.get_record("r", valid_at=5.).version == 1
        with pytest.raises(RuntimeError, match="abort"):
            with store.write_scope():
                store.put_record("r", None, tx_time=2., valid_from=4., valid_to=6.)
                assert store.get_record("r", valid_at=5.).version == 2
                raise RuntimeError("abort")
        assert store.get_record("r", valid_at=5.).version == 1


def test_invalid_selectors_and_refused_publications_preserve_laziness_and_coverage():
    with closing(MemoryStore()) as store:
        store.put_record("r", None, tx_time=2., valid_from=0., valid_to=10.)
        for point in (True, math.inf, math.nan, "5"):
            with pytest.raises(ValueError): store.get_record("r", valid_at=point)
        assert not store._state._validity_indexes
        assert store.get_record("r", valid_at=5.).version == 1
        index = store._state._validity_indexes["r"]
        with pytest.raises(ValueError):
            store.put_record("r", None, tx_time=1., valid_from=4., valid_to=6.)
        assert store._state._validity_indexes["r"] is index
        assert store.get_record("r", valid_at=5.).version == 1


@pytest.mark.parametrize("kind", KINDS)
def test_native_decoded_time_reads_use_point_metadata_instead_of_cloning_history(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path/"store")) as store:
        for i in range(64):
            store.put_record("literal@1", Fraction(i, 7), tx_time=float(i),
                             valid_from=float(2*i), valid_to=float(2*i+1))
        assert store.read_record("literal@1", valid_at=0.5).value == Fraction(0, 7)
        def history(*a, **kw): pytest.fail("native decoded point read cloned retained history")
        monkeypatch.setattr(store, "history", history)
        assert store.read_record("literal@1", valid_at=0.5).record.version == 1
        assert store.read_record("literal@1", as_of=0.5).record.version == 1
        assert store.read_record("literal@1").record.version == 64
        assert store.read_record("literal@1", valid_at=1.) is None
        assert store.read_record("literal@1@2").record.version == 2
        with pytest.raises(ValueError): store.read_record("literal@1@2", valid_at=0.5)
        assert store.read_record("missing", valid_at=0.5) is None


@pytest.mark.parametrize("kind", KINDS)
def test_collection_time_coordinates_keep_the_point_read_comparison_precision(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        store.put_record("r", None, tx_time=0.1, valid_from=0.1, valid_to=0.2, tags=["kept"])
        store.put_record("r", None, tx_time=0.2, valid_from=0.2, valid_to=0.3, tags=["kept"])
        for point in (Fraction(1, 10), Fraction(1, 5), Fraction(3, 10),
                      np.longdouble("0.1"), np.longdouble("0.2"), 0.1, 0.2):
            for key in ("as_of", "valid_at"):
                selector = {key: point}
                expected = store.get_record("r", **selector)
                same_row(next(iter(store.list(**selector)), None), expected)
                same_row(next(iter(store.query(tags_any=["kept"], **selector)), None), expected)
                assert store.list(limit=0, **selector) == store.query(limit=0, **selector) == []
        for selector in ({"as_of": Fraction(1, 5), "valid_at": Fraction(1, 5)},
                         {"as_of": Fraction(3, 10), "valid_at": Fraction(1, 5)}):
            expected = store.get_record("r", **selector)
            same_row(next(iter(store.list(**selector)), None), expected)
            same_row(next(iter(store.query(tags_any=["kept"], **selector)), None), expected)
        for invalid in ({"include_history": True, "valid_at": Fraction(1, 10)},
                        {"include_history": 1, "valid_at": Fraction(1, 10)}):
            with pytest.raises((TypeError, ValueError)): store.list(limit=0, **invalid)
        store.put_record("large", None, tx_time=float(2**53+4),
                         valid_from=float(2**53), valid_to=float(2**53+4), tags=["large"])
        point = 2**53+3  # SQL Float binding rounds this up to the excluded endpoint.
        assert store.get_record("large", valid_at=point).version == 1
        assert [r.id for r in store.list(valid_at=point)] == ["large"]
        assert [r.id for r in store.query(valid_at=point, tags_any=["large"])] == ["large"]


def test_canonical_sql_collection_fallback_checks_pending_projections_before_selection(tmp_path):
    from sqlalchemy import update
    with closing(opened("sql", tmp_path/"store")) as store:
        store.put_record("r", None, tx_time=1., valid_from=0.1, valid_to=0.2)
        with pytest.raises(ValueError, match="projection"):
            with store.write_scope():
                store._sql_connection.execute(update(store.table).values(valid_from=0.15))
                store.query(valid_at=Fraction(1, 5))
        assert store.get_record("r", valid_at=Fraction(1, 5)).version == 1
        assert store.query(valid_at=Fraction(1, 5))[0].version == 1
