"""Collection coverage must agree with retained selection and prune other IDs."""
from contextlib import closing
from fractions import Fraction
import random

import pytest

from rcdb import ComplexRecord, MemoryStore, RCStore
from rcdb._validity_catalog import _Interval, _ValidityCatalog
from rexgraph.value_codec import pack_value

from .test_native_validity_indexes import KINDS, opened, state_of


def row(record_id, start, end, version=1):
    return ComplexRecord(record_id, {}, version=version, valid_from=start, valid_to=end)


def checked_tree(node):
    if node is None:
        return 0, float("-inf"), 0
    left, left_end, left_count = checked_tree(node.left)
    right, right_end, right_count = checked_tree(node.right)
    assert abs(left-right) <= 1
    assert node.height == 1+max(left, right)
    assert node.max_end == max(node.end, left_end, right_end)
    if node.left is not None: assert node.left.key < node.key
    if node.right is not None: assert node.right.key > node.key
    return node.height, node.max_end, left_count+right_count+1


@pytest.mark.parametrize("seed", range(6))
def test_catalog_matches_independent_interval_union_after_every_publication(seed):
    rng, catalog, histories = random.Random(seed), _ValidityCatalog(), {}
    for version in range(256):
        rid = f"r{rng.randrange(24)}"
        start = rng.randrange(-40, 40)/4
        end = None if version % 29 == 0 else start+rng.randrange(1, 24)/4
        histories.setdefault(rid, []).append(row(rid, start, end, version+1))
        catalog.add(histories[rid][-1])
        checked_tree(catalog._all.root)
        for point in (start, end if end is not None else start+100, Fraction(1, 10), -10., 0., 10.):
            expected = [rid for rid, rows in histories.items() if any(
                r.valid_from <= point and (r.valid_to is None or point < r.valid_to) for r in rows)]
            assert catalog.ids_at(point) == expected


@pytest.mark.parametrize("order", ("ascending", "descending", "shuffled"))
def test_sparse_catalog_queries_prune_large_nonmatching_collections(order, monkeypatch):
    positions = list(range(16384))
    if order == "descending": positions.reverse()
    if order == "shuffled": random.Random(19).shuffle(positions)
    catalog = _ValidityCatalog()
    for i in positions:
        catalog.add(row(f"r{i}", float(2*i), float(2*i+1)))
    assert checked_tree(catalog._all.root)[2] == len(positions)
    descriptor, visits = _Interval.max_end, []
    def max_end(node):
        visits.append(1)
        return descriptor.__get__(node, _Interval)
    monkeypatch.setattr(_Interval, "max_end", property(max_end))
    for point in (-1., 0.5, 1., 16384.5, 32766.5, 32767., Fraction(3, 7)):
        visits.clear()
        expected = [f"r{int(point)//2}"] if 0 <= point < 32767 and int(point) % 2 == 0 else []
        assert catalog.ids_at(point) == expected
        assert len(visits) <= 64, (order, point, len(visits))


def test_repeated_versions_touching_ranges_and_broad_overwrites_coalesce():
    catalog = _ValidityCatalog()
    for i in range(2048): catalog.add(row("r", 0., 1., i+1))
    assert checked_tree(catalog._all.root)[2] == 1
    for i in range(1, 1024): catalog.add(row("r", float(2*i), float(2*i+1)))
    assert checked_tree(catalog._all.root)[2] == 1024
    catalog.add(row("r", 1., 2048.))
    assert checked_tree(catalog._all.root)[2] == 1
    assert catalog.ids_at(0.) == ["r"]
    assert catalog.ids_at(2048.) == []
    catalog.add(row("r", -1., None))
    catalog.add(row("r", 4., 5.))
    assert checked_tree(catalog._all.root)[2] == 1
    assert catalog.ids_at(10**100) == ["r"]


@pytest.mark.parametrize("kind", KINDS)
def test_native_catalog_pages_updates_deletes_revival_and_exact_times(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        ids = [f"literal/{i}@1" for i in range(12)]
        for i, rid in enumerate(ids):
            store.put_record(rid, Fraction(i, 7), tx_time=1., valid_from=0.1,
                             valid_to=0.2, tags=["old"])
        assert store.list(valid_at=Fraction(1, 5))
        assert state_of(store)._validity_catalog is not None
        for i, rid in enumerate(ids):
            store.put_record(rid, Fraction(i+1, 7), tx_time=2., valid_from=0.15,
                             valid_to=0.2, tags=["new"])
            if i % 2 == 0: store.delete(rid, tx_time=3.)
            if i % 3 == 0:
                store.put_record(rid, None, tx_time=4., valid_from=4., valid_to=None, tags=["revived"])
        histories = {rid: store.history(rid) for rid in ids}
        for point in (Fraction(1, 10), Fraction.from_float(0.1), Fraction(3, 20),
                      Fraction(1, 5), Fraction.from_float(0.2), 3., 4., 10**30):
            expected = [r for rows in histories.values() if (r := RCStore._select_version(rows, None, point))]
            expected.sort(key=lambda r: (-r.tx_from, r.id, -r.version))
            for limit, offset in ((0, 0), (3, 2), (100, 0), (1, 100)):
                actual = store.list(limit=limit, offset=offset, valid_at=point)
                assert pack_value([r.to_dict() for r in actual]) == pack_value(
                    [r.to_dict() for r in expected[offset:offset+limit]])
            assert [(r.id, r.version) for r in store.query(valid_at=point, tags_all=[])] == [
                (r.id, r.version) for r in expected]
            assert [(r.id, r.version) for r in store.query(valid_at=point, tags_any=["old"])] == [
                (r.id, r.version) for r in expected if "old" in r.signature["tags"]]


def test_warm_collection_never_scans_nonmatching_lineages_or_histories(monkeypatch):
    with closing(MemoryStore()) as store:
        for i in range(512):
            store.put_record(f"r{i}", None, tx_time=float(i), valid_from=float(2*i), valid_to=float(2*i+1))
        assert store.list(valid_at=0.5)[0].id == "r0"
        class NoScan(dict):
            def __iter__(self): pytest.fail("warm catalog iterated unrelated lineages")
            def items(self): pytest.fail("warm catalog iterated unrelated lineages")
            def values(self): pytest.fail("warm catalog iterated unrelated histories")
        state = store._state
        state._rows = NoScan(state._rows)
        original, selected = state._select_lineage, []
        def select(rid, *args): selected.append(rid); return original(rid, *args)
        monkeypatch.setattr(state, "_select_lineage", select)
        assert store.query(valid_at=0.5, tags_all=[])[0].id == "r0"
        assert not store.list(valid_at=1.)
        assert selected == ["r0"]


def test_catalog_is_lazy_and_preserves_engine_lineage_order_and_selector_extensions():
    with closing(MemoryStore()) as store:
        for rid in ("z", "a", "b"):
            store.put_record(rid, None, tx_time=1., valid_from=0., valid_to=1., tags=["kept"])
        state = store._state
        assert store.list(limit=0, valid_at=0.5) == []
        store.get_record("z", valid_at=0.5)
        store.query(valid_at=0.5, tags_any=["kept"])
        store.list(as_of=1., valid_at=0.5)
        assert state._validity_catalog is None
        class CustomTime(float):
            def __ge__(self, other): return True
            def __lt__(self, other): return True
        assert len(store.list(valid_at=CustomTime(4.))) == 3
        assert state._validity_catalog is None
        assert [r.id for r in state.records(valid_at=0.5)] == ["z", "a", "b"]
        catalog = state._validity_catalog
        with pytest.raises(ValueError): store.put_record("z", None, tx_time=0., valid_from=4.)
        assert state._validity_catalog is catalog
        assert not store.list(valid_at=4.)


def test_sql_rollback_rebuilds_collection_coverage_without_provisional_ranges(tmp_path):
    with closing(opened("sql", tmp_path/"store")) as store:
        store.put_record("r", None, tx_time=1., valid_from=0., valid_to=1.)
        assert store.list(valid_at=Fraction(1, 2))
        with pytest.raises(RuntimeError, match="abort"):
            with store.write_scope():
                store.put_record("r", None, tx_time=2., valid_from=4., valid_to=5.)
                assert store.list(valid_at=Fraction(9, 2))
                raise RuntimeError("abort")
        assert not store.list(valid_at=Fraction(9, 2))
        assert store.list(valid_at=Fraction(1, 2))[0].version == 1


@pytest.mark.parametrize("kind", ("local", "sql", "object-file", "object-memory"))
def test_peer_refresh_and_checkpoint_reopen_rebuild_checked_collection_catalog(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as first:
        first.put_record("r", None, tx_time=1., valid_from=0., valid_to=1.)
        assert first.list(valid_at=Fraction(1, 2))
        with closing(opened(kind, path)) as peer:
            assert not peer.list(valid_at=Fraction(9, 2))
            first.put_record("new", Fraction(1, 7), tx_time=2., valid_from=4., valid_to=5.)
            assert [r.id for r in peer.list(valid_at=Fraction(9, 2))] == ["new"]
            peer.delete("new", tx_time=3.)
            assert [r.id for r in first.list(valid_at=Fraction(9, 2))] == ["new"]
        if kind != "sql": first.checkpoint(max_frames=1)
    with closing(opened(kind, path)) as restored:
        assert [r.id for r in restored.list(valid_at=Fraction(9, 2))] == ["new"]
        assert restored.read_record("new", valid_at=Fraction(9, 2)).value == Fraction(1, 7)
