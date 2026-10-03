"""Only unpublished staging is collectible; plans never authorize history loss."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from fractions import Fraction
import hashlib
from threading import Event

import pytest

from rcdb import (LocalStore, MemoryStore, NativeObjectStore, OrphanObject,
                  PublicationUncertainError, RetentionPlan, RetentionPolicy, SQLStore, VersionConflictError)
from rcdb.retention import _inventory
from rexgraph import RexGraph

KINDS = ("memory", "local", "sql", "object-file", "object-memory")


def opened(kind, path, **options):
    if kind == "memory": return MemoryStore()
    if kind == "local": return LocalStore(path)
    if kind == "sql":
        pytest.importorskip("sqlalchemy")
        return SQLStore(f"sqlite:///{path}.sqlite")
    pytest.importorskip("fsspec")
    return NativeObjectStore(f"{'file' if kind == 'object-file' else 'memory'}://{path}", **options)


def seed(store):
    store.put_record("r", Fraction(1, 7), tx_time=1.)
    store.put_record("r", None, tx_time=2.)
    store.delete("r", tx_time=3.)
    store.put_record("r", Fraction(2, 7), tx_time=4.)
    store.commit_mutation("graph", RexGraph.from_graph([0], [1]), analytics=False, tx_time=1.)


def stage(store, kind, raw=b"abandoned", rid="orphan"):
    digest = hashlib.sha256(raw).hexdigest()
    if kind == "memory": store._blobs[(rid, 1)] = raw
    elif kind == "sql": store._store_commit_bytes(rid, 1, raw)
    elif kind == "local":
        with store.write_scope(): store._write_blob(digest, raw)
    else:
        with store.write_scope(): store._write_blob(digest, raw)
    return raw


def inventory(store, kind):
    with store.read_transaction():
        return _inventory(store, "object" if kind.startswith("object-") else kind)


@pytest.mark.parametrize("kind", KINDS)
def test_retention_is_checked_bounded_explicit_and_preserves_all_published_history(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        seed(store); stage(store, kind)
        cursor = store.change_cursor
        assert not store.plan_retention().candidates
        plan = store.plan_retention(RetentionPolicy(grace_seconds=0))
        assert len(plan.candidates) == 1
        assert RetentionPlan.from_bytes(plan.to_bytes()) == plan
        with pytest.raises(ValueError): RetentionPlan.from_bytes(plan.to_bytes()[:-1]+b"!")
        result = store.apply_retention(plan)
        assert result["history_retained"] and result["deleted_objects"] == 1
        assert result["deleted_bytes"] == len(b"abandoned")
        assert store.change_cursor == cursor
        assert store.read_record("r", version=1).value == Fraction(1, 7)
        assert store.read_record("r", version=2).value is None
        assert store.get_record("r", as_of=3.5) is None
        assert store.read_record("r").value == Fraction(2, 7)
        assert store.verify_commits("graph")
        assert not store.plan_retention(RetentionPolicy(grace_seconds=0)).candidates


@pytest.mark.parametrize("kind", KINDS)
def test_stale_or_foreign_plan_is_refused_before_any_deletion(kind, tmp_path):
    with closing(opened(kind, tmp_path/"a")) as store, closing(opened(kind, tmp_path/"b")) as other:
        stage(store, kind); plan = store.plan_retention(RetentionPolicy(grace_seconds=0))
        before = inventory(store, kind)
        with pytest.raises(ValueError, match="stale|another"): other.apply_retention(plan)
        store.put_record("new", None)
        with pytest.raises(ValueError, match="stale"): store.apply_retention(plan)
        assert all(item in inventory(store, kind) for item in before)


@pytest.mark.parametrize("kind", KINDS)
def test_forged_plan_cannot_collect_a_published_address(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        seed(store); stage(store, kind)
        plan = store.plan_retention(RetentionPolicy(grace_seconds=0))
        published = next(item for item in inventory(store, kind)
                         if (item[0], item[1]) not in {(c.namespace, c.address) for c in plan.candidates})
        from rcdb.retention import _read
        k = "object" if kind.startswith("object-") else kind
        with store.read_transaction(): raw = _read(store, k, *published[:2], published[2])
        candidate = OrphanObject(*published[:2], hashlib.sha256(raw).hexdigest(), *published[2:])
        forged = replace(plan, candidates=(plan.candidates[0], candidate))
        before = inventory(store, kind)
        with pytest.raises(ValueError, match="protected"): store.apply_retention(forged)
        assert inventory(store, kind) == before


@pytest.mark.parametrize("kind", KINDS)
def test_changed_candidate_rejects_whole_plan_before_deletion(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        stage(store, kind)
        plan = store.plan_retention(RetentionPolicy(grace_seconds=0))
        bad = replace(plan, candidates=(replace(plan.candidates[0], digest="0"*64),))
        before = inventory(store, kind)
        with pytest.raises(ValueError, match="changed"): store.apply_retention(bad)
        assert inventory(store, kind) == before


@pytest.mark.parametrize("kind", KINDS)
def test_candidate_count_and_byte_limits_are_enforced(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        for i in range(5): stage(store, kind, bytes([i])*10, f"orphan{i}")
        policy = RetentionPolicy(grace_seconds=0, max_objects=2, max_bytes=15)
        first = store.plan_retention(policy)
        assert len(first.candidates) == 1
        assert store.apply_retention(first)["deleted_bytes"] == 10
        second = store.plan_retention(RetentionPolicy(grace_seconds=0, max_objects=2, max_bytes=50))
        assert len(second.candidates) == 2


@pytest.mark.parametrize("policy", ({"grace_seconds": True}, {"grace_seconds": float("nan")},
    {"grace_seconds": -1}, {"max_objects": True}, {"max_objects": 0}, {"max_objects": 4097},
    {"max_bytes": True}, {"max_bytes": 0}, {"max_bytes": 2**41}))
def test_retention_policy_refuses_ambiguous_or_unbounded_values(policy):
    with pytest.raises(ValueError): RetentionPolicy(**policy)


@pytest.mark.parametrize("kind", ("local", "object-file", "object-memory"))
def test_replay_maintenance_and_retention_preserve_published_segments(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        seed(store); old = store.checkpoint(max_frames=2)
        store.put_record("more", Fraction(1, 7))
        new = store.checkpoint(max_frames=2)
        stage(store, kind)
        store.apply_retention(store.plan_retention(RetentionPolicy(grace_seconds=0)))
        assert new.segments[:len(old.segments)] == old.segments
    with closing(opened(kind, path)) as store:
        assert store.get("more") == Fraction(1, 7)
        assert store.verify_commits("graph")


@pytest.mark.parametrize("kind", ("object-file", "object-memory"))
def test_read_only_store_can_plan_but_cannot_apply_retention(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store: seed(store); stage(store, kind)
    with closing(opened(kind, path, read_only=True)) as store:
        plan = store.plan_retention(RetentionPolicy(grace_seconds=0))
        with pytest.raises(PermissionError, match="read-only"): store.apply_retention(plan)


@pytest.mark.parametrize("kind", ("object-file", "object-memory"))
@pytest.mark.parametrize("fault", ("before", "after", "bad_ack"))
def test_unknown_object_gc_fence_refuses_deletion_and_requires_reopen(kind, fault, tmp_path, monkeypatch):
    from rcdb import PublishedObject
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        seed(store); stage(store, kind)
        plan = store.plan_retention(RetentionPolicy(grace_seconds=0))
        before = inventory(store, kind)
        original = store._provider.compare_and_swap
        def broken(key, expected, raw):
            if key != "store.head": return original(key, expected, raw)
            if fault == "before": raise OSError("unknown fence outcome")
            result = original(key, expected, raw)
            if fault == "after": raise OSError("lost fence acknowledgement")
            return PublishedObject(b"wrong", result.token)
        monkeypatch.setattr(store._provider, "compare_and_swap", broken)
        with pytest.raises(PublicationUncertainError, match="fence"): store.apply_retention(plan)
        with pytest.raises(PublicationUncertainError): store.list()
    with closing(opened(kind, path)) as reopened:
        assert inventory(reopened, kind) == before
        assert reopened.verify_commits("graph")


@pytest.mark.parametrize("kind", ("object-file", "object-memory"))
def test_gc_fences_an_optimistic_writer_that_already_staged_against_the_old_head(kind, tmp_path):
    from rcdb import ObjectPublication
    from rcdb.object_publication import publication_for
    staged, release = Event(), Event()
    class Optimistic(ObjectPublication):
        def __init__(self, fs, root): super().__init__(fs, root); self.delegate = publication_for(fs, root)
        def has_objects(self): return self.delegate.has_objects()
        def read(self, key, *, limit): return self.delegate.read(key, limit=limit)
        def compare_and_swap(self, key, expected, raw):
            if key == "store.head" and expected is not None:
                staged.set()
                if not release.wait(10): raise RuntimeError("test writer was not released")
            return self.delegate.compare_and_swap(key, expected, raw)
    path = tmp_path/"store"
    with closing(opened(kind, path)) as cleaner, closing(opened(kind, path, publication_provider=Optimistic)) as writer:
        with pytest.raises(NotImplementedError, match="exclusive"): writer.plan_retention()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(writer.put_record, "pending", Fraction(3, 7))
            try:
                assert staged.wait(10)
                plan = cleaner.plan_retention(RetentionPolicy(grace_seconds=0))
                assert {item.namespace for item in plan.candidates} == {"blobs", "frames"}
                assert cleaner.apply_retention(plan)["deleted_objects"] == 2
            finally: release.set()
            with pytest.raises(VersionConflictError): future.result(timeout=10)
        assert cleaner.get_record("pending") is None
        assert writer.put_record("pending", Fraction(3, 7)).version == 1
        assert cleaner.get("pending") == Fraction(3, 7)
