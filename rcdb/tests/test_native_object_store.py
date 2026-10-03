"""Native object publication has one conditional head and immutable history."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction
import hashlib
import os
from pathlib import Path
from runpy import run_path
from threading import Barrier
import uuid

import pytest

from rcdb import (LocalStore, NativeObjectStore, ObjectPublication, ObjectStore,
                  PublicationUncertainError, PublishedObject, VersionConflictError,
                  open_object_store, open_store, migrate_legacy_batch, plan_legacy_migration)
from rcdb.object_native import ObjectHead, _HEAD_DOMAIN, _HEAD_MAGIC
from rcdb.object_publication import publication_for
from rexgraph import RexGraph
from rexgraph.value_codec import pack_value


def graph(n=2):
    return RexGraph.from_graph(list(range(n-1)), list(range(1, n)), w_E=[Fraction(1, 7)]*(n-1))


@pytest.fixture(params=["file", "memory"])
def uri(request, tmp_path):
    pytest.importorskip("fsspec")
    return "file://"+str(tmp_path / "native") if request.param == "file" else "memory://native-"+uuid.uuid4().hex


class OptimisticPublication(ObjectPublication):
    """Local test double with no transaction arbitration; only CAS serializes."""
    def __init__(self, fs, root, *, barrier=None):
        super().__init__(fs, root)
        self.delegate = publication_for(fs, root)
        self.barrier = barrier
        self.fault = None
    def has_objects(self): return self.delegate.has_objects()
    def read(self, key, *, limit): return self.delegate.read(key, limit=limit)
    def compare_and_swap(self, key, expected, raw):
        if key == "store.head" and expected is not None:
            if self.barrier is not None: self.barrier.wait(timeout=15)
            if self.fault == "reject": raise VersionConflictError("known negative CAS")
            if self.fault == "before": raise OSError("no acknowledgement")
            result = self.delegate.compare_and_swap(key, expected, raw)
            if self.fault == "after": raise OSError("lost acknowledgement")
            if self.fault == "bad_ack": return PublishedObject(b"wrong", result.token)
            return result
        return self.delegate.compare_and_swap(key, expected, raw)


def test_read_only_open_and_operations_never_publish(uri, monkeypatch):
    with closing(NativeObjectStore(uri)) as writer:
        writer.put("r", graph(), analytics=False)
        writer.put_record("none", None)
        expected, identity = writer.change_cursor, writer.store_id
    def read_only_provider(fs, root):
        cap = publication_for(fs, root)
        def refuse(*args, **kwargs): pytest.fail("read-only native source published")
        monkeypatch.setattr(cap, "compare_and_swap", refuse)
        monkeypatch.setattr(cap, "transaction", refuse)
        return cap
    with closing(NativeObjectStore(uri, read_only=True, publication_provider=read_only_provider)) as source:
        assert source.store_id == identity and source.change_cursor == expected
        assert source.get("r").nE == 1 and source.read_record("none").value is None
        assert source.compact()["history_retained"] and source.stats()["native"]
        for op in (lambda: source.put("x", graph()), lambda: source.put_record("x", 1),
                   lambda: source.delete("r"), lambda: source.commit_mutation("r", graph())):
            with pytest.raises(PermissionError, match="read-only"): op()


def test_factory_detects_native_and_keeps_legacy_source_explicit(uri):
    with closing(open_object_store(uri)) as native:
        assert isinstance(native, NativeObjectStore)
        native.put_record("exact", Fraction(2, 7))
    with closing(open_object_store(uri, read_only=True)) as native:
        assert native.get("exact") == Fraction(2, 7)
    if uri.startswith("file://"):
        with closing(open_store("auto://"+uri[7:])) as native:
            assert isinstance(native, NativeObjectStore)
    legacy_uri = uri+"-legacy"
    with closing(ObjectStore(legacy_uri, read_only=False)) as old:
        old.put("r", graph(), analytics=False)
    with closing(open_object_store(legacy_uri)) as source:
        assert type(source) is ObjectStore and source.read_only
    with pytest.raises(ValueError, match="conflicting"):
        open_object_store(legacy_uri, native=True)


def test_legacy_history_migrates_into_native_objects_via_existing_copy_seam(uri):
    with closing(ObjectStore(uri+"-legacy", read_only=False)) as old:
        old.put("literal@1", graph(), meta={"q": Fraction(2, 7)}, analytics=False, _tx_time=10.0)
        old.put("literal@1", graph(3), analytics=False, _tx_time=20.0)
    with closing(ObjectStore(uri+"-legacy")) as source, closing(NativeObjectStore(uri)) as target:
        plan = plan_legacy_migration(source, target)
        result = migrate_legacy_batch(source, target, plan, accept_loss=True)
        assert result.complete and len(result.receipts) == 2
        assert all(r.source_digest == r.destination_digest for r in result.receipts)
        assert target.get_version("literal@1", 1).edge_metric_exact[0] == Fraction(1, 7)
        assert target.history("literal@1")[0].meta["q"] == Fraction(2, 7)


def test_concurrent_initialization_owns_one_identity(uri):
    barrier = Barrier(6)
    def initialize(_):
        barrier.wait(timeout=15)
        with closing(NativeObjectStore(uri)) as store: return store.store_id
    with ThreadPoolExecutor(max_workers=6) as pool:
        identities = list(pool.map(initialize, range(6)))
    assert len(set(identities)) == 1


def test_missing_read_only_prefix_and_unowned_data_never_acquire_an_identity(uri):
    from rcdb.objectstore import _fs_for
    fs, root = _fs_for(uri)
    with pytest.raises(ValueError, match="existing head"):
        NativeObjectStore(uri, read_only=True)
    assert not fs.exists(root)
    fs.makedirs(root, exist_ok=True)
    with fs.open(root+"/unowned", "wb") as stream: stream.write(b"existing")
    with pytest.raises(ValueError, match="existing data"):
        NativeObjectStore(uri)
    assert not fs.exists(root+"/store.head") and not fs.exists(root+"/store.header")


def test_lost_genesis_acknowledgement_recovers_the_same_embedded_identity(uri):
    captured = []
    def provider(fs, root):
        cap = OptimisticPublication(fs, root)
        real = cap.compare_and_swap
        def lost(key, expected, raw):
            result = real(key, expected, raw)
            if key == "store.head":
                captured.append(ObjectHead.from_bytes(raw).header.identity.id)
                raise OSError("lost genesis acknowledgement")
            return result
        cap.compare_and_swap = lost
        return cap
    with pytest.raises(PublicationUncertainError, match="initialization"):
        NativeObjectStore(uri, publication_provider=provider)
    with closing(NativeObjectStore(uri)) as store:
        assert store.store_id == captured[0] and store.change_cursor.sequence == 0
        store.put_record("r", Fraction(1, 7))


def test_missing_authoritative_head_of_an_empty_owned_store_is_not_recreated(uri):
    with closing(NativeObjectStore(uri)) as store:
        header = store.header.to_bytes()
        store.fs.rm(store._provider.path("store.head"))
    with pytest.raises(ValueError, match="head is missing"): NativeObjectStore(uri)
    from rcdb.objectstore import _fs_for
    fs, root = _fs_for(uri)
    with fs.open(root+"/store.header", "rb") as stream: assert stream.read() == header
    assert not fs.exists(root+"/store.head")


def test_independent_writers_allocate_complete_versions(uri):
    peers = [NativeObjectStore(uri) for _ in range(4)]
    barrier = Barrier(len(peers))
    def write(store):
        barrier.wait(timeout=15)
        return [store.put_record("shared", Fraction(n, 7)).version for n in range(3)]
    try:
        with ThreadPoolExecutor(max_workers=len(peers)) as pool:
            versions = [v for group in pool.map(write, peers) for v in group]
        assert sorted(versions) == list(range(1, 13))
        assert peers[0].change_cursor.sequence == 12
    finally:
        for peer in peers: peer.close()


def test_competing_commits_cannot_replace_the_winners_artifact(uri):
    barrier = Barrier(2)
    provider = lambda fs, root: OptimisticPublication(fs, root, barrier=barrier)
    peers = [NativeObjectStore(uri, publication_provider=provider) for _ in range(2)]
    def write(pair):
        i, peer = pair
        try:
            return peer.commit_mutation("r", graph(i+2), analytics=False, actor=str(i), tx_time=10.0)
        except VersionConflictError:
            return None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool: results = list(pool.map(write, enumerate(peers)))
        winner = [i for i, result in enumerate(results) if result is not None]
        assert len(winner) == 1
        with closing(NativeObjectStore(uri)) as reopened:
            assert reopened.verify_commits("r") and reopened.get("r").nE == winner[0]+1
            assert len(reopened.commit_history("r")) == 1 and reopened.change_cursor.sequence == 1
            frame = reopened.changes()[0]
            raw = reopened._load_commit_bytes("r", 1)
            assert hashlib.sha256(raw).hexdigest() == frame.mutation.commit_digest
    finally:
        for peer in peers: peer.close()


@pytest.mark.parametrize("kind", ["graph", "value", "commit", "delete"])
@pytest.mark.parametrize("fault", ["reject", "before", "after", "bad_ack"])
def test_head_acknowledgement_contract_and_recovery(uri, kind, fault):
    provider = lambda fs, root: OptimisticPublication(fs, root)
    with closing(NativeObjectStore(uri, publication_provider=provider)) as store:
        if kind == "delete": store.put("r", graph(), analytics=False, _tx_time=1.0)
        before = store.change_cursor
        store._provider.fault = fault
        exception = VersionConflictError if fault == "reject" else PublicationUncertainError
        with pytest.raises(exception):
            if kind == "graph": store.put("r", graph(), analytics=False, _tx_time=10.0)
            elif kind == "value": store.put_record("r", Fraction(2, 7), tx_time=10.0)
            elif kind == "commit": store.commit_mutation("r", graph(), analytics=False, tx_time=10.0)
            else: store.delete("r", tx_time=10.0)
        if fault == "reject":
            assert store.change_cursor == before
        else:
            for operation in (store.list, lambda: store.put_record("s", 1)):
                with pytest.raises(PublicationUncertainError): operation()
    with closing(NativeObjectStore(uri)) as reopened:
        published = fault in {"after", "bad_ack"}
        assert reopened.change_cursor.sequence == before.sequence+int(published)
        if kind == "delete": assert (reopened.get("r") is None) == published
        else: assert (reopened.read_record("r") is not None) == published
        if kind == "commit" and published: assert reopened.verify_commits("r")


@pytest.mark.parametrize("stage", ["blob", "frame", "commit"])
def test_failed_immutable_staging_never_publishes_a_record(uri, stage, monkeypatch):
    with closing(NativeObjectStore(uri)) as store:
        before = store.change_cursor
        real = store._provider.compare_and_swap
        prefix = {"blob": "blobs/", "frame": "frames/", "commit": "commits/"}[stage]
        def fail(key, expected, raw):
            if key.startswith(prefix): raise OSError("staging interrupted")
            return real(key, expected, raw)
        with monkeypatch.context() as patch:
            patch.setattr(store._provider, "compare_and_swap", fail)
            with pytest.raises(OSError, match="staging"):
                store.commit_mutation("r", graph(), analytics=False)
        assert store.change_cursor == before and store.get("r") is None
        assert store.commit_mutation("r", graph(), analytics=False).version == 1
        assert store.verify_commits("r")


@pytest.mark.parametrize("damage", ["header", "head", "frame", "blob", "commit", "blob-bytes", "commit-bytes", "rollback"])
def test_authoritative_damage_refuses_and_never_regenerates_ownership(uri, damage):
    with closing(NativeObjectStore(uri)) as store:
        genesis = store._provider.read("store.head", limit=20000).data
        store.commit_mutation("r", graph(), analytics=False)
        identity = store.store_id
        frame = store.changes()[0]
        key = {"header": "store.header", "head": "store.head", "frame": "frames/"+frame.digest,
               "blob": "blobs/"+frame.mutation.blob_digest, "commit": "commits/"+frame.mutation.commit_digest,
               "blob-bytes": "blobs/"+frame.mutation.blob_digest, "commit-bytes": "commits/"+frame.mutation.commit_digest,
               "rollback": "store.head"}[damage]
        path = store._provider.path(key)
        if damage.endswith("bytes") or damage == "rollback":
            with store.fs.open(path, "wb") as stream: stream.write(genesis if damage == "rollback" else b"wrong")
        else: store.fs.rm(path)
        with pytest.raises(ValueError):
            if damage in {"blob", "commit", "blob-bytes", "commit-bytes", "rollback"}: store.get("r", verify=False)
            else: NativeObjectStore(uri)
        assert store.store_id == identity


@pytest.mark.parametrize("damage", ["digest", "extra", "bool", "trailing", "ownership", "genesis"])
def test_head_is_closed_canonical_and_owned(uri, damage):
    with closing(NativeObjectStore(uri)) as store:
        head = ObjectHead(store.header, store.change_cursor)
        assert ObjectHead.from_bytes(head.to_bytes()) == head
        record = head.as_record()
        if damage == "extra": record["provider"] = "untrusted.module"
        elif damage == "bool": record["head_version"] = True
        elif damage == "ownership": record["cursor"]["store_id"] = "f"*32
        elif damage == "genesis": record["cursor"]["digest"] = "f"*64
        body = pack_value(record)+(b"extra" if damage == "trailing" else b"")
        raw = _HEAD_MAGIC+hashlib.sha256(_HEAD_DOMAIN+body).digest()+body
        if damage == "digest": raw = raw[:5]+b"0"*32+raw[37:]
        with pytest.raises(ValueError): ObjectHead.from_bytes(raw)


def test_reads_pin_selection_while_another_writer_publishes(uri, monkeypatch):
    with closing(NativeObjectStore(uri)) as owner, closing(NativeObjectStore(uri)) as other:
        owner.put("r", graph(), analytics=False)
        first = owner.change_cursor
        with owner.read_transaction():
            other.put("r", graph(3), analytics=False)
            assert owner.get("r").nE == 1 and owner.change_cursor == first
            with pytest.raises(ValueError, match="pinned"): owner.put_record("x", 1)
        assert owner.get("r").nE == 2
        def refuse(*args, **kwargs): pytest.fail("ordinary native read listed objects")
        monkeypatch.setattr(owner.fs, "ls", refuse)
        monkeypatch.setattr(owner.fs, "find", refuse)
        assert owner.get("r").nE == 2 and len(owner.history("r")) == 2


@pytest.mark.parametrize("backend", ["local", "object"])
def test_corpus_cache_refreshes_between_handles_and_retains_old_images(uri, backend, tmp_path):
    factory = (lambda: LocalStore(tmp_path / "local")) if backend == "local" else (lambda: NativeObjectStore(uri))
    with closing(factory()) as owner, closing(factory()) as other:
        owner.put("r", graph(), analytics=False, tags=["old"])
        old = owner.corpus_snapshot()
        assert old.response(["old"])["scores"] == (Fraction(1),)
        other.put("r", graph(3), analytics=False, tags=["new"])
        new = owner.corpus_snapshot()
        assert new.digest != old.digest and new.versions == (2,)
        assert new.response(["old"])["scores"] == (Fraction(0),)
        assert old.response(["old"])["scores"] == (Fraction(1),)
        other.delete("r")
        assert owner.corpus_snapshot().ids == ()


@pytest.mark.parametrize("backend", ["local", "object"])
def test_pinned_read_transactions_refuse_publication_and_keep_their_cursor(uri, backend, tmp_path):
    factory = (lambda: LocalStore(tmp_path / "local")) if backend == "local" else (lambda: NativeObjectStore(uri))
    with closing(factory()) as store:
        store.put_record("r", Fraction(1, 7))
        cursor = store.change_cursor
        with store.read_transaction():
            for operation in (lambda: store.put_record("r", 2), lambda: store.delete("r"),
                              lambda: store.commit_mutation("graph", graph(), analytics=False)):
                with pytest.raises(ValueError, match="pinned"): operation()
            assert store.change_cursor == cursor and store.get("r") == Fraction(1, 7)


def test_compact_retains_all_history_and_closed_handles_refuse(uri):
    store = NativeObjectStore(uri)
    store.put("r", graph(), analytics=False)
    store.delete("r")
    cursor = store.change_cursor
    assert store.compact()["history_retained"] and store.change_cursor == cursor
    assert store.get_version("r", 1).nE == 1 and store.tombstone("r") is not None
    store.close(); store.close()
    for operation in (store.stats, store.list, lambda: store.get("r"), lambda: store.put_record("r", 1)):
        with pytest.raises(ValueError, match="closed"): operation()


@pytest.mark.parametrize("bounds", [{"include_history": 1}, {"limit": True}, {"as_of": float("nan")}, {"unknown": 1}])
def test_invalid_queries_refuse_on_empty_state(uri, bounds):
    with closing(NativeObjectStore(uri)) as store:
        with pytest.raises((TypeError, ValueError)): store.query(**bounds)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file object publication qualification")
def test_fresh_process_file_writers_share_one_version_axis(tmp_path):
    run_isolated = run_path(str(Path(__file__).resolve().parents[2] / "scripts/test_subprocess.py"))["run_isolated"]
    uri = "file://"+str(tmp_path / "process")
    code = f"""
from rcdb import NativeObjectStore
from fractions import Fraction
s = NativeObjectStore({uri!r})
try:
 for i in range(3): s.put_record('r', Fraction(i, 7))
finally: s.close()
"""
    def child(_): return run_isolated(code, packages=("rcdb", "rexgraph"), cwd=tmp_path, capture_output=True, timeout=45)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(child, range(4)): assert result.returncode == 0, result.stderr.decode()
    with closing(NativeObjectStore(uri)) as store:
        assert [r.version for r in store.history("r")] == list(range(1, 13))
        assert store.change_cursor.sequence == 12


@pytest.mark.skipif(os.name != "posix", reason="POSIX file object publication qualification")
@pytest.mark.parametrize("when", ["before", "after"])
def test_abrupt_process_exit_at_the_head_preserves_a_complete_publication(tmp_path, when):
    run_isolated = run_path(str(Path(__file__).resolve().parents[2] / "scripts/test_subprocess.py"))["run_isolated"]
    uri = "file://"+str(tmp_path / "abrupt")
    with closing(NativeObjectStore(uri)) as store: identity = store.store_id
    code = f"""
import os
from rcdb import NativeObjectStore
from rexgraph import RexGraph
s = NativeObjectStore({uri!r})
real = s._provider.compare_and_swap
def exit_at_head(key, expected, raw):
 if key == 'store.head' and {when!r} == 'before': os._exit(73)
 result = real(key, expected, raw)
 if key == 'store.head': os._exit(73)
 return result
s._provider.compare_and_swap = exit_at_head
s.commit_mutation('r', RexGraph.from_graph([0], [1]), analytics=False)
"""
    result = run_isolated(code, packages=("rcdb", "rexgraph"), cwd=tmp_path, capture_output=True, timeout=45)
    assert result.returncode == 73, result.stderr.decode()
    with closing(NativeObjectStore(uri)) as store:
        assert store.store_id == identity and store.change_cursor.sequence == int(when == "after")
        if when == "after": assert store.get("r").nE == 1 and store.verify_commits("r")
        else: assert store.get("r") is None and store.commit_history("r") == []
