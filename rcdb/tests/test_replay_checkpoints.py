"""Replay coalesces physical reads while preserving canonical history and checks."""
from contextlib import closing
from dataclasses import replace
from fractions import Fraction
from contextlib import contextmanager

import pytest

from rcdb import (LocalStore, MemoryStore, NativeObjectStore, ReplayCheckpoint, ReplaySegment,
                  SQLStore, copy_record, record_packet)
from rcdb.checkpoint import build_checkpoint
from rcdb.object_native import ObjectHead
from rexgraph import RexGraph
from rexgraph.value_codec import pack_value


KINDS = ("local", "object-file", "object-memory")


def opened(kind, root, **options):
    if kind == "local": return LocalStore(root, **options)
    pytest.importorskip("fsspec")
    return NativeObjectStore(f"{'file' if kind == 'object-file' else 'memory'}://{root}", **options)


def seed(store, count=17):
    for i in range(count):
        store.put_record("literal/r@1", None if i == 7 else Fraction(i, 7), tx_time=float(i),
                         valid_from=float(2*i), valid_to=float(2*i+1), meta={"q": Fraction(i, 11)})
    store.delete("literal/r@1", tx_time=float(count))
    store.put_record("literal/r@1", Fraction(9, 7), tx_time=float(count+1))
    graph = RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)])
    store.commit_mutation("graph", graph, analytics=False, tx_time=1.)


def state_bytes(state):
    return pack_value({"rows": tuple(row.to_dict() for row in state.records(include_history=True)),
        "cursor": state.cursor.as_record(), "changes": tuple(frame.to_bytes() for frame in state.changes(limit=2**63-1)),
        "tombstones": tuple((rid, state.tombstone(rid).as_record()) for rid in sorted(state._tombstones))})


def test_canonical_segment_roundtrip_restores_exact_complete_state_and_rejects_content_changes():
    with closing(MemoryStore()) as store:
        seed(store)
        segments = {}
        checkpoint = build_checkpoint(store._state, lambda ref, raw: segments.setdefault(ref.digest, raw), max_frames=4)
        assert len(checkpoint.segments) == 5
        assert ReplayCheckpoint.from_bytes(checkpoint.to_bytes()) == checkpoint
        assert state_bytes(checkpoint.restore(lambda ref: segments[ref.digest])) == state_bytes(store._state)
        reference = checkpoint.segments[0]
        raw = segments[reference.digest]
        assert ReplaySegment.from_bytes(raw, reference=reference).reference == reference
        with pytest.raises(ValueError, match="content address"):
            ReplaySegment.from_bytes(raw[:-1]+bytes([raw[-1]^1]), reference=reference)
        with pytest.raises(ValueError, match="digest"):
            ReplayCheckpoint.from_bytes(checkpoint.to_bytes()[:-1]+b"!")
        with pytest.raises(ValueError):
            replace(checkpoint, segments=tuple(reversed(checkpoint.segments)))


def test_checksum_valid_invalid_transition_is_not_a_trusted_snapshot():
    with closing(MemoryStore()) as store:
        store.put_record("r", None, tx_time=1.)
        first = store.changes()[0]
        bad = replace(first, mutation=replace(first.mutation, previous_version=1))
        end = replace(store.change_cursor, digest=bad.digest)
        segment = ReplaySegment(store._state.initial_cursor(), end, (bad.to_bytes(),))
        checkpoint = ReplayCheckpoint(store.header, (segment.reference,))
        with pytest.raises(ValueError, match="previous visible"):
            checkpoint.restore(lambda ref: segment.to_bytes())


@pytest.mark.parametrize("options", ({"max_frames": True}, {"max_frames": 0}, {"max_frames": 1025},
                                    {"target_bytes": True}, {"target_bytes": 0}, {"target_bytes": 2**40}))
def test_invalid_targets_do_not_publish_segments(options):
    with closing(MemoryStore()) as store:
        store.put_record("r", None)
        writes = []
        with pytest.raises(ValueError): build_checkpoint(store._state, lambda *args: writes.append(args), **options)
        assert not writes


@pytest.mark.parametrize("kind", KINDS)
def test_checkpoint_retains_history_cursors_exact_decode_copy_and_rcql_after_reopen(kind, tmp_path):
    from rcql import Executor, parse
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        seed(store)
        before, cursor = state_bytes(store._state), store.change_cursor
        checkpoint = store.checkpoint(max_frames=4)
        assert checkpoint.cursor == cursor and state_bytes(store._state) == before
        assert store.checkpoint(max_frames=4) == checkpoint
    with closing(opened(kind, path)) as store, closing(MemoryStore()) as target:
        assert state_bytes(store._state) == before and store.change_cursor == cursor
        assert store.read_record("literal/r@1", version=8).value is None
        assert store.read_record("literal/r@1", version=1).value == Fraction(0, 7)
        assert store.get_record("literal/r@1", as_of=17.5) is None
        assert store.get_record("literal/r@1", valid_at=0.5).version == 1
        packet = record_packet(store, "graph", version=1)
        receipt = copy_record(store, target, packet.record, return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest == packet.state_digest
        assert store.verify_commits("graph")
        assert Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_VERSION($db,"graph",1) RETURN RANK(1), 1 / 7')).values == (1, Fraction(1, 7))
        assert store.put_record("literal/r@1", Fraction(10, 7), tx_time=19.).version == 19


@pytest.mark.parametrize("kind", KINDS)
def test_incremental_checkpoint_preserves_old_segments_and_pinned_reader_prefix(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as first:
        seed(first, 8)
        old = first.checkpoint(max_frames=4)
        first.put_record("new", Fraction(1, 7), tx_time=1.)
        with closing(opened(kind, path)) as peer:
            assert peer.read_record("new").value == Fraction(1, 7)
            new = peer.checkpoint(max_frames=4)
        assert new.segments[:len(old.segments)] == old.segments
        assert new.cursor.sequence == old.cursor.sequence+1
        if kind == "local": read = first._read_segment
        else: read = lambda ref: first._read("replay/"+ref.digest, ref.size).data
        assert old.restore(read).cursor == old.cursor
        assert first.read_record("new").value == Fraction(1, 7)


@pytest.mark.parametrize("kind", KINDS)
def test_checkpoint_inside_read_scope_is_refused(kind, tmp_path):
    with closing(opened(kind, tmp_path/"store")) as store:
        seed(store, 2)
        cursor = store.change_cursor
        with store.read_transaction():
            with pytest.raises(ValueError, match="pinned"): store.checkpoint()
        assert store.change_cursor == cursor


@pytest.mark.parametrize("kind", KINDS)
def test_reopen_refuses_missing_or_corrupt_published_segments(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        seed(store, 8)
        checkpoint = store.checkpoint(max_frames=4)
        reference = checkpoint.segments[0]
        if kind == "local":
            (store._replay/reference.digest).write_bytes(b"corrupt")
        else:
            with store.fs.open(store._provider.path("replay/"+reference.digest), "wb") as stream:
                stream.write(b"corrupt")
    with pytest.raises(ValueError, match="content address"):
        opened(kind, path)


def test_local_checkpoint_still_checks_entire_original_physical_prefix(tmp_path):
    path = tmp_path/"store"
    with closing(LocalStore(path)) as store:
        seed(store, 8)
        store.checkpoint(max_frames=4)
        raw = bytearray(store._journal_path.read_bytes())
        raw[200] ^= 1
        store._journal_path.write_bytes(raw)
        with pytest.raises(ValueError): store.checkpoint(max_frames=4)
    with pytest.raises(ValueError, match="prefix"):
        LocalStore(path)


def test_local_absent_optional_cache_falls_back_to_checked_primary_journal(tmp_path):
    path = tmp_path/"store"
    with closing(LocalStore(path)) as store:
        seed(store, 8)
        before = state_bytes(store._state)
        store.checkpoint(max_frames=4)
        store._checkpoint_path.unlink()
    with closing(LocalStore(path)) as store:
        assert state_bytes(store._state) == before


@pytest.mark.parametrize("kind", ("object-file", "object-memory"))
def test_object_fresh_replay_reads_segments_instead_of_each_prefix_frame(kind, tmp_path, monkeypatch):
    from rcdb.object_publication import publication_for
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        seed(store, 32)
        checkpoint = store.checkpoint(max_frames=8)
        store.put_record("tail", Fraction(1, 7), tx_time=1.)
        head = ObjectHead.from_bytes(store._read("store.head", 2**23).data)
        assert head.checkpoint == checkpoint and head.epoch == 1
        expected = store.change_cursor
    reads = []
    def provider(fs, root):
        cap = publication_for(fs, root)
        original = cap.read
        def read(key, **kwargs): reads.append(key); return original(key, **kwargs)
        monkeypatch.setattr(cap, "read", read)
        return cap
    with closing(opened(kind, path, publication_provider=provider)) as reopened:
        assert reopened._state.cursor == expected
        assert len([key for key in reads if key.startswith("frames/")]) == 1
        assert len([key for key in reads if key.startswith("replay/")]) == len(checkpoint.segments)


@pytest.mark.parametrize("kind", ("object-file", "object-memory"))
def test_read_only_object_checkpoint_refuses_before_writing(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store: seed(store, 2)
    with closing(opened(kind, path, read_only=True)) as store:
        with pytest.raises(PermissionError, match="read-only"): store.checkpoint()


@pytest.mark.parametrize("kind", ("memory", "sql"))
def test_providers_without_persistent_segment_capability_are_explicit(kind):
    if kind == "sql": pytest.importorskip("sqlalchemy")
    with closing(MemoryStore() if kind == "memory" else SQLStore("sqlite://")) as store:
        with pytest.raises(NotImplementedError, match="persistent replay"): store.checkpoint()


@pytest.mark.parametrize("kind", KINDS)
def test_empty_checkpoint_preserves_genesis_and_next_publication(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        cursor = store.change_cursor
        checkpoint = store.checkpoint()
        assert not checkpoint.segments and checkpoint.cursor == cursor
    with closing(opened(kind, path)) as store:
        assert store.change_cursor == cursor
        assert store.put_record("first", None).version == 1


@pytest.mark.parametrize("kind", ("object-file", "object-memory"))
@pytest.mark.parametrize("fault", ("reject", "before", "after", "bad_ack"))
def test_checkpoint_conditional_publication_faults_preserve_history_and_require_checked_recovery(kind, fault, tmp_path, monkeypatch):
    from rcdb import PublicationUncertainError, PublishedObject, VersionConflictError
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        seed(store, 2)
        before = state_bytes(store._state)
        original = store._provider.compare_and_swap
        def broken(key, expected, raw):
            if key != "store.head": return original(key, expected, raw)
            if fault == "reject": raise VersionConflictError("known negative")
            if fault == "before": raise OSError("unknown outcome")
            result = original(key, expected, raw)
            if fault == "after": raise OSError("lost acknowledgement")
            return PublishedObject(b"wrong", result.token)
        monkeypatch.setattr(store._provider, "compare_and_swap", broken)
        with pytest.raises(VersionConflictError if fault == "reject" else PublicationUncertainError): store.checkpoint()
        assert state_bytes(store._state) == before
        if fault == "reject": assert store.get("literal/r@1") == Fraction(9, 7)
        else:
            with pytest.raises(PublicationUncertainError): store.list()
    with closing(opened(kind, path)) as store:
        assert state_bytes(store._state) == before
        assert bool(store._head_checkpoint) == (fault in ("after", "bad_ack"))
        assert store.verify_commits("graph")


@pytest.mark.parametrize("applied", (False, True))
def test_local_checkpoint_unknown_pointer_publication_preserves_authoritative_journal(tmp_path, monkeypatch, applied):
    from rcdb import PublicationUncertainError
    import rcdb.localstore as module
    path = tmp_path/"store"
    with closing(LocalStore(path)) as store:
        seed(store, 2)
        before, journal = state_bytes(store._state), store._journal_path.read_bytes()
        original = module.staged_publication
        @contextmanager
        def lost(target, **kwargs):
            if target != store._checkpoint_path:
                with original(target, **kwargs) as staged: yield staged
            elif applied:
                with original(target, **kwargs) as staged: yield staged
                raise OSError("lost pointer acknowledgement")
            else:
                yield tmp_path/"unpublished"
                raise OSError("lost pointer acknowledgement")
        monkeypatch.setattr(module, "staged_publication", lost)
        with pytest.raises(PublicationUncertainError): store.checkpoint()
        assert store._journal_path.read_bytes() == journal
        with pytest.raises(PublicationUncertainError): store.list()
    with closing(LocalStore(path)) as store:
        assert state_bytes(store._state) == before
        assert store.verify_commits("graph")


@pytest.mark.parametrize("kind", ("object-file", "object-memory"))
def test_warm_object_reader_refuses_maintenance_epoch_regression(kind, tmp_path):
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        store.put_record("r", None)
        store.checkpoint()
        store.put_record("tail", None)
        store.checkpoint()
        # Same logical cursor but an earlier maintenance epoch is not an extension.
        raw = ObjectHead(store.header, store._state.cursor, store._head_checkpoint, 1).to_bytes()
        store._provider.compare_and_swap("store.head", store._head_token, raw)
        with pytest.raises(ValueError, match="epoch regressed"): store.list()


def test_native_frame_digest_is_computed_once_but_subclasses_keep_dynamic_behavior(monkeypatch):
    from rcdb.journal import JournalFrame
    with closing(MemoryStore()) as store:
        store.put_record("r", None)
        frame = replace(store.changes()[0])
        calls, original = [], JournalFrame._body
        def body(self): calls.append(self); return original(self)
        monkeypatch.setattr(JournalFrame, "_body", body)
        first = frame.digest
        for _ in range(8): assert frame.digest == first
        assert len(calls) == 1
        changed = replace(frame, sequence=2)
        assert changed.digest != first and len(calls) == 2
        class DynamicFrame(JournalFrame):
            def _body(self): return original(self)+self.suffix
        dynamic = DynamicFrame(**{name: getattr(frame, name) for name in frame.__dataclass_fields__})
        object.__setattr__(dynamic, "suffix", b"a")
        a = dynamic.digest
        object.__setattr__(dynamic, "suffix", b"b")
        assert dynamic.digest != a and "_native_digest" not in dynamic.__dict__


@pytest.mark.parametrize("kind", ("object-file", "object-memory"))
@pytest.mark.parametrize("operation", ("put", "checkpoint", "retention"))
def test_interruption_preserves_control_flow_and_poisons_unknown_head_outcomes(kind, operation, tmp_path, monkeypatch):
    from rcdb import PublicationUncertainError, RetentionPolicy
    path = tmp_path/"store"
    with closing(opened(kind, path)) as store:
        seed(store, 2)
        if operation == "retention":
            import hashlib
            raw = b"orphan"
            with store.write_scope(): store._write_blob(hashlib.sha256(raw).hexdigest(), raw)
            plan = store.plan_retention(RetentionPolicy(grace_seconds=0))
        original = store._provider.compare_and_swap
        def interrupted(key, expected, raw):
            result = original(key, expected, raw)
            if key == "store.head": raise KeyboardInterrupt
            return result
        monkeypatch.setattr(store._provider, "compare_and_swap", interrupted)
        with pytest.raises(KeyboardInterrupt):
            if operation == "put": store.put_record("new", None)
            elif operation == "checkpoint": store.checkpoint()
            else: store.apply_retention(plan)
        with pytest.raises(PublicationUncertainError): store.list()
    with closing(opened(kind, path)) as reopened:
        assert reopened.verify_commits("graph")
        assert (reopened.get_record("new") is not None) == (operation == "put")
