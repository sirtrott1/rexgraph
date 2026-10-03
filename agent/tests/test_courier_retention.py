"""Inactive sender acknowledgements retire only after their checked archive."""
from contextlib import contextmanager, closing
from dataclasses import replace
from fractions import Fraction
import json

import pytest

from agent.courier_ledger import Ledger, LedgerRetentionPlan, LedgerUncertainError
from rcdb import MemoryStore, copy_record


@pytest.mark.parametrize("file_backed", [False, True])
def test_retention_archives_complete_receipts_and_keeps_unselected_entries(file_backed, tmp_path, monkeypatch):
    ledger = Ledger(str(tmp_path/"ledger.json") if file_backed else None)
    monkeypatch.setattr("agent.courier_ledger.time.time", lambda: 1.)
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        record = source.put_record("literal/p@1", Fraction(1, 7))
        receipt = copy_record(source, target, record, return_receipt=True)
        from rcdb import record_packet
        structure = {"state_digest": receipt.source_digest, "record_digest": record_packet(source, record.id).selection_digest}
        ledger.note("peer/p", record.id, receipt.destination_record_id, structure, receipt=receipt)
        ledger.note("other", "keep", "remote", {"state_digest": "1"*64})
        monkeypatch.setattr("agent.courier_ledger.time.time", lambda: 3.)
        ledger.note("peer/p", "active", "remote2", {"state_digest": "2"*64})
        before = ledger.to_dict()
        plan = ledger.plan_retention(before=2., peer="peer/p")
        assert LedgerRetentionPlan.from_bytes(plan.to_bytes()) == plan
        assert ledger.to_dict() == before
        result = ledger.apply_retention(plan, archive_path=tmp_path/"archive.rglr")
        assert result["retired_entries"] == 1 and result["remaining_entries"] == 2
        assert ledger.entry("peer/p", record.id) is None
        saved = LedgerRetentionPlan.from_bytes((tmp_path/"archive.rglr").read_bytes())
        entry = next(iter(json.loads(saved.archive_bytes).values()))
        assert entry == before[ledger._key("peer/p", record.id)]
        assert entry["receipt"] == receipt.to_bytes().hex()
        assert target.read_record(receipt.destination_record_id).value == Fraction(1, 7)


def test_stale_plan_refuses_without_archiving_or_losing_independent_handle_updates(tmp_path):
    first, peer = Ledger(str(tmp_path/"ledger.json")), Ledger(str(tmp_path/"ledger.json"))
    first.note("p", "a", "remote", {})
    plan = first.plan_retention(before=2**40)
    peer.note("p", "b", "remote2", {})
    with pytest.raises(ValueError, match="stale"): first.apply_retention(plan, archive_path=tmp_path/"archive")
    assert not (tmp_path/"archive").exists()
    assert len(first.entries()) == 2


def test_foreign_plan_and_forged_candidate_refuse_before_any_publication(tmp_path):
    first, other = Ledger(), Ledger()
    first.note("p", "a", "remote", {})
    plan = first.plan_retention(before=2**40)
    with pytest.raises(ValueError, match="another"): other.apply_retention(plan, archive_path=tmp_path/"a")
    data = json.loads(plan.archive_bytes)
    next(iter(data.values()))["remote_id"] = "forged"
    from agent.courier_ledger import _canonical
    forged = replace(plan, archive_bytes=_canonical(data))
    with pytest.raises(ValueError, match="candidate"): first.apply_retention(forged, archive_path=tmp_path/"a")
    assert not (tmp_path/"a").exists() and first.remote_id("p", "a") == "remote"


@pytest.mark.parametrize("kind", ["ledger", "gate", "foreign", "link"])
def test_retention_archive_cannot_replace_live_or_unrelated_files(kind, tmp_path):
    ledger = Ledger(str(tmp_path/"ledger.json")); ledger.note("p", "a", "remote", {})
    plan = ledger.plan_retention(before=2**40)
    path = {"ledger": ledger.path, "gate": ledger._gate_path}.get(kind, tmp_path/"archive")
    if kind == "foreign": path.write_bytes(b"unrelated")
    if kind == "link": path.symlink_to(ledger.path)
    original = ledger.path.read_bytes()
    with pytest.raises(ValueError): ledger.apply_retention(plan, archive_path=path)
    assert ledger.path.read_bytes() == original
    if kind == "foreign": assert path.read_bytes() == b"unrelated"


def test_archive_write_failure_leaves_live_ledger_unchanged(tmp_path, monkeypatch):
    ledger = Ledger(str(tmp_path/"ledger.json")); ledger.note("p", "a", "remote", {})
    plan = ledger.plan_retention(before=2**40)
    @contextmanager
    def broken(*args, **kwargs): raise OSError("archive unavailable"); yield
    monkeypatch.setattr("agent.courier_ledger.staged_publication", broken)
    with pytest.raises(OSError, match="archive"): ledger.apply_retention(plan, archive_path=tmp_path/"archive")
    assert ledger.remote_id("p", "a") == "remote"


@pytest.mark.parametrize("applied", [False, True])
def test_unknown_ledger_publication_always_has_a_durable_receipt_archive(tmp_path, monkeypatch, applied):
    import agent.courier_ledger as module
    ledger = Ledger(str(tmp_path/"ledger.json")); ledger.note("p", "a", "remote", {})
    plan = ledger.plan_retention(before=2**40)
    original = module.staged_publication
    @contextmanager
    def lost(path, **kwargs):
        if path != ledger.path:
            with original(path, **kwargs) as staged: yield staged
        elif applied:
            with original(path, **kwargs) as staged: yield staged
            raise OSError("lost acknowledgement")
        else:
            scratch = tmp_path/"unpublished"
            yield scratch
            raise OSError("lost acknowledgement")
    monkeypatch.setattr(module, "staged_publication", lost)
    with pytest.raises(LedgerUncertainError): ledger.apply_retention(plan, archive_path=tmp_path/"archive")
    assert LedgerRetentionPlan.from_bytes((tmp_path/"archive").read_bytes()) == plan
    with pytest.raises(LedgerUncertainError): ledger.entries()
    ledger.load()
    assert bool(ledger.entries()) == (not applied)


@pytest.mark.parametrize("options", [{"before": True}, {"before": float("inf")},
    {"before": 1., "max_entries": 0}, {"before": 1., "max_entries": True}, {"before": 1., "peer": ""}])
def test_retention_requires_explicit_bounded_selection(options):
    with pytest.raises(ValueError): Ledger().plan_retention(**options)


def test_entry_limit_codec_corruption_and_empty_plan(tmp_path, monkeypatch):
    ledger = Ledger()
    monkeypatch.setattr("agent.courier_ledger.time.time", lambda: 1.)
    for i in range(4): ledger.note("p", str(i), "remote", {})
    plan = ledger.plan_retention(before=2., max_entries=2)
    assert len(json.loads(plan.archive_bytes)) == 2
    with pytest.raises(ValueError): LedgerRetentionPlan.from_bytes(plan.to_bytes()[:-1]+b"!")
    ledger.apply_retention(plan, archive_path=tmp_path/"archive")
    assert len(ledger.entries()) == 2
    empty = ledger.plan_retention(before=0.)
    assert ledger.apply_retention(empty, archive_path=tmp_path/"empty")["retired_entries"] == 0
