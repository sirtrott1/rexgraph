"""The hash chain has to survive more than one process writing it.

`record` serialized on a `threading.Lock` and stamped `prev` from a process-local
cached head, so two processes appending to the same journal both extended the same
entry. The chain forked, `verify` walked it linearly and reported a break, and a real
tamper became indistinguishable from ordinary concurrency, which is the whole property.

Concurrency here is not exotic: uvicorn workers, the CLI writing while the server runs,
and a courier delivering while a request is served all reach this file.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import pytest


def _append(journal: str, n: int) -> None:
    os.environ["REXGRAPH_AUDIT_JOURNAL"] = journal
    from agent.server import audit
    audit.reset_cache()
    for i in range(n):
        audit.record("test.write", user=f"p{os.getpid()}", target=str(i))


@pytest.fixture
def journal(tmp_path, monkeypatch):
    p = tmp_path / "audit.jsonl"
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(p))
    from agent.server import audit
    audit.reset_cache()
    yield p
    audit.reset_cache()


def test_the_chain_survives_concurrent_processes(journal):
    from agent.server import audit
    audit.record("test.genesis")
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_append, args=(str(journal), 25)) for _ in range(4)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(60)
    assert all(pr.exitcode == 0 for pr in procs), [pr.exitcode for pr in procs]

    lines = [json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
    assert len(lines) == 101, f"lines lost or interleaved: {len(lines)}"

    seen = [e["prev"] for e in lines]
    assert len(set(seen)) == len(seen), "two entries extended the same head"

    result = audit.verify(journal)
    assert result["valid"], result
    assert result["n_entries"] == 101, result


def test_tampering_is_still_detected(journal):
    """The concurrency fix must not buy a clean chain by making verify vacuous."""
    from agent.server import audit
    for i in range(5):
        audit.record("test.write", target=str(i))
    lines = journal.read_text().splitlines()
    edited = json.loads(lines[2]); edited["target"] = "changed"
    lines[2] = json.dumps(edited, sort_keys=True, separators=(",", ":"))
    journal.write_text("\n".join(lines) + "\n")
    result = audit.verify(journal)
    assert not result["valid"]
    assert result["broken_at"] == 2, result


def test_a_removed_entry_is_still_detected(journal):
    from agent.server import audit
    for i in range(5):
        audit.record("test.write", target=str(i))
    lines = journal.read_text().splitlines()
    del lines[2]
    journal.write_text("\n".join(lines) + "\n")
    assert not audit.verify(journal)["valid"]


def test_the_head_is_read_from_the_end_of_a_long_trail(journal):
    """The tail read has to cross its own block boundary, not just work on short files."""
    from agent.server import audit
    last = None
    for i in range(200):
        last = audit.record("test.write", target="x" * 200 + str(i))
    assert journal.stat().st_size > 4096 * 4, journal.stat().st_size
    assert audit.head(journal) == last["digest"]
    assert audit.verify(journal)["valid"]
