"""An anchor is what makes a rewritten trail detectable.

The chain proves no entry was edited in place. It cannot prove the tail was not
rewritten wholesale, because whoever rewrites it recomputes every digest from the point
they changed and `verify` then reports a perfectly valid chain. An anchor is a statement
kept off the box about where the trail stood at one moment, so a rewrite has to also
change a record the rewriter does not hold.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def trail(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_AUDIT_ANCHORS", str(tmp_path / "anchors.jsonl"))
    from agent.server import audit
    audit.reset_cache()
    yield audit, tmp_path / "audit.jsonl", tmp_path / "anchors.jsonl"
    audit.reset_cache()


def _entries(journal):
    return [json.loads(x) for x in journal.read_text().splitlines() if x.strip()]


def _rewrite(audit, journal, entries):
    """Rebuild a chain that verify() accepts, the way an attacker with write access would."""
    prev = audit.GENESIS
    lines = []
    for e in entries:
        body = {k: v for k, v in e.items() if k != "digest"}
        body["prev"] = prev
        body["digest"] = audit._digest(body)
        prev = body["digest"]
        lines.append(json.dumps(body, sort_keys=True, separators=(",", ":")))
    journal.write_text("\n".join(lines) + "\n")


def test_an_honest_trail_verifies_against_its_anchors(trail):
    audit, journal, _ = trail
    for i in range(5):
        audit.record("test.write", target=str(i))
    audit.anchor()
    for i in range(3):
        audit.record("test.write", target=f"later{i}")
    result = audit.verify_against_anchors()
    assert result["valid"], result
    assert result["n_anchors"] == 1


def test_a_rewrite_passes_verify_but_fails_the_anchor(trail):
    audit, journal, _ = trail
    for i in range(5):
        audit.record("test.write", target=str(i))
    audit.anchor()

    kept = _entries(journal)
    del kept[2]
    kept.append(dict(kept[-1], target="forged"))     # same length, different history
    _rewrite(audit, journal, kept)

    assert audit.verify(journal)["valid"], "the rewrite should look clean to the chain alone"
    result = audit.verify_against_anchors()
    assert not result["valid"]
    assert result["reason"] == "the trail was rewritten behind an anchor", result


def test_truncation_is_caught(trail):
    audit, journal, _ = trail
    for i in range(5):
        audit.record("test.write", target=str(i))
    audit.anchor()
    _rewrite(audit, journal, _entries(journal)[:3])

    assert audit.verify(journal)["valid"]
    result = audit.verify_against_anchors()
    assert not result["valid"]
    assert result["reason"] == "the trail is shorter than an anchor witnessed", result


def test_an_unsigned_anchor_says_so(trail):
    audit, _, _ = trail
    audit.record("test.write")
    assert audit.anchor()["signed"] is False


def test_a_forged_anchor_is_caught_when_anchors_are_signed(trail, monkeypatch):
    audit, journal, anchors = trail
    monkeypatch.setenv("REXGRAPH_ANCHOR_KEY", "ANCHOR_SECRET_REF")
    monkeypatch.setenv("ANCHOR_SECRET_REF", "s3cret-not-on-the-box")
    for i in range(5):
        audit.record("test.write", target=str(i))
    written = audit.anchor()
    assert written["signed"] is True

    kept = _entries(journal)
    del kept[2]
    kept.append(dict(kept[-1], target="forged"))
    _rewrite(audit, journal, kept)
    # the attacker rewrites the anchor to match the new history but cannot sign it
    forged = dict(written, head=audit.head(journal))
    anchors.write_text(json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n")

    result = audit.verify_against_anchors()
    assert not result["valid"]
    assert result["reason"] == "an anchor was forged or the key changed", result


def test_with_no_anchors_it_reduces_to_verify(trail):
    audit, journal, _ = trail
    for i in range(3):
        audit.record("test.write", target=str(i))
    result = audit.verify_against_anchors()
    assert result["valid"] and result["n_anchors"] == 0
    _rewrite(audit, journal, _entries(journal)[:2])
    assert audit.verify_against_anchors()["valid"], \
        "a chain never witnessed cannot detect its own wholesale rewrite"
