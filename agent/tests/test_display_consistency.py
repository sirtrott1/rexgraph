"""What the app shows is what the code computed.

Two complexes built under different rules are not comparable, and a number rendered
from a different computation than the one it names is worse than no number. These
check the chain end to end: one document, through the real HTTP route the screen
calls, against the complex recomputed directly.
"""
from __future__ import annotations


import numpy as np
import pytest
from fastapi.testclient import TestClient

DOC = (b"Alpha connects beta. Beta connects gamma. Gamma connects alpha. "
       b"Delta connects alpha. Alpha connects epsilon. Epsilon connects delta.")


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A session holding a known complex.

    Built directly rather than through /pipeline/stream: that route runs its stages
    in a forkserver subprocess, which does not see a monkeypatched workspace dir, so
    the document bundle lands in the real workspace and no session is registered
    here. The claim under test is that the API reports what the code computed, and
    that does not need the pipeline to establish it.
    """
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.auto import FACE_RULE, auto_rex
    from agent.server.app import app, get_store

    rex = auto_rex(DOC.decode(), face_selection=FACE_RULE)
    session = get_store().create(name="display-consistency")
    session.add_snapshot(rex=rex, action="probe", params={}, results={},
                         summary="display consistency")
    yield TestClient(app), session.session_id, rex
    reset_default_store()


def test_the_api_hodge_split_is_the_computed_one(run):
    client, sid, rex = run
    assert rex is not None, "the run produced no complex"
    api = client.get(f"/api/analysis/{sid}").json()
    h = rex.hodge_full(np.ones(rex.nE))
    got = api.get("hodge") or {}
    for api_key, computed_key in (("pct_gradient", "pct_grad"),
                                  ("pct_curl", "pct_curl"),
                                  ("pct_harmonic", "pct_harm")):
        assert got.get(api_key) == pytest.approx(float(h[computed_key])), (
            f"{api_key}: api {got.get(api_key)} != computed {float(h[computed_key])}")


def test_every_in_process_path_builds_the_same_complex():
    """A document, and the same text through auto_rex and through the adapter, agree.

    Only paths that can be given the same parameters are compared. The pipeline runs
    its stages in a forkserver subprocess with its own adapter settings, so its
    complex is a different configuration of the same text, not a disagreement; a
    test that compared them was reading whatever the real workspace happened to hold.
    """
    import uuid

    from agent.adapters.text import TextAdapter
    from agent.auto import FACE_RULE, auto_rex, build_rex_from_edges
    from agent.corpus import CorpusBuilder
    text = (f"Alpha{uuid.uuid4().hex[:6]} connects beta. Beta connects gamma. "
            "Gamma connects alpha. Delta connects alpha. Alpha connects epsilon. "
            "Epsilon connects delta.")

    def shape(r):
        return (r.nV, r.nE, r.nF, tuple(int(b) for b in r.betti))

    a = shape(auto_rex(text, face_selection=FACE_RULE))
    b = shape(build_rex_from_edges(TextAdapter().build(text), face_selection=FACE_RULE))
    c = CorpusBuilder()
    c.add_text(text, doc_id="d")
    c.build(depth="standard")
    d = shape(c.documents[0].rex)
    assert a == b == d, f"auto_rex {a}, build_rex_from_edges {b}, corpus {d}"
    assert a[2] > 0, "no faces, so curl is identically zero"


def test_the_document_face_rule_is_the_canonical_one():
    """One rule, named once. A document, the query it is scored against and the
    chunks taken from it all build under it."""
    from agent.auto import FACE_RULE
    from agent.corpus import DOC_FACE_RULE
    assert DOC_FACE_RULE is FACE_RULE or DOC_FACE_RULE == FACE_RULE
    assert FACE_RULE == "auto", FACE_RULE


def test_only_attach_faces_reaches_typed_face_selection():
    """The face rule lives in `agent.auto.attach_faces`. A path calling the graph
    method directly is a second rule, free to drift from this one, which is how the
    platform ended up with a document rule, a query rule and a router rule that
    disagreed."""
    import pathlib
    import re

    # Anchored on this file, not on the working directory. The glob used to be the
    # relative "agent/**/*.py", which matches 593 files from the repo root and NOTHING
    # from anywhere else, so a run with any other cwd found no offenders and passed
    # without having scanned a line. A guard that reports success by finding nothing to
    # check is worse than no guard, so the scan asserts its own reach below.
    root = pathlib.Path(__file__).resolve().parents[1] / "agent"
    offenders = []
    scanned = 0
    for f in sorted(root.rglob("*.py")):
        if "tests" in f.parts or f.name == "auto.py":
            continue                       # auto.py defines attach_faces
        scanned += 1
        # read_text closes the handle; the bare open() here leaked one per file and was
        # 418 of the suite's ResourceWarnings on its own
        for i, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            if re.search(r"\.typed_face_selection\(", line):
                offenders.append(f"{f.relative_to(root.parent)}:{i}")

    assert scanned > 100, f"the scan reached only {scanned} files, so it proved nothing"
    assert not offenders, "direct typed_face_selection calls: " + ", ".join(offenders)
