"""Every route has to be able to send what it computes.

The analysis layer returns numpy, and FastAPI's encoder cannot serialize numpy, so
a route that returns its result raw dies AFTER the handler succeeded. That failure
is invisible until the data is shaped a particular way: `/v1/corpus/temporal` was
fine with one document and 500ed with two.

Three of twenty-five route modules sanitize by hand, which means the next route has
to remember. This walks them instead.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

#### two documents, because a single-document corpus short-circuits the paths that
#### produce cross-document arrays
DOCS = [
    ("a.txt", b"Alpha connects beta. Beta connects gamma. Gamma connects delta."),
    ("b.txt", b"Gamma connects delta. Delta connects epsilon. Epsilon connects alpha."),
]

#### streaming routes answer over a socket, not through this client
SKIP = ("stream", "events", "/docs", "/redoc", "/openapi.json")


@pytest.fixture
def populated():
    """Function-scoped on purpose: the full GET walk touches routes that change
    corpus state, so a shared corpus makes the later checks depend on walk order."""
    from agent.server.app import app
    c = TestClient(app)
    for name, body in DOCS:
        c.post("/api/v1/corpus/add",
               files=[("file", (name, io.BytesIO(body), "text/plain"))])
    c.post("/api/v1/corpus/build", json={"depth": "standard"})
    return c, app


def test_every_get_route_can_serialize_its_answer(populated):
    client, app = populated
    paths = [p for p in app.openapi()["paths"] if "{" not in p]
    checked, failures = 0, []
    for path in sorted(paths):
        if any(s in path for s in SKIP):
            continue
        if "get" not in app.openapi()["paths"][path]:
            continue
        checked += 1
        try:
            r = client.get(path)
        except Exception as e:                 # the encoder raises past the handler
            failures.append(f"{path}: {type(e).__name__}: {str(e)[:160]}")
            continue
        if r.status_code >= 500:
            failures.append(f"{path}: {r.status_code} {r.text[:120]}")
    assert checked > 40, f"only {checked} routes probed; the walk is not running"
    assert not failures, "routes that cannot send their answer:\n  " + "\n  ".join(failures)


def test_corpus_temporal_serializes_with_more_than_one_document(populated):
    """The specific shape that failed: numpy arrays and a tuple of arrays."""
    client, _ = populated
    r = client.get("/api/v1/corpus/temporal")
    assert r.status_code == 200, r.text[:200]
    d = r.json()
    assert "tags" in d and "n_phases" in d, d
    assert isinstance(d["tags"], list)


def test_a_missing_trustgraph_is_a_client_answer_not_a_server_fault(populated):
    """Five of six TrustGraph routes answer an unconfigured instance with a 400 and
    a message. Evolution raised a 500, and the error sanitizer then replaced the
    message with "Internal server error", losing the part that says what to do."""
    client, _ = populated
    r = client.post("/api/v1/trustgraph/evolution", json={"flow": "default"})
    assert r.status_code == 400, f"{r.status_code}: {r.text[:160]}"
    assert "TrustGraph URL" in r.text
