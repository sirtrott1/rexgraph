"""/api/v1/db/query separates a malformed request from a server fault.

The route splats the request body into the store predicate, so an unsupported key is
the client's mistake. The store raises TypeError naming the keys it accepts, and the
route turns that into a 400 carrying the same list. A 500 there would report the
server as broken and hide the fix from whoever sent the request.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rexgraph.graph import RexGraph

QUERY = "/api/v1/db/query"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "rcdb.sqlite"))
    import agent.server.routes.rcdb as rcdbroute
    rcdbroute._STORE = None
    from agent.server.app import app
    c = TestClient(app)
    store = rcdbroute._store()
    for i in range(3):
        store.put(f"r{i}", RexGraph(sources=np.arange(3 + i, dtype=np.int32),
                                    targets=np.arange(1, 4 + i, dtype=np.int32)))
    yield c
    rcdbroute._STORE = None


def test_a_supported_predicate_returns_records(client):
    r = client.post(QUERY, json={"min_nE": 1, "limit": 10})
    assert r.status_code == 200
    assert r.json()["count"] >= 1


def test_an_unsupported_key_is_a_400_not_a_500(client):
    r = client.post(QUERY, json={"nE": 4})
    assert r.status_code == 400


def test_the_400_names_the_supported_keys(client):
    detail = client.post(QUERY, json={"nE": 4}).json()["detail"]
    assert "unsupported query key" in detail
    assert "max_nE" in detail


def test_an_empty_body_is_not_an_error(client):
    r = client.post(QUERY, json={})
    assert r.status_code == 200


def test_limit_is_consumed_by_the_route_not_the_predicate(client):
    """`limit` is popped before the splat, so it must not read as an unknown key."""
    r = client.post(QUERY, json={"limit": 1})
    assert r.status_code == 200
    assert r.json()["count"] <= 1
