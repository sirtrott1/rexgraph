"""Record isolation between workspaces, on the routes that predate it.

Authentication is enforced globally, so with auth on every route needs a valid token.
That answers whether a caller is someone, not which records are theirs, and the record
store is one namespace shared by every workspace. A plain user of one workspace could
list and read what another had put there: not the complexes themselves, but their
signatures and channel readings, which is the interesting part.

The restriction lives in `default_store`, so it applies to any route that reaches
records rather than to the ones someone remembered. These tests are written against the
LEGACY routes on purpose: the native surface checks ownership itself, and a fix that
only held there would leave the older door open.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rexgraph import protocol
from rexgraph.graph import RexGraph

TRIANGLE = dict(sources=np.array([0, 1, 2], dtype=np.int32),
                targets=np.array([1, 2, 0], dtype=np.int32))


@pytest.fixture
def two_tenants(tmp_path, monkeypatch):
    """An admin in `default` and a plain user in `beta`, each with a stored record."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")

    from agent.rcdb import reset_default_store
    from agent.server import audit, auth
    auth.reset_auth_manager()
    audit.reset_cache()
    reset_default_store()

    from agent.server.app import app
    mgr = auth.get_auth_manager()
    mgr.enable_auth()
    admin = mgr.bootstrap_admin()
    bob = mgr.create_token("bob", ["beta"], role="user")
    client = TestClient(app)

    rex = RexGraph(**TRIANGLE)
    frame = protocol.encode(rex)
    ah = {"Authorization": f"Bearer {admin}"}
    bh = {"Authorization": f"Bearer {bob}", "X-Workspace": "beta"}
    ct = {"Content-Type": protocol.CONTENT_TYPE}
    a_id = client.post("/rex/v1/store", content=frame,
                       headers={**ah, **ct}).json()["record_id"]
    b_id = client.post("/rex/v1/store", content=frame,
                       headers={**bh, **ct}).json()["record_id"]

    yield client, ah, bh, a_id, b_id
    auth.reset_auth_manager()
    audit.reset_cache()
    reset_default_store()


def test_a_listing_shows_only_this_workspaces_records(two_tenants):
    client, ah, bh, a_id, b_id = two_tenants
    seen = [r["id"] for r in client.get("/api/v1/db/list", headers=bh).json()["records"]]
    assert seen == [b_id], f"bob's listing showed {seen}"


def test_the_owner_still_sees_their_own_record(two_tenants):
    """The half worth checking as hard as the leak: a filter that hides everything
    from everyone also passes the isolation test."""
    client, ah, bh, a_id, b_id = two_tenants
    seen = [r["id"] for r in client.get("/api/v1/db/list", headers=ah).json()["records"]]
    assert seen == [a_id]
    assert client.get(f"/api/v1/db/get/{a_id}", headers=ah).status_code == 200
    assert client.get(f"/api/v1/db/explain/{a_id}", headers=ah).status_code == 200


@pytest.mark.parametrize("route", ["/api/v1/db/get/{}", "/api/v1/db/explain/{}"])
def test_another_workspaces_record_reads_as_absent(two_tenants, route):
    client, ah, bh, a_id, _ = two_tenants
    r = client.get(route.format(a_id), headers=bh)
    assert r.status_code == 404, f"{route} served another workspace's record"


def test_a_record_cannot_be_deleted_from_another_workspace(two_tenants):
    client, ah, bh, a_id, _ = two_tenants
    client.request("DELETE", f"/api/v1/db/{a_id}", headers=bh)
    assert client.get(f"/api/v1/db/get/{a_id}", headers=ah).status_code == 200, \
        "another workspace deleted this record"


def test_absence_is_indistinguishable_from_not_yours(two_tenants):
    """A different answer for the two would turn a guessable id into a way to
    enumerate what other tenants hold."""
    client, ah, bh, a_id, _ = two_tenants
    theirs = client.get(f"/api/v1/db/get/{a_id}", headers=bh)
    nothing = client.get("/api/v1/db/get/rx_does_not_exist_at_all", headers=bh)
    assert theirs.status_code == nothing.status_code == 404


def test_a_write_is_stamped_with_the_workspace_that_made_it(two_tenants):
    client, ah, bh, _, b_id = two_tenants
    body = client.get(f"/api/v1/db/get/{b_id}", headers=bh).json()
    assert body.get("meta", {}).get("workspace") == "beta"


#### the trail over the legacy routes


def test_a_write_through_any_route_lands_in_the_trail(two_tenants):
    """Recording at the store rather than at the routes: a mutation that left no
    entry is exactly what the trail exists to rule out."""
    client, _ah, bh, _a, b_id = two_tenants
    from agent.server import audit
    puts = [e for e in audit.read() if e["action"] == "db.put"]
    assert any(e["target"] == b_id for e in puts), \
        f"no trail entry for {b_id}: {[e['target'] for e in puts]}"


def test_a_delete_attempt_on_someone_elses_record_is_recorded(two_tenants):
    client, _ah, bh, a_id, _b = two_tenants
    client.request("DELETE", f"/api/v1/db/{a_id}", headers=bh)
    from agent.server import audit
    entries = [e for e in audit.read() if e["action"] == "db.delete"]
    assert entries, "a delete attempt left no trace"


def test_the_trail_still_verifies_after_the_routes_have_written_to_it(two_tenants):
    from agent.server import audit
    assert audit.verify()["valid"] is True


#### the shared runtime


@pytest.mark.parametrize("method,path,body", [
    ("post", "/api/v1/models/pull", {"model_id": "x"}),
    ("post", "/api/v1/models/load", {"model_id": "x"}),
    ("post", "/api/v1/models/unload", {"model_id": "x"}),
    ("post", "/api/v1/models/deploy", {"model_id": "x"}),
    ("post", "/api/v1/models/stop", {"model_id": "x"}),
    ("post", "/api/v1/models/set-path", {"model_id": "x", "path": "/tmp/x"}),
    ("delete", "/api/v1/models/cache/x", None),
    ("delete", "/api/v1/models/path/x", None),
])
def test_a_plain_user_cannot_move_the_shared_model_runtime(
        two_tenants, method, path, body):
    """The inference runtime is process-wide, so a stop or an unload takes a model out
    from under whoever else was using it, and a pull spends disk and bandwidth."""
    client, _ah, bh, _a, _b = two_tenants
    kw = {"json": body} if body is not None else {}
    r = getattr(client, method)(path, headers=bh, **kw)
    assert r.status_code == 403, f"{method.upper()} {path} was allowed"


@pytest.mark.parametrize("path", ["/api/v1/models/list", "/api/v1/models/status"])
def test_reading_which_models_exist_stays_ordinary_use(two_tenants, path):
    client, _ah, bh, _a, _b = two_tenants
    assert client.get(path, headers=bh).status_code == 200


#### the parts that must NOT be scoped


def test_a_direct_caller_outside_a_request_sees_the_whole_store(two_tenants):
    """The CLI and anything in-process are not serving a request and are not scoped."""
    from agent.rcdb import default_store
    _client, _ah, _bh, a_id, b_id = two_tenants
    ids = {r.id for r in default_store().list()}
    assert {a_id, b_id} <= ids


def test_scoping_is_off_when_auth_is_off(tmp_path, monkeypatch):
    """Single-operator local use has one tenant, so there is nothing to separate."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    from agent.server import auth, scope
    auth.reset_auth_manager()
    token = scope.set_workspace("beta")
    try:
        assert scope.scoping_active() is False
    finally:
        scope.reset_workspace(token)
        auth.reset_auth_manager()


def test_a_record_with_no_workspace_stays_visible():
    """Records written before ownership existed. Hiding them would read as data
    loss on upgrade; stamping them would mean guessing whose they were."""
    from agent.server.scope import owns
    assert owns({}, "beta") is True
    assert owns({"workspace": None}, "beta") is True
    assert owns({"workspace": "beta"}, "beta") is True
    assert owns({"workspace": "default"}, "beta") is False
