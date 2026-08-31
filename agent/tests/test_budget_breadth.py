"""The compute ceiling, on every route rather than the one that remembered.

The rate limiter counts requests per IP, which bounds how OFTEN a caller asks and says
nothing about what each ask costs: a few hundred bytes can name a complex with millions
of cells. A guard written into one route covers that route, and a caller reaches the
expensive work through whichever route did not have one. So the slot is taken in
middleware, and these tests are mostly about routes that never asked for it.

The ordering matters and is easy to get backwards: Starlette runs the LAST-registered
middleware FIRST, so the budget is registered BEFORE the auth enforcement it has to run
behind. An unauthenticated request must be rejected without ever holding a slot.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    from agent.server import audit, auth
    auth.reset_auth_manager()
    audit.reset_cache()
    yield tmp_path
    auth.reset_auth_manager()
    audit.reset_cache()


@pytest.fixture
def client(isolated):
    from agent.server.app import app
    return TestClient(app)


#### the ceiling reaches routes that never asked for one


@pytest.mark.parametrize("path", [
    "/api/v1/db/list",
    "/api/v1/models/list",
    "/rex/v1/files",
])
def test_a_legacy_route_is_refused_when_the_identity_is_saturated(
        client, path, monkeypatch):
    """The point of moving it to middleware: these routes have no guard of their own."""
    monkeypatch.setenv("REXGRAPH_MAX_INFLIGHT", "0")
    r = client.get(path)
    assert r.status_code == 429, f"{path} ran with no slot available"
    assert r.json()["axis"] == "concurrency"


def test_the_health_probe_answers_while_saturated(client, monkeypatch):
    """A load balancer pulls the node otherwise, which turns a busy server into a
    down one."""
    monkeypatch.setenv("REXGRAPH_MAX_INFLIGHT", "0")
    assert client.get("/api/health").status_code == 200


def test_the_capability_page_answers_while_saturated(client, monkeypatch):
    """`hello` is how a client learns the ceilings; discovering them by being refused
    is the failure this avoids."""
    monkeypatch.setenv("REXGRAPH_MAX_INFLIGHT", "0")
    assert client.get("/rex/v1/hello").status_code == 200


def test_a_slot_is_released_after_the_request(client):
    """A leaked slot turns the ceiling into a countdown."""
    for _ in range(6):
        assert client.get("/rex/v1/files").status_code == 200
    from agent.server.budget import inflight
    assert not inflight(), f"slots outlived their requests: {inflight()}"


def test_ordinary_requests_are_unaffected(client):
    assert client.get("/api/v1/db/list").status_code == 200


#### the ordering the ceiling depends on


def test_an_unauthenticated_request_never_takes_a_slot(isolated, monkeypatch):
    """Registered before auth so it runs after it. Were that backwards, an anonymous
    flood would occupy every slot and the rejection would cost a tenant's concurrency."""
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    mgr = get_auth_manager()
    mgr.enable_auth()
    mgr.bootstrap_admin()
    c = TestClient(app)
    monkeypatch.setenv("REXGRAPH_MAX_INFLIGHT", "1")
    assert c.get("/api/v1/db/list").status_code == 401
    from agent.server.budget import inflight
    assert not inflight(), "a rejected request held a slot"


def _request(headers: dict):
    from starlette.requests import Request
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw,
                    "query_string": b""})


def test_identity_and_workspace_agree_with_the_token(isolated):
    """One reader for "who is this request", because three middlewares needed it and
    three token parsers would eventually disagree about what a bearer header is."""
    from agent.server.auth import get_auth_manager, identity_and_workspace
    mgr = get_auth_manager()
    mgr.enable_auth()
    mgr.bootstrap_admin()
    bob = mgr.create_token("bob", ["beta"], role="user")
    carol = mgr.create_token("carol", ["beta", "gamma"], role="user")

    # no header: the token's own workspace is what scopes the caller
    assert identity_and_workspace(
        _request({"Authorization": f"Bearer {bob}"})) == ("bob", "beta")
    # a workspace the token holds a role in is carried through
    assert identity_and_workspace(
        _request({"Authorization": f"Bearer {carol}",
                  "X-Workspace": "gamma"})) == ("carol", "gamma")
    # one it does not is NOT. This value becomes `scope._current`, so on any route that
    # does not declare `require_workspace` it is what every scoped store filters on, and
    # carrying it through let a caller name their way into another tenant's records,
    # sessions and saved connections.
    assert identity_and_workspace(
        _request({"Authorization": f"Bearer {bob}",
                  "X-Workspace": "gamma"})) == ("bob", "beta")


def test_an_unverifiable_token_is_not_given_an_identity(isolated):
    """Answering "anonymous" would put every bad token in one bucket, so a flood of
    them would meter as a single caller."""
    from agent.server.auth import get_auth_manager, identity_and_workspace
    mgr = get_auth_manager()
    mgr.enable_auth()
    mgr.bootstrap_admin()
    identity, _ws = identity_and_workspace(_request({"Authorization": "Bearer nope"}))
    assert identity == ""


#### the hive answers to the same ceiling


def test_a_bee_is_not_a_way_to_buy_concurrency(isolated, tmp_path, monkeypatch):
    """A requester driving ten bees is the case worth bounding, so the meter is on the
    sender rather than on the worker."""
    from agent.hive import Hive
    from agent.server.budget import BudgetExceeded, guard
    obo = tmp_path / "t.obo"
    obo.write_text("[Term]\nid: GO:1\nname: a\n\n[Term]\nid: GO:2\nname: b\nis_a: GO:1\n")
    h = Hive("metered")
    h.add_tools()
    monkeypatch.setenv("REXGRAPH_MAX_INFLIGHT", "1")
    # the sender already holds their one slot, so the bee cannot take a second
    with guard("user"), pytest.raises(BudgetExceeded):
        h.invoke("rexgraph_homology", {"files": [str(obo)]}, sender="user")


def test_the_hive_runs_headless_with_no_ceiling(isolated, tmp_path):
    """The swarm is used from a CLI and a notebook, where there is no tenant to bound,
    so the budget is a soft dependency rather than the web layer reaching backwards."""
    from agent.hive import _budget_slot
    with _budget_slot("local"):
        pass
