"""The doors onto the courier: the route surface, the tool registry, and the CLI.

The carrier itself was reachable only by importing it, which meant the capability existed
and nothing on the machine could ask for it. These are the three surfaces that make it
askable, and the boundary each one has to keep: a trip reaches another store or another
machine, so it is admin, and a destination is looked up rather than built from whatever
the caller sent.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent import courier as couriermod
from agent import hive as hivemod
from rexgraph.graph import RexGraph


def _rex(n):
    v = np.arange(n, dtype=np.int32)
    return RexGraph(sources=v, targets=np.roll(v, -1).astype(np.int32))


@pytest.fixture
def tenants(tmp_path, monkeypatch):
    """An admin and a plain user, plus two stores on disk for the courier to work between."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", str(tmp_path / "activity.jsonl"))
    from agent.rcdb import open_store, reset_default_store
    from agent.server import audit, auth

    from agent import activity
    auth.reset_auth_manager(); audit.reset_cache(); reset_default_store()
    activity.reset(); hivemod.reset_network(); couriermod.reset_courier()

    from agent.server.app import app
    mgr = auth.get_auth_manager()
    mgr.enable_auth()
    admin = mgr.bootstrap_admin()
    bob = mgr.create_token("bob", ["default"], role="user")

    a_uri, b_uri = f"file://{tmp_path}/a", f"file://{tmp_path}/b"
    src = open_store(a_uri)
    src.put("schema", _rex(3), meta={"kind": "hive-schema"}, tags=["hive-schema"])
    src.put("work", _rex(5), meta={"kind": "interaction"}, tags=["interaction"])

    yield (TestClient(app), {"Authorization": f"Bearer {admin}"},
           {"Authorization": f"Bearer {bob}"}, a_uri, b_uri)
    auth.reset_auth_manager(); audit.reset_cache(); reset_default_store()
    couriermod.reset_courier(); activity.get_log().close()


def _bind(client, ah, a_uri, b_uri):
    assert client.post("/api/v1/courier/routes", headers=ah,
                       json={"hive": "alpha", "store": a_uri}).status_code == 200
    assert client.post("/api/v1/courier/routes", headers=ah,
                       json={"hive": "beta", "store": b_uri}).status_code == 200


def test_a_trip_is_admin_only(tenants):
    """A trip reaches another store or another machine, which is the line the tool
    registry draws for admin next to its own handlers."""
    client, ah, bh, a_uri, b_uri = tenants
    _bind(client, ah, a_uri, b_uri)
    r = client.post("/api/v1/courier/deliver", headers=bh,
                    json={"source": "alpha", "dest": "beta"})
    assert r.status_code == 403
    assert client.post("/api/v1/courier/routes", headers=bh,
                       json={"hive": "x", "store": a_uri}).status_code == 403
    # Reading what the courier is wired to was treated as an ordinary read. It is not:
    # status names the peer hives this deployment reaches and what has already been
    # carried between them, and the courier is a process-wide singleton holding store
    # views bound by whoever bound them, so a survey lists records through someone
    # else's view rather than the caller's.
    assert client.get("/api/v1/courier/status", headers=bh).status_code == 403
    assert client.get("/api/v1/courier/status", headers=ah).status_code == 200


def test_the_route_carries_and_then_holds(tenants):
    client, ah, bh, a_uri, b_uri = tenants
    _bind(client, ah, a_uri, b_uri)

    first = client.post("/api/v1/courier/deliver", headers=ah,
                        json={"source": "alpha", "dest": "beta"}).json()
    assert first["considered"] == 2 and first["carried"] == 2
    again = client.post("/api/v1/courier/deliver", headers=ah,
                        json={"source": "alpha", "dest": "beta"}).json()
    assert again["carried"] == 0 and again["held"] == 2

    from agent.rcdb import open_store
    assert {r.id for r in open_store(b_uri).list()} == {"schema", "work"}


def test_a_destination_cannot_be_named_by_a_caller(tenants):
    """The whole point of looking a destination up: a caller that could name one could
    name a machine the operator never approved."""
    client, ah, bh, a_uri, b_uri = tenants
    _bind(client, ah, a_uri, b_uri)
    r = client.post("/api/v1/courier/deliver", headers=ah,
                    json={"source": "alpha", "dest": "https://somewhere-else:8000"})
    assert r.status_code == 404 and "register it first" in r.json()["detail"]


def test_a_peer_is_named_by_reference_never_by_key(tenants, monkeypatch):
    client, ah, bh, a_uri, b_uri = tenants
    monkeypatch.setenv("PEER_TOKEN", "s3cret")
    # a reference arriving in a REQUEST needs the operator's permission; naming one
    # is no longer enough on its own. See test_request_refs.
    monkeypatch.setenv("REXGRAPH_REQUEST_KEY_REFS", "PEER_TOKEN")
    r = client.post("/api/v1/courier/peers", headers=ah,
                    json={"name": "gpu-box", "url": "https://gpu-box:8000",
                          "api_key_ref": "PEER_TOKEN"})
    assert r.status_code == 200 and r.json()["has_api_key"] is True
    assert "s3cret" not in r.text, "the key must not come back out"
    assert client.get("/api/v1/courier/status", headers=ah).json()["peers"] == ["gpu-box"]

    refused = client.post("/api/v1/courier/peers", headers=ah,
                          json={"name": "raw", "url": "https://x:8000",
                                "api_key": "s3cret"})
    assert refused.status_code == 400 and "reference" in refused.json()["detail"]


def test_survey_reports_without_carrying(tenants):
    client, ah, bh, a_uri, b_uri = tenants
    _bind(client, ah, a_uri, b_uri)
    got = client.get("/api/v1/courier/survey", headers=ah,
                     params={"hive": "alpha", "tags": "hive-schema"}).json()
    assert [r["record_id"] for r in got["records"]] == ["schema"]

    from agent.rcdb import open_store
    assert open_store(b_uri).list() == [], "a survey carried something"


def test_an_unbound_store_is_a_404_not_a_crash(tenants):
    client, ah, bh, a_uri, b_uri = tenants
    assert client.get("/api/v1/courier/survey", headers=ah,
                      params={"hive": "nope"}).status_code == 404


def test_broadcast_reaches_every_bound_destination(tenants):
    client, ah, bh, a_uri, b_uri = tenants
    _bind(client, ah, a_uri, b_uri)
    out = client.post("/api/v1/courier/broadcast", headers=ah,
                      json={"source": "alpha"}).json()
    assert out["dests"] == ["beta"] and out["carried"] == 2


def test_the_tools_are_registered_and_admin_only():
    from agent.mcp_tools import TOOLS, Context, definitions
    assert {"rexgraph_courier_survey", "rexgraph_courier_deliver"} <= set(TOOLS)
    assert all(TOOLS[n].requires == "admin"
               for n in ("rexgraph_courier_survey", "rexgraph_courier_deliver"))
    user = Context(workspace="w", identity="u", is_admin=False, auth_enabled=True)
    assert not [d for d in definitions(user) if "courier" in d["name"]], \
        "a tool that would be refused must not be advertised"


def test_the_tool_refuses_a_destination_it_does_not_route_for(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", "off")
    couriermod.reset_courier()
    from agent.mcp_tools import call
    with pytest.raises(ValueError, match="Register it first"):
        call("rexgraph_courier_deliver", source="alpha", dest="anywhere")
    couriermod.reset_courier()


def test_the_cli_carries_between_two_stores(tmp_path, monkeypatch, capsys):
    """A command that ends when it returns has no hive, so the CLI names stores by uri."""
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", "off")
    from agent.rcdb import open_store
    a, b = f"file://{tmp_path}/a", f"file://{tmp_path}/b"
    open_store(a).put("one", _rex(4), meta={"kind": "x"}, tags=["x"])

    assert couriermod.main(["deliver", a, b]) == 0
    assert '"carried": 1' in capsys.readouterr().out
    assert [r.id for r in open_store(b).list()] == ["one"]

    assert couriermod.main(["survey", a]) == 0
    assert "one" in capsys.readouterr().out
