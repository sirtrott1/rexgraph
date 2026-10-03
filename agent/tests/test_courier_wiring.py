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
from contextlib import closing
from fractions import Fraction
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
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"local://{tmp_path}/rcdb")
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

    a_uri, b_uri = f"local://{tmp_path}/a", f"local://{tmp_path}/b"
    src = open_store(a_uri, **({"read_only": False} if "://" not in a_uri or a_uri.startswith(("file://", "rex://")) else {}))
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
    # carried between them, and the courier is a process wide singleton holding store
    # views bound by whoever bound them, so a survey lists records through someone
    # else's view rather than the caller's.
    assert client.get("/api/v1/courier/status", headers=bh).status_code == 403
    assert client.get("/api/v1/courier/status", headers=ah).status_code == 200
    assert client.post("/api/v1/courier/reconcile", headers=bh,
                       json={"source": "alpha", "dest": "beta"}).status_code == 403


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
    assert {r.id for r in open_store(b_uri, **({"read_only": False} if "://" not in b_uri or b_uri.startswith(("file://", "rex://")) else {})).list()} == {"schema", "work"}


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
    assert open_store(b_uri, **({"read_only": False} if "://" not in b_uri or b_uri.startswith(("file://", "rex://")) else {})).list() == [], "a survey carried something"


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
    a, b = f"local://{tmp_path}/a", f"local://{tmp_path}/b"
    open_store(a, **({"read_only": False} if "://" not in a or a.startswith(("file://", "rex://")) else {})).put("one", _rex(4), meta={"kind": "x"}, tags=["x"])

    assert couriermod.main(["deliver", a, b]) == 0
    assert '"carried": 1' in capsys.readouterr().out
    assert [r.id for r in open_store(b, **({"read_only": False} if "://" not in b or b.startswith(("file://", "rex://")) else {})).list()] == ["one"]

    assert couriermod.main(["survey", a]) == 0
    assert "one" in capsys.readouterr().out


def test_corrupt_ledger_registration_is_refused_and_not_attached(tenants, tmp_path):
    client, ah, _, _, _ = tenants
    path = tmp_path / "ledger.json"
    path.write_bytes(b"{broken")
    response = client.post("/api/v1/courier/peers", headers=ah,
                           json={"name": "p", "url": "http://localhost:1", "ledger": str(path)})
    assert response.status_code == 400 and "ledger" in response.json()["detail"]
    assert couriermod.get_courier().peers() == [] and path.read_bytes() == b"{broken"


@pytest.mark.parametrize("bad", ["unknown-source", "unknown-peer", "receipt", "missing-record",
                                  "source-owner", "unreachable", "destination-owner"])
def test_reconciliation_route_refuses_invalid_selection_or_destination(tenants, bad):
    from agent.courier_remote import Peer
    from rcdb import MemoryStore, copy_record, record_packet
    import httpx
    client, ah, _, _, _ = tenants
    with closing(MemoryStore()) as source, closing(MemoryStore()) as dest:
        source.put_record("r", Fraction(2, 7))
        packet = record_packet(source, "r")
        receipt = copy_record(source, dest, packet.record, return_receipt=True)
        class Remote:
            def rex_fetch_record(self, *args, **kwargs):
                if bad == "unreachable": raise httpx.ConnectError("unreachable")
                if bad == "destination-owner": return packet
                return record_packet(dest, receipt.destination_record_id, version=receipt.destination_version)
        peer = Peer("p", Remote())
        courier = couriermod.get_courier()
        courier.attach_store("source", source); courier.attach_peer(peer)
        body = {"source": "source", "dest": "p", "record_id": "r", "receipt": receipt.as_record()}
        if bad == "unknown-source": body["source"] = "unknown"
        elif bad == "unknown-peer": body["dest"] = "http://arbitrary"
        elif bad == "receipt": body["receipt"] = {"invalid": 1}
        elif bad == "missing-record": body["record_id"] = "missing"
        elif bad == "source-owner": body["receipt"]["source_store_id"] = "e"*32
        response = client.post("/api/v1/courier/reconcile", headers=ah, json=body)
        assert response.status_code == (404 if bad.startswith("unknown-") else 400), response.text
        assert peer.ledger.entries() == [] and len(dest.list()) == 1


def test_reconciliation_route_verifies_without_posting_and_audits_caller(tenants, monkeypatch):
    from agent.courier_remote import Peer
    from agent.server import audit
    from rcdb import MemoryStore, copy_record, record_packet
    client, ah, _, _, _ = tenants
    with closing(MemoryStore()) as source, closing(MemoryStore()) as dest:
        source.put_record("r", None, meta={"q": Fraction(1, 7)})
        packet = record_packet(source, "r")
        receipt = copy_record(source, dest, packet.record, return_receipt=True)
        fetched, audited = [], []
        class Remote:
            def rex_fetch_record(self, record_id, *, version):
                fetched.append((record_id, version))
                return record_packet(dest, record_id, version=version)
            def rex_store_record(self, *args, **kwargs): raise AssertionError("reconciliation posted")
        peer = Peer("p", Remote())
        courier = couriermod.get_courier()
        courier.attach_store("source", source); courier.attach_peer(peer)
        monkeypatch.setattr(audit, "record", lambda action, **kwargs: audited.append((action, kwargs)))
        response = client.post("/api/v1/courier/reconcile", headers=ah,
                               json={"source": "source", "dest": "p", "record_id": "r",
                                     "receipt": receipt.as_record()})
        assert response.status_code == 200, response.text
        assert response.json()["reason"] == "held" and not response.json()["shipped"]
        assert fetched == [(receipt.destination_record_id, receipt.destination_version)]
        assert len(dest.list()) == 1 and peer.ledger.remote_id("p", "r") == receipt.destination_record_id
        entry = next(options for action, options in audited if action == "courier.reconcile")
        assert entry["user"] and entry["workspace"] == "default"
        assert entry["detail"] == {"source": "source", "dest": "p", "record_id": "r"}
