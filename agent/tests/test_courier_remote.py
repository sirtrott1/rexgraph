"""Couriers across machines: the crossing over /rex/v1, and the ledger that makes it idempotent."""
from __future__ import annotations

import numpy as np
import pytest
from agent.courier import CarrySpec, Courier
from agent.courier_remote import Ledger, Peer
from fastapi.testclient import TestClient

from agent import activity, rcdb
from agent import hive as hivemod
from rexgraph.graph import RexGraph

KEY = "a shared deployment key"


def _rex(n):
    v = np.arange(n, dtype=np.int32)
    return RexGraph(sources=v, targets=np.roll(v, -1).astype(np.int32))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", str(tmp_path / "activity.jsonl"))
    from agent.rcdb import reset_default_store
    from agent.server import audit, auth
    auth.reset_auth_manager(); audit.reset_cache(); reset_default_store()
    activity.reset(); hivemod.reset_network()
    yield tmp_path
    activity.get_log().close()
    auth.reset_auth_manager(); audit.reset_cache(); reset_default_store()


def _client(monkeypatch, *, signed: bool = False):
    """A RexClient whose transport is the TestClient, so no socket is involved."""
    if signed:
        monkeypatch.setenv("REXGRAPH_FRAME_KEY", KEY)
    else:
        monkeypatch.delenv("REXGRAPH_FRAME_KEY", raising=False)
    from agent.client import RexClient
    from agent.server.app import app
    transport = TestClient(app)
    rc = RexClient("http://testserver", frame_key=KEY if signed else None)
    import httpx
    # RexClient passes a timeout for a real network call; TestClient deprecates the
    # argument because there is no socket to time out against, so it is dropped here.
    def _strip(kw):
        return {k: v for k, v in kw.items() if k != "timeout"}

    monkeypatch.setattr(httpx, "post", lambda url, **kw: transport.post(
        url.replace("http://testserver", ""), **_strip(kw)))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: transport.get(
        url.replace("http://testserver", ""), **_strip(kw)))
    return rc


@pytest.fixture
def local_store():
    st = rcdb.open_store("memory://")
    st.put("alpha-schema", _rex(4), meta={"kind": "hive-schema"}, tags=["hive-schema"])
    st.put("alpha-work", _rex(5), meta={"kind": "interaction"}, tags=["interaction"])
    return st


def _courier(local_store, peer, **kw):
    c = Courier("mule", **kw)
    c.attach_store("alpha", local_store)
    c.attach_peer(peer)
    return c


def test_a_crossing_ships_what_the_peer_has_not_seen(isolated, monkeypatch, local_store):
    peer = Peer("gpu-box", _client(monkeypatch))
    trip = _courier(local_store, peer).deliver("alpha", "gpu-box")

    assert trip["remote"] is True
    assert trip["considered"] == 2 and trip["carried"] == 2 and trip["held"] == 0
    remote_ids = [s["remote_id"] for s in trip["shipments"]]
    assert all(r and r.startswith("rx_") for r in remote_ids), "the peer names what it took"
    # the complex is really on the far side, rebuilt from the frame
    server_store = rcdb.default_store()
    assert {server_store.get(r).nV for r in remote_ids} == {4, 5}


def test_provenance_crosses_with_the_record(isolated, monkeypatch, local_store):
    peer = Peer("gpu-box", _client(monkeypatch))
    trip = _courier(local_store, peer).deliver("alpha", "gpu-box")

    rid = next(s["remote_id"] for s in trip["shipments"] if s["record_id"] == "alpha-schema")
    meta = rcdb.default_store().get_record(rid).meta
    assert meta["record_id"] == "alpha-schema" and meta["source_hive"] == "alpha"
    assert meta["courier"] == "mule" and meta["kind"] == "hive-schema"
    assert meta["workspace"] == "default", "the server stamps the receiving workspace"


def test_a_repeat_crossing_ships_nothing(isolated, monkeypatch, local_store):
    c = _courier(local_store, Peer("gpu-box", _client(monkeypatch)))
    first = c.deliver("alpha", "gpu-box")
    again = c.deliver("alpha", "gpu-box")

    assert again["carried"] == 0 and again["held"] == 2
    kept = {s["record_id"]: s["remote_id"] for s in first["shipments"]}
    assert {s["record_id"]: s["remote_id"] for s in again["shipments"]} == kept


def test_a_changed_record_crosses_again(isolated, monkeypatch, local_store):
    c = _courier(local_store, Peer("gpu-box", _client(monkeypatch)))
    c.deliver("alpha", "gpu-box")
    local_store.put("alpha-work", _rex(7), meta={"kind": "interaction"}, tags=["interaction"])
    trip = c.deliver("alpha", "gpu-box")

    assert trip["carried"] == 1 and trip["held"] == 1
    rid = next(s["remote_id"] for s in trip["shipments"] if s["record_id"] == "alpha-work")
    assert rcdb.default_store().get(rid).nV == 7


def test_forgetting_an_entry_ships_it_again(isolated, monkeypatch, local_store):
    peer = Peer("gpu-box", _client(monkeypatch))
    c = _courier(local_store, peer)
    c.deliver("alpha", "gpu-box")
    # the ledger records what was sent and cannot see a deletion on the far side
    assert peer.ledger.forget("gpu-box", "alpha-work") is True
    trip = c.deliver("alpha", "gpu-box")
    assert trip["carried"] == 1 and trip["held"] == 1


def test_the_ledger_survives_the_process(tmp_path):
    path = tmp_path / "ledger.json"
    a = Ledger(str(path))
    a.note("gpu-box", "alpha-work", "rx_dead", {"nV": 5})
    b = Ledger(str(path))
    assert b.remote_id("gpu-box", "alpha-work") == "rx_dead"
    assert b.structure("gpu-box", "alpha-work") == {"nV": 5}
    assert b.entries("other") == []


def test_a_record_past_the_peers_ceiling_never_leaves(isolated, monkeypatch, local_store):
    monkeypatch.setenv("REXGRAPH_MAX_CELLS", "4")
    peer = Peer("gpu-box", _client(monkeypatch))
    trip = _courier(local_store, peer).deliver("alpha", "gpu-box")

    assert trip["oversize"] == 1 and trip["carried"] == 1
    over = next(s for s in trip["shipments"] if s["reason"] == "oversize")
    assert over["record_id"] == "alpha-work" and "4-cell limit" in over["detail"]
    assert over["remote_id"] is None, "nothing crossed the wire for it"


def test_signed_frames_cross(isolated, monkeypatch, local_store):
    peer = Peer("gpu-box", _client(monkeypatch, signed=True), confirm=True)
    trip = _courier(local_store, peer).deliver("alpha", "gpu-box")

    assert trip["carried"] == 2, "signing both directions, and the receipt verified"


class _Stub:
    """A peer that answers hello and then misbehaves in one named way."""

    def __init__(self, mode, rex=None):
        self.mode, self.rex, self.stored = mode, rex, 0

    def rex_hello(self):
        return {"limits": {"max_cells": 1000}}

    def rex_store(self, rex, **meta):
        self.stored += 1
        if self.mode == "refuse" and meta.get("record_id") == "alpha-work":
            raise RuntimeError("boom")
        if self.mode == "no_id":
            return {}
        return {"record_id": f"rx_stub{self.stored}"}

    def rex_fetch(self, record_id):
        return self.rex


def test_a_refusing_peer_does_not_strand_the_trip(isolated, local_store):
    peer = Peer("flaky", _Stub("refuse"))
    trip = _courier(local_store, peer).deliver("alpha", "flaky")

    assert trip["refused"] == 1 and trip["carried"] == 1
    bad = next(s for s in trip["shipments"] if s["reason"] == "refused")
    assert "RuntimeError: boom" in bad["detail"]
    assert peer.ledger.remote_id("flaky", "alpha-work") is None, "a refusal is not recorded"


def test_a_peer_that_names_nothing_is_refused(isolated, local_store):
    trip = _courier(local_store, Peer("mute", _Stub("no_id"))).deliver("alpha", "mute")
    assert trip["refused"] == 2 and trip["carried"] == 0


def test_a_peer_returning_a_different_complex_is_refused(isolated, local_store):
    peer = Peer("liar", _Stub("ok", rex=_rex(9)), confirm=True)
    trip = _courier(local_store, peer).deliver("alpha", "liar",
                                               carry=CarrySpec(ids=["alpha-schema"]))
    bad = trip["shipments"][0]
    assert bad["reason"] == "refused" and "different complex" in bad["detail"]
    assert bad["remote_id"] == "rx_stub1", "the id is reported even though it is not trusted"


def test_retrieve_addresses_by_the_local_id(isolated, monkeypatch, local_store):
    peer = Peer("gpu-box", _client(monkeypatch))
    c = _courier(local_store, peer)
    with pytest.raises(ValueError, match="nothing shipped"):
        peer.retrieve("alpha-work")
    c.deliver("alpha", "gpu-box")
    assert peer.retrieve("alpha-work").nV == 5


def test_a_crossing_is_an_edge_in_the_network_complex(isolated, monkeypatch, local_store):
    net = hivemod.get_network()
    net.hive("alpha")
    peer = Peer("gpu-box", _client(monkeypatch))
    c = _courier(local_store, peer, network=net)
    c.deliver("alpha", "gpu-box")

    edges = net.monitor()["edges"]
    assert {"from": "alpha", "to": "gpu-box", "weight": 1} in edges, \
        "a machine across the wire is a cell of the network like any hive"


def test_broadcast_reaches_hives_and_peers(isolated, monkeypatch, local_store):
    other = rcdb.open_store("memory://")
    peer = Peer("gpu-box", _client(monkeypatch))
    c = _courier(local_store, peer)
    c.attach_store("beta", other)
    out = c.broadcast("alpha")

    assert out["dests"] == ["beta", "gpu-box"] and out["carried"] == 4
    assert len(other.list()) == 2
    assert c.status()["peers"] == ["gpu-box"] and c.status()["hives"] == ["alpha", "beta"]
