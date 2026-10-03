"""Typed RCDB records cross the existing socket boundary without graph coercion."""
import asyncio
from contextlib import closing, contextmanager
from dataclasses import replace
from fractions import Fraction

import httpx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from rexgraph import Absent, RexGraph
from rexgraph.value_codec import pack_value
from rcdb import (CopyReceipt, MemoryStore, RecordPacket, record_packet,
                  DECLARATION_CODEC, PROVENANCE_CODEC, COPY_RECEIPT_CODEC)
from rcdb.packet import PACKET_CONTENT_TYPE, RECEIPT_CONTENT_TYPE


@pytest.fixture
def transport(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"local://{tmp_path}/records")
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path/"audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", str(tmp_path/"activity.jsonl"))
    monkeypatch.delenv("REXGRAPH_FRAME_KEY", raising=False)
    from agent import activity
    from agent.rcdb import reset_default_store
    from agent.server import audit, auth
    from agent.server.app import app
    reset_default_store(); auth.reset_auth_manager(); audit.reset_cache(); activity.reset()
    with TestClient(app) as client:
        def strip(kwargs): return {k: v for k, v in kwargs.items() if k != "timeout"}
        monkeypatch.setattr(httpx, "get", lambda url, **kw: client.get(url, **strip(kw)))
        monkeypatch.setattr(httpx, "post", lambda url, **kw: client.post(url, **strip(kw)))
        monkeypatch.setattr(httpx, "stream", lambda method, url, **kw: client.stream(method, url, **strip(kw)))
        yield client
    activity.get_log().close()
    reset_default_store(); auth.reset_auth_manager(); audit.reset_cache()


def client(**kwargs):
    from agent.client import RexClient
    return RexClient("http://testserver", **kwargs)


def packet(value=None, **options):
    with closing(MemoryStore()) as source:
        source.put_record("source/id@7", value, meta={"exact": Fraction(1, 7), "workspace": "forged"},
                          tags=["exact"], valid_from=1.0, valid_to=9.0, **options)
        return record_packet(source, "source/id@7")


@pytest.mark.parametrize("value", [None, Absent, Fraction(2, 7), {"large": 2**90, "tuple": (1, 2)},
                                  np.array([Fraction(1, 7), Absent], dtype=object)])
@pytest.mark.parametrize("signed", [False, True])
def test_exact_payload_metadata_tags_and_validity_round_trip(transport, monkeypatch, value, signed):
    from agent.rcdb import default_store
    key = "record transport key" if signed else None
    if signed: monkeypatch.setenv("REXGRAPH_FRAME_KEY", key)
    rc, original = client(frame_key=key), packet(value)
    receipt = rc.rex_store_record(original, source="hive/λ", courier="mule λ")
    assert receipt.source_store_id == original.source_store_id
    assert receipt.source_digest == receipt.destination_digest == original.state_digest
    assert receipt.destination_record_id.startswith("rx_")
    received = rc.rex_fetch_record(receipt.destination_record_id, version=receipt.destination_version)
    selected = received.snapshot()
    assert pack_value(selected.value) == pack_value(value)
    assert selected.record.meta["exact"] == Fraction(1, 7)
    assert selected.record.meta["workspace"] == "default"
    assert selected.record.meta["source_hive"] == "hive/λ" and selected.record.meta["courier"] == "mule λ"
    assert selected.record.signature["tags"] == ["exact"]
    assert (selected.record.valid_from, selected.record.valid_to) == (1.0, 9.0)
    assert default_store().store_id == received.source_store_id == receipt.destination_store_id
    assert transport.get("/rex/v1/fetch/"+receipt.destination_record_id).status_code == 400


@pytest.mark.parametrize("record_id", [".", "..", "../literal", "slash/id@1", "q?&=#%", "unicode/λ", "ends/"])
def test_client_uses_literal_ids_and_pins_time_or_version_selectors(transport, record_id):
    from agent.rcdb import default_store
    store = default_store()
    store.put_record(record_id, 1, tx_time=10.0, valid_from=1.0, valid_to=5.0)
    store.put_record(record_id, 2, tx_time=20.0, valid_from=5.0, valid_to=9.0)
    rc = client()
    assert rc.rex_fetch_record(record_id, version=1).snapshot().value == 1
    assert rc.rex_fetch_record(record_id, as_of=15.0).snapshot().value == 1
    assert rc.rex_fetch_record(record_id, as_of=20.0, valid_at=5.0).snapshot().value == 2
    with pytest.raises(ValueError, match="combine"):
        rc.rex_fetch_record(record_id, version=1, as_of=15.0)


def test_missing_display_alias_and_cross_workspace_reads_are_absent(transport):
    from agent.rcdb import default_store
    from agent.server.auth import get_auth_manager
    mgr = get_auth_manager(); mgr.enable_auth()
    alpha = mgr.create_token("alice", ["alpha"], role="user")
    beta = mgr.create_token("bob", ["beta"], role="user")
    first = client(api_key=alpha, workspace="alpha")
    receipt = first.rex_store_record(packet(None))
    assert first.rex_fetch_record(receipt.destination_record_id).snapshot().value is None
    second = client(api_key=beta, workspace="beta")
    with pytest.raises(httpx.HTTPStatusError) as denied:
        second.rex_fetch_record(receipt.destination_record_id)
    assert denied.value.response.status_code == 404
    with pytest.raises(httpx.HTTPStatusError) as forbidden:
        client(api_key=beta, workspace="alpha").rex_store_record(packet())
    assert forbidden.value.response.status_code == 403
    default_store().put_record("base", 1, meta={"workspace": "alpha"})
    with pytest.raises(httpx.HTTPStatusError) as absent:
        first.rex_fetch_record("base@1")
    assert absent.value.response.status_code == 404
    assert transport.get("/rex/v1/records/fetch", params={"record_id": receipt.destination_record_id}).status_code == 401


def test_receiver_governance_and_owner_projection_refuse_without_writing(transport):
    from agent.rcdb import default_store
    store = default_store()
    before = store.change_cursor
    store.configure_security(require_commits=True)
    with pytest.raises(httpx.HTTPStatusError) as refused:
        client().rex_store_record(packet())
    assert refused.value.response.status_code == 400 and "cannot govern" in refused.value.response.text
    assert store.change_cursor == before
    store.configure_security(require_commits=False, metadata_fields=[])
    with pytest.raises(httpx.HTTPStatusError) as refused:
        client().rex_store_record(packet())
    assert refused.value.response.status_code == 403 and "workspace" in refused.value.response.text
    assert store.change_cursor == before


def test_packet_publication_uses_copy_record_and_actual_cell_budget(transport, monkeypatch):
    import rcdb
    from agent.rcdb import default_store
    from rcdb.engine import encode_native_payload
    from rcdb import BlobCodecSpec
    with closing(MemoryStore()) as source:
        graph = RexGraph.from_graph([0, 1, 2], [1, 2, 3])
        source.put_prepared("r", encode_native_payload(graph, BlobCodecSpec()), {"nV": 0, "nE": 0, "nF": 0})
        original = record_packet(source, "r")
    called, copy = [], rcdb.copy_record
    def observed(*args, **kwargs):
        called.append(kwargs["destination_id"])
        return copy(*args, **kwargs)
    monkeypatch.setattr(rcdb, "copy_record", observed)
    monkeypatch.setenv("REXGRAPH_MAX_CELLS", "3")
    with pytest.raises(httpx.HTTPStatusError) as refused:
        client().rex_store_record(original)
    assert refused.value.response.status_code == 429
    assert default_store().change_cursor.sequence == 0 and len(called) == 1
    monkeypatch.setenv("REXGRAPH_MAX_CELLS", "4")
    default_store().configure_security(require_commits=True)
    receipt = client().rex_store_record(original)
    assert default_store().verify_commits(receipt.destination_record_id)
    assert len(called) == 2


def test_signature_is_checked_before_packet_decoder(transport, monkeypatch):
    from rexgraph.protocol import sign
    original = packet().to_bytes()
    monkeypatch.setenv("REXGRAPH_FRAME_KEY", "key")
    def parsed(*args, **kwargs): raise AssertionError("decoder reached")
    monkeypatch.setattr(RecordPacket, "from_bytes", parsed)
    for signature in ("", sign(original, b"wrong")):
        response = transport.post("/rex/v1/records/store", content=original,
                                  headers={"Content-Type": PACKET_CONTENT_TYPE, "X-Rex-Signature": signature})
        assert response.status_code == 401


@pytest.mark.parametrize("declared,status", [(None, 413), ("999", 413), ("invalid", 400), ("-1", 400), ("3", 400)])
def test_actual_request_stream_is_bounded_without_trusting_content_length(monkeypatch, declared, status):
    from agent.server.routes.rex import _signed_body
    monkeypatch.delenv("REXGRAPH_FRAME_KEY", raising=False)
    class Request:
        headers = {} if declared is None else {"content-length": declared}
        async def stream(self):
            yield b"ab"
            yield b"cd"
    limit = 3 if declared is None else 100
    with pytest.raises(HTTPException) as refused:
        asyncio.run(_signed_body(Request(), limit))
    assert refused.value.status_code == status


@pytest.mark.parametrize("bad", ["unsigned", "wrong-key", "wrong-source", "wrong-digest", "over-limit", "actual-over-limit"])
def test_client_rejects_bad_or_unbounded_receipts(transport, monkeypatch, bad):
    from rexgraph.protocol import sign
    original = packet(Fraction(1, 7))
    receipt = CopyReceipt(original.source_store_id, original.record.id, original.record.version,
                          original.state_digest, "f"*32, "destination", 1, original.state_digest)
    if bad == "wrong-source": receipt = replace(receipt, source_store_id="e"*32)
    if bad == "wrong-digest": receipt = replace(receipt, destination_digest="e"*64)
    body = receipt.to_bytes()
    headers = {"Content-Type": RECEIPT_CONTENT_TYPE, "X-Rex-Signature": sign(body, b"key")}
    if bad == "unsigned": headers.pop("X-Rex-Signature")
    if bad == "wrong-key": headers["X-Rex-Signature"] = sign(body, b"wrong")
    if bad == "over-limit": headers["Content-Length"] = str(5*1024*1024)
    if bad == "actual-over-limit":
        import rcdb.transfer
        monkeypatch.setattr(rcdb.transfer, "RECEIPT_LIMIT", 100)
    @contextmanager
    def returned(*args, **kwargs):
        yield httpx.Response(200, headers=headers, stream=httpx.ByteStream(body), request=httpx.Request("POST", "http://testserver"))
    monkeypatch.setattr(httpx, "stream", returned)
    with pytest.raises(ValueError): client(frame_key="key").rex_store_record(original)


def test_remote_and_local_couriers_preserve_generic_values_and_metadata_updates(transport):
    from agent.courier import Courier
    from agent.courier_remote import Peer
    from agent.rcdb import default_store
    with closing(MemoryStore()) as source, closing(MemoryStore()) as destination:
        source.put_record("r", None, meta={"q": Fraction(1, 7)})
        courier = Courier("typed")
        courier.attach_store("source", source); courier.attach_store("local", destination)
        peer = Peer("remote", client(), confirm=True); courier.attach_peer(peer)
        assert courier.deliver("source", "remote")["carried"] == 1
        assert courier.deliver("source", "local")["carried"] == 1
        assert peer.retrieve("r") is None
        assert courier.deliver("source", "remote")["held"] == 1
        assert courier.deliver("source", "local")["held"] == 1
        source.put_record("r", None, meta={"q": Fraction(2, 7)})
        remote = courier.deliver("source", "remote")
        assert remote["carried"] == courier.deliver("source", "local")["carried"] == 1
        rid = remote["shipments"][0]["remote_id"]
        assert default_store().get_record(rid).meta["q"] == destination.get_record("r").meta["q"] == Fraction(2, 7)
        entry = peer.ledger.entry("remote", "r")
        receipt = CopyReceipt.from_bytes(bytes.fromhex(entry["receipt"]))
        assert receipt.as_record() == remote["shipments"][0]["receipt"]


def test_ledger_literal_addresses_do_not_collide_and_snapshots_are_owned(tmp_path):
    from agent.courier_remote import Ledger
    path = str(tmp_path/"ledger.json")
    ledger = Ledger(path)
    identity = {"state_digest": "f"*64}
    ledger.note("peer", "part\x1fid", "first", identity)
    ledger.note("peer\x1fpart", "id", "second", identity)
    identity["state_digest"] = "e"*64
    ledger.entry("peer", "part\x1fid")["structure"].clear()
    reopened = Ledger(path)
    assert reopened.remote_id("peer", "part\x1fid") == "first"
    assert reopened.remote_id("peer\x1fpart", "id") == "second"
    assert ledger.structure("peer", "part\x1fid") == {"state_digest": "f"*64}


def test_old_ledger_keys_upgrade_and_invalid_receipts_refuse(tmp_path):
    import json
    from agent.courier_remote import Ledger
    path = tmp_path/"ledger.json"
    entry = {"peer": "p", "record_id": "r", "remote_id": "remote", "structure": {"nV": 5}, "at": 1.0}
    path.write_text(json.dumps({"p\x1fr": entry}))
    assert Ledger(str(path)).remote_id("p", "r") == "remote"
    entry["structure"] = {"record_digest": "f"*64, "state_digest": "f"*64}
    entry["receipt"] = "invalid"
    path.write_text(json.dumps({"p\x1fr": entry}))
    with pytest.raises(ValueError, match="receipt"):
        Ledger(str(path))


def test_declared_datasets_provenance_receipts_and_rcql_results_keep_their_types(transport):
    from rexgraph.io.declaration import DatasetDeclaration
    from rexgraph.io.records import RecordField, RecordSchema
    from rexgraph.relations import RelationSpec
    from rcql import Result, register_result_storage_codec
    from rcql.types import Exactness
    from system.server.app import _result_payload
    declaration = DatasetDeclaration("uninstalled.reader", RecordSchema((RecordField("source"), RecordField("target"))),
                                     RelationSpec("pair", ("source", "target")))
    result = Result(values=(Fraction(1, 7), Absent), exactness=(Exactness.RATIONAL, None),
                    provenance=({"method": "exact"},))
    rc = client()
    with closing(MemoryStore()) as source:
        items = [(DECLARATION_CODEC, declaration), (PROVENANCE_CODEC, {"q": Fraction(1, 7)}),
                 (register_result_storage_codec(), result)]
        first = rc.rex_store_record(packet(1))
        items.append((COPY_RECEIPT_CODEC, first))
        for i, (codec, value) in enumerate(items):
            source.put_record(str(i), value, codec=codec)
            receipt = rc.rex_store_record(record_packet(source, str(i)))
            back = rc.rex_fetch_record(receipt.destination_record_id).snapshot().value
            assert type(back) is type(value)
            if hasattr(value, "to_bytes"): assert back.to_bytes() == value.to_bytes()
            else: assert pack_value(back) == pack_value(value)
            if isinstance(value, Result): assert _result_payload(back) == _result_payload(value)


@pytest.mark.parametrize("temporal", [False, True])
def test_rcql_embedded_graphs_are_admitted_before_result_decode(transport, monkeypatch, temporal):
    from rcql import Result, register_result_storage_codec
    from rexgraph import TemporalRex
    from agent.rcdb import default_store
    value = (TemporalRex([(np.array([0]), np.array([1])), (np.array([0, 1, 2]), np.array([1, 2, 3]))])
             if temporal else RexGraph.from_graph([0, 1, 2], [1, 2, 3]))
    with closing(MemoryStore()) as source:
        source.put_record("result", Result(values=(value,), exactness=(None,)), codec=register_result_storage_codec())
        original = record_packet(source, "result")
    import rexgraph.state as state
    import rexgraph.temporal_state as temporal_state
    def rebuilt(*args, **kwargs): raise AssertionError("result rebuilt before admission")
    monkeypatch.setattr(state, "from_state", rebuilt)
    monkeypatch.setattr(temporal_state, "from_temporal_state", rebuilt)
    monkeypatch.setenv("REXGRAPH_MAX_CELLS", "3")
    with pytest.raises(httpx.HTTPStatusError) as refused:
        client().rex_store_record(original)
    assert refused.value.response.status_code == 429
    assert default_store().change_cursor.sequence == 0


def test_custom_codec_requires_explicit_socket_admission(transport):
    from rcdb import CodecRef, RecordCodec, register_record_codec, unregister_record_codec
    from rcdb.header import DEFAULT_RECORD_CODECS
    from rexgraph.value_codec import unpack_value
    ref = CodecRef("test.operator-admission", 1)
    register_record_codec(RecordCodec(ref, "OperatorValue", pack_value, unpack_value))
    try:
        with closing(MemoryStore(record_codecs=(*DEFAULT_RECORD_CODECS, ref))) as source:
            source.put_record("r", 1, codec=ref)
            original = record_packet(source, "r")
        assert original.snapshot().value == 1
        with pytest.raises(httpx.HTTPStatusError) as refused:
            client().rex_store_record(original)
        assert refused.value.response.status_code == 400 and "admission capability" in refused.value.response.text
    finally:
        unregister_record_codec(ref)


@pytest.mark.parametrize("case", ["receipt", "decoded-limit", "refusal"])
def test_http_compression_preserves_signature_bounds_and_refusal_details(transport, monkeypatch, case):
    import gzip
    from rexgraph.protocol import sign
    import rcdb.transfer
    original = packet(Fraction(1, 7))
    receipt = CopyReceipt(original.source_store_id, original.record.id, original.record.version,
                          original.state_digest, "f"*32, "received", 1, original.state_digest)
    body = b'{"detail":"refused by policy"}' if case == "refusal" else receipt.to_bytes()
    encoded = gzip.compress(body)
    headers = {"Content-Type": "application/json" if case == "refusal" else RECEIPT_CONTENT_TYPE,
               "Content-Encoding": "gzip", "Content-Length": str(len(encoded)),
               "X-Rex-Signature": sign(body, b"key")}
    if case == "decoded-limit":
        assert len(encoded) < len(body)
        monkeypatch.setattr(rcdb.transfer, "RECEIPT_LIMIT", len(encoded)+1)
    @contextmanager
    def returned(*args, **kwargs):
        yield httpx.Response(403 if case == "refusal" else 200, headers=headers,
                             stream=httpx.ByteStream(encoded), request=httpx.Request("POST", "http://testserver"))
    monkeypatch.setattr(httpx, "stream", returned)
    if case == "refusal":
        with pytest.raises(httpx.HTTPStatusError) as refused:
            client(frame_key="key").rex_store_record(original)
        assert refused.value.response.status_code == 403 and "refused by policy" in refused.value.response.text
    elif case == "decoded-limit":
        with pytest.raises(ValueError, match="byte limit"):
            client(frame_key="key").rex_store_record(original)
    else:
        assert client(frame_key="key").rex_store_record(original) == receipt
