"""Sender arbitration, publication failure, and explicit receipt recovery."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
from runpy import run_path
from threading import Barrier

import pytest

from agent.courier_ledger import Ledger, LedgerUncertainError
from agent.courier_remote import Peer
from rcdb import MemoryStore, copy_record, record_packet
from rexgraph.value_codec import pack_value

run_isolated = run_path(str(Path(__file__).resolve().parents[2] / "scripts/test_subprocess.py"))["run_isolated"]


def selected(value=Fraction(2, 7)):
    with closing(MemoryStore()) as store:
        store.put_record("literal/id@1", value, meta={"exact": Fraction(1, 7)}, tags=["exact"])
        return record_packet(store, "literal/id@1")


class Receiver:
    """Trusted capability stub; transfers still use the single RCDB copy seam."""
    def __init__(self, store):
        self.store, self.sent, self.fetched = store, 0, []

    def rex_hello(self):
        return {"record_transfer_version": 1, "limits": {}}

    def rex_store_record(self, packet, **kwargs):
        self.sent += 1
        return copy_record(packet.source(), self.store, packet.record,
                           destination_id=f"remote/{self.sent}", return_receipt=True)

    def rex_fetch_record(self, record_id, *, version):
        self.fetched.append((record_id, version))
        return record_packet(self.store, record_id, version=version)


def ship(peer, packet):
    snapshot = packet.snapshot()
    return peer.ship(snapshot.record, snapshot.value, source="source", courier="test",
                     source_store_id=packet.source_store_id, snapshot=snapshot)


def test_independent_handles_merge_refresh_forget_and_save(tmp_path):
    path = tmp_path / "ledger.json"
    first, second = Ledger(path), Ledger(path)
    first.note("p", "a", "remote/a", {"nV": 3})
    second.note("p", "b", "remote/b", {"nV": 4})
    assert first.remote_id("p", "b") == "remote/b"
    assert second.forget("p", "a") and not first.forget("p", "a")
    first.save()
    assert [e["record_id"] for e in Ledger(path).entries()] == ["b"]
    owned = first.to_dict()
    next(iter(owned.values()))["structure"].clear()
    first.entries()[0]["structure"].clear()
    assert second.structure("p", "b") == {"nV": 4}


def test_threaded_handles_do_not_lose_entries(tmp_path):
    path = tmp_path / "ledger.json"
    ledgers = [Ledger(path) for _ in range(8)]
    start = Barrier(len(ledgers))
    def write(pair):
        i, ledger = pair
        start.wait(timeout=10)
        for j in range(5):
            ledger.note("p/λ", f"{i}/{j}", f"remote/{i}/{j}", {"nV": j})
    with ThreadPoolExecutor(max_workers=len(ledgers)) as pool:
        list(pool.map(write, enumerate(ledgers)))
    assert len(Ledger(path).entries()) == 40


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory process arbitration")
def test_independent_processes_merge_entries(tmp_path):
    path = tmp_path / "ledger.json"
    def write(i):
        code = f"""
from agent.courier_ledger import Ledger
ledger = Ledger({str(path)!r})
for j in range(8):
 ledger.note('p', str({i})+'/'+str(j), 'remote/'+str({i})+'/'+str(j), {{'nV': j}})
"""
        result = run_isolated(code, packages=("agent", "rexgraph", "rcdb"),
                              capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    assert len(Ledger(path).entries()) == 32


def test_reading_a_missing_ledger_creates_nothing(tmp_path):
    parent = tmp_path / "missing"
    ledger = Ledger(parent / "ledger.json")
    assert ledger.entries() == [] and ledger.to_dict() == {}
    assert ledger.entry("p", "r") is None and not parent.exists()


def test_reading_a_historical_ledger_creates_no_sidecar(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"legacy": {"peer": "p", "record_id": "r", "remote_id": "remote",
                                           "structure": {"nV": 3}, "at": 1}}))
    before = {p.name for p in tmp_path.iterdir()}
    assert Ledger(path).remote_id("p", "r") == "remote"
    assert {p.name for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_invalid_sidecar_refuses_before_remote_calls(tmp_path, kind):
    ledger = Ledger(tmp_path / "ledger.json")
    if kind == "symlink": ledger._gate_path.symlink_to(tmp_path / "absent")
    else: ledger._gate_path.mkdir()
    class Unreachable:
        def rex_hello(self): raise AssertionError("network reached")
    assert ship(Peer("p", Unreachable(), ledger), selected()).reason == "refused"
    assert not ledger.path.exists()


def test_same_thread_independent_handles_refresh_inside_delivery_scope(tmp_path):
    path = tmp_path / "ledger.json"
    first, second = Ledger(path), Ledger(path)
    with first.delivery_scope("p", "a"):
        second.note("p", "b", "second", {})
        first.note("p", "a", "first", {})
    assert {e["record_id"] for e in first.entries()} == {"a", "b"}


def test_different_ledgers_in_one_directory_can_send_independently(tmp_path):
    first, second = Ledger(tmp_path / "first.json"), Ledger(tmp_path / "second.json")
    start = Barrier(2)
    def write(ledger):
        with ledger.delivery_scope("p", "r"):
            start.wait(timeout=5)
            ledger.note("p", "r", "remote", {})
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, (first, second)))
    assert first.remote_id("p", "r") == second.remote_id("p", "r") == "remote"


def test_nested_distinct_ledgers_refuse_reverse_lock_order(tmp_path):
    ledgers = [Ledger(tmp_path / name) for name in ("first.json", "second.json")]
    first, second = sorted(ledgers, key=lambda ledger: str(ledger._gate_path))
    with first.delivery_scope("p", "r"):
        with second.delivery_scope("p", "r"): pass
    with second.delivery_scope("p", "r"):
        with pytest.raises(RuntimeError, match="canonical path order"):
            with first.delivery_scope("p", "r"): pass
    assert first.entries() == second.entries() == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX sender process arbitration")
def test_fresh_process_peers_share_one_acknowledged_send_decision(tmp_path):
    packet_path = tmp_path / "source.packet"
    packet_path.write_bytes(selected().to_bytes())
    path, sends = tmp_path / "ledger.json", tmp_path / "sends.txt"
    code = f"""
from pathlib import Path
from contextlib import closing
from agent.courier_remote import Ledger, Peer
from rcdb import MemoryStore, RecordPacket, copy_record
packet = RecordPacket.from_bytes(Path({str(packet_path)!r}).read_bytes())
with closing(MemoryStore()) as dest:
 class Receiver:
  def rex_hello(self): return {{'record_transfer_version': 1, 'limits': {{}}}}
  def rex_store_record(self, packet, **kwargs):
   with Path({str(sends)!r}).open('a') as stream: stream.write('sent\\n')
   return copy_record(packet.source(), dest, packet.record, destination_id='remote', return_receipt=True)
 peer = Peer('p', Receiver(), Ledger({str(path)!r}))
 snapshot = packet.snapshot()
 result = peer.ship(snapshot.record, snapshot.value, source='source', courier='process',
                    source_store_id=packet.source_store_id, snapshot=snapshot)
 assert result.reason in ('shipped', 'held'), result
"""
    def send(_):
        result = run_isolated(code, packages=("agent", "rexgraph", "rcdb"),
                              capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(send, range(4)))
    assert sends.read_text().splitlines() == ["sent"]
    assert len(Ledger(path).entries()) == 1


@pytest.mark.parametrize("bad", [
    "empty", "truncated", "list", "duplicate-key", "duplicate-address", "unknown-field",
    "missing-field", "bad-address", "empty-address", "surrogate", "bad-structure",
    "bool-clock", "bad-clock", "nan", "infinity", "overflow-clock", "overflow-structure",
    "huge-clock", "bad-digest", "missing-receipt", "bad-receipt", "unknown-receipt",
])
def test_corrupt_ledgers_refuse_without_replacing_bytes(tmp_path, bad):
    value = {"peer": "p", "record_id": "r", "remote_id": "remote", "structure": {"nV": 3}, "at": 1.0}
    data = {"old-key": value}
    raw = None
    if bad == "empty": raw = b""
    elif bad == "truncated": raw = b'{"old-key":'
    elif bad == "list": data = []
    elif bad == "duplicate-key": raw = b'{"x":{},"x":{}}'
    elif bad == "duplicate-address": data["second-key"] = dict(value)
    elif bad == "unknown-field": value["secret"] = "unsupported"
    elif bad == "missing-field": value.pop("remote_id")
    elif bad == "bad-address": value["peer"] = 1
    elif bad == "empty-address": value["record_id"] = ""
    elif bad == "surrogate": value["peer"] = "\ud800"
    elif bad == "bad-structure": value["structure"] = []
    elif bad == "bool-clock": value["at"] = True
    elif bad == "bad-clock": value["at"] = "1"
    elif bad == "nan": value["at"] = float("nan")
    elif bad == "infinity": value["structure"] = {"nV": float("inf")}
    elif bad == "overflow-clock": raw = json.dumps(data).replace('1.0', '1e999').encode()
    elif bad == "overflow-structure": raw = json.dumps(data).replace('3}', '1e999}').encode()
    elif bad == "huge-clock": value["at"] = 10**400
    elif bad == "bad-digest": value["structure"] = {"state_digest": "F"*64}
    elif bad == "missing-receipt": value["structure"] = {"state_digest": "f"*64, "record_digest": "e"*64}
    elif bad == "bad-receipt": value["receipt"] = "ff"
    elif bad == "unknown-receipt": value["receipt"] = 7
    path = tmp_path / "ledger.json"
    raw = json.dumps(data).encode() if raw is None else raw
    path.write_bytes(raw)
    with pytest.raises(ValueError): Ledger(path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("bad", ["source-id", "dest-id", "source-digest", "dest-digest", "uppercase", "whitespace"])
def test_persisted_receipt_must_match_its_literal_address_and_state(tmp_path, bad):
    packet = selected()
    path = tmp_path / "ledger.json"
    with closing(MemoryStore()) as destination:
        receipt = Receiver(destination).rex_store_record(packet)
        ledger = Ledger(path)
        identity = {"state_digest": packet.state_digest, "record_digest": packet.selection_digest}
        ledger.note("p", packet.record.id, receipt.destination_record_id, identity, receipt=receipt)
    data = json.loads(path.read_bytes())
    entry = next(iter(data.values()))
    changes = {"source-id": {"source_record_id": "other"}, "dest-id": {"destination_record_id": "other"},
               "source-digest": {"source_digest": "e"*64}, "dest-digest": {"destination_digest": "e"*64}}
    if bad in changes: entry["receipt"] = replace(receipt, **changes[bad]).to_bytes().hex()
    elif bad == "uppercase": entry["receipt"] = entry["receipt"].upper()
    else: entry["receipt"] = " "+entry["receipt"]
    raw = json.dumps(data).encode(); path.write_bytes(raw)
    with pytest.raises(ValueError): Ledger(path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("kind", ["symlink", "directory", "deleted", "truncated"])
def test_invalid_live_ledger_refuses_before_network(tmp_path, kind):
    packet, path = selected(), tmp_path / "ledger.json"
    ledger = Ledger(path)
    ledger.note("p", "other", "remote", {"nV": 1})
    if kind == "deleted": path.unlink()
    elif kind == "truncated": path.write_bytes(b"{")
    else:
        path.rename(tmp_path / "previous.json")
        if kind == "symlink": path.symlink_to(tmp_path / "previous.json")
        else: path.mkdir()
    class Unreachable:
        def rex_hello(self): raise AssertionError("network reached")
    result = ship(Peer("p", Unreachable(), ledger), packet)
    assert result.reason == "refused" and "ledger unavailable" in result.detail


def test_bounded_read_and_write_preserve_previous_publication(tmp_path, monkeypatch):
    import agent.courier_ledger as module
    path = tmp_path / "ledger.json"
    ledger = Ledger(path)
    ledger.note("p", "r", "remote", {"nV": 3})
    before = path.read_bytes()
    monkeypatch.setattr(module, "LEDGER_LIMIT", len(before)-1)
    with pytest.raises(ValueError, match="byte limit"): ledger.entries()
    with pytest.raises(ValueError, match="byte limit"): Ledger().note("p", "r", "remote", {"text": "x"*1000})
    assert path.read_bytes() == before


def test_write_failure_before_preparation_keeps_old_state(tmp_path, monkeypatch):
    import agent.courier_ledger as module
    path = tmp_path / "ledger.json"
    ledger = Ledger(path); ledger.note("p", "a", "first", {})
    before = path.read_bytes()
    @contextmanager
    def refused(target):
        raise PermissionError("cannot stage")
        yield
    with monkeypatch.context() as patch:
        patch.setattr(module, "staged_publication", refused)
        with pytest.raises(PermissionError): ledger.note("p", "b", "second", {})
    assert path.read_bytes() == before and ledger.remote_id("p", "b") is None


@pytest.mark.parametrize("published", [False, True])
def test_unknown_acknowledgement_requires_explicit_verified_reload(tmp_path, monkeypatch, published):
    import agent.courier_ledger as module
    path = tmp_path / "ledger.json"
    ledger = Ledger(path); ledger.note("p", "a", "first", {})
    real = module.staged_publication
    @contextmanager
    def lost_ack(target):
        with real(target) as staged:
            yield staged
            if not published: raise OSError("lost before replace")
        raise OSError("lost after replace")
    # Nested reload must read the actual published image, not the old memory copy.
    with ledger.delivery_scope("p", "b"):
        with monkeypatch.context() as patch:
            patch.setattr(module, "staged_publication", lost_ack)
            with pytest.raises(LedgerUncertainError): ledger.note("p", "b", "second", {})
        for operation in (ledger.entries, ledger.save, lambda: ledger.forget("p", "a"),
                          lambda: ledger.note("p", "c", "third", {})):
            with pytest.raises(LedgerUncertainError): operation()
        ledger.load()
        assert ledger.remote_id("p", "b") == ("second" if published else None)
    assert len(Ledger(path).entries()) == (2 if published else 1)


def test_corrupt_reload_does_not_clear_uncertain_state(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = Ledger(path); ledger.note("p", "a", "first", {})
    ledger._uncertain = True
    path.write_bytes(b"{")
    with pytest.raises(ValueError): ledger.load()
    with pytest.raises(LedgerUncertainError): ledger.entries()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork inheritance")
def test_inherited_handle_refuses_before_entering_inherited_lock(tmp_path):
    code = f"""
import os
from agent.courier_ledger import Ledger
path = {str(tmp_path / 'ledger.json')!r}
ledger = Ledger(path)
with ledger.delivery_scope('p', 'r'):
 child = os.fork()
 if child == 0:
  try:
   ledger.entries()
  except RuntimeError as error:
   assert 'fresh handle' in str(error)
   os._exit(0)
  os._exit(2)
 assert os.waitpid(child, 0)[1] == 0
Ledger(path).note('p', 'r', 'remote', {{}})
"""
    result = run_isolated(code, packages=("agent", "rexgraph", "rcdb"), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert Ledger(tmp_path / "ledger.json").remote_id("p", "r") == "remote"


@pytest.mark.parametrize("after_replace", [False, True])
def test_abrupt_process_exit_leaves_a_complete_old_or_new_image(tmp_path, after_replace):
    path = tmp_path / "ledger.json"
    Ledger(path).note("p", "a", "first", {})
    code = f"""
import os
import rexgraph.io.publication as publication
from agent.courier_ledger import Ledger
real = publication.os.replace
def exit_at_replace(source, destination):
 if str(destination) == {str(path)!r}:
  if {after_replace!r}: real(source, destination)
  os._exit(17)
 return real(source, destination)
publication.os.replace = exit_at_replace
Ledger({str(path)!r}).note('p', 'b', 'second', {{}})
"""
    result = run_isolated(code, packages=("agent", "rexgraph", "rcdb"), capture_output=True, text=True, timeout=20)
    assert result.returncode == 17, result.stderr
    reopened = Ledger(path)
    assert reopened.remote_id("p", "a") == "first"
    assert reopened.remote_id("p", "b") == ("second" if after_replace else None)


@pytest.mark.parametrize("shared_handle", [False, True, "memory"])
def test_concurrent_peers_make_one_send_decision(tmp_path, shared_handle):
    packet = selected()
    with closing(MemoryStore()) as destination:
        receiver, path = Receiver(destination), tmp_path / "ledger.json"
        shared = Ledger() if shared_handle == "memory" else Ledger(path)
        peers = [Peer("p", receiver, shared if shared_handle else Ledger(path), confirm=True) for _ in range(8)]
        start = Barrier(len(peers))
        def send(peer):
            start.wait(timeout=10)
            return ship(peer, packet)
        with ThreadPoolExecutor(max_workers=len(peers)) as pool:
            results = list(pool.map(send, peers))
        assert [r.reason for r in results].count("shipped") == 1
        assert [r.reason for r in results].count("held") == 7
        assert receiver.sent == 1 and len(destination.list()) == 1


@pytest.mark.parametrize("value", [None, Fraction(2, 7), {"large": 2**90, "tuple": (1, 2)}])
def test_known_receipt_repairs_failed_sender_publication_without_resending(tmp_path, monkeypatch, value):
    import agent.courier_ledger as module
    packet, path = selected(value), tmp_path / "ledger.json"
    real = module.staged_publication
    @contextmanager
    def lost_ack(target):
        with real(target) as staged: yield staged
        raise OSError("lost acknowledgement")
    with closing(MemoryStore()) as destination:
        receiver = Receiver(destination)
        peer = Peer("p", receiver, Ledger(path))
        with monkeypatch.context() as patch:
            patch.setattr(module, "staged_publication", lost_ack)
            result = ship(peer, packet)
        assert result.reason == "refused" and not result.shipped
        assert result.remote_id and result.receipt and "not acknowledged" in result.detail
        from rcdb import CopyReceipt
        receipt = CopyReceipt.from_record(result.receipt)
        reconciled = peer.reconcile(packet, receipt)
        assert reconciled.reason == "held" and not reconciled.shipped
        assert receiver.sent == 1 and len(destination.list()) == 1
        assert receiver.fetched == [(receipt.destination_record_id, receipt.destination_version)]
        assert pack_value(peer.retrieve(packet.record.id)) == pack_value(value)
        assert ship(peer, packet).reason == "held" and receiver.sent == 1


@pytest.mark.parametrize("field,foreign", [
    ("source_store_id", "e"*32), ("source_record_id", "foreign"), ("source_version", 2),
    ("source_digest", "e"*64), ("destination_digest", "e"*64),
])
def test_reconciliation_refuses_foreign_source_receipt_before_fetch(tmp_path, field, foreign):
    packet = selected()
    with closing(MemoryStore()) as destination:
        receiver = Receiver(destination); receipt = receiver.rex_store_record(packet)
        peer = Peer("p", receiver, Ledger(tmp_path / "ledger.json"))
        with pytest.raises(ValueError, match="selected source"):
            peer.reconcile(packet, replace(receipt, **{field: foreign}))
        assert receiver.fetched == [] and peer.ledger.entries() == [] and receiver.sent == 1


@pytest.mark.parametrize("field,foreign", [
    ("destination_store_id", "e"*32), ("destination_record_id", "foreign"),
    ("destination_version", 2), ("destination_digest", "e"*64),
])
def test_retrieval_checks_complete_destination_receipt_identity(field, foreign):
    packet = selected()
    with closing(MemoryStore()) as destination:
        receiver = Receiver(destination); receipt = receiver.rex_store_record(packet)
        peer = Peer("p", receiver)
        # Simulate a capability returning a packet from a different receipt address.
        original = record_packet(destination, receipt.destination_record_id)
        receiver.rex_fetch_record = lambda *args, **kwargs: original
        with pytest.raises(ValueError, match="different record"):
            peer._confirm_record(replace(receipt, **{field: foreign}))


def test_retrieve_uses_one_owned_entry_instead_of_split_ledger_reads(monkeypatch):
    packet = selected()
    with closing(MemoryStore()) as destination:
        receiver = Receiver(destination)
        peer = Peer("p", receiver)
        assert ship(peer, packet).shipped
        def split_read(*args): raise AssertionError("split address read")
        monkeypatch.setattr(peer.ledger, "remote_id", split_read)
        assert peer.retrieve(packet.record.id) == Fraction(2, 7)


@pytest.mark.parametrize("confirm", [False, True])
def test_legacy_sender_note_failure_is_refused_with_known_address(tmp_path, monkeypatch, confirm):
    from agent.courier import Courier
    from rexgraph import RexGraph
    graph = RexGraph.from_graph([0], [1])
    with closing(MemoryStore()) as source:
        source.put("graph", graph, analytics=False)
        class Legacy:
            def rex_hello(self): return {"limits": {}}
            def rex_store(self, value, **kwargs): return {"record_id": "known-remote"}
            def rex_fetch(self, record_id): return graph
        peer = Peer("p", Legacy(), Ledger(tmp_path / "ledger.json"), confirm=confirm)
        courier = Courier("test"); courier.attach_store("source", source); courier.attach_peer(peer)
        def refused(entries): raise PermissionError("cannot write ledger")
        monkeypatch.setattr(peer.ledger, "_publish", refused)
        trip = courier.deliver("source", "p")
        assert trip["refused"] == 1 and trip["carried"] == 0
        result = trip["shipments"][0]
        assert result["remote_id"] == "known-remote" and "not acknowledged" in result["detail"]
        assert peer.ledger.entries() == []


def test_interruption_keeps_control_flow_and_requires_verified_reload(tmp_path, monkeypatch):
    import agent.courier_ledger as module
    ledger = Ledger(tmp_path / "ledger.json")
    real = module.staged_publication
    @contextmanager
    def interrupted(target):
        with real(target) as staged: yield staged
        raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(module, "staged_publication", interrupted)
        with pytest.raises(KeyboardInterrupt): ledger.note("p", "r", "remote", {})
    with pytest.raises(LedgerUncertainError): ledger.entries()
    ledger.load()
    assert ledger.remote_id("p", "r") == "remote"


def test_real_socket_transfer_and_receipt_reconciliation(tmp_path):
    """Loopback server, actual RexClient, bearer/workspace/HMAC and durable ledger."""
    env = dict(os.environ, REXGRAPH_CONFIG_DIR=str(tmp_path / "config"),
               REXGRAPH_RCDB_URI=f"local://{tmp_path}/records",
               REXGRAPH_AUDIT_JOURNAL=str(tmp_path / "audit.jsonl"),
               REXGRAPH_ACTIVITY_JOURNAL=str(tmp_path / "activity.jsonl"),
               REXGRAPH_FRAME_KEY="socket-test-key")
    code = f"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction
import socket, threading, time
import uvicorn
from agent.client import RexClient
from agent.courier_remote import Peer, Ledger
from agent.courier import get_courier
from agent.server.auth import get_auth_manager
from agent.server.app import app
from agent import activity
from rcdb import MemoryStore, CopyReceipt, record_packet
manager = get_auth_manager(); manager.enable_auth()
token = manager.create_token('alice', ['alpha'], role='user')
admin = manager.create_token('operator', ['default', 'alpha'], role='admin')
listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen(64)
port = listener.getsockname()[1]
server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
thread.start()
try:
 deadline = time.monotonic()+15
 while not server.started:
  assert thread.is_alive() and time.monotonic() < deadline
  time.sleep(.01)
 with closing(MemoryStore()) as source:
  source.put_record('source/λ', None, meta={{'q': Fraction(1, 7)}})
  packet = record_packet(source, 'source/λ'); snapshot = packet.snapshot()
  peers = [Peer('p', RexClient('http://127.0.0.1:'+str(port), api_key=token,
                 workspace='alpha', frame_key='socket-test-key'), Ledger({str(tmp_path / 'ledger.json')!r}),
                 confirm=True) for i in range(4)]
  def send(peer):
   return peer.ship(snapshot.record, snapshot.value, source='source', courier='socket',
                    source_store_id=packet.source_store_id, snapshot=snapshot)
  with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(send, peers))
  assert [r.reason for r in results].count('shipped') == 1, results
  assert [r.reason for r in results].count('held') == 3, results
  shipped = next(r for r in results if r.shipped)
  receipt = CopyReceipt.from_record(shipped.receipt)
  courier = get_courier(); courier.attach_store('source', source); courier.attach_peer(peers[0])
  control = RexClient('http://127.0.0.1:'+str(port), api_key=admin, workspace='alpha')
  peers[0].ledger.forget('p', packet.record.id)
  reconciled = control._post('/api/v1/courier/reconcile', json={{'source': 'source', 'dest': 'p',
                              'record_id': packet.record.id, 'receipt': receipt.as_record()}})
  assert reconciled['reason'] == 'held' and not reconciled['shipped']
  assert peers[0].retrieve(packet.record.id) is None
  from agent.rcdb import default_store
  assert len(default_store().list()) == 1
  assert default_store().get_record(receipt.destination_record_id).meta['workspace'] == 'alpha'
  peers[0].ledger.forget('p', packet.record.id)
  assert control._post('/api/v1/courier/deliver', json={{'source':'source','dest':'p'}})['carried'] == 1
  assert control._post('/api/v1/courier/broadcast', json={{'source':'source'}})['trips'][0]['held'] == 1
  assert len(default_store().list()) == 2
finally:
 server.should_exit = True; thread.join(timeout=15)
 listener.close(); activity.get_log().close()
 assert not thread.is_alive()
"""
    result = run_isolated(code, packages=("agent", "rexgraph", "rcdb", "rcql", "system"),
                          env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
