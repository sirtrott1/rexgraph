"""Conditional native publication and owned route/selection snapshots.

Races are forced between comparison and the existing copy seam. Each assertion
checks durable state, rather than only an exception or a counter.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from fractions import Fraction
from pathlib import Path
from runpy import run_path
from threading import Barrier, Event
import os

import pytest

from agent import activity, rcdb
from agent.courier import CarrySpec, Courier
from rexgraph.graph import RexGraph

run_isolated = run_path(str(Path(__file__).resolve().parents[2] / "scripts/test_subprocess.py"))["run_isolated"]
KINDS = ("memory", "local", "sql", "object-file", "object-memory")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", str(tmp_path / "activity.jsonl"))
    from agent.courier import reset_courier
    from agent.hive_network import reset_network
    activity.reset(); reset_courier(); reset_network()
    yield
    reset_courier(); reset_network(); activity.get_log().close()


def opened(kind, path):
    if kind == "memory":
        return rcdb.MemoryStore()
    if kind == "local":
        return rcdb.LocalStore(path)
    if kind == "sql":
        return rcdb.SQLStore(f"sqlite:///{path}.sqlite")
    pytest.importorskip("fsspec")
    return rcdb.NativeObjectStore(f"{'file' if kind == 'object-file' else 'memory'}://{path}")


def courier(src, dst, **kw):
    c = Courier("test", **kw)
    c.attach_store("source", src); c.attach_store("dest", dst)
    return c


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", [None, {"q": Fraction(1, 7)}, "graph"])
def test_concurrent_native_arrivals_publish_one_version(kind, value, tmp_path, monkeypatch):
    with ExitStack() as stack:
        src = stack.enter_context(closing(rcdb.MemoryStore()))
        if value == "graph":
            src.put("r", RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)]), analytics=False)
        else:
            src.put_record("r", value)
        first = stack.enter_context(closing(opened(kind, tmp_path / "dest")))
        destinations = [first] + [first if kind == "memory" else
            stack.enter_context(closing(opened(kind, tmp_path / "dest"))) for _ in range(3)]
        workers = [courier(src, dst) for dst in destinations]
        real_copy, barrier = rcdb.copy_record, Barrier(4)

        def racing_copy(*a, **kw):
            assert kw["expected_cursor"] is not None
            barrier.wait(timeout=15)
            return real_copy(*a, **kw)

        monkeypatch.setattr(rcdb, "copy_record", racing_copy)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda c: c.deliver("source", "dest"), workers))
        assert sum(r["carried"] for r in results) == 1, results
        assert sum(r["held"] for r in results) == 3, results
        assert len(first.history("r")) == 1
        assert first.read_record("r").state_digest == src.read_record("r").state_digest
        assert sum(c.status()["trips"] for c in workers) == 4


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("change", ["unrelated", "replace", "delete", "revive", "absent-aba", "same-arrival"])
def test_known_negative_conflict_only_rebases_unchanged_literal_address(kind, change, tmp_path, monkeypatch):
    with closing(rcdb.MemoryStore()) as src, closing(opened(kind, tmp_path / "dest")) as dst:
        src.put_record("r", Fraction(1, 7))
        if change in ("replace", "delete", "revive"):
            dst.put_record("r", "prior")
        c, real_copy, calls = courier(src, dst), rcdb.copy_record, []

        def interrupt(*a, **kw):
            calls.append(kw["expected_cursor"])
            if len(calls) == 1:
                if change == "unrelated":
                    dst.put_record("other", None)
                elif change == "replace":
                    dst.put_record("r", "winner")
                elif change == "delete":
                    dst.delete("r")
                elif change in ("revive", "absent-aba"):
                    if change == "revive":
                        dst.delete("r")
                    dst.put_record("r", "winner")
                    if change == "absent-aba":
                        dst.delete("r")
                else:
                    real_copy(*a, **kw)
            return real_copy(*a, **kw)

        monkeypatch.setattr(rcdb, "copy_record", interrupt)
        trip = c.deliver("source", "dest")
        if change == "unrelated":
            assert trip["carried"] == 1 and len(calls) == 2
            assert dst.get_record("other") is not None and len(dst.history("r")) == 1
        elif change == "same-arrival":
            assert trip["held"] == 1 and len(calls) == 1 and len(dst.history("r")) == 1
        else:
            assert trip["conflict"] == 1 and trip["carried"] == 0 and len(calls) == 1
            snap = dst.read_record("r")
            assert snap is None if change in ("delete", "absent-aba") else snap.value == "winner"


@pytest.mark.parametrize("kind", KINDS)
def test_unrelated_writers_cannot_cause_unbounded_retry(kind, tmp_path, monkeypatch):
    with closing(rcdb.MemoryStore()) as src, closing(opened(kind, tmp_path / "dest")) as dst:
        src.put_record("r", None)
        real_copy, calls = rcdb.copy_record, []
        def interrupt(*a, **kw):
            calls.append(kw["expected_cursor"])
            dst.put_record("other", len(calls))
            return real_copy(*a, **kw)
        monkeypatch.setattr(rcdb, "copy_record", interrupt)
        trip = courier(src, dst, copy_attempts=2).deliver("source", "dest")
        assert trip["conflict"] == 1 and len(calls) == 2 and dst.read_record("r") is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("change_cursor", [False, True])
def test_policy_change_refuses_without_effectful_retry(kind, change_cursor, tmp_path, monkeypatch):
    with closing(rcdb.MemoryStore()) as src, closing(opened(kind, tmp_path / "dest")) as dst:
        src.put_record("r", None)
        real_copy, calls = rcdb.copy_record, []
        def interrupt(*a, **kw):
            calls.append(1)
            if change_cursor:
                dst.put_record("other", None)
            dst.configure_security(metadata_fields=[])
            return real_copy(*a, **kw)
        monkeypatch.setattr(rcdb, "copy_record", interrupt)
        trip = courier(src, dst).deliver("source", "dest")
        assert trip["refused"] == 1 and len(calls) == 1 and dst.read_record("r") is None


@pytest.mark.parametrize("kind", KINDS)
def test_lossy_courier_projection_is_refused_before_publication(kind, tmp_path):
    with closing(rcdb.MemoryStore()) as src, closing(opened(kind, tmp_path / "dest")) as dst:
        src.put_record("r", None)
        dst.configure_security(metadata_fields=["workspace"])
        before = dst.change_cursor
        trip = courier(src, dst).deliver("source", "dest")
        assert trip["refused"] == 1 and dst.change_cursor == before


@pytest.mark.parametrize("kind", KINDS)
def test_revival_reports_live_parent_and_copies_exact_typed_record(kind, tmp_path):
    from rcql import Result
    from rcql.result_codec import register_result_storage_codec
    reference = register_result_storage_codec()
    with closing(rcdb.MemoryStore()) as src, closing(opened(kind, tmp_path / "dest")) as dst:
        result = Result((Fraction(1, 7),))
        src.put_record("r", result, codec=reference, valid_from=2., valid_to=9., meta={"q": Fraction(2, 7)})
        dst.put_record("r", None); dst.delete("r")
        c = courier(src, dst)
        trip = c.deliver("source", "dest")
        assert trip["deliveries"][0]["parent_version"] is None
        assert trip["deliveries"][0]["version"] == 2
        snapshot = dst.read_record("r")
        assert snapshot.value.values == (Fraction(1, 7),)
        assert (snapshot.record.valid_from, snapshot.record.valid_to) == (2., 9.)
        assert snapshot.record.meta["q"] == Fraction(2, 7)
        assert c.deliver("source", "dest")["held"] == 1
        src.put_record("r", result, codec=reference, valid_from=2., valid_to=9., meta={"q": Fraction(3, 7)})
        assert c.deliver("source", "dest")["deliveries"][0]["parent_version"] == 2


@pytest.mark.parametrize("uncertain", [False, True])
def test_publication_failures_are_separate_from_source_read_failures(uncertain, monkeypatch):
    with closing(rcdb.MemoryStore()) as src, closing(rcdb.MemoryStore()) as dst:
        src.put_record("r", None)
        real_copy, calls = rcdb.copy_record, []
        def fail(*a, **kw):
            calls.append(1)
            if uncertain:
                real_copy(*a, **kw)
                raise rcdb.PublicationUncertainError("private backend detail")
            raise OSError("private backend detail")
        monkeypatch.setattr(rcdb, "copy_record", fail)
        trip = courier(src, dst).deliver("source", "dest")
        assert trip["uncertain" if uncertain else "refused"] == 1
        assert trip["unreadable"] == 0 and len(calls) == 1
        assert "private" not in str(trip)
        assert len(dst.history("r")) == int(uncertain)


def test_self_copy_holds_selected_history_after_current_version_advances(monkeypatch):
    with closing(rcdb.MemoryStore()) as st:
        st.put_record("r", "first")
        c, original = courier(st, st), st.list
        def select_then_advance(*a, **kw):
            records = original(*a, **kw)
            st.put_record("r", "newer")
            return records
        monkeypatch.setattr(st, "list", select_then_advance)
        assert c.deliver("source", "dest")["held"] == 1
        assert st.read_record("r").value == "newer" and len(st.history("r")) == 2


def test_equal_owner_does_not_bypass_workspace_visibility():
    from agent.server.scope import ScopedStore
    with closing(rcdb.MemoryStore()) as st:
        src, dst = ScopedStore(st, "alpha"), ScopedStore(st, "beta")
        src.put_record("r", None)
        trip = courier(src, dst).deliver("source", "dest")
        assert trip["refused"] == 1 and trip["held"] == 0
        assert len(st.history("r")) == 1 and st.get_record("r").meta["workspace"] == "alpha"


def test_broadcast_pins_routes_and_source_versions_before_first_destination(monkeypatch):
    with ExitStack() as stack:
        src, a, b, replacement = [stack.enter_context(closing(rcdb.MemoryStore())) for _ in range(4)]
        src.put_record("r", "selected")
        c, real_copy = courier(src, a), rcdb.copy_record
        c.attach_store("second", b)
        def change_bindings(*args, **kw):
            if args[1] is a:
                c.attach_store("second", replacement)
                src.put_record("r", "newer")
            return real_copy(*args, **kw)
        monkeypatch.setattr(rcdb, "copy_record", change_bindings)
        trips = c.broadcast("source", ["dest", "second"])
        assert trips["carried"] == 2
        assert a.read_record("r").value == b.read_record("r").value == "selected"
        assert replacement.read_record("r") is None
        assert c.deliver("source", "second")["carried"] == 1
        assert replacement.read_record("r").value == "newer"


def test_route_rebinding_does_not_redirect_an_inflight_trip_or_hold_registry(monkeypatch):
    with ExitStack() as stack:
        src, dst, new_src, new_dst = [stack.enter_context(closing(rcdb.MemoryStore())) for _ in range(4)]
        src.put_record("r", "old"); new_src.put_record("r", "new")
        c, real_copy, started, release = courier(src, dst), rcdb.copy_record, Event(), Event()
        def blocked(*a, **kw):
            started.set(); assert release.wait(15)
            return real_copy(*a, **kw)
        monkeypatch.setattr(rcdb, "copy_record", blocked)
        with ThreadPoolExecutor(max_workers=2) as pool:
            trip = pool.submit(c.deliver, "source", "dest")
            try:
                assert started.wait(15)
                def rebind():
                    c.attach_store("source", new_src); c.attach_store("dest", new_dst)
                    return c.status()
                assert pool.submit(rebind).result(timeout=10)["trips"] == 0
            finally:
                release.set()
            assert trip.result(timeout=15)["carried"] == 1
        assert dst.read_record("r").value == "old" and new_dst.read_record("r") is None
        assert c.deliver("source", "dest")["carried"] == 1 and new_dst.read_record("r").value == "new"


def test_peer_rebind_during_broadcast_uses_captured_peer():
    from agent.courier_remote import Shipment
    with closing(rcdb.MemoryStore()) as src:
        src.put_record("r", None)
        c, seen = Courier(), []
        c.attach_store("source", src)
        class Peer:
            def __init__(self, name, label): self.name, self.label = name, label
            def ship(self, record, value, **kw):
                seen.append(self.label)
                if self.name == "first": c.attach_peer(Peer("second", "replacement"))
                return Shipment(record.id, "held")
        c.attach_peer(Peer("first", "first")); c.attach_peer(Peer("second", "selected"))
        assert c.broadcast("source", ["first", "second"])["carried"] == 0
        assert seen == ["first", "selected"]


def test_same_name_cannot_route_to_both_store_and_peer(monkeypatch):
    from types import SimpleNamespace
    with closing(rcdb.MemoryStore()) as st:
        c = Courier(); c.attach_store("x", st)
        with pytest.raises(ValueError): c.attach_peer(SimpleNamespace(name="x"))
        c.attach_peer(SimpleNamespace(name="remote"))
        def must_not_open(*a, **kw): pytest.fail("colliding route opened a store")
        monkeypatch.setattr(rcdb, "open_store", must_not_open)
        with pytest.raises(ValueError): c.attach_store("remote", "memory://")


def test_uri_open_race_closes_only_new_owned_handle(monkeypatch):
    from types import SimpleNamespace
    c, closed = Courier(), []
    handle = SimpleNamespace(close=lambda: closed.append("new"))
    def racing_open(uri):
        c.attach_peer(SimpleNamespace(name="remote"))
        return handle
    monkeypatch.setattr(rcdb, "open_store", racing_open)
    with pytest.raises(ValueError): c.attach_store("remote", "memory://")
    assert closed == ["new"] and c.destinations() == ["remote"]


@pytest.mark.parametrize("limit", [-1, True, 1.0, "1", None, 2**63])
def test_selection_rejects_non_native_or_out_of_range_limits(limit):
    with pytest.raises(ValueError): CarrySpec(limit=limit)


@pytest.mark.parametrize("field", ["tags", "ids"])
@pytest.mark.parametrize("value", ["r", False, {"r": 1}, [1], [""], [None]])
def test_selection_rejects_ambiguous_names(field, value):
    with pytest.raises(ValueError): CarrySpec.from_dict({field: value})


def test_owned_spec_status_and_zero_limit_need_no_record_reads():
    from types import SimpleNamespace
    c = Courier(carry=CarrySpec(tags=["configured"]))
    c.attach_store("source", SimpleNamespace())
    status = c.status(); status["carry"]["tags"].append("external")
    assert c.status()["carry"]["tags"] == ["configured"]
    assert c.survey("source", carry=CarrySpec(limit=0)) == []
    assert CarrySpec(limit=0).select(object()) == []
    tags = ["one"]; spec = CarrySpec(tags=tags); tags.append("two")
    assert spec.tags == ["one"]


def test_named_selection_stops_after_limit_successful_reads():
    from types import SimpleNamespace
    reads = []
    def get_record(name):
        reads.append(name)
        return None if name == "missing" else SimpleNamespace(id=name)
    rows = CarrySpec(ids=["missing", "a", "b", "unused"], limit=2).select(SimpleNamespace(get_record=get_record))
    assert [r.id for r in rows] == ["a", "b"] and reads == ["missing", "a", "b"]


def test_explicit_empty_selection_clears_configured_filter():
    with closing(rcdb.MemoryStore()) as src, closing(rcdb.MemoryStore()) as dst:
        src.put_record("r", None)
        c = courier(src, dst, carry=CarrySpec(tags=["absent"]))
        assert c.handler({"source": "source", "dest": "dest"})["considered"] == 0
        assert c.handler({"source": "source", "dest": "dest", "tags": []})["carried"] == 1
        assert c.handler({"source": "source", "dest": "dest", "limit": 0})["considered"] == 0


def test_parallel_trip_counters_and_workspace_network_factory():
    from agent.courier import get_courier
    from agent.hive_network import get_network
    from agent.server.scope import set_workspace, reset_workspace
    with closing(rcdb.MemoryStore()) as src, closing(rcdb.MemoryStore()) as dst:
        src.put_record("r", None)
        c = get_courier(); c.attach_store("source", src); c.attach_store("dest", dst)
        def deliver(index):
            token = set_workspace("alpha" if index % 2 else "beta")
            try:
                assert get_courier() is c
                assert c.network is get_network()
                return c.deliver("source", "dest")
            finally:
                reset_workspace(token)
        with ThreadPoolExecutor(max_workers=8) as pool:
            trips = list(pool.map(deliver, range(16)))
        assert sum(t["carried"] for t in trips) == 1 and sum(t["held"] for t in trips) == 15
        assert (c.status()["trips"], c.status()["carried"]) == (16, 1)
        assert len(get_network("alpha")._net._msgs) == len(get_network("beta")._net._msgs) == 8
        assert len(get_network("default")._net._msgs) == 0


def test_global_network_and_courier_creation_are_singletons_under_contention():
    from agent.courier import get_courier
    from agent.hive_network import get_network
    barrier = Barrier(8)
    def get(index):
        barrier.wait(15)
        return get_courier(), get_network("alpha")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(get, range(8)))
    assert len({id(c) for c, _ in results}) == len({id(n) for _, n in results}) == 1


def test_network_monitor_and_relay_share_one_traffic_gate():
    from agent.hive_network import HiveNetwork
    net, entered, release, monitored = HiveNetwork(), Event(), Event(), Event()
    class Complex:
        def add_message(self, *a, **kw):
            entered.set(); assert release.wait(15)
        def monitor(self): monitored.set(); return {}
    net._net = Complex()
    with ThreadPoolExecutor(max_workers=2) as pool:
        relay = pool.submit(net.relay, "a", "b", "text")
        try:
            assert entered.wait(10)
            monitor = pool.submit(net.monitor)
            assert not monitored.wait(.1)
        finally:
            release.set()
        relay.result(timeout=10); monitor.result(timeout=10)
        assert monitored.is_set()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork inheritance")
def test_inherited_courier_refuses_and_child_singletons_start_fresh(tmp_path):
    code = """
import os, threading
from agent.courier import get_courier, _COURIER_LOCK
from agent.hive_network import get_network, _NETWORKS_LOCK
c = get_courier(); n = get_network(); n.hive('parent')
entered, release = threading.Event(), threading.Event()
def hold():
 with c._lock, _COURIER_LOCK, _NETWORKS_LOCK:
  entered.set(); assert release.wait(15)
t = threading.Thread(target=hold); t.start(); assert entered.wait(10)
child = os.fork()
if child == 0:
 try: c.status()
 except RuntimeError: pass
 else: os._exit(2)
 assert get_courier().hives() == []
 assert get_network().hives() == []
 os._exit(0)
try: assert os.waitpid(child, 0)[1] == 0
finally: release.set(); t.join(10)
assert not t.is_alive()
"""
    result = run_isolated(code, packages=("agent", "rexgraph", "rcdb"), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["local", "sql", "object-file"])
def test_fresh_process_arrivals_use_native_conditional_copy(kind, tmp_path):
    src_path, dst_path = tmp_path / "src", tmp_path / "dst"
    with closing(rcdb.LocalStore(src_path)) as src:
        src.put_record("r", {"q": Fraction(1, 7)})
    # Launch independently imported interpreters. No inherited store handles,
    # pools or native runtime threads enter a child process.
    constructor = {"local": f"rcdb.LocalStore({str(dst_path)!r})",
                   "sql": f"rcdb.SQLStore({'sqlite:///'+str(dst_path)+'.sqlite'!r})",
                   "object-file": f"rcdb.NativeObjectStore({'file://'+str(dst_path)!r})"}[kind]
    code = f"""
from agent import rcdb
from agent.courier import Courier
from contextlib import closing
from pathlib import Path
import os, time
with closing(rcdb.LocalStore({str(src_path)!r})) as src, closing({constructor}) as dst:
 c = Courier(); c.attach_store('source', src); c.attach_store('dest', dst)
 real = rcdb.copy_record
 def copy(*a, **kw):
  Path({str(tmp_path)!r}, 'ready-'+str(os.getpid())).touch()
  deadline = time.monotonic()+15
  while len(list(Path({str(tmp_path)!r}).glob('ready-*'))) < 3:
   assert time.monotonic() < deadline
   time.sleep(.01)
  return real(*a, **kw)
 rcdb.copy_record = copy
 trip = c.deliver('source', 'dest')
 assert trip['carried']+trip['held'] == 1, trip
 print(trip['carried'])
"""
    def child(index):
        return run_isolated(code, packages=("agent", "rexgraph", "rcdb"),
                            capture_output=True, text=True, timeout=30)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(child, range(3)))
    assert all(r.returncode == 0 for r in results), [(r.stdout, r.stderr) for r in results]
    assert sum(int(r.stdout.strip()) for r in results) == 1
    with closing(opened(kind, dst_path)) as dst:
        assert len(dst.history("r")) == 1 and dst.read_record("r").value == {"q": Fraction(1, 7)}


@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from agent.server import auth, audit
    from agent.server.app import app
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"local://{tmp_path}/default")
    auth.reset_auth_manager(); audit.reset_cache(); rcdb.reset_default_store()
    mgr = auth.get_auth_manager(); mgr.enable_auth()
    admin = mgr.bootstrap_admin()
    user = mgr.create_token("user", ["default"], role="user")
    with TestClient(app) as client, closing(rcdb.MemoryStore()) as src, closing(rcdb.MemoryStore()) as dst:
        src.put_record("r", None)
        from agent.courier import get_courier
        c = get_courier(); c.attach_store("source", src); c.attach_store("dest", dst)
        yield client, {"Authorization": f"Bearer {admin}"}, {"Authorization": f"Bearer {user}"}, c
    auth.reset_auth_manager(); audit.reset_cache(); rcdb.reset_default_store()


@pytest.mark.parametrize("endpoint", ["deliver", "broadcast"])
@pytest.mark.parametrize("selection", [{"limit": 0}, {"tags": []}, {"ids": []}])
def test_http_explicit_empty_and_zero_selection(api, endpoint, selection):
    client, admin, user, c = api
    c.carry = CarrySpec(tags=["absent"])
    body = {"source": "source", "dest": "dest", "dests": ["dest"], **selection}
    result = client.post(f"/api/v1/courier/{endpoint}", headers=admin, json=body)
    assert result.status_code == 200, result.text
    trip = result.json() if endpoint == "deliver" else result.json()["trips"][0]
    assert trip["considered"] == (0 if "limit" in selection else 1)
    assert client.post(f"/api/v1/courier/{endpoint}", headers=user, json=body).status_code == 403


@pytest.mark.parametrize("invalid", [{"limit": -1}, {"limit": True}, {"limit": "1"},
    {"limit": 1.0}, {"limit": 2**63}, {"tags": "r"}, {"ids": [1]}, {"source": 3}, {"dest": {}}])
def test_http_bad_typed_selection_is_a_safe_client_error(api, invalid):
    client, admin, _, c = api
    result = client.post("/api/v1/courier/deliver", headers=admin,
                         json={"source": "source", "dest": "dest", **invalid})
    assert result.status_code == 400, result.text
    assert c.status()["trips"] == 0


@pytest.mark.parametrize("invalid", ["dest", False, {"dest": True}, [1], [""]])
def test_http_bad_broadcast_destinations_are_a_client_error(api, invalid):
    client, admin, _, c = api
    assert client.post("/api/v1/courier/broadcast", headers=admin,
        json={"source": "source", "dests": invalid}).status_code == 400
    assert c.status()["trips"] == 0


def test_http_route_namespace_collision_and_literal_names(api):
    client, admin, _, c = api
    collision = client.post("/api/v1/courier/peers", headers=admin,
        json={"name": "dest", "url": "https://example.invalid"})
    assert collision.status_code == 400
    for body in ({"hive": 1}, {"hive": "x", "store": False}):
        assert client.post("/api/v1/courier/routes", headers=admin, json=body).status_code == 400
    c.attach_store(" spaced ", c.store_of("dest"))
    assert client.post("/api/v1/courier/deliver", headers=admin,
        json={"source": "source", "dest": " spaced ", "limit": 0}).status_code == 200


@pytest.mark.parametrize("limit", [True, "1", -1, 1.0])
def test_tools_do_not_coerce_invalid_limits(limit):
    from agent.courier import get_courier
    from agent.mcp_tools import _courier_deliver, _courier_survey
    with closing(rcdb.MemoryStore()) as src, closing(rcdb.MemoryStore()) as dst:
        c = get_courier(); c.attach_store("source", src); c.attach_store("dest", dst)
        with pytest.raises(ValueError): _courier_deliver("source", "dest", limit=limit)
        with pytest.raises(ValueError): _courier_survey("source", limit=limit)
        assert c.status()["trips"] == 0


@pytest.mark.parametrize("kind", KINDS)
def test_opposing_copies_release_source_snapshot_before_destination_writer(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path / "left")) as left, closing(opened(kind, tmp_path / "right")) as right:
        left.put_record("r", "left"); right.put_record("r", "right")
        workers = [courier(left, right), courier(right, left)]
        real_copy, barrier = rcdb.copy_record, Barrier(2)
        def race(*a, **kw):
            barrier.wait(timeout=15)
            return real_copy(*a, **kw)
        monkeypatch.setattr(rcdb, "copy_record", race)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda c: c.deliver("source", "dest"), workers))
        assert sum(r["carried"] for r in results) == 2
        assert left.read_record("r").value == "right" and right.read_record("r").value == "left"


@pytest.mark.parametrize("after_publication", [False, True])
def test_native_local_publication_failure_blocks_reuse_until_reopen(after_publication, tmp_path, monkeypatch):
    path = tmp_path / "dest"
    with closing(rcdb.MemoryStore()) as src, closing(rcdb.LocalStore(path)) as dst:
        src.put_record("r", None)
        def fail(frame):
            if after_publication: raise RuntimeError("failed state finalization")
            raise rcdb.PublicationUncertainError("uncertain journal outcome")
        if after_publication:
            monkeypatch.setattr(dst._state, "apply", fail)
        else:
            monkeypatch.setattr(dst._journal, "publish", fail)
        c = courier(src, dst)
        assert c.deliver("source", "dest")["uncertain"] == 1
        assert c.deliver("source", "dest")["uncertain"] == 1
        with pytest.raises(rcdb.PublicationUncertainError): dst.list()
    with closing(rcdb.LocalStore(path)) as reopened:
        assert len(reopened.history("r")) == int(after_publication)
        if after_publication:
            assert reopened.read_record("r").value is None


def test_loopback_native_courier_threads_preserve_workspace_telemetry(tmp_path):
    env = dict(os.environ, REXGRAPH_CONFIG_DIR=str(tmp_path / "config"),
               REXGRAPH_RCDB_URI=f"local://{tmp_path}/default",
               REXGRAPH_AUDIT_JOURNAL=str(tmp_path / "audit.jsonl"),
               REXGRAPH_ACTIVITY_JOURNAL=str(tmp_path / "activity.jsonl"))
    code = f"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier, Thread
import socket, time
import uvicorn
from agent import rcdb, activity
from agent.client import RexClient
from agent.courier import get_courier
from agent.hive_network import get_network
from agent.server.auth import get_auth_manager
from agent.server.app import app
manager = get_auth_manager(); manager.enable_auth()
token = manager.create_token('operator', ['default', 'alpha'], role='admin')
listener = socket.socket(); listener.bind(('127.0.0.1',0)); listener.listen(64)
server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
thread = Thread(target=lambda: server.run(sockets=[listener]), daemon=True); thread.start()
try:
 deadline = time.monotonic()+15
 while not server.started:
  assert thread.is_alive() and time.monotonic() < deadline
  time.sleep(.01)
 with closing(rcdb.MemoryStore()) as src, closing(rcdb.LocalStore({str(tmp_path / 'dest')!r})) as dst:
  src.put_record('r', None)
  c = get_courier(); c.attach_store('source', src); c.attach_store('dest', dst)
  real, barrier = rcdb.copy_record, Barrier(4)
  def race(*a, **kw):
   barrier.wait(15); return real(*a, **kw)
  rcdb.copy_record = race
  def deliver(index):
   client = RexClient('http://127.0.0.1:'+str(listener.getsockname()[1]), api_key=token, workspace='alpha')
   return client._post('/api/v1/courier/deliver', json={{'source':'source','dest':'dest'}})
  with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(deliver, range(4)))
  assert sum(t['carried'] for t in results) == 1, results
  assert sum(t['held'] for t in results) == 3, results
  assert len(dst.history('r')) == 1 and c.status()['trips'] == 4
  assert len(get_network('alpha')._net._msgs) == 4
  assert len(get_network('default')._net._msgs) == 0
finally:
 server.should_exit = True; thread.join(15); listener.close(); activity.get_log().close()
 assert not thread.is_alive()
"""
    result = run_isolated(code, packages=("agent", "rexgraph", "rcdb", "rcql", "system"),
                          env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
