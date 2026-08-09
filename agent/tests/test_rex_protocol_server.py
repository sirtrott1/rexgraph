"""The native surface: binary bodies, and the boundaries that hold on them.

Two modes to keep apart. Locally auth is off, the caller is the operator on their own
machine, and naming a file by path is the point. Over a network auth is on, and the
things that were conveniences become the attack surface: a path is a read of the
server's disk, a record id is a read of someone else's store, and a tool that costs
more than it took to ask for is a way to occupy the machine.

So the tests are mostly about what does NOT happen when auth is on, because that is
the half nobody notices is broken until it is used.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rexgraph import protocol
from rexgraph.graph import RexGraph


@pytest.fixture
def rex():
    """Two triangles sharing a vertex, so there is homology to compare."""
    src = np.array([0, 1, 2, 2, 3, 4], dtype=np.int32)
    dst = np.array([1, 2, 0, 3, 4, 2], dtype=np.int32)
    return RexGraph(sources=src, targets=dst)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A config dir of this test's own, so nothing touches the real one."""
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


#### the frame itself


def test_a_complex_survives_the_wire_unchanged(rex):
    frame = protocol.decode(protocol.encode(rex))
    back = protocol.to_complex(frame)
    assert back.nV == rex.nV and back.nE == rex.nE
    assert tuple(back.betti) == tuple(rex.betti)


def test_a_face_that_does_not_close_is_refused(rex):
    """What the chain condition actually catches: a payload that arrived intact and
    still is not a complex. The Hodge filter would drop this face silently, so the
    check runs against the raw boundary as sent."""
    rex.add_faces([[0, 1, 2]], [[1.0, 1.0, 1.0]])
    frame = protocol.decode(protocol.encode(rex))
    tampered = dict(frame.tensors)
    key = next(k for k in tampered if "B2" in k and "val" in k.lower())
    vals = tampered[key].copy()
    vals[0] = vals[0] + 3.0                     # the face no longer bounds
    tampered[key] = vals
    frame.header.pop("digest", None)            # isolate the structural check
    bad = protocol.Frame(header=frame.header, tensors=tampered)
    with pytest.raises(protocol.ProtocolError):
        protocol.to_complex(bad, verify=True)


def test_the_chain_condition_alone_does_not_cover_a_face_free_complex(rex):
    """Recorded because it is the tempting wrong belief. `B_d B_{d+1} = 0` relates two
    grades; with only B1 there is no product, so it holds for any B1 at all and cannot
    be what protects a graph in transit. The digest is."""
    assert int(getattr(rex, "_nF", 0) or 0) == 0
    ok, residual = protocol.verify_complex(rex)
    assert ok and residual == 0.0

    # rewire an edge to a different vertex: a real change to the boundary data that
    # the chain condition has no way to object to
    src = np.array([0, 1, 2, 2, 3, 4], dtype=np.int32)
    dst = np.array([1, 2, 0, 3, 4, 3], dtype=np.int32)   # last edge moved
    altered = RexGraph(sources=src, targets=dst)
    assert protocol.verify_complex(altered) == (True, 0.0), \
        "this test asserts the LIMIT; if it fails, the limit moved"


def test_a_flipped_byte_is_caught_by_the_digest(rex):
    """The check that covers the common case: a graph, no faces, one bit wrong."""
    frame = protocol.decode(protocol.encode(rex))
    assert frame.header.get("digest"), "frames are not carrying a digest"
    payload = bytearray(protocol.encode(rex, compress=False))
    head_len = int.from_bytes(payload[8:12], "little")
    payload[16 + head_len] ^= 0x01
    with pytest.raises(protocol.ProtocolError, match="digest"):
        protocol.decode(bytes(payload))


def test_a_truncated_payload_is_refused(rex):
    payload = protocol.encode(rex, compress=False)
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(payload[:-8])


#### signatures: the part a digest cannot do


def test_a_signature_verifies_against_its_key(rex):
    payload = protocol.encode(rex)
    key = b"a shared deployment key"
    assert protocol.verify_signature(payload, protocol.sign(payload, key), key)


def test_a_signature_does_not_verify_under_another_key(rex):
    payload = protocol.encode(rex)
    sig = protocol.sign(payload, b"one key")
    assert not protocol.verify_signature(payload, sig, b"another key")


def test_a_rewritten_frame_fails_its_signature(rex):
    """The case a digest cannot catch: an attacker who rewrites the payload
    recomputes the digest over what they substituted, but cannot produce the HMAC."""
    key = b"a shared deployment key"
    payload = bytearray(protocol.encode(rex, compress=False))
    sig = protocol.sign(bytes(payload), key)
    head_len = int.from_bytes(payload[8:12], "little")
    payload[16 + head_len] ^= 0x01
    assert not protocol.verify_signature(bytes(payload), sig, key)


def test_an_empty_signature_never_verifies(rex):
    payload = protocol.encode(rex)
    assert not protocol.verify_signature(payload, "", b"key")


def test_a_frame_that_lies_about_its_length_is_refused(rex):
    payload = bytearray(protocol.encode(rex))
    payload[8:12] = (99999).to_bytes(4, "little")
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(bytes(payload))


def test_an_oversized_frame_is_refused_before_it_is_built(rex):
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(protocol.encode(rex), max_cells=2)


def test_a_frame_from_a_future_version_is_refused(rex):
    payload = bytearray(protocol.encode(rex))
    payload[4:6] = (protocol.WIRE_VERSION + 1).to_bytes(2, "little")
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(bytes(payload))


def test_faces_survive_the_wire(rex):
    """`to_state` reads the boundary arrays directly, and faces added since the last
    read are pending until they are flushed. A face lost here is lost silently."""
    rex.add_faces([[0, 1, 2]], [[1.0, 1.0, 1.0]])
    back = protocol.to_complex(protocol.decode(protocol.encode(rex)))
    assert back.nF == rex.nF


#### local: the operator on their own machine


def test_local_reports_auth_off_and_paths_allowed(client):
    body = client.get("/rex/v1/hello").json()
    assert body["auth_enabled"] is False
    assert body["paths_allowed"] is True
    assert body["wire_version"] == protocol.WIRE_VERSION


def test_local_can_verify_a_frame_it_posts(client, rex):
    r = client.post("/rex/v1/verify", content=protocol.encode(rex),
                    headers={"Content-Type": protocol.CONTENT_TYPE})
    assert r.status_code == 200, r.text
    fp = r.json()["fingerprint"]
    assert fp["nE"] == rex.nE and fp["chain_valid"] is True


def test_a_tampered_body_is_refused_over_http(client, rex):
    payload = bytearray(protocol.encode(rex, compress=False))
    # corrupt inside the tensor payload, past the header
    head_len = int.from_bytes(payload[8:12], "little")
    payload[16 + head_len + 8] ^= 0xFF
    r = client.post("/rex/v1/verify", content=bytes(payload),
                    headers={"Content-Type": protocol.CONTENT_TYPE})
    assert r.status_code in (400, 422), \
        f"a corrupted body was accepted: {r.status_code}"


def test_a_stored_complex_comes_back_as_the_same_complex(client, rex):
    stored = client.post("/rex/v1/store", content=protocol.encode(rex),
                         headers={"Content-Type": protocol.CONTENT_TYPE})
    assert stored.status_code == 200, stored.text
    rid = stored.json()["record_id"]

    got = client.get(f"/rex/v1/fetch/{rid}")
    assert got.status_code == 200, got.text
    assert got.headers["content-type"].startswith(protocol.CONTENT_TYPE)
    back = protocol.to_complex(protocol.decode(got.content))
    assert back.nE == rex.nE
    assert tuple(back.betti) == tuple(rex.betti)


#### signed deployments


@pytest.fixture
def signed(isolated, monkeypatch):
    """A deployment that signs frames in both directions."""
    monkeypatch.setenv("REXGRAPH_FRAME_KEY", "a shared deployment key")
    from agent.server.app import app
    return TestClient(app), b"a shared deployment key"


def test_a_signed_deployment_says_so(signed):
    client, _ = signed
    assert client.get("/rex/v1/hello").json()["signed_frames"] is True


def test_an_unsigned_frame_is_refused_where_signing_is_required(signed, rex):
    client, _ = signed
    r = client.post("/rex/v1/verify", content=protocol.encode(rex),
                    headers={"Content-Type": protocol.CONTENT_TYPE})
    assert r.status_code == 401, "an unsigned frame was accepted"


def test_a_correctly_signed_frame_is_accepted(signed, rex):
    client, key = signed
    payload = protocol.encode(rex)
    r = client.post("/rex/v1/verify", content=payload,
                    headers={"Content-Type": protocol.CONTENT_TYPE,
                             "X-Rex-Signature": protocol.sign(payload, key)})
    assert r.status_code == 200, r.text


def test_a_frame_rewritten_after_signing_is_refused(signed, rex):
    """The MITM case: the attacker recomputes the header digest over what they
    substituted, and still cannot produce the HMAC."""
    client, key = signed
    payload = bytearray(protocol.encode(rex, compress=False))
    sig = protocol.sign(bytes(payload), key)
    head_len = int.from_bytes(payload[8:12], "little")
    payload[16 + head_len] ^= 0x01
    r = client.post("/rex/v1/verify", content=bytes(payload),
                    headers={"Content-Type": protocol.CONTENT_TYPE,
                             "X-Rex-Signature": sig})
    assert r.status_code == 401


def test_a_signature_from_the_wrong_key_is_refused(signed, rex):
    client, _ = signed
    payload = protocol.encode(rex)
    r = client.post("/rex/v1/verify", content=payload,
                    headers={"Content-Type": protocol.CONTENT_TYPE,
                             "X-Rex-Signature": protocol.sign(payload, b"wrong")})
    assert r.status_code == 401


def test_what_the_server_sends_back_is_signed_too(signed, rex):
    client, key = signed
    payload = protocol.encode(rex)
    sig = {"Content-Type": protocol.CONTENT_TYPE,
           "X-Rex-Signature": protocol.sign(payload, key)}
    rid = client.post("/rex/v1/store", content=payload,
                      headers=sig).json()["record_id"]
    got = client.get(f"/rex/v1/fetch/{rid}")
    assert got.status_code == 200, got.text
    assert protocol.verify_signature(
        got.content, got.headers.get("X-Rex-Signature", ""), key), \
        "the server did not sign what it sent back"


#### handles


def test_an_upload_returns_a_handle_that_resolves(client):
    r = client.post("/rex/v1/upload", content=b"id\tname\na\tAlpha\n",
                    headers={"X-Filename": "terms.tsv"})
    assert r.status_code == 200, r.text
    handle = r.json()["handle"]

    from agent.server.handles import resolve
    assert resolve("default", handle).read_bytes().startswith(b"id\tname")


def test_the_same_bytes_get_the_same_handle(client):
    a = client.post("/rex/v1/upload", content=b"same").json()["handle"]
    b = client.post("/rex/v1/upload", content=b"same").json()["handle"]
    assert a == b, "content-addressed storage kept two copies"


def test_a_handle_does_not_resolve_in_another_workspace(client):
    from agent.server.handles import HandleError, mint, resolve
    handle = mint("alpha", b"alpha's data")["handle"]
    with pytest.raises(HandleError):
        resolve("beta", handle)


@pytest.mark.parametrize("hostile", [
    "../../../../etc/passwd",
    "/etc/passwd",
    "..%2f..%2fetc%2fpasswd",
    "a" * 64,                       # right length, not a real digest
])
def test_a_handle_cannot_be_made_to_name_something_else(isolated, hostile):
    from agent.server.handles import HandleError, resolve
    with pytest.raises(HandleError):
        resolve("default", hostile)


def test_a_workspace_name_cannot_escape_its_directory(isolated):
    from agent.server.handles import HandleError, mint
    with pytest.raises(HandleError):
        mint("../../../etc", b"x")


#### network: auth on


@pytest.fixture
def network(isolated):
    """Auth on, with an admin and a plain user in separate workspaces."""
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    mgr = get_auth_manager()
    mgr.enable_auth()                    # bootstrap only mints once auth is on
    admin_token = mgr.bootstrap_admin()
    user_token = mgr.create_token("bob", ["beta"], role="user")
    assert admin_token and user_token
    return TestClient(app), admin_token, user_token


def test_the_network_surface_refuses_an_anonymous_caller(network):
    client, _, _ = network
    assert client.get("/rex/v1/hello").status_code == 401


def test_the_network_surface_refuses_a_made_up_token(network):
    client, _, _ = network
    r = client.get("/rex/v1/hello",
                   headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_a_valid_token_is_admitted(network):
    client, admin, _ = network
    r = client.get("/rex/v1/hello", headers={"Authorization": f"Bearer {admin}"})
    assert r.status_code == 200, r.text
    assert r.json()["auth_enabled"] is True


def test_paths_are_not_allowed_once_auth_is_on(network):
    client, admin, _ = network
    body = client.get("/rex/v1/hello",
                      headers={"Authorization": f"Bearer {admin}"}).json()
    assert body["paths_allowed"] is False


def test_a_file_path_is_refused_where_a_handle_is_wanted(network, tmp_path):
    """The hole this closes: `files: ["/etc/passwd"]` used to be a valid request."""
    client, admin, _ = network
    secret = tmp_path / "secret.obo"
    secret.write_text("[Term]\nid: GO:0000001\nname: leaked\n")
    r = client.post("/rex/v1/call",
                    headers={"Authorization": f"Bearer {admin}"},
                    json={"name": "rexgraph_join_sources",
                          "arguments": {"files": [str(secret)]}})
    assert r.status_code == 400, f"a raw path was accepted: {r.text[:300]}"
    assert "handle" in r.text.lower()


def test_a_handle_from_another_workspace_is_refused(network):
    client, _, user = network
    from agent.server.handles import mint
    other = mint("alpha", b"[Term]\nid: GO:0000001\nname: private\n")["handle"]
    r = client.post("/rex/v1/call",
                    headers={"Authorization": f"Bearer {user}",
                             "X-Workspace": "beta"},
                    json={"name": "rexgraph_join_sources",
                          "arguments": {"files": [other]}})
    assert r.status_code == 400, f"a cross-workspace handle resolved: {r.text[:300]}"


def test_a_user_cannot_reach_a_workspace_they_are_not_in(network):
    client, _, user = network
    r = client.get("/rex/v1/hello",
                   headers={"Authorization": f"Bearer {user}",
                            "X-Workspace": "alpha"})
    assert r.status_code == 403


def test_a_record_stored_by_one_workspace_is_absent_from_another(network, rex):
    client, admin, user = network
    stored = client.post("/rex/v1/store", content=protocol.encode(rex),
                         headers={"Authorization": f"Bearer {admin}",
                                  "X-Workspace": "default",
                                  "Content-Type": protocol.CONTENT_TYPE})
    assert stored.status_code == 200, stored.text
    rid = stored.json()["record_id"]

    r = client.get(f"/rex/v1/fetch/{rid}",
                   headers={"Authorization": f"Bearer {user}",
                            "X-Workspace": "beta"})
    assert r.status_code == 404, "another workspace's record was served"


def test_an_unknown_tool_is_a_404_not_a_traceback(network):
    client, admin, _ = network
    r = client.post("/rex/v1/call", headers={"Authorization": f"Bearer {admin}"},
                    json={"name": "rexgraph_delete_everything", "arguments": {}})
    assert r.status_code == 404


def test_an_argument_the_tool_does_not_take_is_refused(network):
    client, admin, _ = network
    r = client.post("/rex/v1/call", headers={"Authorization": f"Bearer {admin}"},
                    json={"name": "rexgraph_homology",
                          "arguments": {"files": [], "eval": "__import__('os')"}})
    assert r.status_code == 400


#### the trail


def test_every_call_lands_in_the_trail(network, rex):
    client, admin, _ = network
    client.post("/rex/v1/verify", content=protocol.encode(rex),
                headers={"Authorization": f"Bearer {admin}",
                         "Content-Type": protocol.CONTENT_TYPE})
    from agent.server import audit
    actions = [e["action"] for e in audit.read()]
    assert "rex.verify" in actions


def test_a_refused_call_lands_in_the_trail_too(network, tmp_path):
    """A trail that records only what succeeded cannot show an attempt."""
    client, admin, _ = network
    client.post("/rex/v1/call", headers={"Authorization": f"Bearer {admin}"},
                json={"name": "rexgraph_join_sources",
                      "arguments": {"files": ["/etc/passwd"]}})
    from agent.server import audit
    entries = [e for e in audit.read() if e["action"] == "rex.call"]
    assert entries, "a refused call left no trace"
    assert entries[-1]["outcome"] != "ok"


def test_the_trail_verifies_as_written(isolated):
    from agent.server import audit
    for i in range(5):
        audit.record("test.action", user="u", target=f"t{i}")
    assert audit.verify()["valid"] is True


def test_an_edited_entry_stops_verifying(isolated):
    """The property the whole design is for: changing a record after the fact is
    visible, and the trail names which entry."""
    import json

    from agent.server import audit
    for i in range(5):
        audit.record("test.action", user="u", target=f"t{i}")

    p = audit.journal_path()
    lines = p.read_text().splitlines()
    entry = json.loads(lines[2])
    entry["target"] = "something else"
    lines[2] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")

    result = audit.verify()
    assert result["valid"] is False
    assert result["broken_at"] == 2


def test_a_removed_entry_stops_verifying(isolated):
    from agent.server import audit
    for i in range(5):
        audit.record("test.action", user="u", target=f"t{i}")
    p = audit.journal_path()
    lines = p.read_text().splitlines()
    del lines[2]
    p.write_text("\n".join(lines) + "\n")
    assert audit.verify()["valid"] is False


def test_the_trail_records_no_payload(isolated, rex):
    """A trail holding content is a second copy of the data with different access
    rules."""
    from agent.server import audit
    audit.record("rex.store", user="u", target="rx_1", detail={"nE": rex.nE})
    blob = audit.journal_path().read_text()
    assert "B1_vals" not in blob and "tensors" not in blob


#### ceilings


def test_an_oversized_request_is_refused_by_size(isolated):
    from agent.server.budget import BudgetExceeded, check_size
    with pytest.raises(BudgetExceeded) as e:
        check_size({"nV": 10_000_000}, limit=1000)
    assert e.value.axis == "size"


def test_one_identity_cannot_take_every_slot(isolated):
    from agent.server.budget import BudgetExceeded, guard
    with guard("bob", inflight_limit=1):
        with pytest.raises(BudgetExceeded) as e, guard("bob", inflight_limit=1):
            pass
        assert e.value.axis == "concurrency"


def test_a_slot_is_returned_even_when_the_work_raises(isolated):
    from agent.server.budget import guard
    with pytest.raises(ValueError), guard("bob", inflight_limit=1):
        raise ValueError("boom")
    with guard("bob", inflight_limit=1):
        pass                                    # would raise if the slot leaked


def test_another_identity_is_unaffected(isolated):
    from agent.server.budget import guard
    with guard("bob", inflight_limit=1), guard("alice", inflight_limit=1):
        pass


def test_a_deadline_expires(isolated):
    from agent.server.budget import BudgetExceeded, Deadline
    d = Deadline(seconds=0.0)
    with pytest.raises(BudgetExceeded) as e:
        d.check("work")
    assert e.value.axis == "time"


#### the registry gate


def test_the_tool_list_is_narrowed_to_what_the_caller_may_run(isolated):
    from agent.mcp_tools import TOOLS, Context, definitions
    plain_user = Context(is_admin=False, auth_enabled=True)
    names = {d["name"] for d in definitions(plain_user)}
    admin_only = {n for n, t in TOOLS.items() if t.requires == "admin"}
    assert not (names & admin_only)


def test_a_direct_python_caller_is_not_restricted(isolated, tmp_path):
    """Local use is the operator on their own machine and stays a path away."""
    from agent.mcp_tools import call
    p = tmp_path / "t.obo"
    p.write_text("[Term]\nid: GO:0000001\nname: alpha\n")
    out = call("rexgraph_join_sources", files=[str(p)])
    assert out["n_entities"] >= 0
