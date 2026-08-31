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

    from agent import agent_complex
    auth.reset_auth_manager()
    audit.reset_cache()
    reset_default_store()
    agent_complex.reset_live()                       # keyed by workspace, and process-wide

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
    agent_complex.reset_live()


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


def test_a_write_records_who_made_it(two_tenants):
    """`/api/v1/db/put` stamps no identity of its own, so a `stored_by` here is the
    scoped store's or nothing. Every route that did not stamp its own wrote anonymously
    while the request in hand already knew the caller."""
    client, ah, bh, a_id, b_id = two_tenants
    r = client.post("/api/v1/db/put", headers=bh,
                    json={"id": "bobs", "text": "alpha beta gamma delta"})
    assert r.status_code == 200, r.text

    from agent.rcdb import default_store
    meta = default_store().get_record("bobs").meta
    assert meta["stored_by"] == "bob"
    assert meta["workspace"] == "beta", "the workspace stamp it already had is unchanged"


def test_the_trail_names_the_caller_rather_than_local(two_tenants):
    """The chain was written with user='local' for every caller, because the identity
    was resolved by the middleware and then dropped."""
    client, ah, bh, a_id, b_id = two_tenants
    client.post("/api/v1/db/put", headers=bh, json={"id": "bobs", "text": "alpha beta"})

    from agent.server import audit
    puts = [e for e in audit.read() if e["action"] == "db.put" and e["target"] == "bobs"]
    assert puts, "the write left no trail entry"
    assert puts[-1]["user"] == "bob"
    assert audit.verify()["valid"] is True, "the chain still verifies with the field set"


def test_two_callers_are_told_apart_in_the_trail(two_tenants):
    client, ah, bh, a_id, b_id = two_tenants
    client.post("/api/v1/db/put", headers=ah, json={"id": "mine", "text": "one two three"})
    client.post("/api/v1/db/put", headers=bh, json={"id": "bobs", "text": "four five six"})

    from agent.server import audit
    who = {e["target"]: e["user"] for e in audit.read() if e["action"] == "db.put"}
    assert who["mine"] == "admin" and who["bobs"] == "bob"


#### a workspace name becomes a directory name, so it is held to one rule

def test_dot_dot_is_not_a_workspace():
    """The handle store's rule allowed a dot, so "." and ".." were valid workspaces
    and resolved to the PARENT of the workspace root: a namespace two tenants share
    and neither asked for."""
    from agent.server.handles import valid_workspace
    assert valid_workspace("default") and valid_workspace("_probe")
    assert not valid_workspace("..") and not valid_workspace(".")
    assert not valid_workspace("a/b") and not valid_workspace("")
    assert not valid_workspace("x" * 65)


def test_the_handle_store_refuses_a_traversing_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    from agent.server import handles
    with pytest.raises(ValueError):
        handles.mint("..", b"hello", name="x.txt")
    ok = handles.mint("default", b"hello", name="x.txt")
    assert ok["workspace"] == "default"


def test_one_rule_not_two(monkeypatch):
    """Two regexes had drifted apart and the LOOSER one guarded the header path,
    so the strict rule was watching a door nobody used."""
    from agent.server import auth, handles
    mgr = auth.get_auth_manager()
    for bad in ("..", ".", "a/b"):
        with pytest.raises(ValueError):
            mgr.get_workspace(bad)
        assert not handles.valid_workspace(bad)


def test_a_traversing_workspace_header_is_refused(two_tenants):
    """`X-Workspace` arrives from a request and becomes a directory name downstream.
    It was taken raw; `get_workspace`, the only caller of the strict rule, is never
    on this path."""
    client, ah, bh, a_id, b_id = two_tenants
    r = client.get("/rex/v1/hello", headers={**ah, "X-Workspace": ".."})
    assert r.status_code == 400 and "invalid workspace" in r.json()["detail"]
    assert client.get("/rex/v1/hello", headers=ah).status_code == 200


#### filtering reads was half of it: a write onto another tenant's id is a deletion

def test_a_tenant_cannot_write_onto_another_tenants_id(two_tenants):
    """The demonstrated exploit. Bob owns an id; alice, a plain user of an unrelated
    workspace, writes to the same id; her write becomes the newest version stamped to
    her, and bob's own record then reads as absent TO BOB. Not a leak, a deletion."""
    client, ah, bh, a_id, b_id = two_tenants
    assert client.post("/api/v1/db/put", headers=bh,
                       json={"id": "bobrec", "text": "bob's own data"}).status_code == 200
    assert [r["id"] for r in client.get("/api/v1/db/list", headers=bh).json()["records"]] \
        .count("bobrec") == 1

    hijack = client.post("/api/v1/db/put", headers=ah,
                         json={"id": "bobrec", "text": "alice overwrites"})
    assert hijack.status_code == 403, f"the hijack succeeded: {hijack.status_code}"
    assert "another workspace" in hijack.json()["detail"]

    still = client.get("/api/v1/db/get/bobrec", headers=bh)
    assert still.status_code == 200, "bob lost his own record"


def test_a_tenant_may_still_version_its_own_record(two_tenants):
    """The check must not stop the ordinary case it sits in front of."""
    client, ah, bh, a_id, b_id = two_tenants
    for _ in range(2):
        r = client.post("/api/v1/db/put", headers=bh,
                        json={"id": "mine", "text": "one two three"})
        assert r.status_code == 200
    from agent.rcdb import default_store
    assert default_store().get_record("mine").version == 2


def test_the_refusal_is_recorded(two_tenants):
    """A refused write is a security event and belongs in the trail, named as refused
    rather than absent from it."""
    client, ah, bh, a_id, b_id = two_tenants
    client.post("/api/v1/db/put", headers=bh,
                json={"id": "bobrec2", "text": "bob writes his own data here"})
    client.post("/api/v1/db/put", headers=ah,
                json={"id": "bobrec2", "text": "alice tries to overwrite it"})
    from agent.server import audit
    refused = [e for e in audit.read()
               if e["target"] == "bobrec2" and e["outcome"] == "refused"]
    assert refused, "the refusal left no trail entry"
    assert refused[-1]["user"] == "admin"


def test_a_caller_supplied_workspace_does_not_beat_the_scope():
    """`setdefault` deferred to whatever the caller already set.

    Reading it as "a route that knows better keeps its value" was wrong: inside a request
    the value does not come from the route. /api/v1/db/record-work took the workspace
    from the request BODY and work_recorder stamped a literal "default", so either one
    landed the record in a tenant the caller merely named.
    """
    from agent.server import scope

    class _Inner:
        def __init__(self):
            self.seen = None

        def get_record(self, id, **kw):
            return None

        def put(self, id, rex, meta=None, tags=None, **kw):
            self.seen = meta
            return id

    inner = _Inner()
    store = scope.ScopedStore(inner, "beta", "bob")
    store.put("r1", object(), meta={"workspace": "alpha", "stored_by": "alice"})
    assert inner.seen["workspace"] == "beta", inner.seen
    assert inner.seen["stored_by"] == "bob", inner.seen


def test_record_work_cannot_name_another_tenant(two_tenants):
    """The body carries `workspace`; it must not decide who owns the record."""
    client, ah, bh, a_id, b_id = two_tenants
    r = client.post("/api/v1/db/record-work", headers=bh, json={
        "kind": "pipeline-run", "lineage_id": "lin-x",
        "labels": ["one", "two"], "workspace": "alpha",
    })
    assert r.status_code == 200, r.text
    r2 = client.post("/api/v1/db/record-work", headers=bh, json={
        "kind": "pipeline-run", "lineage_id": "lin-x",
        "labels": ["one", "two", "three"], "workspace": "alpha",
    })
    assert r2.status_code == 200, r2.text
    seen = [x.get("id") for x in client.get("/api/v1/db/records", headers=ah).json()
            .get("records", [])]
    assert "lin-x" not in seen, "bob named alpha in the body and landed a record there"


@pytest.mark.parametrize("bad", ["../..", "a/b", ".", "", "x" * 65, "../_probe"])
def test_a_workspace_name_cannot_escape_the_workspace_root(bad):
    """_ws_dir both joins the name onto a path and creates what it names, so an
    unvalidated form field escaped the root and mkdir made the directories on the way."""
    from agent.server.persistence import _ws_dir
    with pytest.raises(ValueError):
        _ws_dir(bad)


def test_a_usable_workspace_name_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    from agent.server import persistence
    assert persistence._ws_dir("default").is_dir()
    assert persistence._ws_dir("_probe").is_dir()
    assert tmp_path in persistence._ws_dir("default").parents, \
        "the base directory follows the environment rather than the import"


def test_effective_workspace_ignores_the_caller_inside_a_request(two_tenants):
    from agent.server import scope
    token = scope.set_workspace("beta")
    try:
        assert scope.effective_workspace("alpha") == "beta"
        assert scope.effective_workspace("") == "beta"
    finally:
        scope.reset_workspace(token)


def test_effective_workspace_defers_outside_a_request():
    """The CLI and the recorder have no bound workspace and keep their own value."""
    from agent.server import scope
    assert scope.effective_workspace("anything") == "anything"
    assert scope.effective_workspace("") == "default"


@pytest.mark.parametrize("method,path,body", [
    ("post", "/api/v1/model/chat-config", {"url": "http://attacker.example", "api_key": "k"}),
    ("post", "/api/v1/model/local/start", {"model_path": "/tmp/x.gguf"}),
    ("post", "/api/v1/model/local/stop", {}),
    ("post", "/api/v1/model/embedder/start", {"model_path": "/tmp/x.gguf"}),
    ("post", "/api/v1/model/embedder/stop", {}),
    ("post", "/api/v1/model/introspect/attention", {"prompt": "hi", "model_path": "/tmp/x"}),
    ("post", "/api/v1/hive/attach-live", {}),
    ("post", "/api/v1/hive/auto", {}),
    ("post", "/api/v1/hive/remove", {"name": "b"}),
    ("post", "/api/v1/hive/down", {}),
    ("post", "/api/v1/hive/profiles", {"name": "p"}),
    ("delete", "/api/v1/hive/profiles/p", None),
    ("post", "/api/v1/hive/profiles/p/apply", {}),
    ("post", "/api/v1/hive/profiles/active", {"id": "p"}),
    ("post", "/api/v1/ops/compute", {"threads": 1}),
    ("post", "/api/v1/ops/run", {"phase": "test"}),
    ("post", "/api/v1/agents/network/hives", {"name": "h"}),
])
def test_a_plain_user_cannot_move_the_shared_instance(two_tenants, method, path, body):
    """These start and stop subprocesses, spend disk and VRAM, rewrite the compute config
    or delete a profile, all of it process-wide. Repointing chat-config alone sends every
    other tenant's prompts to an endpoint the caller chose."""
    client, _ah, bh, _a, _b = two_tenants
    kw = {"json": body} if body is not None else {}
    r = getattr(client, method)(path, headers=bh, **kw)
    assert r.status_code == 403, f"{method.upper()} {path} was allowed"


@pytest.mark.parametrize("path", [
    "/api/v1/hive/status", "/api/v1/hive/profiles", "/api/v1/ops/phases",
    "/api/v1/ops/compute", "/api/v1/model/local/status",
])
def test_reading_what_the_instance_runs_stays_ordinary_use(two_tenants, path):
    client, _ah, bh, _a, _b = two_tenants
    assert client.get(path, headers=bh).status_code == 200, path


def test_a_document_id_cannot_escape_its_workspace(two_tenants, tmp_path):
    """Validating the workspace name settled which directory this is and said nothing
    about what gets joined onto it. RexBundle.save creates parents, so a doc_id of
    ../../<other>/documents/<name> planted a document in a workspace the caller cannot
    otherwise reach."""
    client, ah, bh, _a, _b = two_tenants
    r = client.post("/api/v1/corpus/add-text", headers=bh, json={
        "text": "alpha beta gamma. delta epsilon zeta.",
        "doc_id": "../../alpha/documents/planted",
    })
    assert r.status_code != 200, r.text
    from agent.server.persistence import _docs_dir
    assert not (_docs_dir("alpha") / "planted.rex").exists(), "bob planted into alpha"


def test_an_ordinary_document_id_still_works(two_tenants):
    client, _ah, bh, _a, _b = two_tenants
    r = client.post("/api/v1/corpus/add-text", headers=bh, json={
        "text": "alpha beta gamma. delta epsilon zeta.", "doc_id": "ordinary",
    })
    assert r.status_code == 200, r.text
    assert r.json()["doc_id"] == "ordinary"


@pytest.mark.parametrize("bad", ["..", "../..", "../../alpha/documents/x"])
def test_deleting_a_document_cannot_reach_the_workspace_root(two_tenants, bad):
    """The delete route rmtree'd whatever the id resolved to."""
    client, _ah, bh, _a, _b = two_tenants
    r = client.delete(f"/api/v1/admin/workspace/files/{bad}", headers=bh)
    assert r.status_code in (400, 404), r.text
    from agent.server.persistence import _docs_dir
    assert _docs_dir("beta").is_dir(), "the workspace root was removed"


def test_an_upload_does_not_stage_where_another_tenant_can_walk_it(two_tenants):
    """The staged file becomes the document's source, so it outlives the request. In the
    shared temp directory that meant one tenant's upload sat next to another's, and both
    /corpus/add and /ocr accept a DIRECTORY, so reading it needed no guessed filename."""
    import tempfile
    from pathlib import Path

    client, _ah, bh, _a, _b = two_tenants
    r = client.post("/api/v1/corpus/add", headers=bh,
                    files={"file": ("secret.txt", b"alpha beta gamma. delta epsilon.",
                                    "text/plain")})
    assert r.status_code == 200, r.text

    from agent.server.persistence import staging_dir
    staged = list(staging_dir("beta").iterdir())
    assert staged, "the upload was not staged under the workspace"

    # The enumeration walked the temp ROOT, so what matters is that nothing lands
    # directly in it. Under pytest the config dir itself sits below tmp_path, so an
    # ancestry test would measure the harness rather than the fix.
    shared = Path(tempfile.gettempdir()).resolve()
    for f in staged:
        assert f.resolve().parent != shared, f"{f} is loose in the shared temp directory"
        assert f.resolve().parent == staging_dir("beta").resolve()

    # In a real deployment this is what closes it: the staging directory lives under the
    # config directory, and path_within refuses that outright, so no caller can name it.
    from agent.server.handles import path_allowed
    assert not path_allowed(str(staging_dir("beta"))), \
        "a caller must not be able to name another workspace's staging directory"


def test_the_live_agentic_complex_is_per_workspace(two_tenants):
    """One process-wide complex meant any tenant could append to the structure every
    other tenant reads through the monitor, the router and the hive's own routing, so a
    forged message moved another workspace's alignment and load-bearing readings."""
    client, ah, bh, _a, _b = two_tenants
    r = client.post("/api/v1/agents/message", headers=bh,
                    json={"sender": "b1", "recipient": "b2", "text": "beta only"})
    assert r.status_code == 200, r.text
    assert r.json()["n_messages"] >= 1

    seen = client.post("/api/v1/agents/message", headers=ah,
                       json={"sender": "a1", "recipient": "a2", "text": "alpha only"})
    assert seen.json()["n_messages"] == 1, "alpha saw beta's messages"


def test_resetting_clears_only_the_callers_own_complex(two_tenants):
    client, ah, bh, _a, _b = two_tenants
    client.post("/api/v1/agents/message", headers=bh,
                json={"sender": "b1", "recipient": "b2", "text": "beta"})
    assert client.post("/api/v1/agents/reset", headers=ah).status_code == 200
    again = client.post("/api/v1/agents/message", headers=bh,
                        json={"sender": "b1", "recipient": "b2", "text": "beta again"})
    assert again.json()["n_messages"] == 2, "alpha's reset wiped beta's complex"


def test_a_named_hive_is_not_shared_between_workspaces(two_tenants):
    """hive(name) is get-or-create and /agents/command reaches it with scope
    'hive:<name>', so any tenant could bring a named hive into being and every tenant
    then shared that one object: the same worker bees, and the same coordination complex
    they each wrote through chat and read through monitor."""
    from agent import hive_network
    client, ah, bh, _a, _b = two_tenants

    r = client.post("/api/v1/agents/command", headers=bh,
                    json={"command": "status", "scope": "hive:victimhive"})
    assert r.status_code == 200, r.text

    # Asked before default's own copy is materialised, because hive() is get-or-create
    # and naming it here would be this test creating what it then complains about.
    seen = client.get("/api/v1/agents/network", headers=ah).json()
    assert "victimhive" not in str(seen), "a hive another tenant created is listed here"

    beta = hive_network.get_network("beta").hive("victimhive")
    default = hive_network.get_network("default").hive("victimhive")
    assert beta is not default, "two workspaces share one named hive"
    assert beta._complex is not default._complex, "and one coordination complex"
