"""A path, a store URI and a connection string are parameters, not permissions.

Several routes took one of those from the request body and acted on it directly: /ml/run
wrote weights to any path and read data from any path, /ml/ingest opened a caller-named
store outside the workspace-scoped view, and the schema router reflected a live database
without consulting the allow-list the other two database routers use.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    from agent.server import audit, auth
    auth.reset_auth_manager(); audit.reset_cache()
    from agent.server.app import app
    yield TestClient(app)
    auth.reset_auth_manager(); audit.reset_cache()


@pytest.mark.parametrize("field", ["save_to", "data"])
def test_ml_run_refuses_a_path_outside_the_allowed_directories(client, field):
    r = client.post("/api/v1/ml/run",
                    json={"archetype": "hgnn", field: "/etc/rexgraph-escape"})
    assert r.status_code == 403, r.text
    assert field in r.json()["detail"]


def test_ml_run_refuses_a_path_into_the_config_directory(client, tmp_path):
    """The config directory holds auth.json, connections.json and the audit journal, so
    it stays refused even when it falls inside an allowed root."""
    r = client.post("/api/v1/ml/run",
                    json={"archetype": "hgnn", "save_to": str(tmp_path / "auth.json")})
    assert r.status_code == 403, r.text


@pytest.mark.parametrize("body", [
    {"connection": "sqlite:////etc/passwd"},
    {"mongo_connection": "mongodb://127.0.0.1:27017"},
])
def test_schema_reflection_answers_to_the_db_allow_list(client, monkeypatch, body):
    monkeypatch.setenv("REXGRAPH_DB_SAFE", "1")
    r = client.post("/api/v1/schema/analyze", json=body)
    assert r.status_code in (400, 403), r.text
    assert "reflect" not in r.text.lower() or r.status_code != 200


def test_the_allowed_roots_are_built_in_one_place(monkeypatch):
    from agent.server.handles import allowed_roots, path_allowed
    monkeypatch.setenv("REXGRAPH_ALLOWED_DIRS", "/srv/data:/srv/more")
    roots = allowed_roots()
    assert "/srv/data" in roots and "/srv/more" in roots
    assert "/tmp" in roots
    assert path_allowed("/srv/data/x.csv")
    assert not path_allowed("/etc/passwd")
    assert not path_allowed("/srv/datafoo/x.csv"), "prefix of the text is not containment"


def test_remote_code_is_off_for_anything_a_caller_named(monkeypatch):
    """trust_remote_code=True downloads and RUNS python from the model repository."""
    from agent.hfguard import remote_code_allowed
    monkeypatch.delenv("REXGRAPH_TRUST_REMOTE_CODE", raising=False)
    assert remote_code_allowed(caller_named=True) is False
    assert remote_code_allowed() is True, "operator configuration keeps its behavior"
    monkeypatch.setenv("REXGRAPH_TRUST_REMOTE_CODE", "1")
    assert remote_code_allowed(caller_named=True) is True


@pytest.fixture
def tenants(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    from agent.server import audit, auth
    auth.reset_auth_manager(); audit.reset_cache()
    from agent.server.app import app
    mgr = auth.get_auth_manager(); mgr.enable_auth()
    admin = mgr.bootstrap_admin()
    bob = mgr.create_token("bob", ["beta"], role="user")
    yield (TestClient(app), {"Authorization": f"Bearer {admin}"},
           {"Authorization": f"Bearer {bob}", "X-Workspace": "beta"})
    auth.reset_auth_manager(); audit.reset_cache()


def test_naming_a_model_to_analyze_is_an_instance_operation(tenants):
    client, _ah, bh = tenants
    r = client.post("/api/v1/huggingface/analyze",
                    headers=bh, json={"text": "a b c", "model": "some/repo"})
    assert r.status_code == 403, r.text


def test_text_level_analysis_stays_ordinary_use(tenants):
    client, _ah, bh = tenants
    r = client.post("/api/v1/huggingface/analyze",
                    headers=bh, json={"text": "alpha beta gamma alpha beta"})
    assert r.status_code == 200, r.text
    assert r.json().get("mode") == "text_cooccurrence"


@pytest.mark.parametrize("path", [
    "/api/v1/courier/status",
    "/api/v1/courier/survey?hive=h",
    "/api/v1/ops/runs",
    "/api/v1/ops/runs/r1",
])
def test_reading_instance_operations_matches_performing_them(tenants, path):
    """The courier and the lifecycle run store are process-wide. Binding a store and
    starting a run are admin operations, so reading what is bound and what was run are
    too; a survey otherwise lists records through a store view bound by someone else."""
    client, _ah, bh = tenants
    assert client.get(path, headers=bh).status_code == 403, path


@pytest.mark.parametrize("url,why", [
    ("file:///etc/passwd", "a non-fetchable scheme is a local file read"),
    ("gopher://example.com/", "a non-fetchable scheme"),
])
def test_an_outbound_fetch_refuses_a_scheme_that_is_not_http(client, url, why):
    """Unconditional, because it is not policy: file:// handed to an HTTP client reads
    a local file, and no deployment wants that from a request body."""
    r = client.post("/api/v1/trustgraph/health", json={"url": url, "flow": "f"})
    assert r.status_code == 400, why


def test_an_outbound_fetch_refuses_a_private_host_when_the_policy_is_on(
        client, monkeypatch):
    monkeypatch.setenv("REXGRAPH_DB_SAFE", "1")
    r = client.post("/api/v1/trustgraph/health",
                    json={"url": "http://169.254.169.254/latest/meta-data/", "flow": "f"})
    assert r.status_code == 400, r.text
    assert "SSRF" in r.text or "loopback" in r.text


def test_a_builder_step_cannot_name_an_output_outside_the_allowed_directories(client):
    """Only the step TYPE was validated; params.output was written to directly."""
    import json as _json
    cfg = _json.dumps({"steps": [
        {"type": "export", "params": {"format": "json", "output": "/etc/rexgraph-escape"}},
    ]})
    r = client.post("/api/v1/builder/run", data={"config": cfg})
    assert r.status_code == 403, r.text


def test_an_ordinary_builder_output_still_runs(client, tmp_path_factory, monkeypatch):
    """Deliberately NOT under the config directory: `client` points REXGRAPH_CONFIG_DIR
    at its own tmp_path, and path_within refuses that outright, so naming it would be
    refused for the right reason and prove nothing about the ordinary case."""
    import json as _json
    outdir = tmp_path_factory.mktemp("builder_out")
    monkeypatch.setenv("REXGRAPH_ALLOWED_DIRS", str(outdir))
    cfg = _json.dumps({"steps": [
        {"type": "export", "params": {"format": "json", "output": str(outdir / "out")}},
    ]})
    r = client.post("/api/v1/builder/run", data={"config": cfg})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("verb,cmd", [
    ("set", "set w1 alpha beta"),
    ("require", "require summarizer"),
    ("forge", "forge m1 hgnn"),
    ("kill", "kill w1"),
])
def test_a_governed_verb_does_nothing_without_confirmation(tenants, verb, cmd):
    """The route's admin check only fires when confirm is true, so a handler that
    ignored confirm had no gate at all. Only kill honoured it; set, require and forge
    executed immediately for anyone holding a token."""
    client, _ah, bh = tenants
    r = client.post("/api/v1/agents/command", headers=bh, json={"command": cmd})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("governed") is True, body
    assert body.get("ok") is False, body


@pytest.mark.parametrize("cmd", ["set w1 alpha", "require summarizer", "forge m1 hgnn",
                                 "kill w1"])
def test_confirming_a_governed_verb_needs_admin(tenants, cmd):
    client, _ah, bh = tenants
    r = client.post("/api/v1/agents/command", headers=bh,
                    json={"command": cmd, "confirm": True})
    assert r.status_code == 403, r.text


def test_help_names_the_verbs_that_are_actually_governed(tenants):
    client, _ah, bh = tenants
    r = client.post("/api/v1/agents/command", headers=bh, json={"command": "help"})
    assert r.status_code == 200, r.text
    assert set(r.json()["governed"]) == {"require", "forge", "set", "kill"}


@pytest.mark.parametrize("field", ["save_to", "data"])
def test_a_tilde_path_is_expanded_before_it_is_checked(client, field):
    """Path.resolve() leaves "~" alone but every sink calls expanduser, so "~/x" resolved
    to "<cwd>/~/x", passed as inside the allow-list, and then wrote to the real home
    directory. Fired: a save_to of "~/x" put 3.6 MB of weights in $HOME."""
    import uuid
    from pathlib import Path
    name = f"REXGRAPH_TILDE_{uuid.uuid4().hex[:8]}"
    r = client.post("/api/v1/ml/run",
                    json={"archetype": "hgnn", field: f"~/{name}"})
    assert r.status_code == 403, r.text
    assert not (Path.home() / name).exists(), "the guard passed and the sink wrote"


@pytest.mark.parametrize("field", ["save_to", "data"])
def test_a_path_that_is_not_a_string_is_a_bad_request(client, field):
    r = client.post("/api/v1/ml/run", json={"archetype": "hgnn", field: 12345})
    assert r.status_code == 400, r.text


def test_a_builder_output_in_defaults_is_checked_too(client):
    """AgentBuilder merges dict(self.defaults) with each step's own params, so naming the
    output in `defaults` rather than in `params` skipped the check entirely. Fired."""
    import json as _json
    cfg = _json.dumps({
        "defaults": {"output": "/etc/rexgraph-escape-via-defaults"},
        "steps": [{"type": "export", "params": {"format": "json"}}],
    })
    r = client.post("/api/v1/builder/run", data={"config": cfg})
    assert r.status_code == 403, r.text


def test_a_builder_step_with_non_dict_params_is_a_bad_request(client):
    import json as _json
    cfg = _json.dumps({"steps": [{"type": "export", "params": ["not", "a", "dict"]}]})
    r = client.post("/api/v1/builder/run", data={"config": cfg})
    assert r.status_code == 400, r.text


@pytest.mark.parametrize("url", ["//169.254.169.254/latest/", "not-a-url", "/etc/passwd"])
def test_a_value_with_no_scheme_is_not_a_fetchable_url(client, url):
    """check_db_uri returns early on a value with no '://' because a bare name like
    'edgelist' is a valid in-memory scheme. Nothing of the sort is fetchable, so the same
    early return let a protocol-relative url past without any check."""
    r = client.post("/api/v1/trustgraph/health", json={"url": url, "flow": "f"})
    assert r.status_code == 400, r.text
