"""The three routes wired into the UI this pass answer, and answer in the shape the
screen reads: models/status splits loaded from available, db/compare returns the
label sets, db/get returns the record itself."""
from __future__ import annotations
import numpy as np, pytest
from fastapi.testclient import TestClient
from rexgraph.graph import RexGraph

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "rcdb.sqlite"))
    import agent.server.routes.rcdb as rr
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app
    c = TestClient(app)
    store = rr._store()
    for i in range(2):
        store.put(f"r{i}", RexGraph(sources=np.arange(3 + i, dtype=np.int32),
                                    targets=np.arange(1, 4 + i, dtype=np.int32)))
    return c

def test_models_status_shape(client):
    r = client.get("/api/v1/models/status")
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("loaded", "available", "n_loaded", "n_available", "vram_total_mb"):
        assert k in d, f"missing {k}: {d}"
    assert isinstance(d["loaded"], list) and isinstance(d["available"], list)

def test_remove_model_path_404s_when_absent(client):
    assert client.delete("/api/v1/models/path/not-registered").status_code == 404

def test_db_get_returns_the_record(client):
    r = client.get("/api/v1/db/get/r0")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["id"] == "r0" and "signature" in d and "version" in d
    assert client.get("/api/v1/db/get/missing").status_code == 404

def test_db_compare_returns_label_sets(client):
    r = client.post("/api/v1/db/compare", json={"a": "r0", "b": "r1"})
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("a", "b", "match", "shared", "only_in_a", "only_in_b"):
        assert k in d, f"missing {k}: {d}"
    assert 0.0 <= d["match"] <= 1.0
    assert client.post("/api/v1/db/compare", json={"a": "r0"}).status_code == 400
    assert client.post("/api/v1/db/compare", json={"a": "r0", "b": "nope"}).status_code == 404


def test_builder_steps_come_from_the_registry(client):
    """The step list is the registry's, so a step added through `register_step`
    reaches the UI without a frontend edit."""
    r = client.get("/api/v1/builder/steps")
    assert r.status_code == 200, r.text
    types = [s["type"] for s in r.json()["steps"]]
    from agent.builder import AgentBuilder
    assert types == sorted(AgentBuilder.available_steps())
    assert types, "no steps registered"


def test_builder_templates_are_runnable_configs(client):
    r = client.get("/api/v1/builder/templates")
    assert r.status_code == 200, r.text
    tpls = r.json()["templates"]
    assert tpls, "no templates"
    for name, cfg in tpls.items():
        assert cfg.get("steps"), f"{name} has no steps"


def test_builder_run_rejects_a_bad_config(client):
    assert client.post("/api/v1/builder/run", data={"config": "not json"}).status_code == 400
    assert client.post("/api/v1/builder/run", data={"config": "{}"}).status_code == 400
    r = client.post("/api/v1/builder/run",
                    data={"config": '{"steps":[{"type":"nonesuch"}]}'})
    assert r.status_code == 400
    assert "nonesuch" in r.text


def test_builder_run_reports_a_failing_step_instead_of_500ing(client):
    """A step that cannot do its work is part of the result, not a server error.
    `required` defaults to True, so the run stops there and says which step and why."""
    cfg = '{"name":"t","steps":[{"type":"corpus"},{"type":"chunk"}]}'
    r = client.post("/api/v1/builder/run", data={"config": cfg, "query": "hello"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["name"] == "t"
    assert d["steps"], "no step was reported"
    assert d["steps"][0]["step_type"] == "corpus"
    assert all("status" in s for s in d["steps"])
    first = d["steps"][0]
    if first["status"] == "error":
        assert first["error"], "an errored step has to say why"
        assert len(d["steps"]) == 1, "a required step that failed stops the run"
    assert "n_documents" in d and "n_chunks" in d
    assert "documents" not in d, "raw working state should not ship in the response"


def test_builder_run_completes_a_config_that_needs_no_documents(client):
    """langgraph_init builds its state from the config alone, so it is the case that
    proves the executor runs rather than only reporting failures."""
    cfg = '{"name":"t","steps":[{"type":"langgraph_init"}]}'
    r = client.post("/api/v1/builder/run", data={"config": cfg})
    assert r.status_code == 200, r.text
    d = r.json()
    assert [s["step_type"] for s in d["steps"]] == ["langgraph_init"]
    assert d["elapsed"] >= 0


def test_builder_run_takes_files_the_way_the_screen_sends_them(client):
    """The Run button posts multipart: the config, an optional query, and the files.
    A pipeline that reads documents gets them and reports what it built."""
    import io
    r = client.post(
        "/api/v1/builder/run",
        data={"config": '{"name":"t","steps":[{"type":"ocr"},{"type":"corpus"},{"type":"chunk"}]}',
              "query": "what does this say?"},
        files=[("files", ("note.txt", io.BytesIO(b"alpha beta gamma. alpha delta."), "text/plain"))])
    assert r.status_code == 200, r.text
    d = r.json()
    assert [s["step_type"] for s in d["steps"]] == ["ocr", "corpus", "chunk"]
    assert all(s["status"] == "ok" for s in d["steps"]), d["steps"]
    assert d["n_documents"] >= 1, "the uploaded file never reached the corpus step"


def test_deploy_preview_returns_every_file_it_lists(client):
    """The screen offers a picker over `files`, so a name in that list without
    content is a file the user is told exists and cannot read."""
    r = client.post("/api/v1/deploy/preview",
                    json={"name": "x", "mode": "service",
                          "builder_config": {"steps": [{"type": "corpus"}]}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["files"], "no files generated"
    missing = [f for f in d["files"] if not isinstance(d.get(f), str) or not d[f]]
    assert not missing, f"listed but not returned: {missing}"
