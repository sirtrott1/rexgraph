"""The tool registry, and the drift it exists to prevent.

A tool definition is a promise that a name resolves to something that runs. Kept in a
list parallel to the handlers, the two drift: a handler gets renamed, the definition
still advertises the old name, and the failure surfaces only when a model calls it.
That happened once already in this tree.

So the first test here is the structural one: every advertised name dispatches.
"""
from __future__ import annotations

import pytest
from agent.mcp_tools import TOOLS, call, definitions
from tests.test_knowledge_roundtrip import BRCA_GAF, BRCA_OBO, GTF


@pytest.fixture
def files(tmp_path):
    out = []
    for name, text in (("genes.gtf", GTF), ("goa.gaf", BRCA_GAF), ("go.obo", BRCA_OBO)):
        p = tmp_path / name
        p.write_text(text)
        out.append(str(p))
    return out


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app
    from fastapi.testclient import TestClient
    yield TestClient(app)
    reset_default_store()


#### the registry cannot advertise what it cannot run


def test_every_advertised_tool_dispatches():
    for d in definitions():
        assert d["name"] in TOOLS, f"{d['name']} is advertised and cannot be called"
        assert callable(TOOLS[d["name"]].handler)


def test_every_tool_is_described_and_schematised():
    for d in definitions():
        assert d["description"].strip(), f"{d['name']} has no description"
        schema = d["input_schema"]
        assert schema["type"] == "object" and schema["properties"]
        for name in schema["required"]:
            assert name in schema["properties"], \
                f"{d['name']} requires {name} but does not declare it"


def test_the_definitions_come_from_the_registry():
    """Derived, not maintained alongside: the two cannot disagree."""
    assert {d["name"] for d in definitions()} == set(TOOLS)


def test_an_unknown_tool_raises_rather_than_answering():
    with pytest.raises(KeyError, match="no tool named"):
        call("rexgraph_not_a_tool")


def test_a_missing_required_argument_is_named():
    with pytest.raises(TypeError, match="study"):
        call("rexgraph_enrich", files=[])


#### the tools answer


def test_join_reports_the_joined_complex(files):
    out = call("rexgraph_join_sources", files=files)
    assert out["n_entities"] == 8 and out["n_relations"] == 9
    assert out["report"]["n_joined"] >= 2


def test_reason_reports_consistency(files):
    out = call("rexgraph_reason_ontology", files=files)
    assert out["consistency"]["consistent"] is True
    assert len(out["betti"]) == 3


def test_enrich_needs_a_study_set_and_uses_it(files):
    out = call("rexgraph_enrich", files=files, study=["BRCA1", "BRCA2"])
    assert out["n_study"] >= 1
    assert isinstance(out["terms"], list)


def test_homology_reports_the_rank_tower(files):
    out = call("rexgraph_homology", files=files)
    assert len(out["betti"]) == 3
    assert out["grades"][0]["harmonic"] == out["betti"][0]


def test_term_similarity_is_exact(files):
    out = call("rexgraph_term_similarity", files=files,
               term_a="DNA repair", term_b="nucleus")
    assert "/" in out["overlap_exact"] or out["overlap_exact"] in ("0", "1")


def test_the_release_tool_needs_a_series(tmp_path):
    a = tmp_path / "a.obo"
    a.write_text("format-version: 1.2\n\n[Term]\nid: A\nname: a\nis_a: B\n\n"
                 "[Term]\nid: B\nname: b\n")
    b = tmp_path / "b.obo"
    b.write_text(a.read_text())
    out = call("rexgraph_release_series", files=[str(a), str(b)])
    assert out["n_releases"] == 2


def test_querying_a_stored_complex(files, client):
    from agent.knowledge import join
    from agent.rcdb import default_store
    join(*files).store(default_store(), "rec")
    out = call("rexgraph_query_stored", record_id="rec", quantity="kappa",
               op="<", threshold=0.99)
    assert out["n_selected"] > 0
    assert all(isinstance(c, str) for c in out["cells"])


def test_querying_a_record_that_does_not_exist_reports_it(client):
    assert "error" in call("rexgraph_query_stored", record_id="nope",
                           quantity="kappa", op=">", threshold=0.0)


#### over HTTP


def test_the_route_lists_the_same_tools(client):
    body = client.get("/api/v1/mcp/tools").json()
    assert {t["name"] for t in body["tools"]} == set(TOOLS)


def test_the_route_runs_a_tool(client, files):
    r = client.post("/api/v1/mcp/call",
                    json={"name": "rexgraph_homology", "arguments": {"files": files}})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["result"]["betti"] == [1, 2, 0]


def test_the_route_returns_json_not_numpy(client, files):
    """Kernel output reaches this route, and FastAPI encodes before the response
    class."""
    import json
    r = client.post("/api/v1/mcp/call",
                    json={"name": "rexgraph_join_sources", "arguments": {"files": files}})
    assert r.status_code == 200
    json.loads(r.text)


def test_the_route_404s_an_unknown_tool(client):
    r = client.post("/api/v1/mcp/call", json={"name": "nope", "arguments": {}})
    assert r.status_code == 404


def test_the_route_400s_a_missing_argument(client, files):
    r = client.post("/api/v1/mcp/call",
                    json={"name": "rexgraph_enrich", "arguments": {"files": files}})
    assert r.status_code == 400 and "study" in r.json()["detail"]


def _uploads(paths):
    """The upload tuples for a list of paths, read and closed."""
    out = []
    for path in paths:
        with open(path, "rb") as fh:
            out.append(("files", (path.rsplit("/", 1)[-1], fh.read(), "text/plain")))
    return out


def test_the_new_routes_answer(client, files):
    """health, propagate and the training bundle, over HTTP."""
    pytest.importorskip("torch")  # trains a model
    up = _uploads(files)

    r = client.post("/api/v1/knowledge/health", files=up)
    assert r.status_code == 200, r.text[:200]
    assert r.json()["health"]["n_nodes"] == 8

    r = client.post("/api/v1/knowledge/propagate", files=up, data={"seed": "BRCA1"})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["reached"]

    r = client.post("/api/v1/knowledge/bundle", files=up)
    assert r.status_code == 200, r.text[:200]
    assert r.headers["content-type"] == "application/octet-stream"


def test_a_seed_that_reaches_nothing_is_a_400(client, files):
    up = _uploads(files)
    r = client.post("/api/v1/knowledge/propagate", files=up, data={"seed": "NOPE"})
    assert r.status_code == 400


def test_the_training_bundle_reloads_as_vectors(client, files, tmp_path):
    pytest.importorskip("torch")  # trains a model
    up = _uploads(files)
    r = client.post("/api/v1/knowledge/bundle", files=up)
    path = str(tmp_path / "b.safetensors")
    with open(path, "wb") as fh:
        fh.write(r.content)
    from rexgraph.io import load_vectors
    matrix, labels, names, meta = load_vectors(path)
    assert matrix.shape[0] == 9 and len(names) == matrix.shape[1]
    assert labels is not None
    assert "is_a" in str(meta.get("classes", ""))
