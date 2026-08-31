"""Recording the platform's own work as temporal relational state.

The lineage is one record holding a TemporalRex: a version is a step, and a moment
resolves to a position. Recording is opt-in per workspace and off by default, so a
workspace that has said nothing records nothing.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "rcdb.sqlite"))
    from agent.rcdb import reset_default_store
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "ws"))
    # `_store()` resolves through agent.rcdb.default_store, which caches process-wide;
    # clearing a route-module global does not reach it.
    reset_default_store()
    from agent.server.app import app
    yield TestClient(app)
    reset_default_store()


def _rec(client, **body):
    body.setdefault("kind", "conversation")
    body.setdefault("lineage_id", "chat:s1")
    return client.post("/api/v1/db/record-work", json=body)


def test_the_platform_records_nothing_by_itself_until_asked(client):
    """The setting governs automatic recording. A chat turn with it off leaves no
    trace; the explicit route is a different thing and is tested below."""
    from agent import work_recorder as wr
    assert wr.enabled("default") is False
    assert wr.record("conversation", ["u: hi", "a: hello"],
                     lineage_id="chat:auto") is None
    assert client.get("/api/v1/db/recorded").json()["lineages"] == []


def test_an_explicit_post_records_without_the_setting(client):
    """Posting to the route is the caller asking, so it does not wait on the switch
    that governs what the platform does on its own."""
    r = _rec(client, labels=["u: hi", "a: hello"])
    assert r.status_code == 200, r.text
    assert r.json()["recorded"] is True


def test_settings_round_trip_and_reject_unknown_keys(client):
    r = client.get("/api/v1/admin/workspace/settings")
    assert r.status_code == 200, r.text
    assert r.json()["defaults"]["record_work"] is False

    r = client.post("/api/v1/admin/workspace/settings", json={"record_work": True})
    assert r.status_code == 200, r.text
    assert r.json()["settings"]["record_work"] is True
    assert client.get("/api/v1/admin/workspace/settings").json()["settings"]["record_work"] is True

    assert client.post("/api/v1/admin/workspace/settings",
                       json={"nonsense": 1}).status_code == 400
    assert client.post("/api/v1/admin/workspace/settings",
                       json={"record_work_kinds": ["nope"]}).status_code == 400


def test_a_lineage_is_one_temporal_rex_that_grows(client):
    client.post("/api/v1/admin/workspace/settings", json={"record_work": True})
    turns = ["u: hi", "a: hello"]
    steps = []
    for i in range(3):
        d = _rec(client, labels=turns).json()
        assert d["recorded"] is True, d
        steps.append((d["version"], d["step"], d["T"]))
        turns = turns + [f"u: q{i}", f"a: a{i}"]

    # every call is a new version AND a new step in the same temporal rex
    assert [s[0] for s in steps] == [1, 2, 3]
    assert [s[1] for s in steps] == [0, 1, 2]
    assert [s[2] for s in steps] == [1, 2, 3]

    from rexgraph.graph import TemporalRex
    import agent.server.routes.rcdb as rr
    stored = rr._store().get("chat:s1")
    assert isinstance(stored, TemporalRex), type(stored)
    assert stored.T == 3
    # the earlier states are still reachable, not overwritten
    assert stored.reconstruct_at(0).nE < stored.reconstruct_at(2).nE


def test_an_unchanged_state_is_not_a_step(client):
    client.post("/api/v1/admin/workspace/settings", json={"record_work": True})
    labels = ["u: hi", "a: hello"]
    first = _rec(client, labels=labels).json()
    again = _rec(client, labels=labels).json()
    assert first["recorded"] is True
    assert again["recorded"] is False and again["unchanged"] is True
    assert again["version"] == first["version"]


def test_a_moment_resolves_to_a_position(client):
    client.post("/api/v1/admin/workspace/settings", json={"record_work": True})
    turns = ["u: hi", "a: hello"]
    times = [1000.0, 2000.0, 3000.0]
    for i, t in enumerate(times):
        _rec(client, labels=turns, when=t)
        turns = turns + [f"u: q{i}", f"a: a{i}"]

    r = client.get("/api/v1/db/recorded/chat:s1/at", params={"when": 2500.0})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["step"] == 1, d           # the state current at 2500 is the one from 2000
    assert d["signature"]["nE"] > 0

    assert client.get("/api/v1/db/recorded/chat:s1/at",
                      params={"when": 1.0}).status_code == 404


def test_per_kind_switches_gate_automatic_recording(client):
    client.post("/api/v1/admin/workspace/settings",
                json={"record_work": True, "record_work_kinds": ["pipeline-run"]})
    from agent import work_recorder as wr
    assert wr.enabled("default", "pipeline-run") is True
    assert wr.enabled("default", "conversation") is False
    assert wr.record("conversation", ["u: hi", "a: hello"],
                     lineage_id="chat:auto") is None
    assert wr.record("pipeline-run", ["ocr", "corpus"],
                     lineage_id="pipeline:auto") is not None


def test_recorded_lists_lineages(client):
    client.post("/api/v1/admin/workspace/settings", json={"record_work": True})
    _rec(client, labels=["u: hi", "a: hello"])
    _rec(client, kind="pipeline-run", lineage_id="pipeline:p1",
         labels=["ocr", "corpus", "chunk"])
    lin = client.get("/api/v1/db/recorded").json()["lineages"]
    assert {x["id"] for x in lin} == {"chat:s1", "pipeline:p1"}
    assert {x["kind"] for x in lin} == {"conversation", "pipeline-run"}
    only = client.get("/api/v1/db/recorded", params={"kind": "conversation"}).json()["lineages"]
    assert [x["id"] for x in only] == ["chat:s1"]


def test_bad_input_is_rejected(client):
    assert _rec(client, labels=[]).status_code == 400
    assert _rec(client, labels=["a", "b"], kind="nonsense").status_code == 400
    r = client.post("/api/v1/db/record-work", json={"labels": ["a", "b"]})
    assert r.status_code == 400 and "lineage_id" in r.text


def test_a_builder_run_records_the_stages_that_actually_ran(client):
    client.post("/api/v1/admin/workspace/settings", json={"record_work": True})
    cfg = '{"name":"p1","steps":[{"type":"langgraph_init"},{"type":"langchain_tools"}]}'
    r = client.post("/api/v1/builder/run", data={"config": cfg})
    assert r.status_code == 200, r.text
    assert r.json().get("recorded"), "the run was not recorded"
    lin = client.get("/api/v1/db/recorded", params={"kind": "pipeline-run"}).json()["lineages"]
    assert [x["id"] for x in lin] == ["pipeline:p1"]
    assert lin[0]["labels"] == ["langgraph_init", "langchain_tools"]


def test_a_failed_stage_is_recorded_as_failed(client):
    """The recorded shape is what ran, not what was composed. corpus needs documents,
    so without them the stage is marked rather than quietly dropped."""
    client.post("/api/v1/admin/workspace/settings", json={"record_work": True})
    cfg = ('{"name":"p2","steps":[{"type":"langgraph_init"},{"type":"corpus"},'
           '{"type":"chunk"}]}')
    r = client.post("/api/v1/builder/run", data={"config": cfg})
    assert r.status_code == 200, r.text
    lin = client.get("/api/v1/db/recorded", params={"kind": "pipeline-run"}).json()["lineages"]
    labels = [x for x in lin if x["id"] == "pipeline:p2"][0]["labels"]
    assert any("!" in x for x in labels), labels


def test_a_single_stage_run_is_not_a_relation(client):
    """One stage has nothing to relate to, so there is no complex and nothing is
    recorded. Silence here is the model being honest, not a dropped write."""
    client.post("/api/v1/admin/workspace/settings", json={"record_work": True})
    r = client.post("/api/v1/builder/run",
                    data={"config": '{"name":"p3","steps":[{"type":"langgraph_init"}]}'})
    assert r.status_code == 200, r.text
    assert "recorded" not in r.json()
    ids = [x["id"] for x in client.get("/api/v1/db/recorded").json()["lineages"]]
    assert "pipeline:p3" not in ids


def test_a_recorded_lineage_works_with_the_rcdb_analytics(client):
    """A lineage is stored as a TemporalRex, so every structural read has to see a
    complex rather than a history. `current_rex` is what makes that true; without it
    similarity, compare and clustering silently returned nothing for recorded work.
    """
    for lid, labels in [("run_A", ["ocr", "corpus", "analysis", "export"]),
                        ("run_B", ["ocr", "corpus", "analysis", "export"]),
                        ("run_C", ["upload", "validate", "reject"])]:
        assert _rec(client, kind="pipeline-run", lineage_id=lid,
                    labels=labels).json()["recorded"] is True

    sim = client.post("/api/v1/db/similar", json={"id": "run_A"}).json()
    ids = [m["id"] for m in sim["matches"]]
    assert "run_B" in ids and "run_C" not in ids, ids

    cmp_ = client.post("/api/v1/db/compare", json={"a": "run_A", "b": "run_C"}).json()
    assert cmp_["shared"] == [] and cmp_["only_in_a"], cmp_

    fam = client.post("/api/v1/db/cluster", json={"threshold": 0.7}).json()
    members = [m for c in fam.get("clusters", []) for m in c["members"]]
    assert "run_A" in members and "run_B" in members, fam


def test_trajectory_reads_a_recorded_lineage(client):
    """The version chain of a recorded lineage is a path, and each step compares two
    reconstructed states rather than two histories."""
    turns = ["u: hi", "a: hello"]
    for i in range(3):
        _rec(client, labels=turns)
        turns = turns + [f"u: q{i}", f"a: a{i}"]
    from agent.rcdb import trajectory
    import agent.server.routes.rcdb as rr
    tj = trajectory(rr._store(), "chat:s1")
    assert len(tj["versions"]) == 3, tj["versions"]
    assert tj["steps"], "no step-to-step comparison was produced"
    assert all("match" in s for s in tj["steps"])
    # each recorded state carries its position in the temporal rex
    assert [v["signature"]["object_type"] for v in tj["versions"]] == ["TemporalRex"] * 3
