"""The app can draw the complexes it already holds.

A session has held its complexes all along, one per step, with an id and a history. What
it could not do is show one: every reading was reachable through a route and the picture
was not, so the app could describe a complex it could not draw.

These routes go through `render_payload` and `render_svg`, the same path the pipeline's
`drawing` stage and the `rexgraph_render` tool take, so what the app draws and what an
agent draws cannot differ.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.faces import autoface
from rexgraph.graph import RexGraph


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A test client with auth off, and the user's real config left alone.

    `disable_auth()` used to PERSIST unconditionally, writing `enabled: false` into the
    host's own `~/.config/rexgraph/auth.json`. Six fixtures did that, which is how this
    suite turned auth off on a live install and left it off. `persist=False` is the
    in-process form and is what a test wants: the server object open, not the host
    reconfigured. The config directory is redirected as well, so nothing here can reach
    a real file even by another path.
    """
    import os

    from fastapi.testclient import TestClient

    from agent.server.app import app
    from agent.server.auth import get_auth_manager

    previous = os.environ.get("REXGRAPH_CONFIG_DIR")
    previous_sessions = os.environ.get("REXGRAPH_SESSION_DIR")
    os.environ["REXGRAPH_CONFIG_DIR"] = str(tmp_path_factory.mktemp("config"))
    # sessions too: this fixture had written 397 of them into the user's own store in a
    # day, which is what buried a real chat at index 1177 of a list they had to scroll
    os.environ["REXGRAPH_SESSION_DIR"] = str(tmp_path_factory.mktemp("sessions"))
    import agent.server.app as server_app
    server_app._store = None
    manager = get_auth_manager()
    was_enabled = getattr(manager, "_auth_enabled", None)
    manager.disable_auth(persist=False)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        if was_enabled is not None:
            manager._auth_enabled = was_enabled
        server_app._store = None
        for name, value in (("REXGRAPH_CONFIG_DIR", previous),
                            ("REXGRAPH_SESSION_DIR", previous_sessions)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def session(client):
    """A session holding a filled triangle with a tail, annotated."""
    from agent.server.app import get_store

    rex = RexGraph(sources=np.array([0, 1, 2, 0, 3], dtype=np.int32),
                   targets=np.array([1, 2, 0, 3, 4], dtype=np.int32))
    autoface(rex)
    rex._ensure_clean()
    for e in range(3):
        rex.attach_metadata(1, e, "kind", "ring")
    rex.attach_metadata(0, 0, "element", "C")
    s = get_store().create(name="graph-route-test")
    s.add_snapshot(rex=rex, action="analyze", results={}, summary="first")
    return s


#### drawing


def test_the_views_are_named_and_described(client):
    body = client.get("/api/v1/graph/views").json()
    assert set(body["views"]) == {"structural", "plane", "character", "embedded", "flow"}
    # the default draws rather than reads: the other four place a cell by what it IS, so
    # structurally identical cells share a point and a star comes out as one dot
    assert body["default"] == "structural"


def test_a_session_renders(client, session):
    body = client.post(f"/api/v1/graph/{session.session_id}/render",
                       json={"view": "plane"}).json()
    assert body["svg"].startswith("<svg")
    assert body["cells_total"] == 5
    assert body["truncated"] is False


def test_every_view_renders(client, session):
    for view in ("structural", "plane", "character", "embedded"):
        res = client.post(f"/api/v1/graph/{session.session_id}/render",
                          json={"view": view})
        assert res.status_code == 200, view
        assert res.json()["view"] == view


def test_an_unknown_view_is_refused(client, session):
    assert client.post(f"/api/v1/graph/{session.session_id}/render",
                       json={"view": "isometric"}).status_code == 400


def test_a_missing_session_is_a_404(client):
    assert client.post("/api/v1/graph/nope/render", json={}).status_code == 404


def test_truncation_is_reported(client, session):
    body = client.post(f"/api/v1/graph/{session.session_id}/render",
                       json={"limit": 2}).json()
    assert body["truncated"] is True
    assert body["cells_drawn"] == 2 and body["cells_total"] == 5


def test_the_readings_come_with_the_picture(client, session):
    """So a reader does not have to call twice to know what they are looking at."""
    body = client.post(f"/api/v1/graph/{session.session_id}/render", json={}).json()
    assert body["state"]["state"] in ("latent", "filled", "closed")
    assert body["field"]["channels"]
    assert body["relations"][0]["quadrance"] == "2"


#### the file


def test_the_image_downloads_as_svg(client, session):
    res = client.post(f"/api/v1/graph/{session.session_id}/image", json={})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/svg+xml")
    assert "attachment" in res.headers["content-disposition"]
    assert res.text.startswith("<svg")


def test_the_filename_says_which_view_and_step(client, session):
    res = client.post(f"/api/v1/graph/{session.session_id}/image",
                      json={"view": "character", "step": 0})
    assert "character" in res.headers["content-disposition"]
    assert "step0" in res.headers["content-disposition"]


def test_the_image_is_not_sniffable(client, session):
    res = client.post(f"/api/v1/graph/{session.session_id}/image", json={})
    assert res.headers.get("x-content-type-options") == "nosniff"


#### history, and drawing a past step


def test_the_history_lists_the_steps(client, session):
    body = client.get(f"/api/v1/graph/{session.session_id}/history").json()
    assert body["n_steps"] >= 1
    assert body["steps"][0]["action"] == "analyze"


def test_a_past_step_can_be_drawn(client, session):
    res = client.post(f"/api/v1/graph/{session.session_id}/render", json={"step": 0})
    assert res.status_code == 200
    assert res.json()["step"] == 0


def test_a_step_that_does_not_exist_is_a_404(client, session):
    assert client.post(f"/api/v1/graph/{session.session_id}/render",
                       json={"step": 99}).status_code == 404


#### interaction


def test_a_click_on_a_relation_returns_what_it_is(client, session):
    body = client.post(f"/api/v1/graph/{session.session_id}/cell",
                       json={"grade": 1, "index": 0}).json()
    assert body["quadrance"] == "2"
    assert body["boundary"] == [0, 1]
    assert body["attributes"]["kind"] == "ring"


def test_a_click_on_a_vertex_returns_its_angles(client, session):
    body = client.post(f"/api/v1/graph/{session.session_id}/cell",
                       json={"grade": 0, "index": 0}).json()
    assert body["attributes"]["element"] == "C"
    assert isinstance(body["angles_at"], list)


def test_a_click_on_a_face_returns_its_sign_context(client, session):
    body = client.post(f"/api/v1/graph/{session.session_id}/cell",
                       json={"grade": 2, "index": 0}).json()
    assert body["reading"]["state"] == "bounds"


def test_an_unknown_grade_is_refused(client, session):
    assert client.post(f"/api/v1/graph/{session.session_id}/cell",
                       json={"grade": 7, "index": 0}).status_code == 400


#### selection


def test_the_attribute_keys_come_from_the_complex(client, session):
    """So a filter box offers what is there rather than a list someone maintains."""
    body = client.get(f"/api/v1/graph/{session.session_id}/attributes").json()
    assert body["keys"]["1"] == ["kind"]
    assert body["keys"]["0"] == ["element"]


def test_a_selection_returns_a_mask(client, session):
    body = client.post(f"/api/v1/graph/{session.session_id}/select",
                       json={"criteria": {"kind": "ring"}, "grade": 1}).json()
    assert body["n_selected"] == 3
    assert body["n_cells"] == 5


def test_a_selection_reaches_the_drawing(client, session):
    """And dims rather than deletes: the cell count does not change."""
    body = client.post(f"/api/v1/graph/{session.session_id}/render",
                       json={"select": {"kind": "ring"}, "select_dim": 1}).json()
    assert body["selection"]["n_selected"] == 3
    assert body["cells_total"] == 5


#### the screen is wired, not just written


def _app_jsx():
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "frontend" / "app.jsx").read_text()


def test_the_screen_exists_and_is_registered():
    """A component nobody can reach is not a feature."""
    source = _app_jsx()
    assert "function GraphView" in source
    assert "graph:GraphView" in source, "not in TAB_MAP"
    assert '{id:"graph"' in source, "not in the nav"


def test_the_screen_calls_the_routes_it_needs():
    source = _app_jsx()
    for path in ("/api/v1/graph/views", "/render", "/image", "/history",
                 "/cell", "/attributes"):
        assert path in source, path


def test_the_frontend_parses():
    """Checked with node rather than by counting braces."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "app.js"
        copy.write_text(_app_jsx())
        result = subprocess.run([node, "--check", str(copy)],
                                capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


#### the edit lineage, in the app


@pytest.fixture
def lineage_id():
    import time

    return f"route-lineage-{int(time.time() * 1e6)}"


def test_a_session_state_can_be_recorded_as_an_edit(client, session, lineage_id):
    """The edit is the COMPLEX, not a description of it, so what is stored reconstructs
    into the same thing that was analysed and drawn."""
    info = client.post(f"/api/v1/graph/lineage/{lineage_id}/record",
                       json={"session_id": session.session_id, "at": 9000.0}).json()
    assert info["version"] == 1 and info["step"] == 0


def test_successive_edits_are_versions_and_steps(client, session, lineage_id):
    for i in range(3):
        client.post(f"/api/v1/graph/lineage/{lineage_id}/record",
                    json={"session_id": session.session_id, "at": 9000.0 + i})
    body = client.get(f"/api/v1/graph/lineage/{lineage_id}").json()
    assert [s["step"] for s in body["steps"]] == [0, 1, 2]


def test_a_recorded_state_draws_like_any_other(client, session, lineage_id):
    client.post(f"/api/v1/graph/lineage/{lineage_id}/record",
                json={"session_id": session.session_id, "at": 9100.0})
    body = client.post(f"/api/v1/graph/lineage/{lineage_id}/render",
                       json={"at": 9100.0}).json()
    assert body["svg"].startswith("<svg")
    assert body["step"] == 0


def test_rendering_a_lineage_needs_a_moment(client, lineage_id):
    assert client.post(f"/api/v1/graph/lineage/{lineage_id}/render",
                       json={}).status_code == 400


def test_an_unrecorded_lineage_is_a_404(client):
    assert client.get("/api/v1/graph/lineage/never-recorded").status_code == 404


def test_the_lineages_are_listable(client):
    assert "lineages" in client.get("/api/v1/graph/lineages").json()


#### the screen shows all three


def test_the_screen_shows_the_lineage_and_the_pipeline_drawing():
    source = _app_jsx()
    assert "function PipelineDrawing" in source, "no pipeline drawing component"
    assert "PipelineDrawing," in source, "the component is never rendered"
    assert "/api/v1/graph/lineage/" in source, "the screen cannot reach a lineage"
    assert "Record this state" in source


def test_the_pipeline_route_takes_the_reader_options():
    """`auto_rex` already forwards them by reader signature; the route is what could not
    say them. `{"aromatic": "pairwise"}` reads an SDF's rings as separate bonds."""
    import inspect

    import agent.server.routes.pipeline as pipeline

    params = inspect.signature(pipeline.stream_pipeline).parameters
    assert "reader_options" in params
    assert "face_selection" in params


def test_the_flow_view_is_reachable_with_a_signal(client, session):
    """For data with no geometry of its own, where a cell sits is where the measurement
    puts it. The view existed in code and the route could not say it."""
    body = client.post(f"/api/v1/graph/{session.session_id}/render",
                       json={"view": "flow", "signal": [1.0, 5.0, 2.0, 9.0, 3.0]}).json()
    assert body["svg"].startswith("<svg")
    assert "potential across" in body["svg"]


def test_the_flow_view_says_when_it_has_no_signal(client, session):
    body = client.post(f"/api/v1/graph/{session.session_id}/render",
                       json={"view": "flow"}).json()
    assert "no signal" in body["svg"]


#### the session list has to be usable, not merely correct


def test_sessions_come_back_newest_first(tmp_path):
    """It sorted by directory name, which is a random hex id, so the order carried no
    meaning. On a real install with 5278 sessions a freshly recorded chat sat at index
    1177: in the list, and unfindable in a dropdown."""
    import json

    from agent.session import list_sessions

    for i, stamp in enumerate([100.0, 300.0, 200.0]):
        d = tmp_path / f"s{i}"
        d.mkdir()
        (d / "session_index.json").write_text(json.dumps({
            "metadata": {"session_id": f"s{i}", "created": stamp, "name": f"n{i}"},
            "snapshots": [], "current_step": 0}))
    names = [s.get("name") for s in list_sessions(str(tmp_path))]
    assert names == ["n1", "n2", "n0"], "not ordered by recency"


def test_the_listing_can_be_bounded(tmp_path):
    """A control populating itself does not want five thousand of anything."""
    import json

    from agent.session import list_sessions

    for i in range(6):
        d = tmp_path / f"s{i}"
        d.mkdir()
        (d / "session_index.json").write_text(json.dumps({
            "metadata": {"session_id": f"s{i}", "created": float(i), "name": f"n{i}"},
            "snapshots": [], "current_step": 0}))
    assert len(list_sessions(str(tmp_path), limit=2)) == 2


def test_the_store_honours_its_own_directory(tmp_path, monkeypatch):
    """So a test holds its sessions somewhere of its own. This suite had written 1065
    sessions into a real install in a day, 397 from one fixture."""
    from agent.server.state import SessionStore

    monkeypatch.setenv("REXGRAPH_SESSION_DIR", str(tmp_path))
    assert SessionStore().storage_dir == str(tmp_path)
