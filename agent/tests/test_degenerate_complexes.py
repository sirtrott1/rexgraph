"""Complexes that carry nothing, and the numbers they report.

An empty complex is a real input: a document that produced no relations, an upload
that OCR could not read, a query naming nothing. The analysis pipeline runs on it,
and what it reports has to be reportable.

An undefined measurement over such a complex is reported as an IEEE non-finite: the
mean over zero edges is NaN and the mixing time of a complex with no cycle is
infinity. Both are the correct values and both are kept. What is not acceptable is
either of them reaching a client, because neither is JSON, so the app renders a
non-finite as null at the response boundary.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from agent.pipeline import AnalysisPipeline
from fastapi.testclient import TestClient

from rexgraph.graph import RexGraph

DEPTHS = ("quick", "standard", "full")


def _empty():
    return RexGraph(sources=np.array([], np.int32), targets=np.array([], np.int32))


def _one_edge():
    return RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32))


def _nonfinite(obj, path=""):
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += _nonfinite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:200]):
            bad += _nonfinite(v, f"{path}[{i}]")
    elif isinstance(obj, (float, np.floating)) and (math.isnan(obj)
                                                    or math.isinf(obj)):
        bad.append(f"{path}={obj}")
    return bad


#### what works today


@pytest.mark.parametrize("depth", DEPTHS)
def test_an_empty_complex_completes_every_depth(depth):
    assert isinstance(AnalysisPipeline(_empty()).run(depth=depth), dict)


@pytest.mark.parametrize("depth", DEPTHS)
def test_a_single_edge_completes_every_depth(depth):
    assert isinstance(AnalysisPipeline(_one_edge()).run(depth=depth), dict)


def test_an_empty_complex_still_reports_its_shape():
    out = AnalysisPipeline(_empty()).run(depth="quick")
    assert out["construction"]["nE"] == 0
    assert out["construction"]["nV"] == 0


#### undefined measurements, and where they stop


@pytest.mark.parametrize("depth", DEPTHS)
def test_an_empty_complex_reports_undefined_measurements_as_non_finite(depth):
    """Not a defect: the mean of no edges is undefined and NaN says so. Recorded
    because it is the input to the serialization contract below."""
    bad = _nonfinite(AnalysisPipeline(_empty()).run(depth=depth))
    assert bad, "an empty complex produced no undefined measurement at all"
    assert any("alpha_G" in b or "chi_mean" in b or "phi" in b for b in bad), \
        f"the undefined values are not the ones expected: {bad[:4]}"


def test_a_single_edge_has_an_infinite_mixing_time():
    """A complex with no cycle has nothing to mix through."""
    out = AnalysisPipeline(_one_edge()).run(depth="standard")
    mixing = out.get("relational", {}).get("mixing_times")
    assert mixing is not None, "mixing_times was not reported at all"
    assert any(math.isinf(float(m)) for m in mixing), \
        f"expected an infinite mixing time on an acyclic complex, got {mixing}"


@pytest.mark.parametrize("depth", DEPTHS)
def test_the_analysis_route_can_send_an_empty_complex(depth, tmp_path, monkeypatch):
    """The defect this file was opened for. An undefined measurement is not JSON, and
    the whole response used to fail rather than one field reporting absent."""
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app, get_store

    client = TestClient(app)
    session = get_store().create(name="empty-probe")
    session.add_snapshot(rex=_empty(), action="probe", params={}, results={},
                         summary="a document that produced no relations")
    r = client.get(f"/api/analysis/{session.session_id}?depth={depth}")
    assert r.status_code == 200, r.text[:300]
    body = json.loads(r.text)          # strict: no bare NaN or Infinity token
    assert isinstance(body, dict)
    reset_default_store()


def test_a_single_edge_survives_the_route(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app, get_store

    client = TestClient(app)
    session = get_store().create(name="one-edge-probe")
    session.add_snapshot(rex=_one_edge(), action="probe", params={}, results={},
                         summary="one relation")
    r = client.get(f"/api/analysis/{session.session_id}?depth=standard")
    assert r.status_code == 200, r.text[:300]
    json.loads(r.text)
    reset_default_store()


def test_an_undefined_measurement_arrives_as_null_not_as_a_zero(tmp_path,
                                                                monkeypatch):
    """null says "no value". 0.0 would be a measurement, and a reader cannot tell the
    difference between a coherence of zero and a coherence over nothing."""
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app, get_store

    client = TestClient(app)
    session = get_store().create(name="null-probe")
    session.add_snapshot(rex=_empty(), action="probe", params={}, results={},
                         summary="empty")
    body = client.get(f"/api/analysis/{session.session_id}?depth=standard").json()
    rel = body.get("relational") or {}
    if "chi_mean" in rel and rel["chi_mean"] is not None:
        assert any(v is None for v in rel["chi_mean"]), \
            f"an undefined channel mean came back as {rel['chi_mean']}"
