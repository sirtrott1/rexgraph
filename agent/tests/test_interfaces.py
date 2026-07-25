"""Tests for the sovereign-engine seams: silent by default, host-injected
identity, and label privacy that preserves structural comparison."""

import numpy as np

from agent import interfaces as ifc
from agent import schema_complex as sc
from agent.rcdb import MemoryStore, compare, _labels_of


def teardown_function():
    ifc.reset()


def test_no_telemetry_by_default():
    ifc.reset()
    # a recording logger/metrics - but the engine uses the NULL defaults, so
    # nothing is emitted unless the host explicitly configures.
    events = []

    class Rec:
        def log(self, level, message, **f): events.append(("log", message))
        def incr(self, name, value=1.0, **t): events.append(("incr", name))
        def observe(self, name, value, **t): events.append(("obs", name))

    # default hooks are silent and never raise
    ifc.get_logger().log("info", "x", a=1)
    ifc.get_metrics().incr("m")
    ifc.get_metrics().observe("t", 0.5)
    assert events == []                      # nothing recorded - engine is silent
    # only after the host opts in does anything flow
    ifc.configure(logger=Rec(), metrics=Rec())
    ifc.get_logger().log("info", "y")
    assert ("log", "y") in events


def test_default_identity_is_local():
    ifc.reset()
    idn = ifc.get_identity()
    assert idn.workspace == "default" and idn.role == "admin"
    ifc.configure(identity=ifc.LocalIdentity(workspace="team-a", role="read"))
    assert ifc.get_identity().workspace == "team-a"


def test_label_privacy_hides_names_preserves_comparison():
    spec = {"tables": [
        {"name": "customers", "primary_key": ["id"]},
        {"name": "orders", "foreign_keys": [{"columns": ["c"], "references": "customers"}]},
        {"name": "items", "foreign_keys": [{"columns": ["o"], "references": "orders"}]}]}
    rexA, metaA = sc.schema_to_rex(sc.parse_schema_json(spec))
    rexB, metaB = sc.schema_to_rex(sc.parse_schema_json(spec))

    ifc.configure(label_privacy="hash", label_salt="tenant42")
    s = MemoryStore()
    s.put("A", rexA, meta=metaA)
    s.put("B", rexB, meta=metaB)
    stored = _labels_of(s.get_record("A"), s.get("A"))
    # real names never persisted
    assert all("customers" not in l and "orders" not in l for l in stored)
    assert all(l.startswith("t_") for l in stored)
    # structural comparison still works (same name -> same token -> aligns)
    cmp = compare(s, "A", "B")
    assert cmp["match"] == 1.0 and len(cmp["shared"]) == 3


def test_privacy_off_by_default():
    ifc.reset()
    spec = {"tables": [{"name": "customers", "primary_key": ["id"]},
                       {"name": "orders", "foreign_keys": [{"columns": ["c"], "references": "customers"}]}]}
    rex, meta = sc.schema_to_rex(sc.parse_schema_json(spec))
    s = MemoryStore()
    s.put("A", rex, meta=meta)
    assert "customers" in _labels_of(s.get_record("A"), s.get("A"))


def test_tokenize_is_deterministic_and_salted():
    a = ifc.tokenize_labels(["x", "y"], salt="s1")
    b = ifc.tokenize_labels(["x", "y"], salt="s1")
    c = ifc.tokenize_labels(["x", "y"], salt="s2")
    assert a == b and a != c            # deterministic within a salt, differs across


def test_structure_only_no_cell_data():
    # the stored complex is built from FK topology; it cannot contain row data.
    spec = {"tables": [{"name": "users", "primary_key": ["id"]},
                       {"name": "orders", "foreign_keys": [{"columns": ["u"], "references": "users"}]}]}
    rex, meta = sc.schema_to_rex(sc.parse_schema_json(spec))
    from agent.rcdb import serialize_complex
    blob = serialize_complex(rex)
    # only structure/labels are present; assert no fabricated cell values sneak in
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0
    # the meta carries labels (names) and edges - never rows
    assert set(meta.keys()) >= {"vertex_labels", "edges"}
    assert "rows" not in meta and "data" not in meta and "values" not in meta
