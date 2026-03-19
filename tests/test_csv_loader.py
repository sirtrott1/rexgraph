"""
Tests for rexgraph.io.csv_loader -- CSV edge list loading and column classification.

No heavy dependencies (only csv, numpy). Tests use temporary CSV files.

Verifies:
    - Column classification: type, polarity, numeric, evidence, description
    - Polarity detection: positive/negative stem matching
    - Weight construction: numeric magnitude, polarity sign, ordinal scaling
    - GraphData: correct nV/nE, to_rex() produces valid RexGraph
    - load_edge_csv: full pipeline with auto column detection
    - Manual role overrides
"""
import os
import tempfile

import numpy as np
import pytest

from rexgraph.io.csv_loader import (
    ColumnRole,
    ColumnProfile,
    classify_columns,
    build_weights,
    build_edge_attrs,
    GraphData,
    load_edge_csv,
)


# Helpers

def _write_csv(rows, path):
    """Write a list of dicts as CSV."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _simple_csv(tmp_path):
    """Simple 3-edge CSV with type and effect columns."""
    path = os.path.join(tmp_path, "edges.csv")
    _write_csv([
        {"source": "A", "target": "B", "type": "binding", "effect": "stimulation", "score": "0.9"},
        {"source": "B", "target": "C", "type": "binding", "effect": "inhibition", "score": "0.5"},
        {"source": "A", "target": "C", "type": "phosphorylation", "effect": "stimulation", "score": "0.7"},
    ], path)
    return path


# Column Classification

class TestClassifyColumns:

    def test_type_by_name(self):
        meta = {"type": ["binding", "phosphorylation", "binding"]}
        profiles = classify_columns(meta)
        assert profiles["type"].role == ColumnRole.TYPE

    def test_polarity_by_name(self):
        meta = {"effect": ["stimulation", "inhibition", "stimulation"]}
        profiles = classify_columns(meta)
        assert profiles["effect"].role == ColumnRole.POLARITY

    def test_numeric_by_name(self):
        meta = {"score": ["0.9", "0.5", "0.7"]}
        profiles = classify_columns(meta)
        assert profiles["score"].role == ColumnRole.NUMERIC

    def test_evidence_by_name(self):
        meta = {"sources": ["SIGNOR;KEGG", "KEGG", "SIGNOR;KEGG;Reactome"]}
        profiles = classify_columns(meta)
        assert profiles["sources"].role == ColumnRole.EVIDENCE

    def test_description_by_name(self):
        meta = {"description": [
            "This is a long description of the mechanism by which A activates B through phosphorylation",
            "Another long description of the inhibitory mechanism through ubiquitination and degradation",
            "A third description that is also quite long and detailed about the pathway interaction",
        ]}
        profiles = classify_columns(meta)
        assert profiles["description"].role == ColumnRole.DESCRIPTION

    def test_numeric_by_values(self):
        """Column with no name match but all numeric values."""
        meta = {"mystery": ["1.0", "2.5", "3.7", "4.2"]}
        profiles = classify_columns(meta)
        assert profiles["mystery"].role == ColumnRole.NUMERIC


# Polarity Detection

class TestPolarityDetection:

    def test_negative_values_detected(self):
        meta = {"effect": ["stimulation", "inhibition", "stimulation", "suppression"]}
        profiles = classify_columns(meta)
        p = profiles["effect"]
        assert "inhibition" in p.negative_values
        assert "suppression" in p.negative_values

    def test_positive_not_in_negative(self):
        meta = {"effect": ["stimulation", "inhibition"]}
        profiles = classify_columns(meta)
        p = profiles["effect"]
        assert "stimulation" not in p.negative_values


# Weight Construction

class TestBuildWeights:

    def test_numeric_magnitude(self):
        meta = {"score": ["2.0", "3.0", "1.0"]}
        profiles = classify_columns(meta)
        w, neg = build_weights(profiles, 3)
        assert np.allclose(w, [2.0, 3.0, 1.0])

    def test_polarity_sign(self):
        meta = {"effect": ["stimulation", "inhibition", "stimulation"]}
        profiles = classify_columns(meta)
        w, neg = build_weights(profiles, 3)
        assert w[0] > 0  # stimulation -> positive
        assert w[1] < 0  # inhibition -> negative
        assert "inhibition" in neg

    def test_default_unit_weights(self):
        """No numeric or polarity columns -> all ones."""
        meta = {"type": ["a", "b", "a"]}
        profiles = classify_columns(meta)
        w, neg = build_weights(profiles, 3)
        assert np.allclose(w, 1.0)
        assert neg == []


# Edge Attrs

class TestBuildEdgeAttrs:

    def test_type_key_present(self):
        meta = {"interaction_type": ["binding", "phosphorylation"]}
        profiles = classify_columns(meta)
        attrs = build_edge_attrs(profiles)
        # Should have "type" key mapped to the type column
        assert "type" in attrs


# GraphData

class TestGraphData:

    def test_basic(self, tmp_path):
        path = _simple_csv(str(tmp_path))
        gd = load_edge_csv(path)
        assert gd.nV == 3  # A, B, C
        assert gd.nE == 3
        assert len(gd.vertices) == 3
        assert gd.src_idx.shape == (3,)
        assert gd.tgt_idx.shape == (3,)

    def test_profiles_populated(self, tmp_path):
        path = _simple_csv(str(tmp_path))
        gd = load_edge_csv(path)
        assert "type" in gd.profiles
        assert "effect" in gd.profiles
        assert "score" in gd.profiles

    def test_w_E_shape(self, tmp_path):
        path = _simple_csv(str(tmp_path))
        gd = load_edge_csv(path)
        assert gd.w_E.shape == (3,)

    def test_to_rex(self, tmp_path):
        from rexgraph.graph import RexGraph
        path = _simple_csv(str(tmp_path))
        gd = load_edge_csv(path)
        rex = gd.to_rex()
        assert isinstance(rex, RexGraph)
        assert rex.nE == 3

    def test_summary_string(self, tmp_path):
        path = _simple_csv(str(tmp_path))
        gd = load_edge_csv(path)
        s = gd.summary()
        assert isinstance(s, str)
        assert "vertices" in s


# Manual Role Overrides

class TestRoleOverrides:

    def test_override(self, tmp_path):
        path = _simple_csv(str(tmp_path))
        gd = load_edge_csv(path, roles={"score": "ordinal"})
        assert gd.profiles["score"].role == "ordinal"


# ColumnProfile Properties

class TestColumnProfileProperties:

    def test_is_categorical(self):
        p = ColumnProfile(name="x", n_unique=5, n_values=100, is_numeric=False)
        assert p.is_categorical

    def test_is_not_categorical_numeric(self):
        p = ColumnProfile(name="x", n_unique=5, n_values=100, is_numeric=True)
        assert not p.is_categorical

    def test_is_binary(self):
        p = ColumnProfile(name="x", n_unique=2, n_values=50)
        assert p.is_binary

    def test_is_freetext(self):
        p = ColumnProfile(name="x", unique_ratio=0.95, avg_length=50.0)
        assert p.is_freetext

    def test_is_delimited(self):
        p = ColumnProfile(name="x", has_delimiter=True)
        assert p.is_delimited_list
