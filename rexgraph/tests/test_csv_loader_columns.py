"""
Tests for load_edge_csv hardening: explicit source/target/weight columns,
usecols restriction, delimiter/header override, and non-finite weight
rejection. Targets arbitrary-schema data such as wide multi-column TSV exports.
"""
import numpy as np
import pytest
from rexgraph.io.csv_loader import load_edge_csv


def test_explicit_columns_wide_schema(tmp_path):
    p = tmp_path / "b.tsv"
    p.write_text("ID1\tX1\tID2\tX2\tY\nL1\tCC\tP1\tMS\t0.5\nL2\tCCO\tP1\tMS\t1.5\nL1\tCC\tP2\tMS\t2.0\n")
    gd = load_edge_csv(str(p), source="ID2", target="ID1", weight="Y")
    assert gd.nE == 3 and set(gd.vertices) == {"P1", "P2", "L1", "L2"}
    assert np.allclose(np.sort(gd.w_E), [0.5, 1.5, 2.0])


def test_usecols_skips_heavy(tmp_path):
    p = tmp_path / "b.tsv"
    p.write_text("ID1\tX1\tID2\tX2\tY\nL1\tHUGE\tP1\tHUGE\t0.5\n")
    gd = load_edge_csv(str(p), source="ID2", target="ID1", weight="Y", usecols=["ID1", "ID2", "Y"])
    assert "X1" not in gd.meta and "X2" not in gd.meta


def test_nonfinite_weight_rejected(tmp_path):
    p = tmp_path / "n.csv"; p.write_text("source,target,w\nA,B,nan\nB,C,2.0\n")
    with pytest.raises(ValueError):
        load_edge_csv(str(p), weight="w")


def test_single_column_delimiter_override(tmp_path):
    p = tmp_path / "s.csv"; p.write_text("source,target\nA,B\nB,C\n")
    gd = load_edge_csv(str(p), delimiter=",")     # explicit delimiter, no Sniffer crash
    assert gd.nE == 2


def test_heuristic_default_back_compat(tmp_path):
    p = tmp_path / "e.csv"; p.write_text("source,target\nA,B\nB,C\n")
    assert load_edge_csv(str(p)).nE == 2
