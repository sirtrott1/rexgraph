"""
Tests for rexgraph.io.bundle -- .rex bundle format.

No heavy dependencies (only numpy and json). Uses temporary directories.

Verifies:
    - save/load roundtrip: RexGraph reconstructed with correct nV/nE/nF
    - MANIFEST.json: correct magic, object_type, metadata
    - Array access: bundle["boundary_ptr"], __contains__, list_arrays
    - Cache: written to cache/ subdirectory, readable
    - Weighted graph: w_E preserved
    - TemporalRex roundtrip
    - RexBundle.from_graph / .save / .load / .to_object
    - Memory-map mode
"""
import json
import os

import numpy as np
import pytest

from rexgraph.io.bundle import (
    RexBundle,
    save_rex,
    load_rex,
)
from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def rex_path(tmp_path):
    return str(tmp_path / "test.rex")


# Basic Roundtrip

class TestRoundtrip:

    def test_basic(self, k4, rex_path):
        save_rex(rex_path, k4)
        loaded = load_rex(rex_path)
        assert isinstance(loaded, RexGraph)
        assert loaded.nV == k4.nV
        assert loaded.nE == k4.nE
        assert loaded.nF == k4.nF

    def test_betti_preserved(self, k4, rex_path):
        save_rex(rex_path, k4)
        loaded = load_rex(rex_path)
        assert loaded.betti == k4.betti

    def test_weighted(self, tmp_path):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2], w_E=w)
        path = str(tmp_path / "weighted.rex")
        save_rex(path, rex)
        loaded = load_rex(path)
        assert np.allclose(loaded._w_E, w)

    def test_suffix_added(self, k4, tmp_path):
        """Saves with .rex suffix even if not provided."""
        path = str(tmp_path / "nosuffix")
        save_rex(path, k4)
        assert os.path.isdir(path + ".rex")


# MANIFEST.json

class TestManifest:

    def test_magic(self, k4, rex_path):
        save_rex(rex_path, k4)
        mf = json.loads(open(os.path.join(rex_path, "MANIFEST.json")).read())
        assert mf["magic"] == "rex-bundle"

    def test_object_type(self, k4, rex_path):
        save_rex(rex_path, k4)
        mf = json.loads(open(os.path.join(rex_path, "MANIFEST.json")).read())
        assert mf["object_type"] == "RexGraph"
        assert mf["nV"] == k4.nV
        assert mf["nE"] == k4.nE


# Array Access

class TestArrayAccess:

    def test_getitem(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        bp = bundle["boundary_ptr"]
        assert bp.shape == (k4.nE + 1,)

    def test_contains(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        assert "boundary_ptr" in bundle
        assert "nonexistent" not in bundle

    def test_list_arrays(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        names = bundle.list_arrays()
        assert "boundary_ptr" in names
        assert "boundary_idx" in names

    def test_missing_raises(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        with pytest.raises(KeyError):
            bundle["nonexistent_array"]


# Cache

class TestCache:

    def test_topology_cache(self, k4, rex_path):
        save_rex(rex_path, k4, cache=["topology"])
        bundle = RexBundle.load(rex_path)
        cache = bundle.read_cache()
        # Betti should be in scalar cache
        assert "betti" in cache or "edge_types" in bundle

    def test_cache_arrays_in_subdir(self, k4, rex_path):
        save_rex(rex_path, k4, cache=["algebra"])
        assert os.path.isdir(os.path.join(rex_path, "cache"))

    def test_all_cache(self, triangle, rex_path):
        save_rex(rex_path, triangle, cache="all")
        loaded = load_rex(rex_path)
        assert loaded.nE == triangle.nE


# TemporalRex

class TestTemporalRex:

    def test_roundtrip(self, tmp_path):
        from rexgraph.graph import TemporalRex
        snaps = [
            (np.array([0, 1, 0], dtype=np.int32),
             np.array([1, 2, 2], dtype=np.int32)),
            (np.array([0, 1, 0, 1], dtype=np.int32),
             np.array([1, 2, 2, 3], dtype=np.int32)),
        ]
        trex = TemporalRex(snaps)
        path = str(tmp_path / "temporal.rex")
        save_rex(path, trex)
        loaded = load_rex(path)
        assert isinstance(loaded, TemporalRex)
        assert loaded.T == 2

    def test_snapshot_files(self, tmp_path):
        from rexgraph.graph import TemporalRex
        snaps = [
            (np.array([0, 1], dtype=np.int32),
             np.array([1, 2], dtype=np.int32)),
        ]
        trex = TemporalRex(snaps)
        path = str(tmp_path / "temporal.rex")
        save_rex(path, trex)
        assert os.path.isdir(os.path.join(path, "snapshots", "0"))


# RexBundle API

class TestRexBundleAPI:

    def test_from_graph_and_save(self, k4, rex_path):
        bundle = RexBundle.from_graph(k4)
        assert bundle.object_type == "RexGraph"
        bundle.save(rex_path)
        assert os.path.exists(rex_path)

    def test_load_and_to_object(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        rex = bundle.to_object()
        assert isinstance(rex, RexGraph)
        assert rex.nV == k4.nV

    def test_repr(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        r = repr(bundle)
        assert "RexBundle" in r
        assert "RexGraph" in r

    def test_mmap_mode(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path, mmap=True)
        bp = bundle["boundary_ptr"]
        assert bp.shape == (k4.nE + 1,)
