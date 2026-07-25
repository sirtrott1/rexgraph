"""
Tests for rexgraph.io.hdf5_format -- HDF5-based storage.

Requires h5py. Skipped if h5py is not installed.

Verifies:
    - save/load roundtrip: RexGraph reconstructed with correct nV/nE/nF
    - Array save/load: save_hdf5_array / load_hdf5_array
    - Cache groups: topology, "all" written and loadable
    - TemporalRex roundtrip
    - Container API: write_to_group / read_from_group / list_groups
    - Weighted graph roundtrip: w_E preserved
    - Compression options: lzf, gzip, None
"""
import os
import tempfile

import numpy as np
import pytest

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

pytestmark = pytest.mark.skipif(not HAS_HDF5, reason="h5py not installed")

if HAS_HDF5:
    from rexgraph.io.hdf5_format import (
        RexHDF5Format,
        save_hdf5,
        load_hdf5,
        save_hdf5_array,
        load_hdf5_array,
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
def h5_path(tmp_path):
    return str(tmp_path / "test.h5")


# Basic Roundtrip

class TestRexGraphRoundtrip:

    def test_basic(self, k4, h5_path):
        save_hdf5(h5_path, k4)
        loaded = load_hdf5(h5_path)
        assert isinstance(loaded, RexGraph)
        assert loaded.nV == k4.nV
        assert loaded.nE == k4.nE
        assert loaded.nF == k4.nF

    def test_betti_preserved(self, k4, h5_path):
        save_hdf5(h5_path, k4)
        loaded = load_hdf5(h5_path)
        assert loaded.betti == k4.betti

    def test_directed_flag(self, triangle, h5_path):
        save_hdf5(h5_path, triangle)
        loaded = load_hdf5(h5_path)
        assert loaded._directed == triangle._directed

    def test_weighted(self, tmp_path):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2], w_E=w)
        path = str(tmp_path / "weighted.h5")
        save_hdf5(path, rex)
        loaded = load_hdf5(path)
        assert np.allclose(loaded._w_E, w)


# Array Save/Load

class TestArrayIO:

    def test_1d(self, h5_path):
        arr = np.array([1.0, 2.0, 3.0])
        save_hdf5_array(arr, h5_path)
        loaded = load_hdf5_array(h5_path)
        assert np.allclose(loaded, arr)

    def test_2d(self, h5_path):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        save_hdf5_array(arr, h5_path)
        loaded = load_hdf5_array(h5_path)
        assert np.allclose(loaded, arr)

    def test_complex(self, h5_path):
        arr = np.array([1+2j, 3+4j], dtype=np.complex128)
        save_hdf5_array(arr, h5_path)
        loaded = load_hdf5_array(h5_path)
        assert np.allclose(loaded, arr)


# Cache Groups

class TestCacheGroups:

    def test_topology_cache(self, k4, h5_path):
        save_hdf5(h5_path, k4, cache=["topology"])
        loaded = load_hdf5(h5_path)
        assert loaded.nV == k4.nV

    def test_all_cache(self, triangle, h5_path):
        save_hdf5(h5_path, triangle, cache="all")
        loaded = load_hdf5(h5_path)
        assert loaded.nE == triangle.nE


# TemporalRex Roundtrip

class TestTemporalRexRoundtrip:

    def test_basic(self, h5_path):
        from rexgraph.graph import TemporalRex
        snaps = [
            (np.array([0, 1, 0], dtype=np.int32),
             np.array([1, 2, 2], dtype=np.int32)),
            (np.array([0, 1, 0, 1], dtype=np.int32),
             np.array([1, 2, 2, 3], dtype=np.int32)),
        ]
        trex = TemporalRex(snaps)
        save_hdf5(h5_path, trex)
        loaded = load_hdf5(h5_path)
        assert isinstance(loaded, TemporalRex)
        assert loaded.T == 2


# Container API

class TestContainerAPI:

    def test_multi_object(self, tmp_path):
        path = str(tmp_path / "container.h5")
        fmt = RexHDF5Format()
        r1 = RexGraph.from_graph([0, 1], [1, 2])
        r2 = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
        fmt.write_to_group(path, "graph_a", r1)
        fmt.write_to_group(path, "graph_b", r2)
        names = fmt.list_groups(path)
        assert "graph_a" in names
        assert "graph_b" in names
        loaded_a = fmt.read_from_group(path, "graph_a")
        loaded_b = fmt.read_from_group(path, "graph_b")
        assert loaded_a.nE == 2
        assert loaded_b.nE == 3


# Compression Options

class TestCompression:

    def test_no_compression(self, k4, h5_path):
        fmt = RexHDF5Format(compression=None)
        fmt.write(h5_path, k4)
        loaded = fmt.read(h5_path)
        assert loaded.nV == k4.nV

    def test_gzip(self, k4, tmp_path):
        path = str(tmp_path / "gzip.h5")
        fmt = RexHDF5Format(compression="gzip", compression_opts=4)
        fmt.write(path, k4)
        loaded = fmt.read(path)
        assert loaded.nE == k4.nE

    def test_overwrite(self, k4, triangle, h5_path):
        save_hdf5(h5_path, k4)
        save_hdf5(h5_path, triangle)
        loaded = load_hdf5(h5_path)
        assert loaded.nE == triangle.nE
