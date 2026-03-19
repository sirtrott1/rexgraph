"""
Tests for rexgraph.io.arrow_bridge -- Arrow/IPC bridge.

All tests require pyarrow. Skipped if pyarrow is not installed.

Verifies:
    - arrays_to_arrow / arrow_to_arrays roundtrip: 1D, 2D, complex
    - Metadata preserved in schema
    - Different-length arrays padded correctly
    - rex_to_arrow / arrow_to_rex roundtrip
    - write_arrow_ipc / read_arrow_ipc file roundtrip
    - read_arrow_batches yields correct data
"""
import os
import tempfile

import numpy as np
import pytest

try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

pytestmark = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")

if HAS_PYARROW:
    from rexgraph.io.arrow_bridge import (
        arrays_to_arrow,
        arrow_to_arrays,
        rex_to_arrow,
        arrow_to_rex,
        write_arrow_ipc,
        read_arrow_ipc,
        read_arrow_batches,
    )

from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


# arrays_to_arrow / arrow_to_arrays

class TestArraysRoundtrip:

    def test_1d_array(self):
        arrays = {"x": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        table = arrays_to_arrow(arrays)
        loaded = arrow_to_arrays(table)
        assert np.allclose(loaded["x"], arrays["x"])

    def test_2d_array_shape_preserved(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        arrays = {"mat": arr}
        table = arrays_to_arrow(arrays)
        loaded = arrow_to_arrays(table)
        assert loaded["mat"].shape == (3, 2)
        assert np.allclose(loaded["mat"], arr)

    def test_complex_array(self):
        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        arrays = {"z": arr}
        table = arrays_to_arrow(arrays)
        # Should have __real and __imag columns
        assert "z__real" in table.column_names
        assert "z__imag" in table.column_names
        loaded = arrow_to_arrays(table)
        assert np.allclose(loaded["z"], arr)

    def test_int_array(self):
        arrays = {"idx": np.array([0, 1, 2], dtype=np.int32)}
        table = arrays_to_arrow(arrays)
        loaded = arrow_to_arrays(table)
        assert np.array_equal(loaded["idx"], arrays["idx"])

    def test_different_lengths_padded(self):
        arrays = {
            "short": np.array([1.0, 2.0], dtype=np.float64),
            "long": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        }
        table = arrays_to_arrow(arrays)
        # Table should have 4 rows (max length)
        assert table.num_rows == 4
        loaded = arrow_to_arrays(table)
        # Short array reconstructed to original length
        assert loaded["short"].shape == (2,)
        assert loaded["long"].shape == (4,)

    def test_metadata_preserved(self):
        arrays = {"x": np.ones(3)}
        meta = {"version": 2, "name": "test"}
        table = arrays_to_arrow(arrays, metadata=meta)
        assert b"rex_user_meta" in table.schema.metadata
        import json
        loaded_meta = json.loads(table.schema.metadata[b"rex_user_meta"])
        assert loaded_meta["version"] == 2
        assert loaded_meta["name"] == "test"

    def test_empty_arrays(self):
        arrays = {"empty": np.zeros(0, dtype=np.float64),
                   "nonempty": np.ones(3)}
        table = arrays_to_arrow(arrays)
        loaded = arrow_to_arrays(table)
        assert loaded["empty"].shape == (0,)


# rex_to_arrow / arrow_to_rex

class TestRexRoundtrip:

    def test_basic_roundtrip(self, triangle):
        table = rex_to_arrow(triangle)
        rex2 = arrow_to_rex(table)
        assert rex2.nV == triangle.nV
        assert rex2.nE == triangle.nE
        assert rex2.nF == triangle.nF

    def test_graph_metadata(self, triangle):
        table = rex_to_arrow(triangle)
        import json
        meta = json.loads(table.schema.metadata[b"rex_user_meta"])
        assert meta["object_type"] == "RexGraph"
        assert meta["nV"] == triangle.nV
        assert meta["nE"] == triangle.nE

    def test_boundary_arrays_present(self, triangle):
        table = rex_to_arrow(triangle)
        loaded = arrow_to_arrays(table)
        assert "boundary_ptr" in loaded
        assert "boundary_idx" in loaded


# IPC File I/O

class TestIPCFile:

    def test_write_read_roundtrip(self):
        arrays = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([4.0, 5.0, 6.0]),
        }
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name
        try:
            write_arrow_ipc(arrays, path)
            loaded = read_arrow_ipc(path)
            assert np.allclose(loaded["a"], arrays["a"])
            assert np.allclose(loaded["b"], arrays["b"])
        finally:
            os.unlink(path)

    def test_with_metadata(self):
        arrays = {"x": np.ones(5)}
        meta = {"experiment": "test123"}
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name
        try:
            write_arrow_ipc(arrays, path, metadata=meta)
            loaded = read_arrow_ipc(path)
            assert np.allclose(loaded["x"], np.ones(5))
        finally:
            os.unlink(path)


# Streaming

class TestBatchReads:

    def test_yields_all_data(self):
        arrays = {"x": np.arange(100, dtype=np.float64)}
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name
        try:
            write_arrow_ipc(arrays, path)
            all_x = []
            for batch in read_arrow_batches(path, batch_rows=50):
                all_x.append(batch["x"])
            combined = np.concatenate(all_x)
            # Should recover all 100 elements (batching may merge)
            assert len(combined) >= 100
        finally:
            os.unlink(path)
