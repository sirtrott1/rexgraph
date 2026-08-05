"""
Tests for rexgraph.io.arrow_bridge: Arrow/IPC bridge.

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
        arrow_to_rex,
        read_arrow_batches,
        read_arrow_ipc,
        rex_to_arrow,
        write_arrow_ipc,
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

    @staticmethod
    def _write_table_multibatch(table, path, max_chunksize):
        """Write a table split into several record batches (multi-batch file)."""
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with pa.OSFile(path, "wb") as sink, ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=max_chunksize)

    def test_multi_batch_streaming_matches_single_read(self):
        # Arrays of different lengths -> padded to a common flat length, then
        # written across MANY small record batches. Each streamed chunk holds
        # only a slice of every column, so the reader must reshape per-batch.
        arrays = {
            "boundary_ptr": np.arange(21, dtype=np.int64),
            "boundary_idx": (np.arange(40, dtype=np.int64) % 20),
            "w_E": np.linspace(0.0, 1.0, 20, dtype=np.float64),
        }
        table = arrays_to_arrow(arrays)
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name
        try:
            # max_chunksize=4 over 40 flat rows -> 10 record batches.
            self._write_table_multibatch(table, path, max_chunksize=4)
            single = read_arrow_ipc(path)

            per_array = {name: [] for name in arrays}
            n_batches = 0
            for batch in read_arrow_batches(path, batch_rows=4):
                n_batches += 1
                for name in arrays:
                    per_array[name].append(batch[name])

            # Confirm the file really spanned multiple streamed batches.
            assert n_batches > 1
            for name, orig in arrays.items():
                combined = np.concatenate(per_array[name])
                assert combined.dtype == orig.dtype
                assert np.array_equal(combined, single[name])
                assert np.array_equal(combined, orig)
        finally:
            os.unlink(path)

    def test_multi_batch_rex_roundtrip(self):
        # A path graph large enough that its CSR arrays span several batches.
        n = 60
        rex = RexGraph.from_graph(list(range(n - 1)), list(range(1, n)))
        table = rex_to_arrow(rex)
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name
        try:
            self._write_table_multibatch(table, path, max_chunksize=8)
            single = read_arrow_ipc(path)

            acc = {}
            n_batches = 0
            for batch in read_arrow_batches(path, batch_rows=8):
                n_batches += 1
                for name, arr in batch.items():
                    acc.setdefault(name, []).append(arr)

            assert n_batches > 1
            for name, full in single.items():
                combined = np.concatenate(acc[name])
                assert np.array_equal(combined, full)

            # Graph reconstructs identically from the full (single-shot) table.
            rex2 = arrow_to_rex(table)
            assert (rex2.nV, rex2.nE, rex2.nF) == (rex.nV, rex.nE, rex.nF)
        finally:
            os.unlink(path)
