"""
Tests for rexgraph.io._compat -- Zarr/HDF5 compatibility layer.

Tests only the pure-Python helpers and backend detection. Zarr and HDF5
storage helpers are tested via their respective format modules.

Verifies:
    - Backend detection flags exist and are booleans
    - to_native: numpy scalars -> Python natives, NaN/Inf -> 0.0
    - json_default: numpy types serializable, sets -> sorted lists
    - NumpyJSONEncoder: json.dumps works with numpy arrays
    - as_str: bytes decoded, str passthrough
    - ensure_zarr_suffix: appends .zarr only when missing
    - rm_rf: removes files and directories
"""
import json
import os
import tempfile

import numpy as np
import pytest

from rexgraph.io._compat import (
    HAS_ZARR,
    ZARR_V3,
    HAS_HDF5,
    HAS_SCIPY,
    to_native,
    json_default,
    NumpyJSONEncoder,
    as_str,
    ensure_zarr_suffix,
    rm_rf,
)


# Backend Detection

class TestBackendDetection:

    def test_flags_are_bool(self):
        assert isinstance(HAS_ZARR, bool)
        assert isinstance(ZARR_V3, bool)
        assert isinstance(HAS_HDF5, bool)
        assert isinstance(HAS_SCIPY, bool)


# to_native

class TestToNative:

    def test_int(self):
        result = to_native(np.int32(42))
        assert result == 42
        assert isinstance(result, int)

    def test_float(self):
        result = to_native(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_bool(self):
        result = to_native(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_nan_becomes_zero(self):
        assert to_native(np.float64(float("nan"))) == 0.0

    def test_inf_becomes_zero(self):
        assert to_native(np.float64(float("inf"))) == 0.0
        assert to_native(np.float64(float("-inf"))) == 0.0

    def test_array_to_list(self):
        result = to_native(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_passthrough(self):
        assert to_native("hello") == "hello"
        assert to_native(42) == 42
        assert to_native(None) is None


# json_default

class TestJsonDefault:

    def test_numpy_array(self):
        result = json_default(np.array([1.0, 2.0]))
        assert result == [1.0, 2.0]

    def test_numpy_int(self):
        result = json_default(np.int64(7))
        assert result == 7
        assert isinstance(result, int)

    def test_numpy_float_nan(self):
        result = json_default(np.float64(float("nan")))
        assert result == 0.0

    def test_set_to_sorted_list(self):
        result = json_default({3, 1, 2})
        assert result == [1, 2, 3]

    def test_unsupported_raises(self):
        with pytest.raises(TypeError):
            json_default(object())


# NumpyJSONEncoder

class TestNumpyJSONEncoder:

    def test_dumps_with_numpy(self):
        data = {"x": np.array([1, 2, 3]), "y": np.int32(5)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["x"] == [1, 2, 3]
        assert parsed["y"] == 5


# as_str

class TestAsStr:

    def test_bytes(self):
        assert as_str(b"hello") == "hello"

    def test_np_bytes(self):
        assert as_str(np.bytes_(b"world")) == "world"

    def test_np_str(self):
        assert as_str(np.str_("foo")) == "foo"

    def test_passthrough(self):
        assert as_str(42) == 42
        assert as_str("bar") == "bar"


# ensure_zarr_suffix

class TestEnsureZarrSuffix:

    def test_appends(self):
        assert ensure_zarr_suffix("graph") == "graph.zarr"

    def test_preserves(self):
        assert ensure_zarr_suffix("graph.zarr") == "graph.zarr"

    def test_path_like(self):
        from pathlib import Path
        result = ensure_zarr_suffix(Path("data/graph"))
        assert result == "data/graph.zarr"


# rm_rf

class TestRmRf:

    def test_removes_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            path = f.name
        assert os.path.exists(path)
        rm_rf(path)
        assert not os.path.exists(path)

    def test_removes_directory(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "file.txt"), "w") as f:
            f.write("test")
        assert os.path.isdir(d)
        rm_rf(d)
        assert not os.path.exists(d)

    def test_nonexistent_is_noop(self):
        rm_rf("/tmp/definitely_does_not_exist_rexgraph_test_12345")
