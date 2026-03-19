"""
Tests for rexgraph.io._serialization -- NamedTuple serialization via adapters.

Tests focus on NpyAdapter (no Zarr/HDF5 dependency) and the field
classification, write/read roundtrip, and result dict logic.

Verifies:
    - _classify_field: correct type dispatch for all field kinds
    - NpyAdapter: array, scalar, json, string roundtrip
    - NpyAdapter subgroup: creates subdirectory
    - write/read_namedtuple: roundtrip for a simple NamedTuple
    - write/read_result_dict: roundtrip for mixed array/scalar dicts
    - register_type: manual type registration
"""
import json
import os
import tempfile
from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
import pytest

from rexgraph.io._serialization import (
    NpyAdapter,
    _classify_field,
    write_namedtuple,
    read_namedtuple,
    write_result_dict,
    read_result_dict,
    register_type,
)


# Field Classification

class TestClassifyField:

    def test_none(self):
        assert _classify_field(None) == "none"

    def test_ndarray(self):
        assert _classify_field(np.array([1, 2, 3])) == "array"

    def test_int(self):
        assert _classify_field(42) == "scalar"

    def test_float(self):
        assert _classify_field(3.14) == "scalar"

    def test_bool(self):
        assert _classify_field(True) == "scalar"

    def test_numpy_int(self):
        assert _classify_field(np.int32(5)) == "scalar"

    def test_string(self):
        assert _classify_field("hello") == "string"

    def test_dict(self):
        assert _classify_field({"a": 1}) == "dict"

    def test_list_of_dict(self):
        assert _classify_field([{"a": 1}, {"b": 2}]) == "list_of_dict"

    def test_tuple_of_ints(self):
        # Numeric tuple -> treated as array
        assert _classify_field((1, 2, 3)) == "array"

    def test_tuple_of_strings(self):
        assert _classify_field(("a", "b")) == "tuple"

    def test_list_numeric(self):
        assert _classify_field([1.0, 2.0, 3.0]) == "array"


# NpyAdapter

class TestNpyAdapter:

    def test_array_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            adapter.put_array("test", arr)
            loaded = adapter.get_array("test")
            assert np.allclose(loaded, arr)

    def test_scalar_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            adapter.put_scalar("x", 42)
            assert adapter.get_scalar("x") == 42

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            obj = {"key": [1, 2, 3], "nested": {"a": True}}
            adapter.put_json("data", obj)
            loaded = adapter.get_json("data")
            assert loaded == obj

    def test_string_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            adapter.put_string("name", "hello world")
            assert adapter.get_string("name") == "hello world"

    def test_has(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            assert not adapter.has("missing")
            adapter.put_array("present", np.zeros(3))
            assert adapter.has("present")
            adapter.put_scalar("scalar_val", 5)
            assert adapter.has("scalar_val")

    def test_subgroup(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            sub = adapter.subgroup("child")
            sub.put_scalar("x", 99)
            assert sub.get_scalar("x") == 99
            assert os.path.isdir(os.path.join(d, "child"))

    def test_missing_array_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            assert adapter.get_array("nonexistent") is None

    def test_missing_scalar_returns_default(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            assert adapter.get_scalar("x", default=7) == 7


# NamedTuple Write/Read

SimpleResult = namedtuple("SimpleResult", ["data", "score", "label"])
register_type(SimpleResult)


class TestNamedTupleRoundtrip:

    def test_simple_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            obj = SimpleResult(
                data=np.array([1.0, 2.0, 3.0]),
                score=0.95,
                label="test",
            )
            write_namedtuple(adapter, "result", obj)
            loaded = read_namedtuple(adapter, "result", type_class=SimpleResult)
            assert np.allclose(loaded.data, obj.data)
            assert loaded.score == 0.95
            assert loaded.label == "test"

    def test_none_obj_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            write_namedtuple(adapter, "empty", None)
            result = read_namedtuple(adapter, "empty")
            assert result is None

    def test_none_fields_preserved(self):
        NullableResult = namedtuple("NullableResult", ["arr", "optional_arr"])
        register_type(NullableResult)
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            obj = NullableResult(arr=np.ones(3), optional_arr=None)
            write_namedtuple(adapter, "result", obj)
            loaded = read_namedtuple(adapter, "result", type_class=NullableResult)
            assert np.allclose(loaded.arr, np.ones(3))
            assert loaded.optional_arr is None


# Result Dict Write/Read

class TestResultDictRoundtrip:

    def test_mixed_dict(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            data = {
                "values": np.array([1.0, 2.0, 3.0]),
                "count": 42,
                "name": "test",
                "valid": True,
            }
            write_result_dict(adapter, "output", data)
            loaded = read_result_dict(adapter, "output")
            assert np.allclose(loaded["values"], data["values"])
            assert loaded["count"] == 42
            assert loaded["name"] == "test"
            assert loaded["valid"] is True

    def test_empty_dict(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = NpyAdapter(d)
            write_result_dict(adapter, "empty", {})
            loaded = read_result_dict(adapter, "empty")
            assert isinstance(loaded, dict)
