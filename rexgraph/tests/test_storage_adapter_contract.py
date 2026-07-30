"""StorageAdapter is a contract, so every adapter is held to the same behaviour.

It used to be a bare class of NotImplementedError stubs with nothing enforcing them:
a subclass that forgot get_json constructed fine and blew up later, deep inside a
write, and no test ever ran the three adapters through the same paces. These tests
are the contract -- one parametrized suite, every implementation.
"""

import numpy as np
import pytest

from rexgraph.io._compat import HAS_HDF5, HAS_ZARR
from rexgraph.io._serialization import (
    HDF5Adapter, NpyAdapter, StorageAdapter, ZarrAdapter,
)


@pytest.fixture(params=["npy", "zarr", "hdf5"])
def adapter(request, tmp_path):
    kind = request.param
    if kind == "npy":
        yield NpyAdapter(tmp_path / "store")
    elif kind == "zarr":
        if not HAS_ZARR:
            pytest.skip("zarr not installed")
        import zarr
        yield ZarrAdapter(zarr.open_group(str(tmp_path / "s.zarr"), mode="w"))
    else:
        if not HAS_HDF5:
            pytest.skip("h5py not installed")
        import h5py
        with h5py.File(tmp_path / "s.h5", "w") as f:
            yield HDF5Adapter(f)


def test_array_round_trips_with_dtype_and_shape(adapter):
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    adapter.put_array("a", arr)
    back = adapter.get_array("a")
    assert back is not None
    assert back.dtype == arr.dtype and back.shape == arr.shape
    assert np.array_equal(back, arr)


def test_missing_array_is_none_not_an_exception(adapter):
    assert adapter.get_array("never_written") is None


def test_scalar_round_trips(adapter):
    adapter.put_scalar("n", np.int64(7))
    adapter.put_scalar("x", 1.5)
    assert int(adapter.get_scalar("n")) == 7
    assert float(adapter.get_scalar("x")) == 1.5


def test_missing_scalar_returns_the_default(adapter):
    assert adapter.get_scalar("nope", default=42) == 42
    assert adapter.get_scalar("nope") is None


def test_string_round_trips_and_defaults(adapter):
    adapter.put_string("s", "relational complex")
    assert adapter.get_string("s") == "relational complex"
    assert adapter.get_string("nope") == ""
    assert adapter.get_string("nope", default="fallback") == "fallback"


def test_json_round_trips_nested_structures(adapter):
    payload = {"a": [1, 2, {"b": "c"}], "d": {"e": [3.5]}}
    adapter.put_json("j", payload)
    assert adapter.get_json("j") == payload


def test_missing_json_returns_the_default(adapter):
    assert adapter.get_json("nope") is None
    assert adapter.get_json("nope", default={}) == {}


def test_json_survives_a_nonfinite_value(adapter):
    """put_json fed json.dumps directly, so a NaN metric wrote a bare NaN token into
    a zarr/hdf5 attr -- unreadable by anything but Python's lenient parser."""
    adapter.put_json("j", {"kappa": np.float64("nan"), "ok": 1.0})
    back = adapter.get_json("j")
    assert back["ok"] == 1.0
    assert back["kappa"] == 0.0


def test_has_reports_arrays_and_metadata_alike(adapter):
    assert not adapter.has("x")
    adapter.put_array("x", np.zeros(3))
    assert adapter.has("x")
    assert not adapter.has("y")
    adapter.put_scalar("y", 1)
    assert adapter.has("y")


def test_subgroup_is_a_working_adapter_and_is_isolated(adapter):
    sub = adapter.subgroup("child")
    assert isinstance(sub, StorageAdapter)
    sub.put_array("a", np.ones(2))
    assert np.array_equal(sub.get_array("a"), np.ones(2))
    # the child's namespace is its own
    assert adapter.get_array("a") is None


def test_subgroup_is_reentrant(adapter):
    adapter.subgroup("child").put_scalar("k", 5)
    assert adapter.subgroup("child").get_scalar("k") == 5


def test_the_base_class_refuses_to_be_instantiated():
    """A stub that raises at call time hides an incomplete adapter until it is used
    on real data. Missing methods must fail at construction, naming what is missing."""
    with pytest.raises(TypeError) as ei:
        StorageAdapter()
    assert "abstract" in str(ei.value).lower()


def test_an_incomplete_adapter_fails_at_construction():
    class Partial(StorageAdapter):
        def put_array(self, name, arr): pass

    with pytest.raises(TypeError) as ei:
        Partial()
    assert "get_array" in str(ei.value)
