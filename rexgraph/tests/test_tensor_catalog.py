"""Bounded metadata reads search the complete safetensors header."""
import numpy as np
import pytest

from rexgraph.io.catalog import FileCatalog


@pytest.fixture
def catalog(tmp_path):
    tensors = pytest.importorskip("safetensors.numpy")
    tensors.save_file({"weight": np.arange(6, dtype=np.float32).reshape(2, 3),
                       "scalar": np.array(7, dtype=np.int64),
                       "empty": np.empty((0, 3), dtype=np.float64)}, str(tmp_path / "m.safetensors"))
    return FileCatalog([tmp_path])


def test_shapes_dtypes_and_limit(catalog):
    rows = catalog.tensors("root0/m.safetensors")
    assert {r["name"]: (r["shape"], r["dtype"]) for r in rows} == {
        "empty": ([0, 3], "F64"), "scalar": ([], "I64"), "weight": ([2, 3], "F32")}
    assert len(catalog.tensors("root0/m.safetensors", limit=1)) == 1
    assert all(set(row) == {"name", "shape", "dtype"} for row in rows)


def test_search_finds_late_tensor_without_fetching_unmatched_slices(tmp_path, monkeypatch):
    tensors = pytest.importorskip("safetensors.numpy")
    tensors.save_file({**{f"a{i:04d}": np.zeros(1, np.int8) for i in range(1100)},
                       "zz_TARGET_weight": np.ones(2, np.float32)}, str(tmp_path / "m.safetensors"))
    catalog = FileCatalog([tmp_path])
    import safetensors
    original, slices = safetensors.safe_open, []

    class HeaderOnly:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)
        def __enter__(self):
            self.reader.__enter__()
            return self
        def __exit__(self, *args):
            return self.reader.__exit__(*args)
        def keys(self):
            return self.reader.keys()
        def get_slice(self, key):
            slices.append(key)
            return self.reader.get_slice(key)
        def get_tensor(self, key):
            pytest.fail("tensor payload read")

    monkeypatch.setattr(safetensors, "safe_open", HeaderOnly)
    assert catalog.search_tensors("root0/m.safetensors", "target WEIGHT", limit=1) == [
        {"name": "zz_TARGET_weight", "shape": [2], "dtype": "F32"}]
    assert slices == ["zz_TARGET_weight"]
    slices.clear()
    assert len(catalog.tensors("root0/m.safetensors", limit=2)) == 2 and len(slices) == 2
    assert catalog.search_tensors("root0/m.safetensors", "missing") == []
    assert len(slices) == 2
    assert len(catalog.tensors("root0/m.safetensors", limit=2000)) == 1000


def test_empty_search_and_literal_terms(catalog):
    assert catalog.search_tensors("root0/m.safetensors", "", limit=2) == catalog.tensors("root0/m.safetensors", limit=2)
    assert catalog.search_tensors("root0/m.safetensors", ".*") == []
    assert [r["name"] for r in catalog.search_tensors("root0/m.safetensors", "WEI GHT")] == ["weight"]


def test_missing_corrupt_and_non_tensor_files(tmp_path, catalog):
    with pytest.raises(KeyError):
        catalog.tensors("root0/absent.safetensors")
    (tmp_path / "bad.safetensors").write_bytes(b"bad")
    (tmp_path / "not.h5").write_bytes(b"bad")
    catalog.refresh()
    from safetensors import SafetensorError
    with pytest.raises(SafetensorError):
        catalog.tensors("root0/bad.safetensors")
    with pytest.raises(ValueError, match="safetensors"):
        catalog.tensors("root0/not.h5")


def test_replacement_header_is_read_fresh(tmp_path, catalog):
    from safetensors.numpy import save_file
    save_file({"replacement": np.ones(4, np.int32)}, str(tmp_path / "m.safetensors"))
    assert catalog.tensors("root0/m.safetensors") == [{"name": "replacement", "shape": [4], "dtype": "I32"}]
