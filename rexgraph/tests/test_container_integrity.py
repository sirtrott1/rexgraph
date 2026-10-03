"""Derived values and appended results must be verified before consumption."""
from contextlib import contextmanager

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.io.container_integrity import ATTRIBUTE, container_digest
from rexgraph.rextypes import PersistenceDiagram
from rexgraph.state import to_state


@pytest.fixture(params=["hdf5", "zarr"])
def backend(request):
    pytest.importorskip({"hdf5": "h5py", "zarr": "zarr"}[request.param])
    if request.param == "hdf5":
        from rexgraph.io.hdf5_format import RexHDF5Format
        return RexHDF5Format(), ".h5"
    from rexgraph.io.zarr_format import RexZarrFormat
    return RexZarrFormat(), ".zarr"


@contextmanager
def opened(path, mode="a"):
    if path.suffix == ".h5":
        import h5py
        with h5py.File(path, mode) as root:
            yield root
    else:
        import zarr
        yield zarr.open_group(str(path), mode=mode)


def graph():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.mark.parametrize("change", ["array", "scalar", "missing", "added", "header", "removed_seal"])
def test_cache_and_graph_reads_refuse_modified_containers(backend, tmp_path, change):
    fmt, suffix = backend
    path = tmp_path / ("cache"+suffix)
    fmt.write(path, graph(), cache=["B1", "betti"])
    with opened(path) as root:
        if change == "array":
            root["algebra"]["B1"][0, 0] = 777
        elif change == "scalar":
            root["topology"].attrs["betti"] = "[777, 0, 0]"
        elif change == "missing":
            del root["algebra"]["B1"]
        elif change == "added":
            root["algebra"].attrs["extra"] = 777
        elif change == "header":
            root.attrs["object_type"] = "array"
        else:
            del root.attrs[ATTRIBUTE]
    for read in (fmt.read, fmt.read_cache):
        with pytest.raises(ValueError, match="integrity"):
            read(str(path))


def test_update_refuses_to_reseal_corrupt_old_content(backend, tmp_path):
    fmt, suffix = backend
    path = tmp_path / ("cache"+suffix)
    fmt.write(path, graph(), cache=["B1"])
    with opened(path) as root:
        root["algebra"]["B1"][0, 0] = 777
    with pytest.raises(ValueError, match="integrity"):
        fmt.write_to_group(path, "new", graph())
    with opened(path, "r") as root:
        assert root["algebra"]["B1"][0, 0] == 777
        assert "objects" not in root
    assert list(tmp_path.iterdir()) == [path]


def test_legacy_cache_reads_require_explicit_opt_in(backend, tmp_path):
    fmt, suffix = backend
    path = tmp_path / ("legacy"+suffix)
    original = graph()
    fmt.write(path, original, cache=["B1"])
    with opened(path) as root:
        del root.attrs[ATTRIBUTE]
        root.attrs["format_version"] = "2.0.0"
    assert to_state(fmt.read(str(path))).header["digest"] == to_state(original).header["digest"]
    with pytest.raises(ValueError, match="legacy auxiliary"):
        fmt.read_cache(str(path))
    assert np.array_equal(fmt.read_cache(str(path), allow_unsealed=True)["B1"], original.B1)


def test_appended_result_and_native_graph_remain_readable(backend, tmp_path):
    fmt, suffix = backend
    path = tmp_path / ("results"+suffix)
    original = graph()
    fmt.write(path, original, cache=["B1"])
    diagram = PersistenceDiagram(np.array([[0., 1., 0., 0., 1.]]), np.empty((0, 3)),
                                 (1, 0, 0), np.array([0, 1], np.int64))
    fmt.write_persistence_result(path, diagram)
    assert to_state(fmt.read(str(path))).header["digest"] == to_state(original).header["digest"]
    restored = fmt.read_persistence_result(str(path))["diagram"]
    for before, after in zip(diagram, restored, strict=True):
        assert np.array_equal(before, after)
    assert np.array_equal(fmt.read_cache(str(path))["B1"], original.B1)
    with opened(path) as root:
        root["persistence"].attrs["unclaimed"] = "modified"
    with pytest.raises(ValueError, match="integrity"):
        fmt.read_persistence_result(str(path))


def test_unknown_cache_request_preserves_old_publication(backend, tmp_path):
    fmt, suffix = backend
    path = tmp_path / ("cache"+suffix)
    fmt.write(path, graph())
    with pytest.raises(ValueError, match="unknown cache"):
        fmt.write(path, graph(), cache="misspelled_cache")
    assert fmt.read(str(path)).nE == 3


def test_hdf5_external_links_and_group_cycles_are_refused(tmp_path):
    h5py = pytest.importorskip("h5py")
    for cyclic in [False, True]:
        with h5py.File(tmp_path / "links.h5", "w") as root:
            if cyclic:
                root["self"] = root
            else:
                root["outside"] = h5py.ExternalLink("other.h5", "/data")
            with pytest.raises(ValueError, match="cyclic|external"):
                container_digest(root)


def test_large_array_digest_uses_bounded_slices(monkeypatch):
    from rexgraph.io import container_integrity as integrity
    monkeypatch.setattr(integrity, "_BLOCK_BYTES", 32)
    values = np.arange(105, dtype=np.int64).reshape(3, 5, 7)
    class Array:
        attrs = {}
        shape = values.shape
        dtype = values.dtype
        calls = []
        def __getitem__(self, selection):
            chunk = values[selection]
            self.calls.append(chunk.nbytes)
            return chunk
    first = Array()
    second = Array()
    assert container_digest(first) == container_digest(second)
    assert first.calls and max(first.calls) <= 32
