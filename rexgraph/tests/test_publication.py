"""A failed native write must preserve the last complete publication."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.io.publication import staged_publication
from rexgraph.state import to_state


@pytest.fixture(params=["hdf5", "zarr"])
def backend(request):
    pytest.importorskip({"hdf5": "h5py", "zarr": "zarr"}[request.param])
    if request.param == "hdf5":
        from rexgraph.io.hdf5_format import RexHDF5Format
        return RexHDF5Format(), ".h5"
    from rexgraph.io.zarr_format import RexZarrFormat
    return RexZarrFormat(), ".zarr"


def graph(n=1):
    return RexGraph.from_graph(list(range(n)), list(range(1, n+1)))


def test_encoding_failure_preserves_previous_graph(backend, tmp_path):
    fmt, suffix = backend
    path = tmp_path / ("graph"+suffix)
    original = graph()
    fmt.write(path, original)
    broken = graph(2)
    broken.set_provenance({"unsupported": object()})
    with pytest.raises(TypeError, match="unsupported native value"):
        fmt.write(path, broken)
    assert to_state(fmt.read(str(path))).header["digest"] == to_state(original).header["digest"]
    assert list(tmp_path.iterdir()) == [path]


def test_partial_storage_failure_preserves_previous_graph(backend, tmp_path, monkeypatch):
    fmt, suffix = backend
    path = tmp_path / ("graph"+suffix)
    fmt.write(path, graph())
    store = fmt._store
    count = 0
    def fail_after_first(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("storage failure")
        return store(*args)
    monkeypatch.setattr(fmt, "_store", fail_after_first)
    with pytest.raises(OSError, match="storage failure"):
        fmt.write(path, graph(2))
    assert fmt.read(str(path)).nE == 1
    assert list(tmp_path.iterdir()) == [path]


def test_group_failure_preserves_all_existing_groups(backend, tmp_path):
    fmt, suffix = backend
    path = tmp_path / ("groups"+suffix)
    fmt.write_to_group(path, "first", graph())
    fmt.write_to_group(path, "second", graph(2))
    with pytest.raises(TypeError, match="Unsupported type"):
        fmt.write_to_group(path, "first", object())
    assert fmt.read_from_group(str(path), "first").nE == 1
    assert fmt.read_from_group(str(path), "second").nE == 2
    assert sorted(fmt.list_groups(str(path))) == ["first", "second"]
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_group_updates_preserve_both_writers(backend, tmp_path):
    fmt, suffix = backend
    path = tmp_path / ("groups"+suffix)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fmt.write_to_group, path, str(i), graph(i+1)) for i in range(2)]
        for future in futures:
            future.result(timeout=30)
    assert sorted(fmt.list_groups(str(path))) == ["0", "1"]
    assert [fmt.read_from_group(str(path), str(i)).nE for i in range(2)] == [1, 2]


@pytest.mark.parametrize("directory", [False, True])
def test_rename_failure_restores_previous_publication(directory, tmp_path, monkeypatch):
    path = tmp_path / "target"
    if directory:
        path.mkdir()
        (path / "value").write_text("old")
    else:
        path.write_text("old")
    replace = os.replace
    def fail_staged(source, destination):
        if ".tmp-" in Path(source).name:
            raise OSError("rename failure")
        replace(source, destination)
    monkeypatch.setattr(os, "replace", fail_staged)
    with pytest.raises(OSError, match="rename failure"):
        with staged_publication(path, directory=directory) as staged:
            (staged / "value" if directory else staged).write_text("new")
    assert (path / "value" if directory else path).read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_rollback_retains_recoverable_previous_directory(tmp_path, monkeypatch):
    path = tmp_path / "target"
    path.mkdir()
    (path / "value").write_text("old")
    replace = os.replace
    def fail_after_backup(source, destination):
        if ".tmp-" in Path(source).name:
            # A concurrent uncoordinated writer also creates a destination.
            path.mkdir()
            raise OSError("publish failure")
        if Path(source).name == "previous":
            raise OSError("rollback failure")
        replace(source, destination)
    monkeypatch.setattr(os, "replace", fail_after_backup)
    with pytest.raises(OSError, match="rollback failure"):
        with staged_publication(path, directory=True) as staged:
            (staged / "value").write_text("new")
    recovery, = tmp_path.glob(".target.recovery-*")
    assert (recovery / "previous" / "value").read_text() == "old"
    assert not list(tmp_path.glob(".target.tmp-*"))


def test_array_writer_accepts_named_arguments_and_path_objects(tmp_path):
    pytest.importorskip("h5py")
    from rexgraph.io.hdf5_format import save_hdf5_array, load_hdf5_array
    path = tmp_path / "array"
    save_hdf5_array(arr=np.array([3, 4]), path=path)
    assert load_hdf5_array(str(path)+".h5").tolist() == [3, 4]


@pytest.mark.parametrize("directory", [False, True])
def test_replacement_preserves_permissions(directory, tmp_path):
    path = tmp_path / "target"
    if directory:
        path.mkdir()
    else:
        path.touch()
    path.chmod(0o700 if directory else 0o600)
    with staged_publication(path, directory=directory) as staged:
        (staged / "value" if directory else staged).write_text("new")
    assert path.stat().st_mode & 0o777 == (0o700 if directory else 0o600)
