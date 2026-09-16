"""The planned tensor manifest name shares the native bounded catalog reader."""
from dataclasses import replace

import numpy as np
import pytest

from rexgraph.io.catalog import FileCatalog
from rcql import BoundSource, Executor, SourcePolicy, lookup, parse
from rcql.operators import get_operator


@pytest.fixture
def catalog(tmp_path):
    from safetensors.numpy import save_file
    save_file({"a": np.zeros(2, np.int64), "b": np.ones((2, 3), np.float32)}, str(tmp_path / "m.safetensors"))
    return FileCatalog([tmp_path])


def test_alias_has_one_reader_and_one_contract(catalog, monkeypatch):
    assert get_operator("TENSOR_MANIFEST").fn is get_operator("TENSORS").fn
    assert replace(lookup("TENSOR_MANIFEST"), name="TENSORS") == lookup("TENSORS")
    monkeypatch.setattr(catalog, "load", lambda *a: pytest.fail("payload loaded"))
    result = Executor(sources={"c": catalog}).execute(parse(
        'FROM CATALOG("c") RETURN TENSOR_MANIFEST(name="root0/m.safetensors",limit=1),TENSORS("root0/m.safetensors",1)'))
    assert result.values[0] == result.values[1] == [{"name": "a", "shape": [2], "dtype": "I64"}]


@pytest.mark.parametrize("permissions", [(), ("files",), ("file_read",), ("files", "search")])
@pytest.mark.parametrize("explain", [False, True])
def test_both_metadata_permissions_required(catalog, permissions, explain, monkeypatch):
    monkeypatch.setattr(catalog, "tensors", lambda *a, **kw: pytest.fail("reader reached"))
    engine = Executor(sources={"c": BoundSource(catalog, SourcePolicy.allow(*permissions))})
    with pytest.raises(PermissionError):
        engine.execute(replace(parse('FROM $c RETURN TENSOR_MANIFEST("root0/m.safetensors")'), explain=explain))


def test_explain_is_not_header_io(catalog, monkeypatch):
    monkeypatch.setattr(catalog, "tensors", lambda *a, **kw: pytest.fail("header read"))
    result = Executor(sources={"c": catalog}).execute(parse(
        'EXPLAIN FROM $c RETURN TENSOR_MANIFEST("root0/m.safetensors")'))
    assert result.execution == ()


@pytest.mark.parametrize("expression", ['TENSOR_MANIFEST()', 'TENSOR_MANIFEST(1)',
    'TENSOR_MANIFEST("root0/m.safetensors",true)', 'TENSOR_MANIFEST("root0/m.safetensors",1.5)'])
def test_bad_arguments_refused_before_read(catalog, expression, monkeypatch):
    monkeypatch.setattr(catalog, "tensors", lambda *a, **kw: pytest.fail("reader reached"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"c": catalog}).execute(parse(f'FROM $c RETURN {expression}'))


def test_registered_catalog_names_only(catalog):
    engine = Executor(sources={"c": catalog})
    for name in ("/etc/passwd", "../m.safetensors", "root0/../m.safetensors"):
        with pytest.raises((KeyError, ValueError)):
            engine.execute(parse(f'FROM $c RETURN TENSOR_MANIFEST("{name}")'))


def test_catalog_reopen_and_native_rcdb_export(tmp_path):
    from contextlib import closing
    import rcdb
    from rexgraph import RexGraph
    from rexgraph.io import save
    r = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]])
    with closing(rcdb.open_store(f"rex://{tmp_path / 'db'}")) as db:
        db.put("r", r)
        save(str(tmp_path / "state.safetensors"), db.get("r"))
    first = FileCatalog([tmp_path])
    query = parse('FROM $c RETURN TENSOR_MANIFEST("root0/state.safetensors"),TENSORS("root0/state.safetensors")')
    before = Executor(sources={"c": first}).execute(query).values
    after = Executor(sources={"c": FileCatalog([tmp_path])}).execute(query).values
    assert before == after and before[0] == before[1] and before[0]


def test_query_search_reaches_names_beyond_listing_limit(tmp_path):
    from safetensors.numpy import save_file
    save_file({**{f"a{i:04d}": np.zeros(1) for i in range(1001)}, "zz_target": np.ones(1)}, str(tmp_path / "m.safetensors"))
    result = Executor(sources={"c": FileCatalog([tmp_path])}).execute(parse(
        'FROM $c RETURN SEARCH_TENSORS("root0/m.safetensors","target",1)'))
    assert result.values == ([{"name": "zz_target", "shape": [1], "dtype": "F64"}],)
