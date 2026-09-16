"""Native file navigation and recoverable RCQL publication."""
from fractions import Fraction

import pytest

from rcql import Executor, parse, mutation, call, param, BoundSource, SourcePolicy
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.io import save, load
from rexgraph.io.catalog import FileCatalog, object_digest


@pytest.mark.parametrize("suffix", ["rcbd", "rex", "safetensors", "h5", "hdf5", "zarr"])
@pytest.mark.parametrize("temporal", [False, True])
def test_file_replacement_is_exact_recoverable_and_queryable(tmp_path, suffix, temporal):
    pytest.importorskip({"rcbd": "safetensors", "rex": "safetensors", "safetensors": "safetensors", "h5": "h5py", "hdf5": "h5py", "zarr": "zarr"}[suffix])
    first = RexGraph.from_graph([0], [1])
    candidate = RexGraph.from_hypergraph([0, 3], [0, 1, 2], w_E=[Fraction(1, 3)])
    if temporal:
        history = TemporalRex([], general=True)
        history.append_snapshot(candidate, at=7)
        candidate = history
    name = f"root0/test.{suffix}"
    path = tmp_path / f"test.{suffix}"
    save(str(path), first)
    catalog = FileCatalog([tmp_path])
    before = catalog.hash(name)
    executor = Executor(sources={"files": catalog}, params={"candidate": candidate})
    text = f'FROM CATALOG("files") MUTATE "{name}" SET state=$candidate, expected_hash="{before}" COMMIT'
    assert parse(text) == mutation(source_expr=call("CATALOG", "files"), record_id=name,
                                   resulting=param("candidate"), expected_hash=before)
    explanation = executor.execute(parse("EXPLAIN " + text))
    assert not explanation.execution and catalog.hash(name) == before
    result = executor.execute(parse(text))
    record = result.values[0]
    assert record.version is None  # no fabricated RCDB version on a plain file
    assert record.previous_hash == before and record.sha256 != before
    assert object_digest(load(str(path))) == object_digest(candidate)
    backup = tmp_path / record.backup.split("/", 1)[1]
    assert object_digest(load(str(backup))) == object_digest(first)
    assert len(catalog.list()) == 1  # recovery copies are not discovered as live data
    reopened = FileCatalog([tmp_path])
    read = Executor(sources={"files": reopened}).execute(parse(
        f'FROM FILE("files","{name}") AS document RETURN document, STATE_HASH()'))
    assert object_digest(read.values[0]) == read.values[1] == object_digest(candidate)
    with pytest.raises(ValueError, match="expected_hash"):
        executor.execute(parse(text))
    assert object_digest(load(str(path))) == object_digest(candidate)


@pytest.mark.parametrize("fields", ['expected_version=1', 'valid_from=1', 'expected_hash="bad"'])
def test_file_mutation_fields_fail_during_explain(tmp_path, fields):
    rex = RexGraph.from_graph([0], [1])
    catalog = FileCatalog([tmp_path])
    with pytest.raises(ValueError):
        Executor(sources={"files": catalog}, params={"r": rex}).execute(parse(
            f'EXPLAIN FROM $files MUTATE "root0/missing.rcbd" SET state=$r, {fields} COMMIT'))
    assert not list(tmp_path.iterdir())


def test_file_publication_requires_file_write_before_io(tmp_path, monkeypatch):
    rex = RexGraph.from_graph([0], [1])
    catalog = FileCatalog([tmp_path])
    def forbidden(*args, **kwargs):
        pytest.fail("denied mutation reached file publication")
    monkeypatch.setattr(catalog, "commit_mutation", forbidden)
    bound = BoundSource(catalog, SourcePolicy.allow("identity", "mutate"))
    with pytest.raises(PermissionError):
        Executor(sources={"f": bound}, params={"r": rex}).execute(parse(
            'FROM $f MUTATE "root0/a.rcbd" SET state=$r COMMIT'))


def test_whole_source_alias_checks_read_capability():
    rex = RexGraph.from_graph([0], [1])
    bound = BoundSource(rex, SourcePolicy.allow("identity"))
    with pytest.raises(PermissionError):
        Executor(sources={"r": bound}).execute(parse('FROM $r AS r RETURN r'))


@pytest.mark.parametrize('suffix', ['rcbd', 'safetensors'])
def test_failed_publication_restores_original(tmp_path, monkeypatch, suffix):
    import os
    first = RexGraph.from_graph([0], [1])
    path = tmp_path / f'original.{suffix}'
    save(str(path), first)
    catalog = FileCatalog([tmp_path])
    name = f'root0/original.{suffix}'
    before = catalog.hash(name)
    replace = os.replace
    def fail_staged(source, target):
        from pathlib import Path
        if Path(source).name == f'next.{suffix}' and Path(target) == path:
            raise OSError('injected publication failure')
        return replace(source, target)
    monkeypatch.setattr(os, 'replace', fail_staged)
    with pytest.raises(OSError, match='injected'):
        catalog.commit_mutation(name, RexGraph.from_graph([0, 1], [1, 2]), expected_hash=before)
    assert catalog.hash(name) == before
    assert object_digest(load(str(path))) == object_digest(first)
