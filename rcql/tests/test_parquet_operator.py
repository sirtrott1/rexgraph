"""Parquet names use the Core exporter, not a second tabular state codec."""
from dataclasses import replace

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.partition_state import build_rex_partition
from rcql import ArtifactServices, BoundSource, Executor, SourcePolicy, parse


def setup(columns=None):
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2]]], relation_ids=[7, 13])
    partition = build_rex_partition(rex, [1, 1])
    columns = {"id": np.array([7, 13]), "value": np.array([[1., 2.], [3., 4.]])} if columns is None else columns
    return rex, partition, columns


def test_core_parity_members_and_schema():
    pytest.importorskip("pyarrow")
    from rexgraph.io.export import ExportManifest, export_parquet, verify_export
    rex, partition, columns = setup()
    engine = Executor(sources={"r": rex}, params={"p": partition, "columns": columns})
    out = engine.execute(parse('FROM $r LET a=EXPORT_PARQUET(columns=$columns,partition=$p) '
                               'RETURN a, a.payload, a.manifest, a.digest, HASH(a.payload,"bytes")')).values
    payload, manifest = export_parquet(columns, partition_digest=partition.digest)
    assert out[1] == payload and out[3] == manifest.digest and out[4] == manifest.payload_sha256
    assert verify_export(payload, ExportManifest(**out[2]))
    assert out[0] == {"payload": out[1], "manifest": out[2], "digest": out[3]}


def test_explicit_encryption_composes_without_another_export_implementation():
    pytest.importorskip("pyarrow")
    pytest.importorskip("cryptography")
    from rexgraph.io.security import StaticKeyProvider
    rex, partition, columns = setup()
    e = Executor(sources={"r": rex}, params={"p": partition, "c": columns},
        artifacts=ArtifactServices(keys=StaticKeyProvider({"key": b"K"*32})))
    result = e.execute(parse('FROM $r LET a=EXPORT_PARQUET($c,$p) '
        'LET b=ENCRYPT(a.payload,"key","Parquet") RETURN DECRYPT(b),a.payload')).values
    assert result[0] == result[1]


def test_explain_never_loads_arrow_or_encodes(monkeypatch):
    import builtins
    import rexgraph.io.export as core
    original = builtins.__import__
    def blocked(name, *args, **kw):
        if name.startswith("pyarrow"):
            pytest.fail("optional import during explain")
        return original(name, *args, **kw)
    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setattr(core, "export_parquet", lambda *a, **kw: pytest.fail("exported"))
    rex, p, c = setup()
    result = Executor(sources={"r": rex}, params={"p": p, "c": c}).execute(
        parse('EXPLAIN FROM $r RETURN EXPORT_PARQUET($c,$p).payload'))
    assert result.execution == ()


@pytest.mark.parametrize("columns", [{"a": [1], "b": [1, 2]}, {"a": 1}, {1: [1]}, {"a": np.zeros((2, 0))}])
@pytest.mark.parametrize("explain", [False, True])
def test_column_shape_failures_precede_execution(columns, explain, monkeypatch):
    import rcql.executor
    rex, p, c = setup(columns)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": rex}, params={"p": p, "c": c}).execute(replace(
            parse('FROM $r RETURN EXPORT_PARQUET($c,$p)'), explain=explain))


@pytest.mark.parametrize("mode", ["stale", "foreign", "permission"])
def test_lineage_and_policy_are_not_grants(mode):
    rex, p, c = setup()
    if mode == "stale":
        p.rex.set_cell_attrs([0], w_E=[4])
    bound = RexGraph.from_cells([2, [[0, 1]]]) if mode == "foreign" else rex
    if mode == "permission":
        bound = BoundSource(rex, SourcePolicy.allow("read"))
    with pytest.raises((TypeError, ValueError, PermissionError)):
        Executor(sources={"r": bound}, params={"p": p, "c": c}).execute(
            parse('FROM $r RETURN EXPORT_PARQUET($c,$p)'))
