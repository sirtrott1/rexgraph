"""SQL exports share exact declarations, transactions and native state identity."""
from fractions import Fraction as Q

import numpy as np
import pytest

sa = pytest.importorskip("sqlalchemy")
from rexgraph import Absent, ExactArray, Relations, RexGraph, VertexTable
from rexgraph.state import to_state
from rexgraph.io import sql_bridge as sql


@pytest.fixture
def engine(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'exact.db'}")
    yield engine
    engine.dispose()


def graph():
    rex = RexGraph.from_relations(Relations.from_supports(
        [[0, 1, 2], [2, 3]], heads=[1, 0], shares=[Q(1, 3), 0, Q(2, 3), Absent, Absent],
        vertices=VertexTable(("a", "b", "c", "d", "isolated")),
        weights=[Q(2, 7), Absent], signs=[-1, 1], relation_ids=["branch", "pair"],
        attributes={1: {0: {"assertion": [Q(1, 7), Absent]}}}),
        g_channel="normalized", c_channel="count")
    return rex


def test_sql_boundary_restores_complete_declared_state(engine):
    original = graph()
    sql.write_boundary_sql(original, engine, "boundary")
    restored = sql.reconstruct_rex_sql(engine, boundary="boundary")
    assert to_state(restored).header["digest"] == to_state(original).header["digest"]
    data = sql.read_boundary_sql(engine, "boundary")
    assert data["coefficient"].tolist() == [Q(1, 3), Q(-1), Q(2, 3), Q(-1), Q(1)]
    assert data["head"].tolist() == [False, True, False, True, False]
    assert data["weight"].tolist() == [Q(2, 7)]*3 + [Absent]*2
    assert data["sign"].tolist() == [-1]*3 + [1]*2


def test_empty_declared_basis_with_isolates_roundtrips(engine):
    original = RexGraph.from_relations(Relations.from_supports([], vertices=VertexTable(("isolated",))))
    sql.write_boundary_sql(original, engine)
    assert to_state(sql.reconstruct_rex_sql(engine, boundary="boundary")).header["digest"] == to_state(original).header["digest"]


def test_exact_metrics_bigints_absence_matrices_projection_and_batches(engine):
    values = np.array([Q(1, 7), Absent, 2**2000+1, Q(-3, 5)], dtype=object)
    matrix = ExactArray.from_values([[Q(1, 3), Absent], [2**1500, 0], [Q(2, 7), Q(1)], [1, 2]])
    sql.write_metrics_sql({"value": values, "matrix": matrix, "plain": np.arange(4, dtype=np.int16)}, engine)
    restored = sql.read_metrics_sql(engine)
    assert restored["value"].tolist() == values.tolist()
    assert restored["matrix"].tolist() == matrix.values().tolist()
    assert restored["plain"].dtype == np.int16
    projected = sql._read_table(engine, "metrics", columns=["value"])
    assert list(projected) == ["value"] and projected["value"].tolist() == values.tolist()
    batches = list(sql.read_sql_batches(engine, "metrics", chunksize=3))
    assert [len(batch["value"]) for batch in batches] == [3, 1]
    assert np.concatenate([batch["value"] for batch in batches]).tolist() == values.tolist()
    assert np.concatenate([batch["matrix"] for batch in batches]).tolist() == matrix.values().tolist()


def test_uint64_values_above_sql_signed_width_use_exact_escape(engine):
    values = np.array([2**63+1, 2**64-1], np.uint64)
    sql.write_metrics_sql({"value": values}, engine)
    assert sql.read_metrics_sql(engine)["value"].tolist() == values.tolist()


def test_sql_failure_after_table_replacement_rolls_back_old_state(engine, monkeypatch):
    original = graph()
    sql.write_boundary_sql(original, engine)
    def fail(*args):
        raise OSError("state storage failure")
    monkeypatch.setattr(sql, "_write_sql_state", fail)
    with pytest.raises(OSError, match="storage failure"):
        sql.write_boundary_sql(RexGraph.from_graph([0], [1]), engine)
    assert to_state(sql.reconstruct_rex_sql(engine, boundary="boundary")).header["digest"] == to_state(original).header["digest"]


def test_sql_writer_participates_in_caller_transaction(engine):
    original = graph()
    sql.write_boundary_sql(original, engine)
    with pytest.raises(RuntimeError, match="caller rollback"):
        with engine.begin() as connection:
            sql.write_boundary_sql(RexGraph.from_graph([0], [1]), connection)
            raise RuntimeError("caller rollback")
    assert to_state(sql.reconstruct_rex_sql(engine, boundary="boundary")).header["digest"] == to_state(original).header["digest"]


def test_tampered_projection_cannot_override_its_native_basis(engine):
    sql.write_boundary_sql(graph(), engine)
    with engine.begin() as connection:
        table = sa.Table("boundary", sa.MetaData(), autoload_with=connection)
        connection.execute(table.update().where(table.c.edge_idx == 0).values(vertex_idx=4))
    with pytest.raises(ValueError, match="projection disagrees"):
        sql.reconstruct_rex_sql(engine, boundary="boundary")


@pytest.mark.parametrize("field, value", [("presence", 2), ("kind", 256), ("denominator", 0)])
def test_invalid_exact_sql_coefficients_are_not_narrowed(engine, field, value):
    sql.write_metrics_sql({"value": [Q(1, 7)]}, engine)
    with engine.begin() as connection:
        connection.exec_driver_sql('UPDATE metrics SET "value__rex_'+field+'" = ?', (value,))
    with pytest.raises(ValueError, match="exact|denominator"):
        sql.read_metrics_sql(engine)


def test_sql_exact_append_checks_schema_and_retains_old_values(engine):
    sql.write_metrics_sql({"value": [Q(1, 7)]}, engine)
    sql.write_metrics_sql({"value": [Q(2, 7), Absent]}, engine, if_exists="append")
    assert sql.read_metrics_sql(engine)["value"].tolist() == [Q(1, 7), Q(2, 7), Absent]
    with pytest.raises(ValueError, match="schema disagrees"):
        sql.write_metrics_sql({"value": [1.]}, engine, if_exists="append")
    assert sql.read_metrics_sql(engine)["value"].tolist() == [Q(1, 7), Q(2, 7), Absent]
