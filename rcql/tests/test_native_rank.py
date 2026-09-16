"""Full tower exact rank/homology query contracts and native method provenance."""
import builtins

import pytest
from rexgraph import RexGraph
from rexgraph.core import _sparse
from rexgraph.linear_operator import RexOperator, boundary_operator
from rexgraph.native_sparse import empty_native

from rcql import Executor, parse
from rcql.types import Exactness


def _run(rex, expression, *, explain=False, **params):
    return Executor(sources={"r": rex}, params=params).execute(parse(
        ("EXPLAIN " if explain else "") + "FROM $r RETURN " + expression))


def test_full_grade_four_queries_without_scipy(monkeypatch):
    rex = RexGraph.from_graph([0], [1])
    rex._graded_duals = [empty_native((0, 3)).dual,
        _sparse.dual_from_coo([0, 1], [0, 0], [-1., 1.], 3, 1)]
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            pytest.fail("native RCQL rank imported SciPy")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = _run(rex, "RANK(1), NULLITY(3), RANK(BOUNDARY(4)), NULLITY(COBOUNDARY(4)), "
                      "BETTI(0), BETTI(1), BETTI(2), BETTI(3), BETTI(4)")
    assert result.values == (1, 3, 1, 1, 1, 0, 0, 2, 0)
    assert result.exactness == (Exactness.INTEGER,)*9
    events = [e["methods"][0] for e in result.execution if e["operator"] == "BETTI"]
    assert events[0]["cache_hit"] is False
    assert all(e["cache_hit"] is True and e["algorithms"] == [] for e in events[1:])
    assert all(e["chain_condition"] == "exact-zero" for e in events)


def test_explain_checks_domain_and_chain_without_computing_any_rank(monkeypatch):
    import rexgraph.graded_boundary as gb
    import rexgraph.native_rank as nr
    rex = RexGraph.from_hypergraph([0, 4], [0, 1, 2, 3])
    def forbidden(*args, **kwargs):
        pytest.fail("EXPLAIN ran rank reduction")
    monkeypatch.setattr(gb, "_rank_integer_columns", forbidden)
    monkeypatch.setattr(nr, "_rank_integer_columns", forbidden)
    monkeypatch.setattr(gb, "_exact_column_rank_reduction", forbidden)
    result = _run(rex, "RANK(1), NULLITY(1), BETTI(0)", explain=True)
    nodes = [n for n in result.native_plan["nodes"] if n.get("operator") in {"RANK", "NULLITY", "BETTI"}]
    assert [n["physical"]["method"] for n in nodes] == ["certified-exact-rank", "certified-exact-rank", "native-exact-betti"]
    assert "_betti_tower_reading" not in rex.__dict__


@pytest.mark.parametrize("bad", ["domain", "chain", "shape"])
@pytest.mark.parametrize("explain", [False, True])
def test_invalid_homology_refused_before_adapter(bad, explain, monkeypatch):
    import rcql.executor
    rex = RexGraph(sources=[0, 1, 2], targets=[1, 2, 0],
        B2_col_ptr=[0, 3, 6], B2_row_idx=[0, 1, 2]*2, B2_vals=[1.]*6)
    rex._graded_duals = [_sparse.dual_from_coo([0, 1], [0, 0],
        [1., -.999999 if bad == "domain" else 1.], 3 if bad == "shape" else 2, 1)]
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter resolved"))
    with pytest.raises(ValueError, match="integer higher|chain condition|grade axes"):
        _run(rex, "BETTI(3)", explain=explain)


@pytest.mark.parametrize("name", ["RANK", "NULLITY"])
@pytest.mark.parametrize("explain", [False, True])
def test_higher_rank_noninteger_domain_refused_before_adapter(name, explain, monkeypatch):
    import rcql.executor
    rex = RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    rex._graded_duals = [_sparse.dual_from_coo([0], [0], [1.0000001], 1, 1)]
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter resolved"))
    with pytest.raises(ValueError, match="integer higher"):
        _run(rex, f"{name}(3)", explain=explain)


def test_rank_method_records_structural_elimination_and_memo_branches():
    import rexgraph.graded_boundary as gb
    gb._RANK_MEMO.clear()
    rex = RexGraph.from_hypergraph([0, 4], [0, 1, 2, 3])
    result = _run(rex, "RANK(1), NULLITY(1)")
    assert result.values == (1, 0)
    assert [e["methods"][0]["algorithm"] for e in result.execution] == [
        "sparse-integer-elimination", "exact-rank-content-memo"]
    rex = RexGraph.from_hypergraph([0, 3, 5, 7], [0, 1, 2, 0, 1, 0, 2])
    assert _run(rex, "RANK(1)").execution[-1]["methods"][0]["algorithm"] == "spanned-branching-union-find"


def test_operator_name_does_not_infer_rank_of_the_source_boundary():
    import scipy.sparse as sp
    rex = RexGraph.from_graph([0, 1], [1, 2])
    matrix = sp.csr_matrix((3, 2))
    operator = RexOperator("B1", (3, 2), 1, 0, lambda v: matrix @ v,
        matrix=matrix, source=rex, construction="boundary", variance="chain")
    result = _run(rex, "RANK($op), NULLITY($op), RANK(1)", op=operator)
    assert result.values == (0, 2, 2)
    assert result.execution[0]["methods"][0]["method"] == "explicit-matrix-exact-rank"


def test_rank_and_betti_cache_invalidate_after_primary_mutation():
    rex = RexGraph.from_graph([0, 1], [1, 2])
    before = _run(rex, "BETTI(1), NULLITY(1), RANK(1)")
    assert before.values == (0, 0, 2)
    rex.add_edges([2], [0])
    after = _run(rex, "BETTI(1), NULLITY(1), RANK(1)")
    assert after.values == (1, 1, 2)
    assert after.execution[0]["methods"][0]["cache_hit"] is False
    assert rex.betti_tower == (1, 1)
    assert rex.betti == (1, 1, 0)


@pytest.mark.parametrize("expression", ["RANK(true)", "NULLITY(false)", "BETTI(-1)", "BETTI(3)"])
def test_native_rank_grade_contracts(expression):
    with pytest.raises((TypeError, ValueError)):
        _run(RexGraph.from_graph([0], [1]), expression)


def test_operator_rank_source_identity_is_not_rebound():
    rex = RexGraph.from_graph([0], [1])
    foreign = RexGraph.from_graph([0], [1])
    with pytest.raises((TypeError, ValueError)):
        _run(rex, "RANK($op)", op=boundary_operator(foreign, 1))
