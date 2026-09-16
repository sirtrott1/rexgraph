"""Native overlap planning, coefficient contracts and RCDB round trips."""
from contextlib import closing
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rcql import Executor, parse
from rexgraph import RexGraph
from rexgraph.cochain import Cochain
from rexgraph.text_overlap import TextOverlapView


def fixture():
    return RexGraph.from_cells([4, [[0, 1, 2], [2, 3], [1]]], w_E=[Q(2), Q(3), Q(5)])


def test_query_actions_and_parameter_descriptors_match_core():
    r = fixture()
    view = TextOverlapView(r)
    x = Cochain(1, np.array([1, 2, 3], object), source=r)
    result = Executor(sources={"r": r}, params={"x": x, "a": view}).execute(parse(
        'FROM $r LET a=TEXT_OVERLAP_VIEW() RETURN APPLY(a,$x,true),'
        'APPLY(ADJOINT(a),$x,true),APPLY($a,$x,false)'))
    for index, value in enumerate(result.values):
        expected = view.apply(x.values, exact=index < 2)
        np.testing.assert_array_equal(value.values, expected)
        assert value.grade == 1 and value.source is r


def test_explain_does_not_build_a_tensor_or_solve(monkeypatch):
    monkeypatch.setattr(TextOverlapView, "__init__", lambda *a, **kw: pytest.fail("view built"))
    out = Executor(sources={"r": fixture()}).execute(parse('EXPLAIN FROM $r RETURN TEXT_OVERLAP_VIEW()'))
    assert not out.execution
    node = next(n for n in out.native_plan["nodes"] if n.get("operator") == "TEXT_OVERLAP_VIEW")
    assert node["physical"]["method"] == "core-primary-text-overlap"


@pytest.mark.parametrize("expression", [
    'TEXT_OVERLAP_VIEW(policy="tensor")', 'TEXT_OVERLAP_VIEW("sentence")',
    'APPLY(TEXT_OVERLAP_VIEW(),INDICATOR(CELL(0,0)),true)',
    'PAGERANK(TEXT_OVERLAP_VIEW())'])
@pytest.mark.parametrize("explain", [False, True])
def test_invalid_contracts_fail_before_adapters(expression, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError, SyntaxError)):
        Executor(sources={"r": fixture()}).execute(replace(parse(f'FROM $r RETURN {expression}'), explain=explain))


def test_foreign_view_is_refused():
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": fixture()}, params={"a": TextOverlapView(fixture())}).execute(
            parse('FROM $r RETURN APPLY($a,INDICATOR(CELL(1,0)),true)'))


def test_reopened_document_uses_original_relation_axis(tmp_path):
    import rcdb
    from rexgraph.sectioning import add_sectioning, add_coarsening
    r = fixture()
    add_sectioning(r, "span", {"span0": [0], "span1": [1, 2]})
    add_coarsening(r, "sentence", "span", [0, 0], ["sentence0"])
    uri = f"rex://{tmp_path / 'store'}"
    with closing(rcdb.open_store(uri)) as store:
        store.put("document", r)
    with closing(rcdb.open_store(uri)) as store:
        out = Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_GET($db,"document") RETURN APPLY(TEXT_OVERLAP_VIEW(),'
            'INDICATOR(CELL(1,0)),true),PAGERANK(MARKOV_VIEW())'))
    np.testing.assert_array_equal(out.values[0].values,
                                  TextOverlapView(r).apply(np.array([1, 0, 0], object), exact=True))
    assert len(out.values[0].values) == 3
    assert sum(out.values[1].values) == pytest.approx(1)
