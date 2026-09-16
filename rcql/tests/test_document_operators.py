"""Typed primary responses and section identity after RCDB reopening."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.partition import document_field, section_response
from rexgraph.sectioning import add_sectioning, add_coarsening, sectionings_of
from rexgraph.tower import semantic_closure
from rcql import Executor, BoundSource, SourcePolicy, parse


def fixture():
    r = RexGraph.from_cells([4, [[0, 1, 2], [0, 0], [2], [1, 3]]], relation_ids=[2, 3, 5, 7])
    add_sectioning(r, "sentence", {"a": [0, 1], "b": [2, 3]})
    add_coarsening(r, "chapter", "sentence", [0, 0], ["all"])
    return r


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("reading", ["mass", "coverage"])
@pytest.mark.parametrize("weight", ["flat", "invdeg"])
def test_field_and_section_contracts_match_core(exact, reading, weight):
    r = fixture()
    e = Executor(sources={"r": r}, params={"exact": exact, "reading": reading, "weight": weight})
    out = e.execute(parse('FROM $r LET x=DOCUMENT_FIELD(CELLS(0,[0,1]),reading=$reading,seed_weight=$weight,exact=$exact) '
        'LET s=SECTION_RESPONSE("sentence",[0,1],$reading,$weight,$exact) '
        'RETURN x,s,s.scores,s.labels,COUNT(s.scores),QUADRANCE(x,$exact)'))
    field = document_field(r, [0, 1], reading=reading, seed_weight=weight, exact=exact)
    scores, labels = section_response(r, sectionings_of(r)["sentence"], [0, 1],
                                     propagator=reading, seed_weight=weight, exact=exact)
    np.testing.assert_array_equal(out.values[0].numpy(), field.numpy())
    assert out.values[0].source is r and out.values[0].grade == 1
    assert out.values[2:5] == (tuple(scores), tuple(labels), 2)
    assert out.values[-1] == sum(v*v for v in field.numpy())
    assert out.exactness[0].value == ("rational" if exact else "approximate")


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("expression", [
    'DOCUMENT_FIELD([0.2])', 'DOCUMENT_FIELD([true])', 'DOCUMENT_FIELD([-1])',
    'DOCUMENT_FIELD([4])', 'DOCUMENT_FIELD([0],"similarity")',
    'DOCUMENT_FIELD([0],seed_weight="random")', 'DOCUMENT_FIELD(CELLS(1))',
    'SECTION_RESPONSE("missing",[0])', 'SECTION_RESPONSE("sentence",[0],exact=1)',
    'SEMANTIC_CLOSURE(-1)', 'SEMANTIC_CLOSURE(0,max_depth=0)', 'SEMANTIC_CLOSURE(0,grade=1)',
    'CLOSURE(0,max_depth=0)'])
def test_invalid_contracts_fail_before_adapter(expression, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        Executor(sources={"r": fixture()}).execute(replace(parse(f'FROM $r RETURN {expression}'), explain=explain))


def test_explain_does_not_run_response_or_closure(monkeypatch):
    import rexgraph.partition as core
    import rexgraph.tower as tower
    monkeypatch.setattr(core, "document_field", lambda *a, **k: pytest.fail("response ran"))
    monkeypatch.setattr(core, "section_response", lambda *a, **k: pytest.fail("response ran"))
    monkeypatch.setattr(tower, "semantic_closure", lambda *a, **k: pytest.fail("closure ran"))
    result = Executor(sources={"r": fixture()}).execute(parse(
        'EXPLAIN FROM $r RETURN DOCUMENT_FIELD([0]),SECTION_RESPONSE("sentence",[0]),SEMANTIC_CLOSURE(0)'))
    assert not result.execution


def test_semantic_closure_is_the_existing_core_rule_with_full_grades():
    r = RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]], [[(0, 1), (1, 1), (2, -1)]]])
    out = Executor(sources={"r": r}).execute(parse(
        'FROM $r LET c=SEMANTIC_CLOSURE(0) RETURN c,CLOSURE(0),c.converged,c.relations,c.steps'))
    expected = semantic_closure(r, 0)
    assert out.values == (expected, expected, True, [0, 1, 2], expected["steps"])
    assert expected["steps"][0]["betti"] == [1, 0, 0]


def test_section_labels_require_identity_permission():
    r = fixture()
    e = Executor(sources={"r": BoundSource(r, SourcePolicy.allow("read"))})
    assert e.execute(parse('FROM $r RETURN DOCUMENT_FIELD([0])')).values[0].grade == 1
    with pytest.raises(PermissionError):
        e.execute(parse('EXPLAIN FROM $r RETURN SECTION_RESPONSE("sentence",[0])'))


def test_reopened_section_layers_and_rational_shares(tmp_path):
    import rcdb
    uri = f"rex://{tmp_path / 'store'}"
    r = fixture()
    store = rcdb.open_store(uri)
    Executor(sources={"db": store}, params={"r": r}).execute(parse(
        'FROM $db MUTATE "doc" SET state=$r,actor="Art" COMMIT'))
    store.close()
    store = rcdb.open_store(uri)
    try:
        result = Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_GET($db,"doc") RETURN DOCUMENT_FIELD([0]),SECTION_RESPONSE("chapter",[0])'))
        assert result.values[0].numpy().tolist() == [Q(1, 3), Q(0), Q(0), Q(0)]
        assert result.values[1]["scores"] == (Q(1, 3),) and result.values[1]["labels"] == ("all",)
    finally:
        store.close()
