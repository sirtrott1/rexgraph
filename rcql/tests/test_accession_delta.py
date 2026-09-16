"""Accession changes use Core, explicit target coordinates and stored sources."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from rcql import Executor, parse
from rexgraph.accession_delta import accession_delta
from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.graph import RexGraph
from rexgraph.type_accession import CoordinateSpace, TypeAccession


def fixture(rex=None, rectangular=False):
    rex = rex if rex is not None else RexGraph.from_cells([3, [[0, 1, 2], [0]]])
    target = RexGraph.from_cells([4, [[0, 1], [1, 2], [1, 3]]])
    a, b = CoordinateComplex.from_rex(rex), CoordinateComplex.from_rex(target)
    j = GradedMap(a, b, (((0, 0, 1), (1, 1, 1), (2, 2, 1)), ((0, 0, 1), (1, 1, 1), (2, 0, 1))))
    coords = CoordinateSpace("reading", ("a", "b")) if rectangular else None
    old = TypeAccession(rex, 1, "old", ((0, 1, Q(1, 3)),), coordinates=coords)
    new = TypeAccession(target, 1, "new", ((1, 0, 2),), coordinates=coords)
    return rex, old, new, j


@pytest.mark.parametrize("rectangular", [False, True])
def test_query_and_members_match_core(rectangular):
    rex, a, b, j = fixture(rectangular=rectangular)
    engine = Executor(sources={"r": rex}, params={"a": a, "b": b, "j": j})
    result = engine.execute(parse('FROM $r LET d=ACCESSION_DELTA(old=$a,new=$b,correspondence=$j) '
        'RETURN d,d.entries,d.shape,d.formula,d.grade,d.implicit_zero,d.domain_keys,d.codomain_keys'))
    expected = accession_delta(a, b, j)
    assert result.values == (expected, expected["entries"], expected["shape"], expected["formula"], 1, Q(0),
                             expected["domain_keys"], expected["codomain_keys"])
    assert result.exactness[5].value == "rational"


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("kind", ["float", "foreign", "keys", "coordinates", "stale"])
def test_bad_inputs_refuse_before_adapter(kind, explain, monkeypatch):
    import rcql.executor
    rex, a, b, j = fixture()
    if kind == "float":
        a = replace(a, entries=((0, 0, 0.0),))
    elif kind == "foreign":
        b = fixture()[2]
    elif kind == "keys":
        a = replace(a, cell_keys=(0, 1))
    elif kind == "coordinates":
        a = replace(a, coordinates=CoordinateSpace("different", ("a", "b")))
    else:
        rex.add_edges([0], [1])
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter executed"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": rex}, params={"a": a, "b": b, "j": j}).execute(replace(
            parse('FROM $r RETURN ACCESSION_DELTA($a,$b,$j)'), explain=explain))


def test_explain_does_not_compose_and_repeated_call_reuses_core(monkeypatch):
    import rexgraph.accession_delta as core
    rex, a, b, j = fixture()
    engine = Executor(sources={"r": rex}, params={"a": a, "b": b, "j": j})
    calls = []
    original = core.accession_delta
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(core, "accession_delta", counted)
    q = parse('FROM $r RETURN ACCESSION_DELTA($a,$b,$j),ACCESSION_DELTA($a,$b,$j).entries')
    assert engine.execute(replace(q, explain=True)).execution == () and calls == []
    result = engine.execute(q)
    assert calls == [True] and result.values[1] == result.values[0]["entries"]
    with pytest.raises(TypeError, match="no declared member"):
        engine.execute(parse('EXPLAIN FROM $r RETURN ACCESSION_DELTA($a,$b,$j).source'))


def test_stored_endpoint_uses_native_source_without_scipy(tmp_path):
    from rcdb import RexStore
    store = RexStore(str(tmp_path / "db"))
    try:
        rex, _, _, _ = fixture()
        store.put("r", rex)
        restored = Executor(sources={"db": store}).execute(parse('FROM $db RETURN RCDB_GET("r")')).values[0]
        _, a, b, j = fixture(restored)
        expected = accession_delta(a, b, j)
        result = Executor(sources={"r": restored}, params={"a": a, "b": b, "j": j}).execute(parse(
            'FROM $r RETURN ACCESSION_DELTA($a,$b,$j)'))
        assert result.values[0] == expected
    finally:
        store.close()
