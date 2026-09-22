"""Effective mode counts are exact rationals, the harmonic log is their float logarithm."""
import math
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.graph import RexGraph

from rcql import Executor, call, parse, query, source
from rcql.operators import get_operator


NAMES = ("EFFECTIVE_MODES", "HARMONIC_LOG")


def _k4_one_face(**kw):
    rex = RexGraph(sources=np.array([0, 0, 0, 1, 1, 2], np.int32),
                   targets=np.array([1, 2, 3, 2, 3, 3], np.int32), **kw)
    rex.add_faces([[0, 1, 3]])
    rex._ensure_clean()
    return rex


def test_text_and_direct_readings_agree():
    rex = _k4_one_face()
    q = parse('FROM $r RETURN EFFECTIVE_MODES(1), EFFECTIVE_MODES(grade=1, sector="hodge"), '
              'HARMONIC_LOG(1), EFFECTIVE_MODES(0)')
    result = Executor(sources={"r": rex}).execute(q)
    assert result.values[:2] == (Q(289, 59), Q(75, 19))
    assert result.values[2] == pytest.approx(math.log(Q(289, 59)), abs=1e-15)
    assert result.values[3] == get_operator("EFFECTIVE_MODES").fn(rex, 0)
    assert [e.value for e in result.exactness] == ["rational", "rational", "approximate", "rational"]


def test_each_harmonic_weight_reads_through_text():
    rex = _k4_one_face()
    q = parse('FROM $r RETURN EFFECTIVE_MODES(1, "completed", "mean"), '
              'EFFECTIVE_MODES(grade=1, weight="energy"), EFFECTIVE_MODES(1, "hodge"), BETTI(1), '
              'HARMONIC_LOG(1, weight="energy")')
    values = Executor(sources={"r": rex}).execute(q).values
    assert values[:2] == (Q(1350, 227), Q(113, 19))
    assert values[1] == values[2] + values[3]
    assert values[4] == pytest.approx(math.log(Q(113, 19)), abs=1e-15)


def test_the_contract_names_mirror_the_core_module():
    from rexgraph import harmonic_modes

    from rcql import harmonic_modes_contracts
    assert harmonic_modes_contracts.SECTORS == harmonic_modes.SECTORS
    assert harmonic_modes_contracts.WEIGHTS == harmonic_modes.WEIGHTS


def test_a_stored_weighted_complex_keeps_its_metric(tmp_path):
    import rcdb
    rex = _k4_one_face(w_E=[Q(1), Q(2), Q(3), Q(1, 2), Q(5, 3), Q(7)])
    expected = tuple(get_operator("EFFECTIVE_MODES").fn(rex, 1, sector) for sector in ("completed", "up"))
    assert expected != tuple(get_operator("EFFECTIVE_MODES").fn(_k4_one_face(), 1, sector)
                             for sector in ("completed", "up"))
    store = rcdb.open_store(f"rex://{tmp_path / 'store'}")
    try:
        store.put("r", rex, analytics=False)
        result = Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_GET($db,"r") RETURN EFFECTIVE_MODES(1), EFFECTIVE_MODES(1, "up")'))
        assert result.values == expected
    finally:
        store.close()


@pytest.mark.parametrize("name", NAMES)
def test_explain_is_typed_and_reads_no_trace(name, monkeypatch):
    import rcql.executor
    import rexgraph.graded_boundary
    rex = _k4_one_face()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("EXPLAIN ran adapter"))
    monkeypatch.setattr(rexgraph.graded_boundary, "_exact_gram_traces",
                        lambda *a: pytest.fail("EXPLAIN read a trace"))
    plan = Executor(sources={"r": rex}).execute(replace(query(source("r"), call(name, 1)), explain=True))
    declared = plan.values[0]["returns"][0]
    assert declared["result"]["exactness"] == ("rational" if name == "EFFECTIVE_MODES" else "approximate")
    statuses = {p["name"]: p["status"] for p in declared["predicates"]}
    assert statuses["chain_condition"] == statuses["sector_traces"] == "verified"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("args", [(-1,), (3,), (True,), (1.0,), ("1",), (1, "curl"), (1, 1),
                                  (1, "completed", "median"), (1, "hodge", "energy"), (1, "up", "mean")])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_arguments_are_refused_before_any_adapter(name, args, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("invalid input ran adapter"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": _k4_one_face()}).execute(
            replace(query(source("r"), call(name, *args)), explain=explain))


def test_a_sector_with_no_modes_has_no_log():
    rex = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    executor = Executor(sources={"r": rex})
    assert executor.execute(parse('FROM $r RETURN EFFECTIVE_MODES(1, "up")')).values == (Q(0),)
    with pytest.raises(ValueError, match="no modes"):
        executor.execute(parse('FROM $r RETURN HARMONIC_LOG(1, "up")'))
