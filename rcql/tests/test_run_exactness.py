"""Public run preserves declarations and refuses numeric approximation explicitly."""
from contextlib import closing
from fractions import Fraction as Q

import numpy as np
import pytest

from rcql import Exactness, Executor, run
from rcql.query_cache import QueryCache
from rexgraph import Approx, ExactArray, RexGraph


@pytest.mark.parametrize("value,contract", [(2**53+1, Exactness.INTEGER), (Q(1,10), Exactness.RATIONAL),
    (ExactArray.from_values([Q(1,10)]), Exactness.RATIONAL)])
def test_run_reports_exact_native_value_contracts(value, contract):
    result = run("FROM $r RETURN $value", sources={"r": RexGraph.from_graph([0], [1])},
                 params={"value": value}, exactness="exact")
    assert result.exactness == (contract,)
    assert result.native_plan["evaluation_policy"] == {"requested": "exact", "conversion": "none"}
    assert type(result).from_bytes(result.to_bytes()).exactness == result.exactness


@pytest.mark.parametrize("value", [1.0, Q(1,3)+0.0, [1, .5], {"integer": 1, "approximate": .5},
    np.array([1, .5], dtype=object), Approx(.1, "declared approximation")])
def test_exact_policy_refuses_mixed_and_explicit_approximation(value):
    graph = RexGraph.from_graph([0], [1])
    result = run("FROM $r RETURN $value", sources={"r": graph}, params={"value": value})
    assert result.values[0] is value
    with pytest.raises(ValueError, match="approximate or rounded"):
        run("FROM $r RETURN $value", sources={"r": graph}, params={"value": value}, exactness="exact")


def test_exact_cache_keys_and_results_retain_requested_policy():
    rcdb = pytest.importorskip("rcdb")
    graph = RexGraph.from_graph([0], [1])
    with closing(rcdb.MemoryStore()) as store:
        cache = QueryCache(store); text = "FROM $r RETURN BETTI(0)"
        ordinary = run(text, sources={"r": graph}, cache=cache)
        exact = run(text, sources={"r": graph}, cache=cache, exactness="exact")
        hit = run(text, sources={"r": graph}, cache=cache, exactness="exact")
        assert ordinary.values == exact.values == hit.values == (1,)
        assert ordinary.native_plan["cache"]["key"] != exact.native_plan["cache"]["key"]
        assert hit.native_plan["cache"]["hit"]
        assert hit.native_plan["evaluation_policy"]["requested"] == "exact"


def test_exact_policy_refuses_mutation_before_resolving_source():
    from rcql import mutation, source
    with pytest.raises(ValueError, match="read-only"):
        Executor(exactness="exact").execute(mutation(source("missing"), "id", RexGraph.from_graph([0], [1])))


@pytest.mark.parametrize("policy", [True, "approximate", "", None])
def test_invalid_policy_refuses_before_reading(policy):
    with pytest.raises(ValueError, match="exactness"):
        run("FROM $missing RETURN BETTI(0)", exactness=policy)
