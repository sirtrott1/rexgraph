"""Typed queries over live agent values, and the policy that bounds them.

RCQL evaluates against sources it is handed rather than anything it imports, so this is
where the agent decides what a query may reach. A source registered bare answers whatever
the operator registry can ask; one bound to a policy answers only what the policy permits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# rexgraph-rcql is a separate distribution and an optional extra. Prefer an installed one
# and fall back to the copy in this repository, so the behaviour is exercised here rather
# than silently skipped, without installing anything into the caller's environment.
if "rcql" not in sys.modules:
    try:
        import rcql as _probe
        if not hasattr(_probe, "parse"):
            raise ImportError("namespace shadow")
    except ImportError:
        root = Path(__file__).resolve().parents[2] / "rcql"
        if root.is_dir():
            sys.modules.pop("rcql", None)
            sys.path.insert(0, str(root))

rcql = pytest.importorskip("rcql")
pytest.importorskip("rcql.parser")


def _rex():
    from rexgraph.graph import RexGraph
    return RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])


@pytest.fixture
def runtime():
    from agent.rcql_runtime import RCQLRuntime
    return RCQLRuntime()


def test_a_registered_source_answers_a_typed_query(runtime):
    from rcql import parse
    runtime.register("main", _rex())
    result = runtime.execute(parse('FROM REX("main") RETURN BETTI(1)'))
    assert result is not None


def test_a_string_is_refused(runtime):
    """Parsing belongs to the caller, so what reaches here is already an AST a policy can
    be reasoned about against, rather than text this layer would have to trust."""
    runtime.register("main", _rex())
    with pytest.raises(TypeError):
        runtime.execute('FROM REX("main") RETURN BETTI(1)')


def test_weighted_dirac_uses_the_shared_typed_runtime(runtime):
    from fractions import Fraction

    from rcql import parse
    runtime.register("main", _rex())
    result = runtime.execute(parse('FROM REX("main") LET x=GRADED_CHAIN([SHARE(CELL(1,0))]) '
                                   'RETURN APPLY(DIRAC(),x,true), APPLY(ANTI_DIRAC(),x,true)'))
    assert result.values[0].component(1).values.tolist() == [Fraction(1), Fraction(-1)]
    assert result.values[1].component(1).values.tolist() == [Fraction(-1), Fraction(1)]
    assert result.provenance[0]["result_type"]["kind"] == "GradedChain"


def test_metric_green_uses_the_shared_chain_runtime(runtime):
    import numpy as np

    from rcql import parse
    from rexgraph.cochain import Chain
    runtime.register("main", _rex())
    result = runtime.execute(parse('FROM REX("main") LET x=SHARE(CELL(1,0)) '
        'LET g=RESOLVENT(HODGE_UP(0)) RETURN GREEN_SOLVE(g,x), APPLY(g,x)'))
    assert all(isinstance(v, Chain) for v in result.values)
    np.testing.assert_allclose(result.values[0].values, result.values[1].values)
    assert result.provenance[0]["result_type"]["variance"] == "chain"


def test_brackets_use_shared_exact_graded_runtime(runtime):
    from fractions import Fraction

    from rcql import parse
    runtime.register("main", _rex())
    result = runtime.execute(parse('FROM REX("main") LET x=GRADED_CHAIN([SHARE(CELL(1,0))]) '
        'RETURN APPLY(COMMUTATOR(DIRAC(),ANTI_DIRAC()),x,true)'))
    assert result.values[0].component(0).values.tolist() == [Fraction(2), Fraction(-4), Fraction(2)]
    assert result.provenance[0]["result_type"]["kind"] == "GradedChain"


def test_exact_channels_use_shared_native_runtime(runtime):
    from fractions import Fraction

    from rcql import parse
    runtime.register("main", _rex())
    result = runtime.execute(parse('FROM REX("main") LET x=INDICATOR(CELL(1,0)) '
        'RETURN APPLY(CHANNEL("F"),x,true)'))
    assert result.values[0].values.tolist() == [Fraction(2), Fraction(-2)]
    assert result.exactness[0].value == "rational"
    assert result.provenance[0]["result_type"]["kind"] == "Field"


def test_sheaf_check_uses_shared_exact_runtime_and_rectangular_maps(runtime):
    from fractions import Fraction

    from rexgraph.sheaf import ExactSheaf
    rex = _rex()
    section = ExactSheaf(rex, stalk_dims=(2, 1), mediator_dims=(1, 1, 1))
    section.assign(0, [Fraction(1, 3), Fraction(1, 6)])
    section.assign(1, [Fraction(1, 2)])
    section.restrict(0, [[1, 1]])
    runtime.register("main", rex)
    result = runtime.execute(rcql.parse('FROM REX("main") RETURN SECTION_CHECK($section).compatible'),
                             params={"section": section})
    assert result.values == (True,)
    section.assign(1, [Fraction(2, 3)])
    assert runtime.execute(rcql.parse('FROM REX("main") RETURN SECTION_CHECK($section).compatible'),
                            params={"section": section}).values == (False,)


def test_sources_are_named_and_removable(runtime):
    runtime.register("a", _rex())
    runtime.register("b", _rex())
    assert runtime.sources() == ("a", "b")
    runtime.remove("a")
    assert runtime.sources() == ("b",)
    runtime.remove("never-registered")


def test_an_empty_source_name_is_refused(runtime):
    with pytest.raises(ValueError):
        runtime.register("   ", _rex())


def test_a_policy_bounds_what_the_query_may_reach(runtime):
    """The point of registering with a policy rather than bare."""
    from rcql import SourcePolicy
    runtime.register("main", _rex(), policy=SourcePolicy.allow("read"))
    assert runtime.sources() == ("main",)


def test_typed_let_and_exact_literals_reach_the_shared_runtime(runtime):
    from fractions import Fraction

    from rcql import call, let, parse, query, ref, source
    runtime.register("main", _rex())
    text = 'FROM $main LET x = INDICATOR(CELL(1, 0)) LET q = 17/20 RETURN QUADRANCE(x, TRUE), q'
    built = query(source("main"), call("QUADRANCE", ref("x"), True), ref("q"),
                  bindings=(let("x", call("INDICATOR", call("CELL", 1, 0))),
                            let("q", Fraction(17, 20))))
    assert built == parse(text)
    result = runtime.execute(built)
    assert result.values == (Fraction(1), Fraction(17, 20))
    assert result.native_plan["bindings"][1]["name"] == "q"


def test_unused_let_cannot_bypass_the_runtime_source_policy(runtime):
    from rcql import SourcePolicy, parse
    runtime.register("main", _rex(), policy=SourcePolicy.allow("identity"))
    with pytest.raises(PermissionError):
        runtime.execute(parse('FROM $main LET unused = BETTI(0) RETURN 17/20'))


def test_mutation_explain_and_commit_use_the_shared_native_runtime(runtime):
    from dataclasses import replace

    from rcdb import MemoryStore, VersionConflictError
    from rcql import mutation, param, source

    store = MemoryStore().configure_security(require_commits=True)
    runtime.register("db", store)
    request = mutation(source("db"), "candidate", param("rex"), expected_version=0)
    explained = runtime.execute(replace(request, explain=True), params={"rex": _rex()})
    assert explained.native_plan["nodes"][-1]["kind"] == "commit"
    assert store.history("candidate") == []
    result = runtime.execute(request, params={"rex": _rex()})
    assert result.provenance[0]["record_version"] == 1
    assert store.verify_commits("candidate")
    with pytest.raises(VersionConflictError):
        runtime.execute(request, params={"rex": _rex()})


def test_typed_composition_uses_the_same_named_result_contract(runtime):
    from fractions import Fraction

    from rcql import alias, call, member, parse, query, source
    runtime.register("main", _rex())
    text = ('FROM $main AS r RETURN r.DESCRIBE().nE AS relations, '
            '[BETTI(grade=0), 17/20] AS readings')
    built = query(source("main"), alias(member(call("DESCRIBE"), "nE"), "relations"),
                  alias([call("BETTI", grade=0), Fraction(17, 20)], "readings"), source_alias="r")
    assert built == parse(text)
    result = runtime.execute(built)
    assert result.named_values == {"relations": 2, "readings": [1, Fraction(17, 20)]}
    assert result.native_plan["return_aliases"] == ["relations", "readings"]
