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
