"""Phase 1: a call is typed, checked and refused before anything executes.

The Phase 1 exit gate is that an invalid grade, basis, capability or exactness request
fails before an adapter is called. These tests assert the ordering itself, not only the
outcome: a source that explodes when touched proves a refusal happened first, where an
assertion on the exception type alone would pass even if the adapter had already run.
"""

from __future__ import annotations

import pytest

from rcql.binding import SourceKindError, UnreachableOperator, bind, classify
from rcql.capabilities import SourcePolicy
from rcql.inference import infer
from rcql.signatures import catalogued, lookup
from rcql.types import (
    Domain,
    Effect,
    Exactness,
    RCType,
    ValueKind,
    Variance,
)

rcdb = pytest.importorskip("rcdb")


class Exploding:
    """A store-shaped source whose methods fail if anything actually calls them.

    The methods must EXIST so the binder's surface scan classifies it as a store; they
    must fail when invoked so a refusal that happened too late is visible. Probing with
    hasattr is not a call, which is the distinction this class turns into a test.
    """

    def _boom(self, *args, **kwargs):  # pragma: no cover - reaching this is the failure
        raise AssertionError("an adapter ran before the call was refused")

    get = history = stats = list = search = _boom


@pytest.fixture
def rex():
    from rexgraph.graph import RexGraph

    # plain lists: RCQL computes on the core library's exact carriers, so its tests do
    # not reach for numpy to build a complex either
    return RexGraph.from_graph(sources=[0, 1, 2], targets=[1, 2, 0])


@pytest.fixture
def store(tmp_path, rex):
    s = rcdb.open_store(f"rex://{tmp_path / 'store'}")
    s.put("r1", rex)
    yield s
    s.close()


@pytest.fixture
def catalog(tmp_path):
    from rexgraph.io.catalog import FileCatalog

    root = tmp_path / "cat"
    root.mkdir()
    cat = FileCatalog([root])
    cat.refresh()
    return cat


# ------------------------------------------------------------------ types


def test_the_extended_type_is_backward_compatible():
    """name, grade and exactness keep their positions, because that was the whole type."""
    legacy = RCType("Integer", 1, Exactness.INTEGER)
    assert (legacy.name, legacy.grade, legacy.exactness) == ("Integer", 1, Exactness.INTEGER)
    assert legacy.kind is ValueKind.UNKNOWN
    assert not legacy.is_determined()


def test_domain_and_exactness_answer_different_questions():
    """A rational value rendered to a float is RATIONAL over ROUNDED, not one or the other."""
    rendered = RCType("q", kind=ValueKind.EXACT_RATIONAL,
                      domain=Domain.RATIONAL, exactness=Exactness.ROUNDED)
    assert rendered.domain is Domain.RATIONAL
    assert rendered.exactness is Exactness.ROUNDED


def test_chains_and_cochains_do_not_share_a_space():
    chain = RCType("c", 1, kind=ValueKind.CHAIN, variance=Variance.CHAIN)
    cochain = RCType("c", 1, kind=ValueKind.COCHAIN, variance=Variance.COCHAIN)
    assert not chain.same_space(cochain)
    assert chain.same_space(chain.with_(name="renamed"))


def test_a_declared_basis_mismatch_is_not_a_shared_space():
    from rcql.types import BasisRef

    left = RCType("c", 1, kind=ValueKind.CHAIN, variance=Variance.CHAIN,
                  basis=BasisRef("a", 1))
    right = left.with_(basis=BasisRef("b", 1))
    assert not left.same_space(right)


def test_a_declared_temporal_mismatch_is_not_a_shared_space():
    from rcql.types import TemporalRef

    left = RCType("c", 1, kind=ValueKind.COCHAIN, variance=Variance.COCHAIN,
                  temporal=TemporalRef(version=1))
    right = left.with_(temporal=TemporalRef(version=2))
    assert not left.same_space(right)


# ------------------------------------------------------------------ classification


def test_sources_are_classified_by_surface_not_by_import(store, catalog, rex):
    """RCQL must not import rcdb or the catalog module to decide what it was handed."""
    assert classify(store) is ValueKind.REX
    assert classify(catalog) is ValueKind.CATALOG_ENTRY_SET
    assert classify(rex) is ValueKind.REX
    assert classify(object()) is ValueKind.UNKNOWN


# ------------------------------------------------------------------ refusal ordering


def test_an_unreachable_operator_is_refused_without_touching_the_source():
    binding = bind("db", Exploding(), SourcePolicy.allow("*"))
    with pytest.raises(UnreachableOperator, match="state_digest"):
        infer(binding, "RCDB_STATE_HASH")


def test_a_wrong_source_kind_is_refused_without_touching_the_source(catalog):
    binding = bind("cat", catalog, SourcePolicy.allow("*"))
    with pytest.raises(SourceKindError, match="reads a Rex source"):
        infer(binding, "RCDB_GET", ("r1",))


def test_a_withheld_capability_is_refused_without_touching_the_source():
    binding = bind("db", Exploding(), SourcePolicy.allow("history"))
    with pytest.raises(PermissionError, match="identity"):
        infer(binding, "RCDB_GET", ("r1",))


def test_bad_arity_and_bad_argument_kinds_are_refused_without_touching_the_source():
    binding = bind("db", Exploding(), SourcePolicy.allow("*"))
    with pytest.raises(TypeError, match="takes 1 arguments"):
        infer(binding, "RCDB_GET")
    with pytest.raises(TypeError, match="rejected"):
        infer(binding, "RCDB_GET", (7,))


def test_state_hash_cannot_be_handed_the_catalog_it_is_filed_beside(catalog):
    """Today this fails inside the digest with AttributeError on a private attribute.

    A declared source kind turns that into a refusal at the boundary, which is the whole
    reason the signature names what it reads.
    """
    binding = bind("cat", catalog, SourcePolicy.allow("*"))
    with pytest.raises(SourceKindError):
        infer(binding, "STATE_HASH")


# ------------------------------------------------------------------ typing and explain


def test_a_result_is_typed_and_carries_its_source_before_execution(store):
    binding = bind("db", store, SourcePolicy.allow("*"))
    typed = infer(binding, "RCDB_GET", ("r1",))

    assert typed.result.kind is ValueKind.REX
    assert typed.result.source.name == "db"
    assert typed.result.source.policy_digest
    assert Effect.READ in typed.effects


def test_a_projected_reading_is_structural_and_a_digest_is_bytes(store):
    binding = bind("db", store, SourcePolicy.allow("*"))
    listed = infer(binding, "RCDB_LIST", ()).result
    digest = infer(binding, "RCDB_HASH", ("r1",)).result

    assert listed.kind is ValueKind.RECORD_SET
    assert listed.exactness is Exactness.STRUCTURAL
    assert digest.kind is ValueKind.DIGEST
    assert digest.domain is Domain.BYTES


def test_explain_answers_without_running_and_without_leaking(store):
    binding = bind("db", store, SourcePolicy.allow("*"))
    explained = infer(binding, "RCDB_SECURITY", ()).explain()

    assert explained["operator"] == "RCDB_SECURITY"
    assert explained["result"]["kind"] == "SecurityStatus"
    assert explained["requires"] == ["security"]

    # A plan description is structural. It names the operator, the source binding and the
    # declared result, and carries no backend path, no store URI and no live object. The
    # check is for those specific leaks rather than for the substring "key", which appears
    # legitimately in implementation_key and in a precondition warning against key material.
    rendered = repr(explained)
    assert "rex://" not in rendered
    assert "/tmp" not in rendered
    assert repr(binding.value) not in rendered
    assert explained["policy_digest"] and len(explained["policy_digest"]) == 64


# ------------------------------------------------------------------ catalogue shape


def test_every_storage_operator_has_a_signature():
    """The storage inventory remains declared while native math has its own contracts."""
    storage = {
        "FILES", "SEARCH", "FILE_INFO", "FILE_HASH", "HASH_FILES", "TENSORS",
        "SEARCH_TENSORS", "STATE_HASH", "RCDB_LIST", "RCDB_SEARCH", "RCDB_GET",
        "RCDB_HISTORY", "RCDB_STATS", "RCDB_HASH", "RCDB_COMMITS", "RCDB_VERIFY",
        "RCDB_STATE_HASH", "RCDB_SECURITY",
    }
    assert storage <= catalogued()


def test_every_runtime_operator_except_source_syntax_has_a_static_signature():
    """A whole phrase cannot be preflighted if one executable leaf has no contract."""
    from rcql.operators import _REGISTRY

    # REX(name) is source syntax handled before expression evaluation; its adapter is a
    # compatibility name passthrough rather than a value-producing operator.
    assert set(_REGISTRY) - {"REX"} <= catalogued()


def test_primary_cell_contracts_carry_source_grade_basis_and_exact_share_domain(rex):
    """A C1 relation stays primary in the plan before any adapter executes."""
    binding = bind("complex", rex, SourcePolicy.allow("*"))

    relation = infer(binding, "CELL", (1, 0)).result
    assert (relation.kind, relation.grade, relation.variance) == (
        ValueKind.CELL, 1, Variance.CELL,
    )
    assert relation.source == binding.ref
    assert relation.basis.source_id == "complex"
    assert relation.basis.grade == 1

    binary = infer(binding, "COMPOSITE", (relation,)).result
    share = infer(binding, "SHARE", (binary,)).result
    boundary = infer(binding, "BOUNDARY", (relation,)).result
    co_relations = infer(binding, "CORELATIONS", (relation,)).result

    assert (binary.kind, binary.domain, binary.exactness) == (
        ValueKind.COMPOSITE_BINARY, Domain.RATIONAL, Exactness.RATIONAL,
    )
    assert (share.kind, share.grade, share.variance, share.domain, share.exactness) == (
        ValueKind.CHAIN, 0, Variance.CHAIN, Domain.RATIONAL, Exactness.RATIONAL,
    )
    assert (boundary.kind, boundary.grade, boundary.exactness) == (
        ValueKind.CELL_BOUNDARY, 0, Exactness.RATIONAL,
    )
    assert (co_relations.kind, co_relations.grade, co_relations.variance) == (
        ValueKind.CELL_SET, 2, Variance.CELL,
    )


def test_cell_contract_refuses_a_foreign_source_before_execution(rex):
    """Equal-looking cells from another basis are not silently reinterpreted."""
    left = bind("left", rex, SourcePolicy.allow("*"))
    right = bind("right", rex, SourcePolicy.allow("*"))
    foreign = infer(right, "CELL", (1, 0)).result

    with pytest.raises(TypeError, match="source-bound"):
        infer(left, "COMPOSITE", (foreign,))


def test_temporal_contracts_keep_transition_time_and_channel_exactness():
    """Planning distinguishes exact structure from a measured C1 amplitude field."""
    from rexgraph.graph import RexGraph, TemporalRex

    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_graph(sources=[0], targets=[1]))
    timeline.append_snapshot(RexGraph.from_graph(sources=[0, 1], targets=[1, 2]))
    binding = bind("timeline", timeline, SourcePolicy.allow("*"))
    assert binding.schema.kind is ValueKind.TEMPORAL_REX

    delta = infer(binding, "TEMPORAL_DELTA", (1,)).result
    event = infer(binding, "SIGNAL_AT", (delta, (0, 1))).result
    structural = infer(binding, "SIGNAL_SOURCE", (delta,)).result
    amplitude = infer(binding, "RELATION_SIGNAL", (delta,)).result
    existence = infer(binding, "RELATION_SIGNAL", (delta, "existence")).result
    flow = infer(binding, "SIGNAL_FLOW", (delta,)).result
    hodge = infer(binding, "SIGNAL_HODGE", (delta,)).result

    assert (delta.kind, delta.grade, delta.temporal.version) == (ValueKind.DELTA, 1, 1)
    assert event.kind is ValueKind.TEMPORAL_EVENT
    assert (structural.kind, structural.grade, structural.domain, structural.exactness) == (
        ValueKind.CHAIN, 0, Domain.RATIONAL, Exactness.RATIONAL,
    )
    assert (amplitude.kind, amplitude.grade, amplitude.domain, amplitude.exactness) == (
        ValueKind.COCHAIN, 1, Domain.REAL, Exactness.APPROXIMATE,
    )
    assert (existence.domain, existence.exactness) == (Domain.INTEGER, Exactness.INTEGER)
    assert (flow.kind, flow.domain, flow.exactness) == (
        ValueKind.SIGNAL_FLOW, Domain.RATIONAL, Exactness.RATIONAL,
    )
    assert (hodge.kind, hodge.grade, hodge.exactness) == (
        ValueKind.HODGE_SPLIT, 1, Exactness.APPROXIMATE,
    )
    assert all(value.source == binding.ref and value.temporal.version == 1
               for value in (delta, event, structural, amplitude, existence, flow, hodge))


def test_metric_curvature_contract_refuses_wrong_space_and_names_rational_shares(rex):
    """A C1 cochain is required and integer metric values can yield rational curvature."""
    from rcql.types import BasisRef

    binding = bind("complex", rex, SourcePolicy.allow("*"))
    metric = RCType(
        "C1Metric", grade=1, kind=ValueKind.COCHAIN, variance=Variance.COCHAIN,
        domain=Domain.INTEGER, exactness=Exactness.INTEGER, source=binding.ref,
        basis=BasisRef("complex", 1),
    )
    reading = infer(binding, "METRIC_CURVATURE", (metric,)).result
    assert (reading.kind, reading.domain, reading.exactness, reading.source) == (
        ValueKind.METRIC_CURVATURE, Domain.RATIONAL, Exactness.RATIONAL, binding.ref,
    )

    wrong_grade = metric.with_(grade=0, basis=BasisRef("complex", 0))
    with pytest.raises(TypeError, match="grade=1"):
        infer(binding, "METRIC_CURVATURE", (wrong_grade,))


def test_identity_history_and_mutation_are_separate_capabilities():
    """Reading a projected summary is not the right to resolve an identity."""
    assert lookup("RCDB_LIST").requires == frozenset()
    assert lookup("RCDB_GET").requires == frozenset({"identity"})
    assert lookup("RCDB_HISTORY").requires == frozenset({"history"})
    assert lookup("RCDB_SECURITY").requires == frozenset({"security"})


def test_the_hashing_operator_declares_its_filesystem_effect():
    assert Effect.FILESYSTEM in lookup("HASH_FILES").effects
    assert Effect.FILESYSTEM not in lookup("RCDB_LIST").effects


def test_typing_a_call_does_not_require_the_machinery_that_would_run_it():
    """Deciding what a call WOULD produce must not import numpy or the operator registry.

    This is what lets EXPLAIN answer for a plan without paying for it, and it is why the
    type, signature, binding and inference surfaces are eager while the executor stays
    behind __getattr__. Run in a subprocess because the check is about what a fresh import
    pulls, and this suite has already imported everything.
    """
    import subprocess
    import sys
    import tempfile

    # The whole pre-execution surface, not just the type layer: binding, signature lookup,
    # inference and planning. EXPLAIN has to be able to answer for a plan without paying
    # for it, so a planner that reached the operator registry would defeat the point even
    # though it never runs an adapter.
    program = (
        "import sys, rcql, rcql.planning;"
        "_ = rcql.RCType, rcql.bind, rcql.infer, rcql.lookup, rcql.ValueKind, rcql.Effect;"
        "_ = rcql.planning.plan_query, rcql.planning.QueryPlan;"
        "print('numpy' in sys.modules, 'rcql.operators' in sys.modules)"
    )
    # A neutral working directory on purpose. A subprocess does not inherit the sys.path
    # repair in conftest, so launching it from the repository root would resolve rcql to
    # the outer directory that shadows the package and fail on the first attribute. That
    # is the same shadowing this suite guards against elsewhere, and it reaches into a
    # child process precisely because the fix lives in the parent's interpreter state.
    with tempfile.TemporaryDirectory() as neutral:
        out = subprocess.run([sys.executable, "-c", program], capture_output=True,
                             text=True, check=True, cwd=neutral).stdout.strip()
    assert out == "False False", f"typing pulled the numeric stack: {out}"


def test_declared_dependencies_match_what_the_package_actually_imports():
    """RCQL declares what it uses, and only what it uses.

    Two failures are possible and both matter. A missing declaration means an install can
    import the package and then fail at the first call, which is how safetensors escaped
    in rcdb. An unused declaration is the inverse claim: it asserted rcql depends directly
    on scipy when scipy is only ever reached through a rexgraph method.

    RCQL is meant to compute on the exact tensor carriers of the core library, so numpy
    here is a temporary consequence of the Rex-math adapters still holding raw arrays. When
    those move onto the core carriers this test will fail on the now-unused numpy
    declaration, which is the intended signal rather than a nuisance.
    """
    import ast as ast_module
    import pathlib
    import re
    import sys

    import tomllib

    package = pathlib.Path(__file__).resolve().parents[1]
    meta = tomllib.loads((package / "pyproject.toml").read_text())["project"]
    declared = {
        re.split(r"[<>=!\[ ]", spec, maxsplit=1)[0].lower().replace("-", "_")
        for spec in meta["dependencies"]
    }
    declared.discard("rexgraph")

    stdlib = set(sys.stdlib_module_names)
    imported: set[str] = set()
    for path in sorted((package / "rcql").rglob("*.py")):
        for node in ast_module.walk(ast_module.parse(path.read_text())):
            if isinstance(node, ast_module.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast_module.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    third_party = {
        name for name in imported
        if name not in stdlib and name not in {"rcql", "rexgraph", "__future__"}
    }
    # rcdb is an optional extra and is imported inside functions, never at module scope
    third_party.discard("rcdb")

    assert third_party == declared, {
        "imported but not declared": sorted(third_party - declared),
        "declared but not imported": sorted(declared - third_party),
    }


def test_explain_carries_the_state_and_the_frame_it_read(store):
    """A result's meaning includes when it was read and in which basis.

    An identical-looking reading from another version, or in another ordered basis, is a
    different value. An account that omitted both would let EXPLAIN imply a state and a
    frame it never established, which is the thing the blueprint forbids.
    """
    binding = bind("db", store, SourcePolicy.allow("*"))
    explained = infer(binding, "RCDB_GET", ("r1",)).explain()

    assert "temporal" in explained["result"]
    assert "basis" in explained["result"]

    # and the payload stays plain: no descriptors, no live objects, no key material
    def plain(value):
        if isinstance(value, dict):
            return all(plain(v) for v in value.values())
        if isinstance(value, list):
            return all(plain(v) for v in value)
        return isinstance(value, (str, int, float, bool, type(None)))

    assert plain(explained), "the explain payload must be renderable without objects"
