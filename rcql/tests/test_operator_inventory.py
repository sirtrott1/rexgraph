"""Inventory is presence plus a current contract, never a conformance claim."""
from __future__ import annotations

import inspect
import json
from dataclasses import replace

import numpy as np
import pytest
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.cochain import Chain
from rexgraph.sheaf import ExactSheaf
from rexgraph.type_accession import AccessionFamily, TypeAccession
from rexgraph.io.catalog import object_digest

from rcql import (
    BoundSource,
    Call,
    Executor,
    ListExpr,
    Literal,
    SourceKindError,
    SourcePolicy,
    call,
    catalogued,
    lookup,
    operator_inventory,
    query,
    source,
)
from rcql.executor import value_exactness
from rcql.inventory import SOURCE_FORMS
from rcql.operators import _REGISTRY, get_operator


@pytest.fixture
def rex():
    return RexGraph.from_simplicial(
        np.array([0, 1, 0], dtype=np.int32),
        np.array([1, 2, 2], dtype=np.int32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )


@pytest.fixture
def timeline():
    value = TemporalRex([])
    value.append_snapshot(RexGraph(sources=np.array([0, 1]), targets=np.array([1, 2]),
                                   w_E=np.array([2.0, 1.0])))
    value.append_snapshot(RexGraph(sources=np.array([1, 1, 2]), targets=np.array([0, 2, 3]),
                                   w_E=np.array([3.0, 1.0, 4.0])))
    return value


@pytest.fixture
def catalog(tmp_path):
    from rexgraph.io.catalog import FileCatalog
    save_file = pytest.importorskip("safetensors.numpy").save_file
    save_file({"w": np.ones((2, 3))}, str(tmp_path / "m.safetensors"))
    value = FileCatalog([tmp_path])
    value.refresh()
    return value


@pytest.fixture
def store(tmp_path, rex):
    rcdb = pytest.importorskip("rcdb")
    value = rcdb.open_store(f"rex://{tmp_path / 'db'}")
    value.put("r1", rex)
    yield value
    value.close()


# One direct adapter + planned execution per current native name. These are smoke
# contracts, supplemented by the arity/basis/branching/gluing semantic suites.
C1 = call("CELL", 1, 0)
FIELD = call("INDICATOR", C1)
DELTA = call("TEMPORAL_DELTA", 1)


def _accessions(rex):
    return AccessionFamily((TypeAccession(rex, 1, "first", ((0, 0, 1), (1, 1, 1))),
                            TypeAccession(rex, 1, "second", ((1, 1, 1), (2, 2, 1)))))


def _chain_map(rex):
    from rexgraph.chain_map import CoordinateComplex, GradedMap
    c = CoordinateComplex.from_rex(rex)
    return GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))


def _replay(rex, replicate=False):
    from rexgraph.io.catalog import object_digest
    from rexgraph.io.mutation import mutation_to_bytes, prepare_mutation
    from rexgraph.io.replication import pack_replication
    delta = mutation_to_bytes(prepare_mutation(rex, rex, tx_time=1))
    return (pack_replication(b"", [delta], checkpoint_state=object_digest(rex))[0] if replicate else delta,)


NATIVE_CASES = {
    "SYMMETRY": lambda rex: ([_chain_map(rex)], [1]),
    "TRAINING_PARTITION": lambda rex: ({"source_state": object_digest(rex), "cells": [[1, [0]]]},),
    "VOID": (call("CELLS", 1),),
    "RESOLVENT_GROUP": ([call("CHANNEL", "T")], [1], [1]),
    "VALIDATE_RELATIONS": ([[(0, 1), (1, 1), (2, -1)], [(0, 1)]],),
    "QUOTIENT": (call("CELL", 1, 0),),
    "APPLY_DELTA": _replay,
    "REPLICATE": lambda rex: _replay(rex, True),
    "ACCESSION_DELTA": lambda rex: (*_accessions(rex).accessions, _chain_map(rex)),
    "TEXT_OVERLAP_VIEW": (),
    "MARKOV_VIEW": (),
    "PAGERANK": (call("MARKOV_VIEW"),),
    "CAYLEY": (call("COMMUTATOR", call("CHANNEL", "T"), call("CHANNEL", "G")), 1),
    "COMPLEX_STRUCTURE": (call("COMMUTATOR", call("CHANNEL", "T"), call("CHANNEL", "T")),),
    "RATIONAL_ROTATION": (call("COMPLEX_STRUCTURE", call("COMMUTATOR", call("CHANNEL", "T"), call("CHANNEL", "T"))), 3, 4, 5),
    "DIFF": lambda rex: (rex,),
    "FIELD_DELTA": lambda rex: (Chain(1, np.ones(rex.nE, dtype=int), source=rex), _chain_map(rex)),
    "FIELD_DELTA_MOMENT": lambda rex: (Chain(1, np.ones(rex.nE, dtype=int), source=rex), _chain_map(rex)),
    "ORIENTED_FIELD_DELTA_MOMENT": lambda rex: (Chain(1, np.ones(rex.nE, dtype=int), source=rex), _chain_map(rex)),
    "FILL": lambda rex: (Chain(1, np.array([1, 1, -1]), source=rex),),
    "HARMONIC_SHADOW": (),
    "HASH": (), "MANIFEST": (call("PARTITION", C1),), "LINEAGE": (call("PARTITION", C1),),
    "TRANSPORT": (b"sample", "bytes"), "SHOW_CAPABILITIES": (),
    "SIMPLE_HOMOLOGY": (1,), "MULTIPLICITY_HOMOLOGY": (1,),
    "FACES": (call("CELLS", 1),), "RESTRICT": (call("CELL", 1, 0),),
    "PARTITION": (call("CELL", 1, 0),),
    "COLUMN_EXPANSION": (call("BOUNDARY", 1),),
    "PRIMARY_LIFT": lambda rex: _expansion_factors(rex),
    "HYPERSLICE": (C1,),
    "ADJUGATE": (call("CHANNEL", "T"),),
    "HOMOTOPY": lambda rex: (lambda m: (m, m, [(), (), ()]))(_chain_map(rex)),
    "SIGMA_OPERATOR": (0.5, [1, 2, 3], [-1, -2, -3], ["T", "G", "F", "C"], "share"),
    "CRITICAL_COMMUTATOR": (0.6, [1, 2, 3], [-1, -2, -3], ["T", "G", "F", "C"], "share"),
    "CRITICAL_RATE": ([1, 2, 3], [-1, -2, -3], ["T", "G", "F", "C"], "share"),
    "TEMPORAL": (),
    "CHAIN": ([call("BOUNDARY", 1)],),
    "TRANSFER": lambda rex: (call("CHAIN", [call("BOUNDARY", 1)]), _unit_chain(rex)),
    "DEPENDENCE": ([FIELD, FIELD],),
    "STRAIN": (1, call("METRIC", 1)),
    "METRIC_HOMOTOPY": (call("METRIC", 1), call("METRIC", 1), 0),
    "CROSS_METRIC": lambda rex: (*_accessions(rex).accessions, [(0, 0, 1)]),
    "ACTION": lambda rex: (_unit_chain(rex), call("HODGE_SUM", 1)),
    "VARIATION": lambda rex: (call("HODGE_SUM", 1), _unit_chain(rex)),
    "DIFFERENTIAL": lambda rex: (_unit_chain(rex), _unit_chain(rex)),
    "SUPPORT": (C1,), "DEGREE": (C1,), "SHARED_BOUNDARY": (C1, C1),
    "FIELD": (FIELD,), "GRADIENT": (FIELD,), "CURL": (FIELD,),
    "GRAM": ([FIELD, FIELD],), "GRAM_RANK": ([FIELD, FIELD],),
    "ORIENTED_MOMENT": lambda rex: (_unit_chain(rex), _unit_chain(rex)),
    "COBOUNDARY_MOMENT": lambda rex: (_unit_chain(rex), _unit_chain(rex)),
    "FIELD_QUOTIENT": lambda rex: (_unit_chain(rex), _unit_chain(rex)),
    "COFIELD_QUOTIENT": lambda rex: (_unit_chain(rex), _unit_chain(rex)),
    "RATE": (FIELD, 2), "TEMPORAL_RATE": (3, 2), "MOMENT_RATE": (3, 2),
    "MASS": (FIELD,), "ARGMIN": ([C1, C1], [2, 1]), "ARGMAX": ([C1, C1], [1, 2]),
    "TRACE": (call("CHANNEL", "G"), True),
    "TYPE_VIEW": lambda rex: (FIELD, _accessions(rex).accessions[0], True),
    "TYPES": lambda rex: (_accessions(rex),),
    "CURVATURE": (FIELD,), "WEIGHT": (C1,), "SIGNING": (C1,), "ORIENTATION": (C1,),
    "PARITY": ([C1, C1],), "CHAIN_VALID": (),
    "GREEN_OPERATOR": (1,), "GREEN_FIELD": (C1,),
    "GREEN_GRAM": ([C1, C1],), "GREEN_SPREAD": (C1, C1),
    "COUNT": (call("CELLS", 1),), "SUM": ([1, 2, 3],), "MEAN": ([1, 2],),
    "COMMUTATOR": (call("DIRAC"), call("ANTI_DIRAC")),
    "ANTICOMMUTATOR": (call("HODGE_DOWN", 1), call("HODGE_UP", 1)),
    "DIRAC": (), "ANTI_DIRAC": (), "GRADED_CHAIN": ([call("ZERO", 1, "chain")],),
    "GRADE_COMPONENT": (call("GRADED_CHAIN", [call("ZERO", 1, "chain")]), 1),
    "HODGE_DOWN": (1,), "HODGE_UP": (1,), "HODGE_SUM": (1,), "HODGE_DIFFERENCE": (1,),
    "ADJOINT": (call("BOUNDARY", 1),),
    "CHAIN_MAP": lambda rex: (_chain_map(rex),),
    "ACCUMULATE": (FIELD, FIELD),
    "APPLY": (call("HODGE_OPERATOR", 1), FIELD),
    "ARITY": (C1,), "BETTI": (1,), "BOUNDARY": (C1,), "CELL": (1, 0),
    "CELLS": (1,), "CHARACTER": (True,), "CLOSURE": (0,),
    "SEMANTIC_CLOSURE": (0,), "DOCUMENT_FIELD": ([0, 1],),
    "SECTION_RESPONSE": lambda rex: _section_case(rex),
    "COBOUNDARY": (call("CELL", 0, 0),), "COMPOSITE": (C1,),
    "CORELATIONS": (C1,), "DESCRIBE": (), "ENCLOSURE": (C1,), "EXISTENCE": (C1,),
    "GRADE": (), "GREEN": (), "HARMONIC": (FIELD,), "HEAD": (C1,), "HODGE": (FIELD,),
    "HODGE_COORDS": (FIELD,), "HODGE_OPERATOR": (1,), "INDICATOR": (C1,),
    "METRIC_CURVATURE": (FIELD,), "NULLITY": (1,), "QUADRANCE": (FIELD, True),
    "RANK": (1,), "SHARE": (C1,), "SHARE_SUPPORT": (C1,), "SIGNIFICANCE": (0,),
    "SPREAD": (FIELD, FIELD, True), "STAR": (C1,), "STATE_HASH": (), "WINDING": (FIELD,),
    "ZERO": (1,), "SHOW_OPERATORS": (),
    "CHANNEL": ("G",), "STAR_CHARACTER": (call("CELL", 0, 0), True),
    "SCALE_MOMENT": (call("CHANNEL", "T"), 1, True, True),
    "CHARACTER_ENERGY": (call("CHANNEL", "T"),),
    "METRIC": (1,), "MOMENT": (FIELD, FIELD, call("METRIC", 1), True),
    "INTEGRATE": (call("ZERO", 1), call("ZERO", 1, "chain"), True),
    "RESOLVENT": (call("HODGE_OPERATOR", 1),),
    "GREEN_SOLVE": (call("RESOLVENT", call("HODGE_OPERATOR", 1)), FIELD),
    "ACCESS": lambda rex: (FIELD, _accessions(rex).accessions[0], True),
    "ACCESS_TYPES": lambda rex: (FIELD, _accessions(rex), True),
    "CO_RELATE": lambda rex: (call("ACCESS", FIELD, _accessions(rex).accessions[0], True),
                                call("ACCESS", FIELD, _accessions(rex).accessions[1], True), None, True),
    "MOMENT_TENSOR": lambda rex: (call("ACCESS_TYPES", FIELD, _accessions(rex), True), None, True),
}


def _section_case(rex):
    from rexgraph.sectioning import add_sectioning
    add_sectioning(rex, "all", {"all": list(range(rex.nE))})
    return "all", [0]


def _unit_chain(rex):
    from rexgraph.cochain import Chain
    return Chain(1, np.array([1, 0, 0]), source=rex)


def _expansion_factors(rex):
    from rexgraph.column_expansion import ColumnExpansion
    from rexgraph.linear_operator import boundary_operator
    value = ColumnExpansion(boundary_operator(rex, 1))
    return value.legs, value.lift
TEMPORAL_CASES = {
    "ALIGN_BY_LINEAGE": ([[0, 3], [7, 0, 2]],),
    "DELTA": (1,), "EXISTENCE_DELTA": (DELTA,), "ORIENTATION_DELTA": (DELTA,),
    "SIGNING_DELTA": (DELTA,), "HEAD_DELTA": (DELTA,), "STRUCTURAL_DELTA": (DELTA,),
    "METRIC_DELTA": (DELTA,), "EXISTENCE_HISTORY": (), "BIOES": (), "BETWEEN": (0, 1, "step"),
    "TEMPORAL_DELTA": (1,), "SIGNAL_AT": (DELTA, (0, 1)),
    "SIGNAL_SOURCE": (DELTA,), "RELATION_SIGNAL": (DELTA,),
    "SIGNAL_FLOW": (DELTA,), "SIGNAL_HODGE": (DELTA,),
}
CATALOG_CASES = {
    "TENSOR_MANIFEST": ("root0/m.safetensors",),
    "FILES": (), "SEARCH": ("m",), "FILE_INFO": ("root0/m.safetensors",),
    "FILE_HASH": ("root0/m.safetensors",), "HASH_FILES": (), "TENSORS": ("root0/m.safetensors",),
    "SEARCH_TENSORS": ("root0/m.safetensors", "w"),
}
STORE_CASES = {
    "CORPUS_FIELD": (["r"],),
    "RCDB_STATE_HASH": (),
    "RCDB_LIST": (), "RCDB_SEARCH": ("r",), "RCDB_GET": ("r1",),
    "RCDB_HISTORY": ("r1",), "RCDB_STATS": (), "RCDB_HASH": ("r1",),
    "RCDB_COMMITS": ("r1", 1), "RCDB_VERIFY": ("r1",), "RCDB_SECURITY": (),
}

TURN_CASES = {"TURN_FIELD": (), "PATH_CHANGE": ("alpha delta",)}


@pytest.mark.parametrize("name", sorted(TURN_CASES))
def test_conversation_direct_adapter_and_typed_contract_agree(name):
    from rexgraph.flow.turn_field import TurnField
    field = TurnField()
    field.observe("alpha beta gamma")
    _check_contract(field, name, TURN_CASES[name])


def _direct(value, expression):
    if isinstance(expression, ListExpr):
        return [_direct(value, item) for item in expression.items]
    """Call adapters themselves, bypassing parser/planner/executor."""
    if isinstance(expression, Literal):
        return expression.value
    assert isinstance(expression, Call)
    return get_operator(expression.name).fn(value, *(_direct(value, arg) for arg in expression.args))


def _check_contract(value, name, args):
    expression = call(name, *args)
    direct = _direct(value, expression)
    executor = Executor(sources={"s": value})
    actual = executor.execute(query(source("s"), expression))
    explained = executor.execute(replace(query(source("s"), expression), explain=True)).values[0]
    declared = explained["returns"][0]
    assert value_exactness(direct) == actual.exactness[0]
    assert declared["result"]["exactness"] == actual.exactness[0].value
    assert declared["requires"] == sorted(lookup(name).requires)
    assert declared["memoizable"] == lookup(name).memoizable


@pytest.mark.parametrize("name", sorted(NATIVE_CASES))
def test_native_direct_adapter_and_typed_contract_agree(rex, name):
    args = NATIVE_CASES[name]
    _check_contract(rex, name, args(rex) if callable(args) else args)


@pytest.mark.parametrize("name", sorted(TEMPORAL_CASES))
def test_temporal_direct_adapter_and_typed_contract_agree(timeline, name):
    _check_contract(timeline, name, TEMPORAL_CASES[name])


def test_exact_glue_direct_adapter_and_typed_contract_agree(rex):
    section = ExactSheaf(rex, stalk_dim=1)
    for index in range(rex.nE):
        section.assign(index, [1])
    _check_contract(rex, "GLUE", (section,))
    _check_contract(rex, "SECTION_CHECK", (section,))


@pytest.mark.parametrize("name", sorted(CATALOG_CASES))
def test_catalog_direct_adapter_and_typed_contract_agree(catalog, name):
    _check_contract(catalog, name, CATALOG_CASES[name])


@pytest.mark.parametrize("name", sorted(STORE_CASES))
def test_store_direct_adapter_and_typed_contract_agree(store, name):
    _check_contract(store, name, STORE_CASES[name])


def test_registry_signatures_and_direct_cases_are_a_closed_inventory():
    assert set(_REGISTRY) == catalogued() | {"REX"}
    covered = NATIVE_CASES.keys() | TEMPORAL_CASES.keys() | CATALOG_CASES.keys() | STORE_CASES.keys() | TURN_CASES.keys()
    from rcql.artifact_contracts import OBSERVABLE
    assert covered | OBSERVABLE | {"GLUE", "SECTION_CHECK", "REX", "EXPORT_PARQUET"} == set(_REGISTRY)
    for name in catalogued():
        parameters = list(inspect.signature(get_operator(name).fn).parameters.values())[1:]
        assert all(p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD} for p in parameters)
        required = sum(p.default is p.empty for p in parameters)
        assert lookup(name).arity == (required, len(parameters)), name


def test_signature_requirements_preserve_the_previous_effective_permissions():
    special = {
        "TURN_FIELD": {"read", "identity", "history", "agent_read"},
        "PATH_CHANGE": {"read", "identity", "history", "agent_read"},
        "CORPUS_FIELD": {"read", "records", "search", "identity"},
        "TRAINING_PARTITION": {"read", "identity", "train"},
        "QUOTIENT": {"read", "identity"},
        "APPLY_DELTA": {"read", "identity", "history", "security"},
        "REPLICATE": {"read", "identity", "history", "security"},
        "HASH": {"read", "identity"}, "MANIFEST": {"read", "identity"},
        "LINEAGE": {"read", "identity"}, "SHOW_CAPABILITIES": set(),
        "ENCRYPT": {"read", "security"}, "DECRYPT": {"read", "security"},
        "SIGN": {"read", "security"}, "VERIFY_SIGNATURE": {"read", "security"},
        "PSEUDONYMIZE": {"read", "identity", "security"},
        "RESTRICT": {"read", "identity"}, "PARTITION": {"read", "identity"},
        "FILL": {"read", "identity"}, "EXPORT_PARQUET": {"read", "identity"},
        "DIFF": {"read", "identity"},
        "SECTION_RESPONSE": {"read", "identity"},
        "RCDB_STATE_HASH": {"read", "history", "identity"},
        "RCDB_LIST": {"records"}, "RCDB_SEARCH": {"records", "search"},
        "RCDB_GET": {"identity"}, "RCDB_HASH": {"identity"},
        "RCDB_HISTORY": {"history", "identity"}, "RCDB_COMMITS": {"history", "identity"},
        "RCDB_VERIFY": {"history", "identity"}, "RCDB_SECURITY": {"security", "admin"},
        "FILES": {"files", "file_read"}, "FILE_INFO": {"files", "file_read"},
        "FILE_HASH": {"files", "file_read"}, "HASH_FILES": {"files", "file_read"},
        "TENSORS": {"files", "file_read"}, "TENSOR_MANIFEST": {"files", "file_read"}, "SEARCH": {"files", "search"},
        "SEARCH_TENSORS": {"files", "search"},
        "MODEL_TRAIN": {"read", "train"},
    }
    for name in catalogued():
        assert lookup(name).requires == special.get(name, {"read"}), name
    reused = NATIVE_CASES.keys() | TEMPORAL_CASES.keys() | {"EXPORT_PARQUET"}
    assert {name for name in catalogued() if lookup(name).memoizable} == (
        reused - {"STATE_HASH", "SHOW_OPERATORS", "APPLY_DELTA", "REPLICATE",
                  "PROGRAM_RUN", "PROGRAM_READ"}
    )


def test_source_form_inventory_matches_the_separate_from_evaluator():
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(Executor._eval_source)))
    handled = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute)
                and node.left.attr == "name" and isinstance(node.left.value, ast.Name)
                and node.left.value.id == "expr"):
            for item in ast.walk(node.comparators[0]):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    handled.add(item.value)
    assert handled == SOURCE_FORMS.keys()


def test_planned_only_rows_stay_refused_and_logical_store_hash_now_executes(store):
    executor = Executor(sources={"s": store})
    for explain in (False, True):
        result = executor.execute(replace(query(source("s"), call("RCDB_STATE_HASH")), explain=explain))
        assert result.values if explain else result.values == (store.state_digest(),)
        with pytest.raises(KeyError, match="DEIDENTIFY"):
            executor.execute(replace(query(source("s"), call("DEIDENTIFY")), explain=explain))


def test_unweighted_temporal_amplitude_can_refine_the_conservative_numeric_contract():
    from rcql import Exactness
    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_graph(sources=[0, 1], targets=[1, 2]))
    timeline.append_snapshot(RexGraph.from_graph(sources=[1, 1, 2], targets=[0, 2, 3]))
    expression = query(source("s"), call("RELATION_SIGNAL", DELTA))
    executor = Executor(sources={"s": timeline})
    result = executor.execute(expression)
    explained = executor.execute(replace(expression, explain=True)).values[0]
    # Keep exact native coefficients; do not round merely to match a conservative
    # source independent signature. Weighted cases above exercise its other branch.
    assert result.exactness == (Exactness.RATIONAL,)
    assert explained["returns"][0]["result"]["exactness"] == "approximate"


def test_inventory_distinguishes_source_forms_and_refusals():
    rows = operator_inventory()
    named = {row["name"]: row for row in rows}
    # Every row is a contract that exists. A name with no contract has no row.
    assert set(named) == catalogued() | SOURCE_FORMS.keys()
    assert [row["name"] for row in rows] == sorted(named)
    assert named["RCDB_STATE_HASH"]["status"] == "implemented"
    assert named["RCDB_STATE_HASH"]["current"]["refusal"] is None
    assert named["REX"]["status"] == "source-only"
    assert named["REX"]["current"] is None
    assert named["RCDB_GET"]["roles"] == ["expression", "source"]
    assert named["CHANNEL"]["status"] == "implemented"
    assert named["CHANNEL"]["current"]["arity"] == [1, 1]
    assert named["GRAM"]["status"] == "implemented"
    assert named["ADJUGATE"]["status"] == "implemented"
    assert named["GLUE"]["status"] == "implemented"
    assert "not a multi-Rex mutation" in named["GLUE"]["note"]
    assert named["HEAD"]["current"]["result"]["kind"] == "Chain"
    assert named["CORELATIONS"]["status"] == "implemented"
    assert named["CO_RELATE"]["status"] == "implemented"
    assert named["TYPE_VIEW"]["status"] == "implemented"
    assert named["SHOW_OPERATORS"]["status"] == "implemented"
    json.dumps(rows)  # no live carriers, functions, arrays, source paths or addresses
    rows[0]["name"] = "changed"
    assert operator_inventory()[0]["name"] != "changed"
    assert operator_inventory(limit=3, offset=4) == operator_inventory()[4:7]
    assert operator_inventory(limit=0) == operator_inventory(offset=10**9) == []


@pytest.mark.parametrize("kwargs", [
    {"limit": -1}, {"limit": 1001}, {"limit": True}, {"limit": 1.5},
    {"offset": -1}, {"offset": True}, {"offset": 1.5},
])
def test_inventory_pagination_rejects_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        operator_inventory(**kwargs)


@pytest.mark.parametrize("name", sorted(CATALOG_CASES))
def test_catalog_contract_rejects_non_catalog_before_adapter(rex, monkeypatch, name):
    def boom(*args):
        pytest.fail("wrong-source query reached an adapter")
    monkeypatch.setitem(_REGISTRY, name, replace(_REGISTRY[name], fn=boom))
    request = query(source("s"), call(name, *CATALOG_CASES[name]))
    for explain in (False, True):
        with pytest.raises(SourceKindError, match="CatalogEntrySet"):
            Executor(sources={"s": rex}).execute(replace(request, explain=explain))


@pytest.mark.parametrize("kind,name", [
    *(("catalog", name) for name in CATALOG_CASES),
    *(("store", name) for name in STORE_CASES),
    ("rex", "DESCRIBE"), ("rex", "SHOW_OPERATORS"),
])
def test_execution_and_explain_enforce_identical_requirements(request, monkeypatch, kind, name):
    value = request.getfixturevalue(kind)
    args = (CATALOG_CASES | STORE_CASES | NATIVE_CASES)[name]
    expression = query(source("s"), call(name, *args))
    needed = lookup(name).requires
    executor = Executor(sources={"s": BoundSource(value, SourcePolicy.allow(*needed))})
    for explain in (False, True):
        executor.execute(replace(expression, explain=explain))

    def boom(*args):
        pytest.fail("denied query reached an adapter")
    monkeypatch.setitem(_REGISTRY, name, replace(_REGISTRY[name], fn=boom))
    for missing in needed:
        executor = Executor(sources={
            "s": BoundSource(value, SourcePolicy.allow(*(needed - {missing}))),
        })
        for explain in (False, True):
            with pytest.raises(PermissionError, match=missing):
                executor.execute(replace(expression, explain=explain))


def test_cache_observable_catalog_reads_are_not_memoized(catalog):
    expression = call("FILE_INFO", "root0/m.safetensors")
    result = Executor(sources={"s": catalog}).execute(query(
        source("s"), expression, call("FILE_HASH", "root0/m.safetensors"), expression,
    ))
    assert result.values[0].sha256 is None
    assert result.values[2].sha256 == result.values[1]
    assert not lookup("FILE_INFO").memoizable
    assert not lookup("FILE_HASH").memoizable
    assert lookup("TEMPORAL_DELTA").memoizable


def test_inventory_query_is_bounded_and_does_not_touch_source():
    class Opaque:
        def __getattr__(self, name):
            raise AssertionError(f"inventory adapter inspected source: {name}")
    assert get_operator("SHOW_OPERATORS").fn(Opaque(), 2, 3) == operator_inventory(limit=2, offset=3)


def test_structural_inventory_import_does_not_load_numeric_stack():
    import os
    import subprocess
    import sys
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = (
        "import sys; from rcql import operator_inventory; "
        "assert len(operator_inventory()) >= 191; "
        "assert not {'numpy', 'rcql.operators', 'rcql.executor'} & sys.modules.keys()"
    )
    # -I keeps the working directory off sys.path. Run from the repository root, the
    # rcql/ directory shadows the installed package as a namespace package and the
    # import fails for a reason that has nothing to do with what this test measures.
    subprocess.run([sys.executable, "-I", "-c", code], check=True, env=env)
