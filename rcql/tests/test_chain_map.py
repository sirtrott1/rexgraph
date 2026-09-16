"""Native graded map certificates, honest EXPLAIN and execution provenance."""
import json
from dataclasses import replace

import pytest
from rexgraph.chain_map import ChainMap, CoordinateComplex, GradedMap
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graph import RexGraph
from rexgraph.type_accession import CoordinateSpace, TypeAccession

from rcql import BoundSource, Executor, SourcePolicy, bind, call, param, parse, query, source
from rcql.planning import _carrier_literal
from rcql.types import Domain, TemporalRef


def make():
    rex = RexGraph.from_graph([0, 1], [1, 2])
    target = CoordinateComplex((CoordinateSpace("q0", ("left", "right")), CoordinateSpace("q1", ("edge",))),
                               (((0, 0, -1), (1, 0, 1)),))
    components = (((0, 0, 1), (0, 1, 1), (1, 2, 1)), ((0, 1, 1),))
    accessions = tuple(TypeAccession(rex, k, "quotient", e, coordinates=target.spaces[k]) for k, e in enumerate(components))
    return rex, GradedMap.from_accessions(accessions, target)


def run(rex, *exprs, explain=False, **params):
    return Executor(sources={"r": rex}, params=params).execute(query(source("r"), *exprs, explain=explain))


def test_text_query_returns_checked_map_and_actual_method_provenance():
    rex, p = make()
    result = Executor(sources={"r": rex}, params={"p": p}).execute(parse("FROM $r RETURN CHAIN_MAP($p)"))
    certificate = result.values[0]
    assert isinstance(certificate, ChainMap) and certificate.declaration is p
    assert certificate.commutation_residuals == (0,)
    assert result.exactness[0].value == "structural"
    desc = result.provenance[0]["result_type"]
    assert desc["kind"] == "ChainMap" and desc["grade"] is None and desc["shape"] is None
    assert desc["basis"] is None and desc["domain"] == "rational"
    graded = desc["graded_map"]
    assert graded["shapes"] == [[2, 3], [1, 2]] and graded["nnz"] == [3, 1]
    assert graded["chain_preserving"] is True
    assert graded["domain"][0]["keys"] == ["0", "1", "2"]
    assert graded["codomain"][0]["keys"] == ["left", "right"]
    assert graded["domain_digest"] == p.domain.coefficient_digest
    assert graded["codomain_digest"] == p.codomain.coefficient_digest
    assert graded["coefficient_digest"] == p.coefficient_digest
    assert "entries" not in json.dumps(graded)
    method = result.execution[0]["methods"][0]
    assert method["method"] == "exact-sparse-chain-map-verification"
    assert method["source_residual"] == method["target_residual"] == "0"
    assert method["commutation_residuals"] == ["0"]
    node = next(n for n in result.native_plan["nodes"] if n.get("operator") == "CHAIN_MAP")
    assert node["physical"]["method"] == method["method"]


def test_explain_checks_sparse_algebra_without_running_adapters(monkeypatch):
    import rcql.executor
    rex, p = make()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    result = run(rex, call("CHAIN_MAP", param("p")), explain=True, p=p)
    node = result.values[0]["returns"][0]
    facts = {f["name"]: f for f in node["predicates"]}
    for name in ("source_chain_law", "target_chain_law", "chain_map_squares", "chain_map_boundary_state"):
        assert facts[name]["status"] == "verified"
    assert node["result"]["graded_map"]["chain_preserving"] is True
    json.dumps(result.values[0])


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("defect", ["square", "foreign", "incomplete", "target-chain", "source-chain", "stale"])
def test_invalid_maps_fail_before_any_adapter(explain, defect, monkeypatch):
    import rcql.executor
    rex, p = make()
    if defect == "square":
        p = replace(p, components=(p.components[0], ()))
    elif defect == "foreign":
        p = make()[1]
    elif defect == "incomplete":
        typ = _carrier_literal(bind("r", rex, SourcePolicy.allow("*")), p)
        desc = replace(typ.graded_map, domain=typ.graded_map.domain[:1], codomain=typ.graded_map.codomain[:1],
                       shapes=typ.graded_map.shapes[:1], nnz=typ.graded_map.nnz[:1])
        p = replace(typ, graded_map=desc)
    elif defect in {"target-chain", "source-chain"}:
        rex = RexGraph.from_cells(solid_octahedron_3rex())
        c = CoordinateComplex.from_rex(rex)
        if defect == "target-chain":
            d = CoordinateComplex(c.spaces, (((0, 0, 1),), ((0, 0, 1),), ()))
        else:
            # A malformed stored B3 is live source data, not a forged proof flag.
            from rexgraph.native_sparse import as_native
            upper = as_native(rex._graded_duals[0])
            data = upper.data.copy()
            data[0] += 1
            rex._graded_duals[0] = upper.with_data(data).dual
            c = CoordinateComplex.from_rex(rex)
            d = CoordinateComplex(c.spaces, ((), (), ()))
        p = GradedMap(c, d, ((), (), (), ()))
    else:
        p = p.verify()
        rex.remove_edges([1, 1])
        rex._ensure_clean()
        rex.add_edges([1, 2], [0, 1])
        rex._ensure_clean()
        assert (rex.nV, rex.nE) == (3, 2)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((ValueError, TypeError)):
        run(rex, call("CHAIN_MAP", param("p")), explain=explain, p=p)


@pytest.mark.parametrize("defect", ["missing", "shapes", "nnz", "oversize-nnz", "coordinates", "time", "domain", "construction"])
def test_descriptor_checks_are_not_length_only(defect, monkeypatch):
    import rcql.executor
    rex, p = make()
    typ = _carrier_literal(bind("r", rex, SourcePolicy.allow("*")), p)
    desc = typ.graded_map
    if defect == "missing":
        typ = replace(typ, graded_map=None)
    elif defect == "shapes":
        typ = replace(typ, graded_map=replace(desc, shapes=((3, 3), (1, 2))))
    elif defect in {"nnz", "oversize-nnz"}:
        typ = replace(typ, graded_map=replace(desc, nnz=(-1, 1) if defect == "nnz" else (3, 10)))
    elif defect == "coordinates":
        typ = replace(typ, graded_map=replace(desc, domain=(replace(desc.domain[0], keys=("2", "1", "0")), desc.domain[1])))
    elif defect == "time":
        typ = replace(typ, temporal=TemporalRef(version=2))
    elif defect == "domain":
        typ = replace(typ, domain=Domain.REAL)
    else:
        typ = replace(typ, graded_map=replace(desc, construction="unverified-guess"))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("CHAIN_MAP", param("p")), explain=True, p=typ)


def test_abstract_or_computed_declaration_defers_proof_never_claims_it_checked():
    rex, p = make()
    typ = _carrier_literal(bind("r", rex, SourcePolicy.allow("*")), p)
    result = run(rex, call("CHAIN_MAP", param("p")), explain=True, p=typ)
    node = result.values[0]["returns"][0]
    assert node["result"]["graded_map"]["chain_preserving"] is None
    assert any(f["name"] == "chain_map_certificate" and f["status"] == "deferred" for f in node["predicates"])
    result = run(rex, call("CHAIN_MAP", call("CHAIN_MAP", param("p"))), p=p)
    assert isinstance(result.values[0], ChainMap)
    assert len(result.execution) == 2
    assert result.provenance[0]["result_type"]["graded_map"]["chain_preserving"] is None
    assert result.execution[-1]["methods"][0]["chain_preserving"] is True


def test_repeated_query_expression_reuses_one_execution_not_a_global_proof_cache():
    rex, p = make()
    expr = call("CHAIN_MAP", param("p"))
    result = run(rex, expr, expr, p=p)
    assert result.values[0] is result.values[1]
    assert len(result.execution) == 1
    again = run(rex, expr, p=p)
    assert again.values[0] is not result.values[0]


def test_read_policy_checked_before_any_proof_or_adapter(monkeypatch):
    import rcql.executor
    rex, p = make()
    monkeypatch.setattr(GradedMap, "verify", lambda *a: pytest.fail("proof ran without read"))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(PermissionError):
        Executor(sources={"r": BoundSource(rex, SourcePolicy.allow())}, params={"p": p}).execute(
            query(source("r"), call("CHAIN_MAP", param("p"))))


def test_core_composition_roundtrips_through_rcql_as_one_explicit_map():
    rex, p = make()
    d = p.codomain
    q = GradedMap(d, d, tuple(tuple((i, i, -2) for i in range(n)) for n in d.sizes))
    composed = p.verify().then(q.verify())
    result = run(rex, call("CHAIN_MAP", param("p")), p=composed)
    assert result.values[0].declaration.components == composed.declaration.components
    assert result.values[0].commutation_residuals == (0,)


def test_raw_declaration_metadata_does_not_claim_chain_preservation():
    rex, p = make()
    result = run(rex, param("p"), p=p)
    assert result.values[0] is p
    assert result.provenance[0]["result_type"]["graded_map"]["chain_preserving"] is None
    assert result.exactness[0].value == "structural"


@pytest.mark.parametrize("fixture", ["branch", "witness", "loop", "empty", "grade3"])
def test_native_full_tower_not_a_pairwise_only_read(fixture):
    rex = (RexGraph.from_cells(solid_octahedron_3rex()) if fixture == "grade3" else
           RexGraph.from_cells({"branch": [4, [[0, 1, 2, 3]]], "witness": [1, [[0]]],
                                "loop": [1, [[0, 0]]], "empty": [1, []]}[fixture]))
    c = CoordinateComplex.from_rex(rex)
    p = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))
    result = run(rex, call("CHAIN_MAP", param("p")), p=p)
    assert result.values[0].commutation_residuals == (0,) * (len(c.sizes)-1)
    assert result.provenance[0]["result_type"]["graded_map"]["shapes"] == [[n, n] for n in c.sizes]
