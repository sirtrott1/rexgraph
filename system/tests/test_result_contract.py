"""System renders RCQL's native result/provenance rather than inventing query semantics."""
import json
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest
from rexgraph.graph import RexGraph

from system.serialize import json_value


def test_artifact_bytes_are_identified_without_dumping_payloads():
    import hashlib
    payload = b"private artifact\x00\xff"
    assert json_value(payload) == {"kind": "ArtifactBytes", "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(), "payload": "omitted"}
    assert "private artifact" not in str(json_value(payload))


def test_exact_and_complex_arrays_are_strict_json_not_rounded_or_repr():
    value = np.array([[Fraction(17, 20), Fraction(-1, 3)]], dtype=object)
    result = json_value(value)
    assert result["values"] == [[{"numerator": 17, "denominator": 20}, {"numerator": -1, "denominator": 3}]]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    complex_result = json_value(np.array([complex(1, 2), complex(float("nan"), 0)]))
    assert complex_result["values"][0] == {"real": 1.0, "imaginary": 2.0}
    json.dumps(complex_result, allow_nan=False)


def test_bounded_samples_and_nonfinite_metadata_are_explicit():
    result = json_value(np.array([1., float("inf"), float("nan"), 2.]), max_values=2)
    assert result["sample"] == [1.0, {"nonfinite_float": "inf"}]
    assert result["min"] == 1.0 and result["max"] == 2.0
    assert json_value(np.float64("nan")) == {"nonfinite_float": "nan"}
    with pytest.raises(ValueError, match="max_values"):
        json_value(np.zeros(1), max_values=-1)
    json.dumps(result, allow_nan=False)


def test_dataclass_rendering_never_deepcopies_a_live_source_or_exposes_opaque_repr():
    class Opaque:
        def __repr__(self):
            pytest.fail("opaque repr was exposed")
        def __deepcopy__(self, memo):
            pytest.fail("live value was deep-copied")
    @dataclass
    class Carrier:
        source: object
        handle: object
        grade: int = 1
    assert json_value(Carrier(Opaque(), Opaque())) == {
        "handle": {"python_type": "Opaque"}, "grade": 1,
    }


def test_sheaf_result_transport_names_diagnostics_and_keeps_rational_residuals():
    from rcql import Executor, parse
    from rexgraph.sheaf import ExactSheaf

    rex = RexGraph.from_hypergraph([0, 3], [0, 1, 2])
    sh = ExactSheaf(rex, grade=0, stalk_dims=(2, 1, 1), mediator_dims=(1,))
    sh.assign(0, [Fraction(1, 3), Fraction(1, 6)])
    sh.assign(1, [Fraction(2, 3)])
    sh.assign(2, [Fraction(2, 3)])
    sh.restrict(0, [[1, 1]], mediator=0)
    result = Executor(sources={"s": rex}, params={"section": sh}).execute(parse(
        'FROM $s RETURN SECTION_CHECK($section), GLUE($section)'))
    compact, full = [json_value(value) for value in result.values]
    assert compact["kind"] == "ExactSectionCheck" and not compact["compatible"]
    assert compact["diagnostic"] == "anchor-incidence-residuals"
    assert compact["obstructions"][0]["residual"] == [{"numerator": -1, "denominator": 6}]
    assert full["diagnostic"] == "all-pair-incidence-residuals"
    assert full["ratio"] == {"numerator": 1, "denominator": 3}
    assert full["agreement_components"] == 2
    assert "H1" not in full and "H0" not in full
    json.dumps([compact, full], allow_nan=False)


def test_phrase_result_transport_retains_intersection_policy_and_contributors():
    from rcql import PhraseCorrespondence, PhraseSheaf, PhraseStalk, SourcePolicy, bind

    policy = SourcePolicy.allow("read", "identity")
    sh = PhraseSheaf((PhraseStalk("a", bind("state-a", object(), policy)),
                     PhraseStalk("b", bind("state-b", object(), SourcePolicy.allow("read")))),
                    (PhraseCorrespondence("compare", ("a", "b")),))
    sh.identity("a", "compare")
    sh.identity("b", "compare")
    sh.assign("a", [Fraction(1, 3)])
    sh.assign("b", [Fraction(1, 2)])
    out = json_value(sh.check_section())
    assert out["policy"]["permissions"] == ["read"]
    assert out["policy"]["digest"] == sh.policy.digest
    assert [r["name"] for r in out["contributors"]] == ["state-a", "state-b"]
    assert out["named_obstructions"][0]["correspondence"] == "compare"
    assert out["named_obstructions"][0]["exact"]["residual"] == [{"numerator": -1, "denominator": 6}]
    json.dumps(out, allow_nan=False)


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from system.server.app import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sources():
    from rcql import SourcePolicy

    from system.state import sources
    rcdb = pytest.importorskip("rcdb")
    graph = RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])
    store = rcdb.MemoryStore().configure_security(require_commits=True)
    store.commit_mutation("r", graph, analytics=False)
    sources.register("contract-graph", graph)
    sources.register("contract-db", store)
    sources.register("contract-denied", store, policy=SourcePolicy.allow("read"))
    yield store
    for name in ("contract-graph", "contract-db", "contract-denied"):
        sources.remove(name)


def test_query_api_preserves_exact_native_plan_and_provenance(client, sources):
    response = client.post("/api/query", json={
        "query": 'FROM REX("contract-graph") RETURN SHARE(CELL(1, 0)), CHARACTER(TRUE)',
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["native_plan"]["schema"] == "rcql.native-plan"
    assert body["execution"] and body["provenance"]
    assert body["values"][0]["kind"] == "Chain"
    assert {"numerator": 1, "denominator": 3} in body["values"][0]["values"]["values"]
    json.dumps(body, allow_nan=False)


def test_hash_and_governed_mutation_endpoints_use_rcql_contracts(client, sources):
    response = client.get("/api/rcdb/contract-db/state-hash")
    assert response.status_code == 200, response.text
    assert response.json()["digest"] == sources.state_digest()
    text = ('FROM RCDB("contract-db") MUTATE "copy" '
            'SET state = RCDB_GET("r"), expected_version = 0 COMMIT')
    explained = client.post("/api/query", json={"query": "EXPLAIN " + text})
    assert explained.status_code == 200, explained.text
    assert sources.history("copy") == []
    committed = client.post("/api/query", json={"query": text})
    assert committed.status_code == 200, committed.text
    assert committed.json()["provenance"][0]["record_version"] == 1
    assert sources.verify_commits("copy")
    stale = client.post("/api/query", json={"query": text})
    assert stale.status_code == 400
    assert "expected version 0" in stale.json()["detail"]
    denied = client.get("/api/rcdb/contract-denied/state-hash")
    assert denied.status_code == 400


def test_composed_named_results_preserve_labels_and_exact_values(client, sources):
    response = client.post("/api/query", json={
        "query": ('FROM REX(name="contract-graph") AS r '
                  'RETURN r.DESCRIBE().nE AS relations, [17/20, 1] AS exact_values'),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["aliases"] == ["relations", "exact_values"]
    assert body["values"] == [2, [{"numerator": 17, "denominator": 20}, 1]]
    assert body["exactness"] == ["integer", "rational"]
    assert body["native_plan"]["source_alias"] == "r"
    assert [item["alias"] for item in body["provenance"]] == body["aliases"]
    json.dumps(body, allow_nan=False)


def test_exact_integration_reads_one_pinned_rcdb_version(client, sources):
    response = client.post("/api/query", json={
        "query": ('FROM RCDB_VERSION(RCDB("contract-db"), "r", 1) '
                  'RETURN INTEGRATE(INDICATOR(CELL(0,0)), SHARE(CELL(1,0)), true) AS integral'),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["values"] == [{"numerator": 1, "denominator": 3}]
    assert body["aliases"] == ["integral"]
    assert body["exactness"] == ["rational"]
    assert body["native_plan"]["source"]["state"]["record_version"] == 1


def test_exact_adjoint_over_pinned_rcdb_preserves_math_and_http_contract(client, sources):
    response = client.post("/api/query", json={
        "query": ('FROM RCDB_VERSION(RCDB("contract-db"), "r", 1) '
                  'RETURN QUADRANCE(APPLY(ADJOINT(BOUNDARY(1)), SHARE(CELL(1,0)), true), true) AS energy'),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    # share=(1/3,1/3,0,1/3); B.T share=(1/3,-1/3), Q=2/9.
    assert body["values"] == [{"numerator": 2, "denominator": 9}]
    assert body["aliases"] == ["energy"] and body["exactness"] == ["rational"]
    assert body["native_plan"]["source"]["state"]["record_version"] == 1
    adjoint = next(n for n in body["native_plan"]["nodes"] if n.get("operator") == "ADJOINT")
    assert adjoint["result"]["operator"]["action_variance"] == "chain"
    json.dumps(body, allow_nan=False)


def test_weighted_dirac_preserves_graded_rcdb_state_and_exact_http_values(client, sources):
    response = client.post("/api/query", json={
        "query": ('FROM RCDB_VERSION(RCDB("contract-db"), "r", 1) '
                  'LET x=GRADED_CHAIN([SHARE(CELL(1,0))]) '
                  'LET d=DIRAC() LET a=ANTI_DIRAC() '
                  'RETURN APPLY(d,x,true) AS raised, APPLY(a,x,true) AS reversed, '
                  'QUADRANCE(GRADE_COMPONENT(APPLY(d,x,true),1),true) AS energy, d AS operator'),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["values"][0]["kind"] == "GradedChain"
    assert body["values"][0]["sizes"] == [4, 2]
    assert body["values"][0]["components"][1]["values"]["values"] == [
        {"numerator": 1, "denominator": 3}, {"numerator": -1, "denominator": 3}]
    assert body["values"][1]["components"][1]["values"]["values"] == [
        {"numerator": -1, "denominator": 3}, {"numerator": 1, "denominator": 3}]
    assert body["values"][2] == {"numerator": 2, "denominator": 9}
    assert body["values"][3]["kind"] == "GradedOperator"
    assert body["native_plan"]["source"]["state"]["record_version"] == 1
    assert [b["grade"] for b in body["provenance"][0]["result_type"]["graded_bases"]] == [0, 1]
    json.dumps(body, allow_nan=False)


def test_weighted_hodge_energy_preserves_pinned_rcdb_and_exact_http_values(client, sources):
    response = client.post("/api/query", json={
        "query": ('FROM RCDB_VERSION(RCDB("contract-db"), "r", 1) '
                  'LET x=SHARE(CELL(1,0)) '
                  'RETURN MOMENT(x,APPLY(HODGE_UP(0),x,true),none,true) AS energy'),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["values"] == [{"numerator": 2, "denominator": 9}]
    assert body["aliases"] == ["energy"] and body["exactness"] == ["rational"]
    assert body["native_plan"]["source"]["state"]["record_version"] == 1
    hodge = next(n for n in body["native_plan"]["nodes"] if n.get("operator") == "HODGE_UP")
    assert hodge["result"]["operator"]["metric_self_adjoint"] is True
    assert hodge["result"]["operator"]["metric_psd"] is True
    assert hodge["result"]["operator"]["psd"] is False
    json.dumps(body, allow_nan=False)


def test_metric_resolvent_preserves_chain_and_pinned_rcdb_http_contract(client, sources):
    response = client.post("/api/query", json={
        "query": ('FROM RCDB_VERSION(RCDB("contract-db"), "r", 1) '
                  'LET x=SHARE(CELL(1,0)) LET g=RESOLVENT(HODGE_UP(0),alpha=1/2) '
                  'LET y=GREEN_SOLVE(g,x) RETURN y, g, QUADRANCE(y)'),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["values"][0]["kind"] == "Chain"
    assert body["values"][0]["grade"] == 0
    assert body["values"][1]["solve_form"] == "positive-diagonal-metric"
    assert body["values"][1]["action_variance"] == "chain"
    assert len(body["values"][1]["metric_digest"]) == 64
    assert body["exactness"] == ["approximate", "structural", "approximate"]
    r = sources.get_version("r", 1)
    b = r.graded_boundaries()[0].toarray()
    expected = np.linalg.solve(np.eye(4)+.5*b@b.T, np.array([1/3, 1/3, 0, 1/3]))
    np.testing.assert_allclose(body["values"][0]["values"]["values"], expected)
    assert body["values"][2] == pytest.approx(expected@expected)
    assert body["native_plan"]["source"]["state"]["record_version"] == 1
    event = next(e for e in body["execution"] if e["operator"] == "GREEN_SOLVE")
    assert event["methods"][0]["kernel"] == "native-metric-block-cg"
    json.dumps(body, allow_nan=False)


def test_exact_bracket_keeps_pinned_rcdb_grades_and_operand_metadata(client, sources):
    response = client.post("/api/query", json={
        "query": ('FROM RCDB_VERSION(RCDB("contract-db"), "r", 1) '
                  'LET x=GRADED_CHAIN([SHARE(CELL(1,0))]) LET d=DIRAC() LET a=ANTI_DIRAC() '
                  'LET k=COMMUTATOR(d,a) RETURN APPLY(k,x,true), APPLY(ANTICOMMUTATOR(d,a),x,true), k'),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["exactness"] == ["rational", "rational", "structural"]
    values = body["values"][0]["components"][0]["values"]["values"]
    assert values == [{"numerator": n, "denominator": d} for n, d in [(-8, 9), (-2, 9), (4, 3), (-2, 9)]]
    for c in body["values"][1]["components"]:
        assert all(v == {"numerator": 0, "denominator": 1} for v in c["values"]["values"])
    handle = body["values"][2]
    assert handle["kind"] == "GradedOperatorBracket" and handle["metric_self_adjoint"]
    assert [op["name"] for op in handle["operands"]] == ["DIRAC", "ANTI_DIRAC"]
    assert body["native_plan"]["source"]["state"]["record_version"] == 1
    json.dumps(body, allow_nan=False)


def test_exact_channel_field_keeps_pinned_rcdb_and_rational_http_values(client, sources):
    response = client.post("/api/query", json={
        "query": ('FROM RCDB_VERSION(RCDB("contract-db"), "r", 1) '
                  'LET f=CHANNEL("F") LET x=INDICATOR(CELL(1,0)) '
                  'RETURN APPLY(f,x,true), f'),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    field, handle = body["values"]
    assert field["kind"] == "Field" and field["grade"] == 1
    assert field["values"]["values"] == [{"numerator": 8, "denominator": 3}, {"numerator": -8, "denominator": 3}]
    assert handle["exact_action"] and handle["exact_transpose"] and handle["exact_diagonal"]
    assert handle["channel"] == "F" and handle["frustration_reference"] == "raw-G"
    assert body["exactness"] == ["rational", "structural"]
    assert body["native_plan"]["source"]["state"]["record_version"] == 1
    event = next(e for e in body["execution"] if e["operator"] == "APPLY")
    assert event["methods"][0]["method"] == "rational-channel-action"
    json.dumps(body, allow_nan=False)


def test_normalized_channel_json_does_not_overstate_exact_capability():
    from rexgraph.channel_operator import channel_operator
    r = RexGraph.from_graph([0, 1], [1, 2], g_channel="normalized")
    body = json_value(channel_operator(r, "G"))
    assert body["exact_diagonal"] and not body["exact_action"] and not body["exact_transpose"]


def test_native_rank_betti_pinned_state_and_integer_transport(client, sources):
    response = client.post("/api/query", json={"query":
        'FROM RCDB_VERSION(RCDB("contract-db"), "r", 1) RETURN RANK(1), NULLITY(1), BETTI(0)'})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["values"] == [2, 0, 2]
    assert body["exactness"] == ["integer"]*3
    assert body["native_plan"]["source"]["state"]["record_version"] == 1
    assert body["execution"][0]["methods"][0]["method"] == "native-exact-rank"
    assert body["execution"][-1]["methods"][0]["chain_condition"] == "exact-zero"
    json.dumps(body, allow_nan=False)
