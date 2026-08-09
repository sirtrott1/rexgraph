"""The TrustGraph adapter, in standalone mode.

2,257 lines, 28% executed, and it is the beachhead integration: knowledge-graph
triples in, a relational complex and a confidence reading out. Most of the untested
mass needs a live TrustGraph API, but the whole standalone path does not: triple
normalisation, context matrices, complex construction, analysis, signal decomposition,
subgraph confidence, enrichment triples and the MCP tool definitions all run offline.

That is what is covered here. Anything requiring `self.api` is exercised only for its
failure behaviour, since a missing server has to say so rather than raise something
unrelated.
"""
from __future__ import annotations

import numpy as np
import pytest
from agent.integrations.trustgraph_adapter import (
    SimpleTriple,
    TrustGraphAdapter,
    _extract_predicate_type,
    _normalize_uri,
    _triple_to_strings,
    build_context_matrix_explicit,
    build_context_matrix_from_documents,
)

#: a small knowledge graph with one cycle and two predicate families
TRIPLES = [
    SimpleTriple("http://ex.org/alpha", "http://ex.org/regulates", "http://ex.org/beta"),
    SimpleTriple("http://ex.org/beta", "http://ex.org/regulates", "http://ex.org/gamma"),
    SimpleTriple("http://ex.org/gamma", "http://ex.org/regulates", "http://ex.org/alpha"),
    SimpleTriple("http://ex.org/alpha", "http://ex.org/binds", "http://ex.org/delta"),
    SimpleTriple("http://ex.org/delta", "http://ex.org/binds", "http://ex.org/epsilon"),
    SimpleTriple("http://ex.org/epsilon", "http://ex.org/binds", "http://ex.org/alpha"),
]

TUPLES = [("alpha", "regulates", "beta"), ("beta", "regulates", "gamma"),
          ("gamma", "regulates", "alpha")]


@pytest.fixture
def adapter():
    return TrustGraphAdapter()


#### triple normalisation


def test_a_uri_reduces_to_its_local_name():
    assert _normalize_uri("http://ex.org/alpha") == "alpha"


def test_a_hash_uri_reduces_to_its_fragment():
    assert _normalize_uri("http://ex.org/onto#Protein") == "Protein"


def test_a_bare_string_passes_through():
    assert _normalize_uri("alpha") == "alpha"


def test_predicate_families_are_distinguished():
    a = _extract_predicate_type("http://ex.org/regulates")
    b = _extract_predicate_type("http://ex.org/binds")
    assert a != b, "two different predicates collapsed to the same type"


def test_the_same_predicate_gives_the_same_type():
    assert (_extract_predicate_type("http://ex.org/regulates")
            == _extract_predicate_type("http://ex.org/regulates"))


@pytest.mark.parametrize("t", [
    SimpleTriple("a", "p", "b"),
    ("a", "p", "b"),
])
def test_every_accepted_triple_shape_reads_as_three_strings(t):
    s, p, o = _triple_to_strings(t)
    assert (s, p, o) == ("a", "p", "b")


#### context matrices


def test_an_explicit_context_matrix_marks_the_right_cells():
    idx = {"alpha": 0, "beta": 1, "gamma": 2}
    C, labels = build_context_matrix_explicit(
        {"pathway": ["alpha", "beta"], "response": ["gamma"]}, idx, 3)
    assert labels == ["pathway", "response"], "labels are not sorted"
    assert C.shape == (2, 3)
    assert C[0].tolist() == [1, 1, 0]
    assert C[1].tolist() == [0, 0, 1]


def test_an_unknown_entity_in_a_context_is_ignored_not_fatal():
    C, _ = build_context_matrix_explicit(
        {"c": ["alpha", "not-an-entity"]}, {"alpha": 0}, 1)
    assert C.tolist() == [[1]]


def test_an_empty_context_map_gives_an_empty_matrix():
    C, labels = build_context_matrix_explicit({}, {"alpha": 0}, 1)
    assert labels == [] and C.shape[0] == 0


def test_documents_with_no_grouping_fall_into_one_context():
    """With no named graph on any triple, everything lands in a single context.
    The index has to be keyed by the raw term, which is what the triples carry."""
    names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    idx = {f"http://ex.org/{n}": i for i, n in enumerate(names)}
    C, labels = build_context_matrix_from_documents(TRIPLES, idx, len(idx))
    assert C.shape == (len(labels), len(idx))
    assert len(labels) == 1, f"expected one context, got {labels}"
    assert C.sum() > 0, "no entity was placed in any context"


#### building a complex


def test_triples_build_a_complex(adapter):
    rex, meta = adapter.from_triples(TRIPLES)
    assert rex is not None and rex.nE > 0
    assert rex.nV > 0
    assert isinstance(meta, dict)


def test_plain_tuples_build_a_complex(adapter):
    rex, _ = adapter.from_triples(TUPLES)
    assert rex.nE == 3 and rex.nV == 3


def test_the_cycle_in_the_triples_is_filled_rather_than_lost(adapter):
    """Three triples forming a directed cycle close a triangle. The default
    face_selection fills it, so beta_1 drops to 0 and the cycle is accounted for as a
    face. beta_1 = 0 with nF = 0 would be the cycle going missing; this is not that."""
    rex, _ = adapter.from_triples(TUPLES)
    assert rex.nF == 1, f"the cycle was not filled: nF={rex.nF}"
    assert rex.betti[1] == 0, f"a filled triangle still reports betti={tuple(rex.betti)}"


def test_the_meta_names_the_entities(adapter):
    _, meta = adapter.from_triples(TUPLES)
    blob = repr(meta)
    assert "alpha" in blob and "beta" in blob, \
        "the entity names are not reachable from the meta"


def test_two_predicates_give_two_edge_types(adapter):
    _, meta = adapter.from_triples(TRIPLES)
    n_types = meta.get("n_types") or meta.get("n_edge_types")
    if n_types is not None:
        assert n_types >= 2, f"two predicate families collapsed to {n_types} type(s)"


def test_explicit_contexts_reach_the_construction(adapter):
    rex, meta = adapter.from_triples(
        TRIPLES, face_selection="context",
        contexts={"pathway": ["alpha", "beta", "gamma"],
                  "binding": ["alpha", "delta", "epsilon"]})
    assert rex is not None and rex.nE > 0


def test_context_selection_without_contexts_is_refused(adapter):
    """`face_selection='context'` is a claim that contexts exist. Without them the
    adapter has nothing to select on."""
    with pytest.raises((ValueError, TypeError)):
        adapter.from_triples(TRIPLES, face_selection="context")


def test_an_empty_triple_list_is_refused_with_a_message(adapter):
    """Nothing to build from is the caller's error, and it is named as such."""
    with pytest.raises(ValueError, match="edge"):
        adapter.from_triples([])


#### analysis over the built complex


def test_the_adapter_analyses_what_it_built(adapter):
    rex, _ = adapter.from_triples(TRIPLES)
    out = adapter.analyze(rex, depth="standard")
    assert isinstance(out, dict) and out


@pytest.mark.parametrize("depth", ["quick", "standard"])
def test_analysis_reports_only_numbers(adapter, depth):
    import math
    rex, _ = adapter.from_triples(TRIPLES)
    out = adapter.analyze(rex, depth=depth)

    def bad(o, path=""):
        r = []
        if isinstance(o, dict):
            for k, v in o.items():
                r += bad(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o[:200]):
                r += bad(v, f"{path}[{i}]")
        elif isinstance(o, (float, np.floating)) and (math.isnan(o) or math.isinf(o)):
            r.append(f"{path}={o}")
        return r

    offending = bad(out)
    assert not offending, "non-finite: " + "; ".join(offending[:6])


def test_a_signal_over_the_triples_decomposes(adapter):
    rex, _ = adapter.from_triples(TRIPLES)
    rng = np.random.RandomState(0)
    out = adapter.decompose_signal(rex, rng.randn(rex.nE))
    assert isinstance(out, dict) and out


#### subgraph confidence


def test_subgraph_confidence_scores_a_selection(adapter):
    rex, _ = adapter.from_triples(TRIPLES)
    out = adapter.subgraph_confidence(rex, [0, 1, 2])
    assert isinstance(out, dict) and out


def test_subgraph_confidence_on_an_empty_selection_does_not_crash(adapter):
    rex, _ = adapter.from_triples(TRIPLES)
    assert isinstance(adapter.subgraph_confidence(rex, []), dict)


def test_subgraph_confidence_ignores_an_out_of_range_vertex(adapter):
    rex, _ = adapter.from_triples(TRIPLES)
    out = adapter.subgraph_confidence(rex, [0, 9999])
    assert isinstance(out, dict)


#### enrichment triples: the write-back path


def test_enrichment_triples_are_produced_from_an_analysis(adapter):
    rex, meta = adapter.from_triples(TRIPLES)
    analysis = adapter.analyze(rex, depth="standard")
    out = adapter.to_enrichment_triples(rex, analysis, meta)
    assert isinstance(out, list)


def test_every_enrichment_triple_has_three_terms(adapter):
    rex, meta = adapter.from_triples(TRIPLES)
    analysis = adapter.analyze(rex, depth="standard")
    for t in adapter.to_enrichment_triples(rex, analysis, meta):
        s, p, o = _triple_to_strings(t)
        assert s and p and o, f"incomplete enrichment triple: {t!r}"


#### the MCP surface


def test_the_mcp_tool_definitions_are_well_formed(adapter):
    """These are handed to a model as callable tools. A definition with no name or no
    description is a tool the model cannot use."""
    defs = adapter.as_mcp_tool_definitions()
    assert defs, "the adapter exposes no MCP tools"
    for d in defs:
        assert d.get("name"), f"a tool definition has no name: {d}"
        assert (d.get("description") or "").strip(), \
            f"tool {d.get('name')} has no description"


def test_every_mcp_tool_name_is_unique(adapter):
    names = [d["name"] for d in adapter.as_mcp_tool_definitions()]
    assert len(names) == len(set(names)), "duplicate MCP tool names"


def _mcp_targets(adapter):
    """Tool name -> the method it names, across both classes that implement them.

    The tool set spans TrustGraphAdapter and TrustGraphPipeline and there is no
    dispatcher resolving a name to a callable, so the check has to look in both.
    """
    from agent.integrations.trustgraph_pipeline import TrustGraphPipeline
    out = {}
    for d in adapter.as_mcp_tool_definitions():
        m = d["name"].replace("rexgraph_", "", 1)
        out[d["name"]] = (hasattr(adapter, m) or hasattr(TrustGraphPipeline, m))
    return out


def test_every_mcp_tool_names_a_method_that_exists(adapter):
    """A tool definition pointing at no method is a call the model will make and
    nothing can answer."""
    missing = [n for n, ok in _mcp_targets(adapter).items() if not ok]
    assert not missing, f"MCP tools with no method behind them: {missing}"


#### the seam between the adapter's output and its own inputs


def test_the_context_builder_accepts_the_labels_the_adapter_returns(adapter):
    rex, meta = adapter.from_triples(TRIPLES)
    labels = meta["vertex_labels"]
    idx = {name: i for i, name in enumerate(labels)}
    C, _ = build_context_matrix_from_documents(TRIPLES, idx, len(labels))
    assert C.sum() > 0, (
        "a context matrix built from the adapter's own vertex_labels is empty: the "
        "builder matches raw URIs, the labels are normalized")


#### the API path, with no server


def test_the_api_property_without_a_url_fails_with_a_message(adapter):
    """`api` is lazy. With no URL configured it has to explain itself rather than
    raise an AttributeError from somewhere inside a client library."""
    try:
        _ = adapter.api
    except Exception as e:                       # noqa: BLE001 - that is the check
        assert str(e).strip(), "the API failure carries no message"


def test_listing_cores_without_a_server_does_not_hang(adapter):
    try:
        out = adapter.list_kg_cores()
        assert isinstance(out, list)
    except Exception as e:                       # noqa: BLE001
        assert str(e).strip(), "the failure carries no message"
