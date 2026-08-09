"""The stack end to end: files in, a trained model and a stored, queryable complex out.

Every layer the agent exposes, exercised on one joined complex rather than each in
isolation, because the thing worth checking is that the layers compose. A capability
that works alone and cannot be reached from the layer above it is not wired.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
from agent.knowledge import join
from tests.test_knowledge_roundtrip import BRCA_GAF, BRCA_OBO, GTF

pytest.importorskip("torch")  # the ml extra: these build or train models


def _big_ontology(n_terms=60, n_annotations=80, seed=0):
    """An ontology and annotation set large enough for a split to mean something."""
    random.seed(seed)
    obo = "format-version: 1.2\n"
    for i in range(n_terms):
        obo += f"\n[Term]\nid: T:{i}\nname: t{i}\n"
        if i:
            obo += f"is_a: T:{random.randrange(0, i)}\n"
    rows = ["!gaf-version: 2.2"]
    for i in range(n_annotations):
        gene = f"G{i % 30}"
        rows.append("\t".join([
            "UniProtKB", f"P{i:05d}", gene, "involved_in",
            f"T:{random.randrange(n_terms)}", "PMID:1", "IDA", "", "P", gene, gene,
            "protein", "taxon:9606", "2026", "U", "", ""]))
    return obo, "\n".join(rows) + "\n"


@pytest.fixture
def study(tmp_path):
    paths = []
    for name, text in (("genes.gtf", GTF), ("goa.gaf", BRCA_GAF), ("go.obo", BRCA_OBO)):
        p = tmp_path / name
        p.write_text(text)
        paths.append(str(p))
    return join(*paths)


@pytest.fixture
def large(tmp_path):
    obo, gaf = _big_ontology()
    a, b = tmp_path / "big.obo", tmp_path / "big.gaf"
    a.write_text(obo)
    b.write_text(gaf)
    return join(str(a), str(b))


#### the NN layer


def test_a_joined_complex_becomes_a_training_bundle(large):
    from agent.warehouse.source import knowledge_bundle
    b = knowledge_bundle(large)
    assert b.kind == "hypergraph"
    assert b.X.shape[0] == large.nE
    assert b.meta["feat_dim"] == b.X.shape[1]
    assert set(b.splits) == {"train", "val", "test"}
    assert b.meta["classes"], "the target values are unnamed"


def test_the_relation_target_is_the_relation_type(large):
    """Learning to tell an `is_a` from an annotation out of structure alone needs no
    external label: the type channel already carries it."""
    from agent.warehouse.source import knowledge_bundle
    b = knowledge_bundle(large, target="relation")
    assert b.meta["n_classes"] >= 2
    assert any("is_a" in c for c in b.meta["classes"])


def test_a_model_actually_trains_on_it(large):
    """The claim the whole NN path rests on. Anything less than chance would mean the
    structural features carry nothing about the relation."""
    from agent.models.archetypes import get
    from agent.models.train import train_one
    from agent.warehouse.source import knowledge_bundle

    b = knowledge_bundle(large)
    spec = get("hgnn")
    cfg = dict(spec["defaults"])
    cfg.update({"feat_dim": b.meta["feat_dim"], "n_classes": b.meta["n_classes"]})
    out = train_one(spec["build"](cfg, b), b, steps=80, seed=0)
    assert out["steps"] == 80
    assert out["metric"] > 0.5, (
        f"test accuracy {out['metric']} is at or below chance, so the structural "
        f"features carry nothing about the relation type")


def test_an_unknown_target_is_refused(large):
    from agent.warehouse.source import knowledge_bundle
    with pytest.raises(ValueError, match="target"):
        knowledge_bundle(large, target="bogus")


#### structural health, and what a seed reaches


def test_health_reads_the_hodge_split_of_the_structure(study):
    h = study.health()
    assert h["n_nodes"] == study.nV
    assert h["status"] in ("acyclic", "draining", "circulating")
    assert h["draining"] + h["circulating"] == pytest.approx(1.0, abs=1e-6)


def test_health_names_the_entities_every_path_runs_through(study):
    h = study.health()
    assert h["bottlenecks"], "no bottleneck was identified at all"
    assert h["bottlenecks"][0]["node"] == "BRCA1", \
        "the most connected entity is not the top bottleneck"


def test_an_empty_complex_has_no_health_to_report():
    from agent.knowledge import Knowledge
    empty = Knowledge({}, [], {}, {"n_collisions": 0})
    assert empty.health()["status"] == "empty"


def test_a_seed_reaches_what_it_is_connected_to(study):
    field = study.propagate(["BRCA1"], t=1.0)
    assert field.shape[0] == study.nE
    assert np.abs(field).sum() > 0
    reached = {i for i, v in enumerate(field) if abs(v) > 1e-9}
    touching = {i for i, (a, _r, b, _o) in enumerate(study.edges)
                if "BRCA1" in (study.display(a), study.display(b))}
    assert touching <= reached, "a relation on the seed itself was not reached"


def test_a_seed_that_touches_nothing_is_refused(study):
    with pytest.raises(ValueError, match="touches no relation"):
        study.propagate(["NOT_AN_ENTITY"])


def test_a_seed_of_the_wrong_length_is_refused(study):
    with pytest.raises(ValueError, match="values for"):
        study.propagate(np.ones(study.nE + 3))


#### every layer, reachable from the agent layer


def test_every_capability_has_an_mcp_tool():
    """The agent layer is the exposed software, so a capability with no tool is a
    capability a model driving the stack cannot use."""
    from agent.mcp_tools import TOOLS
    for name in ("rexgraph_join_sources", "rexgraph_reason_ontology",
                 "rexgraph_enrich", "rexgraph_release_series",
                 "rexgraph_term_similarity", "rexgraph_homology",
                 "rexgraph_structure_health", "rexgraph_propagate",
                 "rexgraph_query_stored"):
        assert name in TOOLS, f"{name} is not reachable through MCP"


def test_every_capability_has_a_builder_step():
    from agent.builder import AgentBuilder
    steps = set(AgentBuilder.available_steps())
    for name in ("knowledge", "ontology_reason", "enrichment", "releases"):
        assert name in steps, f"{name} cannot be composed in the builder"


def test_the_health_and_propagate_tools_run(tmp_path):
    from agent.mcp_tools import call
    paths = []
    for name, text in (("genes.gtf", GTF), ("goa.gaf", BRCA_GAF), ("go.obo", BRCA_OBO)):
        p = tmp_path / name
        p.write_text(text)
        paths.append(str(p))
    health = call("rexgraph_structure_health", files=paths)
    assert health["health"]["n_nodes"] == 8
    reached = call("rexgraph_propagate", files=paths, seed=["BRCA1"])
    assert reached["reached"], "propagation reported nothing"
