"""Every file type, through every consumer, as one complex.

The point of this file is the matrix. A format that parses but cannot be stored, or
stores but cannot be searched, or is searchable but produces no features, is not
usable; and which of those it fails at is not obvious from the parser passing.

So every ontology format and every scientific container is taken through the same
eight stations: join, complex, RCDB store, RCDB search, `.rex` archive, corpus
document, analysis pipeline, structural features, and triples out.

The join itself is checked on the case that matters: a genome annotation, a GO
annotation set and the ontology, which name the same gene three different ways and
have no join key in common until each file's own cross-references are followed.
"""
from __future__ import annotations

import numpy as np
import pytest
from agent.knowledge import Knowledge, join
from tests.test_ontology_formats import GAF, NT, OBO, OBOGRAPH, OWL, TTL
from tests.test_science_formats import BED, FASTA, GFF, PDB, SDF, VCF

#: a GTF, a GAF and an OBO that describe the same genes with no shared spelling
GTF = (
    'chr17\tHAVANA\tgene\t43044295\t43125483\t.\t-\t.\t'
    'gene_id "ENSG00000012048"; gene_name "BRCA1"; Dbxref "HGNC:HGNC:1100";\n'
    'chr17\tHAVANA\ttranscript\t43044295\t43125483\t.\t-\t.\t'
    'gene_id "ENSG00000012048"; transcript_id "ENST00000357654";'
    ' gene_name "BRCA1";\n'
    'chr13\tHAVANA\tgene\t32315474\t32400266\t.\t+\t.\t'
    'gene_id "ENSG00000139618"; gene_name "BRCA2";\n'
    'chr17\tHAVANA\tgene\t43100000\t43130000\t.\t-\t.\t'
    'gene_id "ENSG00000267595"; gene_name "RPL21P4";\n'
)

BRCA_GAF = "\n".join([
    "!gaf-version: 2.2",
    "\t".join(["UniProtKB", "P38398", "BRCA1", "involved_in", "GO:0006281",
               "PMID:1", "IDA", "", "P", "BRCA1 protein", "BRCA1|RNF53",
               "protein", "taxon:9606", "20260101", "UniProt", "", ""]),
    "\t".join(["UniProtKB", "P51587", "BRCA2", "involved_in", "GO:0006281",
               "PMID:2", "IDA", "", "P", "BRCA2 protein", "BRCA2|FANCD1",
               "protein", "taxon:9606", "20260101", "UniProt", "", ""]),
    "\t".join(["UniProtKB", "P38398", "BRCA1", "located_in", "GO:0005634",
               "PMID:3", "IDA", "", "C", "BRCA1 protein", "BRCA1",
               "protein", "taxon:9606", "20260101", "UniProt", "", ""]),
]) + "\n"

BRCA_OBO = """format-version: 1.2
ontology: go

[Term]
id: GO:0006281
name: DNA repair
namespace: biological_process
is_a: GO:0006974 ! cellular response to DNA damage stimulus
alt_id: GO:0006284

[Term]
id: GO:0006974
name: cellular response to DNA damage stimulus
namespace: biological_process
is_a: GO:0008150

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0005634
name: nucleus
namespace: cellular_component
"""

#: every format that reaches a complex from a file, by extension
ONTOLOGY_FILES = {"go.obo": OBO, "go.obojson": OBOGRAPH, "o.owl": OWL,
                  "o.ttl": TTL, "o.nt": NT, "a.gaf": GAF}
SCIENCE_FILES = {"m.sdf": SDF, "p.pdb": PDB, "s.fasta": FASTA, "v.vcf": VCF,
                 "a.gff": GFF, "g.gtf": GTF, "b.bed": BED}
ALL_FILES = {**ONTOLOGY_FILES, **SCIENCE_FILES}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import default_store, reset_default_store
    reset_default_store()
    yield default_store()
    reset_default_store()


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


@pytest.fixture
def study(tmp_path):
    """The three-file case: genome annotation, GO annotation, ontology."""
    return [_write(tmp_path, "genes.gtf", GTF),
            _write(tmp_path, "goa.gaf", BRCA_GAF),
            _write(tmp_path, "go.obo", BRCA_OBO)]


#### the join


def test_three_files_that_share_no_spelling_still_join(study):
    """The GTF calls it `ENSG00000012048` and `BRCA1`; the GAF calls it `P38398`,
    `BRCA1` and `RNF53`; the OBO calls the term `GO:0006281`. Nothing matches across
    files until each file's own cross-references are followed."""
    k = join(*study)
    joined = {j["entity"] for j in k.report["joined"]}
    assert "BRCA1" in joined, f"the gene did not join across files: {joined}"
    assert k.report["n_joined"] >= 2


def test_a_gene_reaches_the_ontology_through_the_annotation(study):
    """The whole point: a genomic feature connected to a term hierarchy."""
    triples = set(join(*study).triples())
    assert ("BRCA1", "involved_in", "DNA repair") in triples
    assert ("DNA repair", "is_a",
            "cellular response to DNA damage stimulus") in triples


def test_a_bare_accession_resolves_to_the_name_the_ontology_gives_it(study):
    """The GAF says `located_in GO:0005634` and nothing else. The OBO names that
    term and never relates it, so it is declared and unreferenced there; without
    carrying declared terms into the index the entity reads as an accession."""
    assert ("BRCA1", "located_in", "nucleus") in set(join(*study).triples())


def test_a_transcript_does_not_merge_into_its_own_gene(study):
    """`gene_name` on a transcript row names the transcript's GENE. Treating it as an
    alias collapses the two into one vertex and turns their relation into a
    self-loop, which is a silent loss of resolution."""
    k = join(*study)
    assert "ENST00000357654" in k.entities, "the transcript lost its identity"
    assert ("ENST00000357654", "part_of", "BRCA1") in set(k.triples())
    assert not [e for e in k.edges if e[0] == e[2]], "a self-loop was produced"


def test_the_join_reports_no_collisions_on_clean_input(study):
    assert join(*study).report["collisions"] == []


def test_a_collision_is_reported_rather_than_merged(tmp_path):
    """One file naming two features by one identifier is a fact about the file. It
    must be reported, because a wrong join produces a complex richer than the
    evidence."""
    gtf = ('chr1\t.\tgene\t1\t100\t.\t+\t.\tgene_id "A"; gene_name "SHARED";\n'
           'chr1\t.\tgene\t200\t300\t.\t+\t.\tgene_id "B"; gene_name "SHARED";\n')
    k = join(_write(tmp_path, "dup.gtf", gtf))
    assert k.report["n_collisions"] >= 1
    assert any(c["identifier"] == "shared" for c in k.report["collisions"])


def test_edge_types_say_which_file_asserted_the_relation(study):
    ec = join(*study).edge_construction()
    assert any(n.startswith("go.obo:") for n in ec.type_names)
    assert any(n.startswith("goa.gaf:") for n in ec.type_names)
    assert any(n.startswith("genes.gtf:") for n in ec.type_names)


def test_joining_one_file_is_the_same_as_reading_it(tmp_path):
    k = join(_write(tmp_path, "go.obo", BRCA_OBO))
    assert k.nE == 2 and k.report["n_sources"] == 1


def test_joining_nothing_returns_an_empty_result():
    k = join()
    assert isinstance(k, Knowledge) and k.nV == 0 and k.nE == 0


def test_an_unjoinable_type_is_refused():
    with pytest.raises(TypeError, match="cannot join"):
        join(42)


#### the matrix: every format through every station


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_file_joins(tmp_path, name):
    k = join(_write(tmp_path, name, ALL_FILES[name]))
    assert k.nE > 0, f"{name} produced no relations"
    assert k.nV > 0


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_file_becomes_a_complex(tmp_path, name):
    rex = join(_write(tmp_path, name, ALL_FILES[name])).rex()
    assert rex.nE > 0 and rex.nV > 0
    assert rex.betti is not None


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_file_stores_and_reloads(tmp_path, store, name):
    k = join(_write(tmp_path, name, ALL_FILES[name]))
    k.store(store, "rec")
    back = store.get("rec")
    assert (back.nV, back.nE) == (k.rex().nV, k.rex().nE)
    assert tuple(back.betti) == tuple(k.rex().betti)


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_stored_file_is_searchable_by_its_entities(tmp_path, store, name):
    """A record nothing can find is a record that is not stored."""
    k = join(_write(tmp_path, name, ALL_FILES[name]))
    k.store(store, "rec")
    entity = k.display(next(iter(k.entities)))
    hits = [r.id for r in store.query(labels_any=[entity])]
    assert "rec" in hits, f"{name}: searching for {entity!r} did not find the record"


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_stored_file_is_searchable_by_its_source_kind(tmp_path, store, name):
    k = join(_write(tmp_path, name, ALL_FILES[name]))
    k.store(store, "rec")
    kind = k.parts[0].kind
    assert "rec" in [r.id for r in store.query(tags_any=[kind])], \
        f"{name}: not findable by its own kind {kind!r}"


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_file_archives_to_rex_and_back(tmp_path, name):
    from rexgraph.io import load_rex, save_rex
    k = join(_write(tmp_path, name, ALL_FILES[name]))
    rex = k.rex()
    path = str(tmp_path / "archive.rex")
    save_rex(path, rex)
    back = load_rex(path)
    assert (back.nV, back.nE) == (rex.nV, rex.nE)
    assert tuple(back.betti) == tuple(rex.betti)
    labels = (getattr(back, "_agent_meta", {}) or {}).get("vertex_labels")
    assert labels, f"{name}: the archive lost its entity names"


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_file_becomes_a_corpus_document(tmp_path, name):
    from agent.corpus import CorpusBuilder
    k = join(_write(tmp_path, name, ALL_FILES[name]))
    c = CorpusBuilder()
    c.add_document(source=name, doc_id=name, text="",
                   edge_construction=k.edge_construction())
    c.build(depth="standard")
    doc = c.documents[0]
    assert doc.rex is not None and doc.rex.nE > 0
    assert doc.analysis, f"{name}: the corpus produced no analysis"


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_file_runs_the_analysis_pipeline(tmp_path, name):
    from agent.pipeline import AnalysisPipeline
    out = AnalysisPipeline(join(_write(tmp_path, name, ALL_FILES[name])).rex()).run(
        depth="full")
    assert isinstance(out, dict) and out


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_file_produces_structural_features(tmp_path, name):
    """The training signal for a file with no prose in it."""
    k = join(_write(tmp_path, name, ALL_FILES[name]))
    X, names, y, classes = k.features()
    assert X.shape[0] == k.nE, f"{name}: {X.shape[0]} feature rows for {k.nE} edges"
    assert X.shape[1] == len(names) and X.shape[1] > 0
    assert np.isfinite(X).all(), f"{name}: features contain a non-finite value"
    assert y.shape[0] == X.shape[0]
    assert len(classes) >= 1


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_a_file_comes_out_as_triples(tmp_path, name):
    k = join(_write(tmp_path, name, ALL_FILES[name]))
    triples = k.triples()
    assert len(triples) == k.nE
    for s, p, o in triples:
        assert s and p and o


#### the joined study, through the same stations


def test_the_study_stores_with_every_source_as_a_tag(study, store):
    k = join(*study)
    k.store(store, "brca")
    tags = store.get_record("brca").signature["tags"]
    assert {"knowledge", "gff", "gaf", "obo"} <= set(tags), tags


def test_the_study_is_found_by_a_gene_and_by_a_term(study, store):
    join(*study).store(store, "brca")
    assert [r.id for r in store.query(labels_any=["BRCA1"])] == ["brca"]
    assert [r.id for r in store.query(labels_any=["DNA repair"])] == ["brca"]
    assert [r.id for r in store.query(labels_all=["BRCA1", "nucleus"])] == ["brca"]


def test_the_study_keeps_its_join_report_through_the_store(study, store):
    join(*study).store(store, "brca")
    meta = store.get_record("brca").meta
    assert meta["join"]["n_joined"] >= 2
    assert meta["join"]["sources"], "the record forgot which files built it"


def test_the_study_feeds_trustgraph(study):
    from agent.integrations.trustgraph_adapter import TrustGraphAdapter
    a = TrustGraphAdapter()
    rex, meta = a.from_triples(join(*study).triples())
    assert rex.nE > 0
    assert isinstance(a.analyze(rex, depth="standard"), dict)


def test_the_study_feeds_a_session(study, tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "s.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import get_store
    k = join(*study)
    s = get_store().create(name="brca-study")
    s.add_snapshot(rex=k.rex(), action="join", params={}, results={},
                   summary="genome + annotation + ontology")
    assert len(s.snapshots) == 1
    reset_default_store()


def test_the_study_features_separate_the_sources(study):
    """The relation type is a real supervised target: which file asserted this, and
    what did it assert."""
    X, _names, y, classes = join(*study).features()
    assert len(set(y.tolist())) >= 3, "every relation landed in one class"
    assert len(classes) == len(set(classes))


def test_a_complex_can_be_joined_back_in(study):
    """A stored complex is a source like any other, so a study can be extended."""
    base = join(study[2])                      # the ontology alone
    extended = join(base.rex(), study[1], origins=["go", "goa.gaf"])
    assert extended.nE > base.nE
    assert extended.report["n_sources"] == 2


#### mixing kinds


def test_an_ontology_and_a_molecule_coexist_without_joining(tmp_path):
    """Two files about unrelated things produce one complex with two components and
    no invented bridge between them."""
    k = join(_write(tmp_path, "go.obo", BRCA_OBO),
             _write(tmp_path, "m.sdf", SDF))
    assert k.report["n_joined"] == 0, "unrelated files were joined"
    assert int(k.rex().betti[0]) >= 2, "the components were merged"


def test_every_ontology_format_joins_with_a_science_file(tmp_path):
    """No pair may raise, whatever the two describe."""
    for oname, otext in ONTOLOGY_FILES.items():
        for sname, stext in SCIENCE_FILES.items():
            k = join(_write(tmp_path, oname, otext),
                     _write(tmp_path, sname, stext))
            assert k.nE > 0, f"{oname} + {sname} produced nothing"


#### features that can be batched


def test_every_format_produces_the_same_feature_layout(tmp_path):
    """`nhats` is adaptive: a channel that is identically zero is not carried, so a
    pair of disjoint edges reports two channels and a connected complex reports four.
    Reading one complex, that is right. Learning across many, column 2 has to mean
    the same thing in every row, so the character block is a fixed four."""
    widths, layouts = set(), set()
    for name, text in ALL_FILES.items():
        _X, names, _y, _c = join(_write(tmp_path, name, text)).features()
        widths.add(len(names))
        layouts.add(tuple(names))
    assert len(widths) == 1, f"feature width varies across formats: {sorted(widths)}"
    assert len(layouts) == 1, "the feature layout is not stable across formats"


def test_the_four_channels_are_always_present(tmp_path):
    _X, names, _y, _c = join(_write(tmp_path, "b.bed", BED)).features()
    for channel in ("L1_down", "L_O", "L_SG", "L_C"):
        assert f"char_{channel}" in names, f"{channel} is missing from the features"


def test_an_inactive_channel_is_zero_not_absent(tmp_path):
    """Two overlapping pairs on different sequences give two relations that share no
    entity, so co-participation carries nothing. The column stays and reads zero."""
    bed = ("chr1\t0\t10\ta\nchr1\t5\t15\tb\n"
           "chr2\t0\t10\tc\nchr2\t5\t15\td\n")
    k = join(_write(tmp_path, "d.bed", bed))
    assert k.nE == 2, f"expected two disjoint relations, got {k.triples()}"
    X, names, _y, _c = k.features()
    col = names.index("char_L_C")
    assert np.allclose(X[:, col], 0.0), "an inactive channel carried a value"


def test_a_file_with_no_relations_says_so(tmp_path):
    """Two intervals that never overlap relate nothing. An empty complex is not a
    result, and building one silently would store a record with no content."""
    bed = "chr1\t0\t10\ta\nchr2\t100\t200\tb\n"
    with pytest.raises(ValueError, match="no relations"):
        join(_write(tmp_path, "d.bed", bed)).rex()


def test_features_from_two_files_stack(tmp_path):
    """The batching case the fixed layout exists for."""
    a = join(_write(tmp_path, "go.obo", BRCA_OBO)).features()[0]
    b = join(_write(tmp_path, "m.sdf", SDF)).features()[0]
    stacked = np.vstack([a, b])
    assert stacked.shape == (a.shape[0] + b.shape[0], a.shape[1])


#### the routes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app
    from fastapi.testclient import TestClient
    yield TestClient(app)
    reset_default_store()


def _post(client, *files, **data):
    return client.post("/api/v1/knowledge/join",
                       files=[("files", (n, t.encode(), "text/plain"))
                              for n, t in files],
                       data=data)


def test_the_formats_route_lists_both_registries(client):
    body = client.get("/api/v1/knowledge/formats").json()
    assert ".obo" in body["ontology_extensions"]
    assert ".gtf" in body["science_extensions"]
    assert set(body["extensions"]) >= set(body["ontology_extensions"])


def test_no_ontology_reader_leaks_into_the_science_list(client):
    """Ontology readers are registered in the file registry too, so the science list
    has to exclude them or the screen offers `.obo` twice under two names."""
    body = client.get("/api/v1/knowledge/formats").json()
    assert not [r for r in body["science"].values() if r.startswith("ontology_")]


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_any_single_file_joins_through_the_route(client, name):
    r = _post(client, (name, ALL_FILES[name]))
    assert r.status_code == 200, r.text[:200]
    assert r.json()["n_relations"] > 0


def test_the_study_joins_through_the_route(client):
    r = _post(client, ("genes.gtf", GTF), ("goa.gaf", BRCA_GAF),
              ("go.obo", BRCA_OBO))
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["report"]["n_joined"] >= 2
    assert [s["origin"] for s in body["report"]["sources"]] == [
        "genes.gtf", "goa.gaf", "go.obo"]
    rels = {tuple(t[:3]) for t in body["relations"]}
    assert ("BRCA1", "involved_in", "DNA repair") in rels


def test_a_relation_says_which_file_asserted_it(client):
    body = _post(client, ("genes.gtf", GTF), ("go.obo", BRCA_OBO)).json()
    assert {t[3] for t in body["relations"]} == {"genes.gtf", "go.obo"}


def test_the_route_stores_and_the_record_is_searchable(client):
    body = _post(client, ("genes.gtf", GTF), ("goa.gaf", BRCA_GAF),
                 ("go.obo", BRCA_OBO), store_id="brca", tags="study").json()
    assert body.get("stored_as") == "brca", body.get("store_error")
    from agent.rcdb import default_store
    hits = [r.id for r in default_store().query(labels_any=["BRCA1"])]
    assert hits == ["brca"]


def test_a_file_that_cannot_be_read_is_named_not_fatal(client):
    r = _post(client, ("go.obo", BRCA_OBO), ("junk.owl", "<not xml"))
    assert r.status_code == 200, r.text[:200]
    assert r.json()["failed_files"][0]["file"] == "junk.owl"


def test_an_upload_of_only_unreadable_files_is_refused(client):
    r = _post(client, ("junk.owl", "<not xml"))
    assert r.status_code == 400


def test_files_that_relate_nothing_are_refused_with_a_reason(client):
    r = _post(client, ("d.bed", "chr1\t0\t10\ta\nchr2\t100\t200\tb\n"))
    assert r.status_code == 400
    assert "no relations" in r.json()["detail"]


def test_the_route_sends_only_numbers(client):
    """The join report reaches the browser, so nothing in it may be non-finite."""
    import json
    r = _post(client, ("genes.gtf", GTF), ("goa.gaf", BRCA_GAF))
    assert r.status_code == 200
    json.loads(r.text)


#### schemas, joined with the rest


DDL = """CREATE TABLE gene (id INT PRIMARY KEY, symbol TEXT);
CREATE TABLE transcript (id INT PRIMARY KEY, gene_id INT REFERENCES gene(id));
CREATE TABLE annotation (id INT PRIMARY KEY, gene_id INT REFERENCES gene(id));"""

MODEL_OBO = """format-version: 1.2
ontology: model

[Term]
id: gene
name: gene

[Term]
id: transcript
name: transcript
is_a: gene
"""


@pytest.fixture
def schema():
    from agent.schema_complex import parse_schema_ddl
    return parse_schema_ddl(DDL)


def test_a_schema_joins_as_tables_and_foreign_keys(schema):
    k = join(schema, origins=["warehouse.sql"])
    assert ("transcript", "references", "gene") in set(k.triples())
    assert k.parts[0].kind == "schema"


def test_a_schema_and_an_ontology_align_on_one_complex(tmp_path, schema):
    """The conformance question: does the table the database calls `transcript`
    correspond to the class the ontology calls `transcript`, and does the foreign key
    between two tables agree with the subsumption between two classes. Both relations
    land on the same pair of vertices, so the comparison is a type filter rather than
    a second model."""
    k = join(schema, _write(tmp_path, "model.obo", MODEL_OBO),
             origins=["warehouse.sql", "model.obo"])
    joined = {j["entity"] for j in k.report["joined"]}
    assert {"gene", "transcript"} <= joined
    triples = set(k.triples())
    assert ("transcript", "references", "gene") in triples
    assert ("transcript", "is_a", "gene") in triples


def test_a_table_the_ontology_does_not_model_is_not_joined(tmp_path, schema):
    """`annotation` exists in the database and not in the ontology. It stays, and it
    stays unjoined: a schema element the model does not cover is the finding."""
    k = join(schema, _write(tmp_path, "model.obo", MODEL_OBO),
             origins=["warehouse.sql", "model.obo"])
    joined = {j["entity"] for j in k.report["joined"]}
    assert "annotation" in k.entities
    assert "annotation" not in joined


def test_a_schema_reaches_the_store_and_the_features(schema, store):
    k = join(schema, origins=["warehouse.sql"])
    k.store(store, "schema-rec")
    assert "schema" in store.get_record("schema-rec").signature["tags"]
    X, names, y, classes = k.features()
    assert X.shape[0] == k.nE and len(names) == X.shape[1]


def test_a_schema_an_ontology_and_a_genome_are_one_complex(tmp_path, schema):
    """All three kinds at once, which is the whole point."""
    k = join(schema,
             _write(tmp_path, "model.obo", MODEL_OBO),
             _write(tmp_path, "genes.gtf", GTF),
             origins=["warehouse.sql", "model.obo", "genes.gtf"])
    assert k.report["n_sources"] == 3
    kinds = {p.kind for p in k.parts}
    assert kinds == {"schema", "obo", "gff"}
    assert k.rex().nE == k.nE


#### recommendations: what would add to this join


def test_an_annotation_alone_asks_for_its_ontology(tmp_path):
    """A GAF names GO terms and defines none of them."""
    k = join(_write(tmp_path, "goa.gaf", BRCA_GAF))
    kinds = {r["kind"] for r in k.recommendations()}
    assert "annotation_without_ontology" in kinds
    go = [r for r in k.recommendations() if r.get("namespace") == "GO"]
    assert go, "the GO terms were not noticed"
    assert "go.obo" in go[0]["action"]
    assert go[0]["n_affected"] >= 1
    assert go[0]["examples"], "the recommendation names no example"


def test_a_recommendation_names_where_the_file_is_published(tmp_path):
    """Naming a file without saying where it lives is not actionable. Nothing is
    fetched: this is a pointer, not a download."""
    k = join(_write(tmp_path, "goa.gaf", BRCA_GAF))
    go = [r for r in k.recommendations() if r.get("namespace") == "GO"][0]
    assert "obolibrary" in go["published"]


def test_an_ontology_alone_asks_for_data(tmp_path):
    k = join(_write(tmp_path, "go.obo", BRCA_OBO))
    assert "ontology_without_data" in {r["kind"] for r in k.recommendations()}


def test_the_complete_study_is_not_told_to_add_anything(study):
    """A genome annotation, its GO annotations and the ontology is the whole set.
    Suggesting anything there would be noise, which is what makes the other
    suggestions worth reading."""
    assert join(*study).recommendations() == []


def test_a_namespace_is_not_recommended_when_its_definer_is_loaded(tmp_path):
    """The GTF is present, so its Ensembl identifiers having no prettier name is a
    fact about the format, not a missing file."""
    k = join(_write(tmp_path, "genes.gtf", GTF),
             _write(tmp_path, "goa.gaf", BRCA_GAF))
    assert not [r for r in k.recommendations() if r.get("namespace") == "Ensembl"]
    assert [r for r in k.recommendations() if r.get("namespace") == "GO"]


def test_files_that_share_nothing_are_told_so(tmp_path):
    k = join(_write(tmp_path, "go.obo", BRCA_OBO), _write(tmp_path, "m.sdf", SDF))
    assert "no_shared_entities" in {r["kind"] for r in k.recommendations()}


def test_a_collision_becomes_a_recommendation(tmp_path):
    gtf = ('chr1\t.\tgene\t1\t100\t.\t+\t.\tgene_id "A"; gene_name "SHARED";\n'
           'chr1\t.\tgene\t200\t300\t.\t+\t.\tgene_id "B"; gene_name "SHARED";\n')
    k = join(_write(tmp_path, "dup.gtf", gtf))
    rec = [r for r in k.recommendations() if r["kind"] == "collisions"]
    assert rec and rec[0]["n_affected"] >= 1


def test_every_recommendation_says_what_to_do(tmp_path):
    """A finding with no action is an observation, and the screen has those already."""
    for files in ([_write(tmp_path, "goa.gaf", BRCA_GAF)],
                  [_write(tmp_path, "go.obo", BRCA_OBO)],
                  [_write(tmp_path, "go.obo", BRCA_OBO),
                   _write(tmp_path, "m.sdf", SDF)]):
        for r in join(*files).recommendations():
            assert r.get("detail", "").strip(), f"{r['kind']} explains nothing"
            assert r.get("action", "").strip(), f"{r['kind']} suggests nothing"


def test_unresolved_groups_by_namespace(tmp_path):
    k = join(_write(tmp_path, "goa.gaf", BRCA_GAF))
    assert "GO" in k.unresolved()
    assert all(i.upper().startswith("GO:") for i in k.unresolved()["GO"])


def test_a_named_term_is_not_unresolved(study):
    """Once the ontology is loaded, GO:0006281 reads as DNA repair and is not
    reported as an unnamed reference."""
    assert "GO" not in join(*study).unresolved()


def test_namespace_detection_is_exact():
    from agent.knowledge import namespace_of
    assert namespace_of("GO:0006281") == "GO"
    assert namespace_of("UniProtKB:P38398") == "UniProtKB"
    assert namespace_of("ENSG00000012048") == "Ensembl"
    assert namespace_of("BRCA1") is None
    assert namespace_of("") is None


def test_the_route_carries_the_recommendations(client):
    body = _post(client, ("goa.gaf", BRCA_GAF)).json()
    recs = body["report"]["recommendations"]
    assert recs and any(r["kind"] == "annotation_without_ontology" for r in recs)


#### scale


def test_the_join_is_not_quadratic():
    """A regression guard with a lot of headroom.

    The join was O(groups x identifier-sets): it scanned every identifier set once
    per group. At 20k terms that was 322 seconds, and at real GO scale it did not
    finish. Attributing each set to its group in one pass makes it near-linear, and
    45k terms with 400k annotations now joins in under three seconds.

    The bound below is ~30x the linear time and ~1/10th the quadratic time, so it
    fails on a return of the old shape without being sensitive to the machine.
    """
    import random
    import time

    random.seed(0)
    n_terms, n_annotations = 10000, 40000
    obo = ["format-version: 1.2", ""]
    for i in range(n_terms):
        obo.append(f"[Term]\nid: GO:{i:07d}\nname: term {i}")
        if i > 10:
            obo.append(f"is_a: GO:{random.randrange(0, i):07d}")
        obo.append("")
    gaf = ["!gaf-version: 2.2"]
    for _ in range(n_annotations):
        g = random.randrange(2000)
        gaf.append("\t".join([
            "UniProtKB", f"P{g:05d}", f"GENE{g}", "involved_in",
            f"GO:{random.randrange(n_terms):07d}", "PMID:1", "IDA", "", "P",
            f"protein {g}", f"GENE{g}", "protein", "taxon:9606", "2026", "U",
            "", ""]))

    from agent.adapters.ontology_formats import parse
    parts = [parse("\n".join(obo), "obo"), parse("\n".join(gaf), "gaf")]
    start = time.monotonic()
    k = join(*parts, origins=["go.obo", "goa.gaf"])
    elapsed = time.monotonic() - start

    assert k.nE == n_terms - 11 + n_annotations
    assert k.report["n_joined"] > 0, "nothing joined, so the timing means nothing"
    assert elapsed < 30.0, (
        f"joining {n_terms} terms took {elapsed:.1f}s. The join is scanning all "
        f"identifier sets per group again.")


#### querying inside a stored complex


def test_cells_can_be_selected_by_a_structural_invariant(client, study):
    """`/query` filters which records match; this filters which cells inside one, on
    quantities the complex computes rather than on attributes anyone recorded."""
    from agent.rcdb import default_store
    join(*study).store(default_store(), "brca")
    r = client.post("/api/v1/db/cells/brca",
                    json={"where": [{"quantity": "kappa", "op": "<",
                                     "threshold": 0.9}]})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["grade"] == "vertex"
    assert 0 < body["n_selected"] <= body["n_cells"]
    assert all(isinstance(c, str) for c in body["cells"]), \
        "cells came back as indices rather than names"


def test_two_clauses_compose(client, study):
    from agent.rcdb import default_store
    join(*study).store(default_store(), "brca")
    one = client.post("/api/v1/db/cells/brca", json={"where": [
        {"quantity": "kappa", "op": ">", "threshold": 0.0}]}).json()["n_selected"]
    both = client.post("/api/v1/db/cells/brca", json={"where": [
        {"quantity": "kappa", "op": ">", "threshold": 0.0},
        {"quantity": "phi", "op": ">", "threshold": 0.2, "channel": 1},
    ], "combine": "and"}).json()["n_selected"]
    assert both <= one, "AND selected more than one of its clauses alone"


def test_predicates_on_different_grades_are_refused(client, study):
    """kappa is per vertex and chi is per edge. Combining them into one mask would
    silently compare arrays of different lengths."""
    from agent.rcdb import default_store
    join(*study).store(default_store(), "brca")
    r = client.post("/api/v1/db/cells/brca", json={"where": [
        {"quantity": "kappa", "op": ">", "threshold": 0.5},
        {"quantity": "chi", "op": ">", "threshold": 0.1}]})
    assert r.status_code == 400
    assert "grade" in r.json()["detail"]


def test_an_unknown_quantity_is_refused_with_the_list(client, study):
    from agent.rcdb import default_store
    join(*study).store(default_store(), "brca")
    r = client.post("/api/v1/db/cells/brca", json={"where": [
        {"quantity": "pagerank", "op": ">", "threshold": 0.5}]})
    assert r.status_code == 400 and "kappa" in r.json()["detail"]


def test_explain_names_the_vertex_it_describes(client, study):
    from agent.rcdb import default_store
    join(*study).store(default_store(), "brca")
    body = client.get("/api/v1/db/explain/brca?dim=0&idx=0").json()
    assert body["label"]
    assert "kappa" in body and "dominant_channel" in body


def test_explain_returns_json_not_numpy(client, study):
    """The kernels answer in arrays, and FastAPI's encoder runs before the response
    class. An explanation that computes perfectly came back as a 500."""
    import json

    from agent.rcdb import default_store
    join(*study).store(default_store(), "brca")
    for dim in (0, 1):
        r = client.get(f"/api/v1/db/explain/brca?dim={dim}&idx=0")
        assert r.status_code == 200, r.text[:200]
        json.loads(r.text)


def test_explain_out_of_range_says_the_size(client, study):
    from agent.rcdb import default_store
    join(*study).store(default_store(), "brca")
    r = client.get("/api/v1/db/explain/brca?dim=0&idx=9999")
    assert r.status_code == 400 and "out of range" in r.json()["detail"]


def test_querying_a_record_that_does_not_exist_is_a_404(client):
    r = client.post("/api/v1/db/cells/nope",
                    json={"where": [{"quantity": "kappa", "op": ">",
                                     "threshold": 0.0}]})
    assert r.status_code == 404


#### parity against the dense join kernel


#: two ontology fragments that overlap on named terms, so an exact label match is
#: enough to align them and the dense kernel has something to join on
OBO_LEFT = """format-version: 1.2

[Term]
id: T:1
name: alpha
is_a: T:2

[Term]
id: T:2
name: beta
is_a: T:3

[Term]
id: T:3
name: gamma
"""

OBO_RIGHT = """format-version: 1.2

[Term]
id: T:2
name: beta
is_a: T:4

[Term]
id: T:4
name: delta
is_a: T:3

[Term]
id: T:3
name: gamma
"""


def test_the_sparse_join_agrees_with_the_dense_kernel(tmp_path):
    """`_joins.outer_join` is the oracle, not the path.

    The kernel takes DENSE `B1`/`B2`, which at 65k x 490k is 255 GB, so the assembly
    in `agent.knowledge` stays sparse and union-find based. That is only defensible if
    the two agree where the dense one can run, so this pins them on a size it can.

    The input is two fragments that overlap on NAMED terms, because the kernel aligns
    by exact label match and nothing else. That is a second reason it cannot be the
    path: a GAF calls a term `GO:0006281` where the OBO calls it `DNA repair`, and
    resolving that is what the cross-reference join does and what the kernel does not.
    """
    from rexgraph.core import _joins

    left = join(_write(tmp_path, "left.obo", OBO_LEFT), origins=["left"])
    right = join(_write(tmp_path, "right.obo", OBO_RIGHT), origins=["right"])
    rl, rr = left.rex(face_selection="none"), right.rex(face_selection="none")
    labels_l = [left.display(c) for c in left.entities]
    labels_r = [right.display(c) for c in right.entities]

    shared = np.asarray(_joins.build_shared_vertex_map(labels_l, labels_r))
    assert (shared >= 0).sum() == 2, \
        f"expected beta and gamma to align, got {shared.tolist()}"

    dense = _joins.outer_join(
        np.asarray(rl.B1, dtype=np.float64), np.asarray(rl.B2, dtype=np.float64),
        rl.nV, rl.nE, rl.nF,
        np.asarray(rr.B1, dtype=np.float64), np.asarray(rr.B2, dtype=np.float64),
        rr.nV, rr.nE, rr.nF,
        shared.astype(np.int32))

    both = join(_write(tmp_path, "left.obo", OBO_LEFT),
                _write(tmp_path, "right.obo", OBO_RIGHT),
                origins=["left", "right"])
    mine = both.rex(face_selection="none")

    assert int(mine.nV) == int(dense["nVj"]), (
        f"entity count differs: sparse {mine.nV}, dense kernel {dense['nVj']}")
    assert int(mine.nE) == int(dense["nEj"]), (
        f"relation count differs: sparse {mine.nE}, dense kernel {dense['nEj']}")
    assert float(dense["chain_residual"]) == pytest.approx(0.0), \
        "the kernel's own join broke the chain condition, so it is not a valid oracle"

    kernel_betti = [int(b) for b in np.asarray(dense["beta"]).ravel()]
    mine_betti = [int(b) for b in mine.betti]
    assert mine_betti[:len(kernel_betti)] == kernel_betti, (
        f"homology differs: sparse {mine_betti}, dense kernel {kernel_betti}")


def test_the_dense_kernel_cannot_resolve_cross_references(tmp_path):
    """The other reason it is an oracle and not the path.

    A GAF names a term by accession and an OBO names it by label. `build_shared_vertex_map`
    matches labels exactly, so it aligns nothing; the cross-reference join aligns them
    because the files declare the correspondence themselves.
    """
    from rexgraph.core import _joins

    onto = join(_write(tmp_path, "go.obo", BRCA_OBO), origins=["obo"])
    anno = join(_write(tmp_path, "goa.gaf", BRCA_GAF), origins=["gaf"])
    shared = np.asarray(_joins.build_shared_vertex_map(
        [onto.display(c) for c in onto.entities],
        [anno.display(c) for c in anno.entities]))
    assert (shared < 0).all(), "the labels happened to match; pick a sharper fixture"

    both = join(_write(tmp_path, "go.obo", BRCA_OBO),
                _write(tmp_path, "goa.gaf", BRCA_GAF), origins=["obo", "gaf"])
    assert both.report["n_joined"] >= 2
    assert ("BRCA1", "involved_in", "DNA repair") in set(both.triples())


def test_the_dense_join_kernel_is_not_the_scale_path():
    """Recorded so the choice is not revisited by accident: the kernel materialises
    B1 densely, so its memory is nV x nE floats. At the scale this stack targets that
    is not a tuning question."""
    n_vertices, n_edges = 65_000, 490_000
    gigabytes = n_vertices * n_edges * 8 / 1e9
    assert gigabytes > 100, (
        "the dense join's footprint stopped being prohibitive; re-evaluate whether "
        "the sparse assembly is still the right path")


#### reachable from the agent builder


def test_the_ontology_template_runs_end_to_end(tmp_path, study):
    """knowledge -> ontology_reason -> enrichment as one composed agent, which is what
    makes the new work usable rather than merely present."""
    from agent.builder import AgentBuilder

    cfg = AgentBuilder.template("ontology")
    for step in cfg["steps"]:
        if step["type"] == "enrichment":
            step["params"]["study"] = "BRCA1 BRCA2"
    res = AgentBuilder(cfg).run(files=study, query="")

    assert [s.step_type for s in res.steps] == [
        "knowledge", "ontology_reason", "enrichment"]
    failed = [(s.step_type, s.error) for s in res.steps if s.status != "ok"]
    assert not failed, failed
    assert res.steps[0].data["n_entities"] == 8
    assert res.steps[1].data["consistent"] is True


def test_every_new_step_is_registered_and_documented():
    from agent.builder import AgentBuilder
    steps = AgentBuilder.available_steps()
    for name in ("knowledge", "ontology_reason", "enrichment", "releases"):
        assert name in steps, f"{name} is not reachable from the builder"
        assert (AgentBuilder.step_help(name) or "").strip(), \
            f"{name} shows an empty description on the builder screen"


def test_a_reasoning_step_without_a_knowledge_step_says_so():
    """Skipped with a reason beats failing with a traceback."""
    from agent.builder import AgentBuilder
    res = AgentBuilder({"name": "x", "steps": [{"type": "ontology_reason"}]}).run(
        files=[], query="")
    assert res.steps[0].status == "ok"
    assert "skipped" in res.steps[0].data


@pytest.mark.parametrize("name", ["ontology", "releases"])
def test_the_new_templates_name_only_registered_steps(name):
    from agent.builder import AgentBuilder
    known = set(AgentBuilder.available_steps())
    used = {s["type"] for s in AgentBuilder.template(name)["steps"]}
    assert used <= known, f"{name} uses unregistered steps: {used - known}"


#### the warehouse takes a joined complex


def test_a_joined_complex_feeds_the_warehouse(study):
    """The warehouse pipeline reads `EdgeData`. A joined complex becomes one, so the
    existing tier split, feature build and bundle work unchanged."""
    import agent.warehouse.source as S

    k = join(*study)
    ed = S.edge_data_from_knowledge(k)
    rex = S.edge_complex(ed)
    assert int(rex.nE) == k.nE
    assert int(rex.nV) == k.nV

    mask = np.arange(len(ed.src_idx))
    X, names = S.edge_features(rex, ed, mask)
    assert X.shape == (k.nE, len(names))
    assert S.labels(ed, mask).shape[0] == k.nE


def test_an_entity_on_both_sides_stays_one_node(study):
    """`load_edges` indexes the source and target columns into disjoint ranges. A
    joined complex has ONE entity space, so an entity appearing as both a subject and
    an object must not become two nodes."""
    import agent.warehouse.source as S
    k = join(*study)
    ed = S.edge_data_from_knowledge(k)
    both = set(ed.src_idx.tolist()) & set(ed.dst_idx.tolist())
    assert both, "no entity appears on both sides, so the case is untested"
    assert S.edge_complex(ed).nV == k.nV


def test_the_tier_split_separates_regions_when_weighted_by_degree(study):
    import agent.warehouse.source as S
    k = join(*study)
    tiers = S.tier_split(S.edge_data_from_knowledge(k, weight_by="degree"), 3)
    assert sum(len(t) for t in tiers) == k.nE
    assert sum(1 for t in tiers if len(t)) > 1, "every relation landed in one tier"


def test_an_unknown_weighting_is_refused(study):
    import agent.warehouse.source as S
    with pytest.raises(ValueError, match="weight_by"):
        S.edge_data_from_knowledge(join(*study), weight_by="bogus")
