"""Check ingest validation and secret store handling through Agent entry points."""
import tempfile

import numpy as np
import pytest


def test_correlation_signs_and_weights_survive_auto_rex():
    from agent.auto import auto_rex
    matrix = np.array([[1, -.9, .1], [-.9, 1, .8], [.1, .8, 1]])
    rex = auto_rex(matrix, threshold=.5, typing="none", face_selection="none")
    np.testing.assert_array_equal(rex._edge_signs, [-1, 1])
    np.testing.assert_allclose(rex.w_E, [.9, .8])


def test_dataframe_ingest_needs_no_temporary_file(monkeypatch):
    pandas = pytest.importorskip("pandas")
    from agent.auto import auto_rex
    def forbidden(*args, **kwargs):
        raise AssertionError("DataFrame ingest created a temporary file")
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", forbidden)
    rex = auto_rex(pandas.DataFrame({"source": ["a", "b"], "target": ["b", "c"]}))
    assert rex.nE == 2
    assert rex.nV == 3


@pytest.mark.parametrize("module", ["rcdb", "integrations"])
def test_http_text_never_opens_a_server_path(tmp_path, monkeypatch, module):
    from agent.adapters.text import TextAdapter
    from importlib import import_module
    path = tmp_path / "private.txt"
    path.write_text("server file contents must remain private")
    seen = []
    sentinel = object()
    def capture(self, text, **kwargs):
        seen.append(text)
        return sentinel
    monkeypatch.setattr(TextAdapter, "build", capture)
    import agent.auto as auto
    monkeypatch.setattr(auto, "build_rex_from_edges", lambda *args, **kwargs: sentinel)
    routes = import_module("agent.server.routes." + module)
    rex, source = routes._rex_from_body({"text": str(path)})
    assert rex is sentinel
    assert seen == [str(path)]


def test_corpus_supplied_text_never_opens_a_server_path(tmp_path, monkeypatch):
    from agent import cache
    from agent.adapters import EdgeConstruction
    from agent.adapters.text import TextAdapter
    from agent.corpus import CorpusBuilder
    path = tmp_path / "private.txt"
    path.write_text("the server file must stay separate from request text")
    seen = []
    keys = []
    original_key = cache.content_key
    def capture_key(content, **kwargs):
        keys.append(content)
        return original_key(content, **kwargs)
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "content_key", capture_key)
    def capture(self, text, **kwargs):
        seen.append(text)
        return EdgeConstruction(
            sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32),
            weights=np.ones(2), signs=np.ones(2), type_labels=np.zeros(2, np.int32),
            vertex_labels=["a", "b", "c"], n_types=1, type_names=["edge"],
        )
    monkeypatch.setattr(TextAdapter, "build", capture)
    corpus = CorpusBuilder()
    corpus.add_text(str(path))
    corpus.build(depth="quick")
    assert seen == [str(path)]
    assert keys == [str(path).encode("utf-8")]


@pytest.mark.parametrize("backend", ["file", "env"])
@pytest.mark.parametrize("damaged", ["{broken", "[]", '{"saved": {"wrong": "field"}}'])
@pytest.mark.parametrize("operation", ["put", "delete"])
def test_corrupt_secrets_refuse_mutation(tmp_path, backend, damaged, operation):
    from agent.secrets import EnvSecretStore, FileSecretStore
    path = tmp_path / "secrets.json"
    path.write_text(damaged)
    cls = FileSecretStore if backend == "file" else EnvSecretStore
    store = cls(str(path))
    with pytest.raises(ValueError, match="refusing to modify"):
        if operation == "put":
            store.put("new", "reference")
        else:
            store.delete("saved")
    assert path.read_text() == damaged


def test_source_document_ids_do_not_collide_by_basename(tmp_path):
    from agent.corpus.ingest import doc_id_for
    a = tmp_path / "one" / "foo.txt"
    b = tmp_path / "two" / "foo.txt"
    assert doc_id_for(a) != doc_id_for(b)
    assert doc_id_for(a) == doc_id_for(a.parent / "." / a.name)


def test_legacy_resume_requires_an_unambiguous_basename(tmp_path):
    from agent.corpus.ingest import pending
    from types import SimpleNamespace
    store = SimpleNamespace(_idx={"foo": []})
    a, b = str(tmp_path / "one/foo.txt"), str(tmp_path / "two/foo.txt")
    assert pending(store, [a, b]) == [a, b]
    assert pending(store, [a]) == []


def test_corpus_identity_uses_content_and_preserves_display_label(tmp_path):
    from agent.corpus import CorpusBuilder
    a, b = tmp_path / "one/foo.txt", tmp_path / "two/foo.txt"
    a.parent.mkdir(); b.parent.mkdir()
    a.write_text("first document"); b.write_text("second document")
    corpus = CorpusBuilder()
    aid = corpus.add_document(str(a))
    bid = corpus.add_document(str(b))
    assert aid != bid
    c = tmp_path / "renamed.txt"
    c.write_bytes(a.read_bytes())
    assert corpus.add_document(str(c)) == aid
    assert corpus.documents[0].meta["source_label"] == "foo.txt"


@pytest.mark.parametrize("extension", ["txt", "csv"])
def test_directory_identity_uses_content_and_retains_source(tmp_path, extension):
    from agent.corpus import CorpusBuilder
    first, second, copied = (tmp_path / name for name in ("one", "two", "copy"))
    for directory in (first, second, copied):
        directory.mkdir()
    filename = "foo." + extension
    (first / filename).write_text("first document")
    (second / filename).write_text("second document")
    (copied / filename).write_bytes((first / filename).read_bytes())
    corpus = CorpusBuilder()
    aid, = corpus.add_directory(str(first))
    bid, = corpus.add_directory(str(second))
    cid, = corpus.add_directory(str(copied))
    assert aid != bid
    assert cid == aid
    assert corpus.documents[0].source == str(first / filename)
    assert corpus.documents[0].meta["source_label"] == filename


def test_declared_isolated_vertices_invalidate_topology_caches():
    from rexgraph.graph import RexGraph
    rex = RexGraph.from_graph([0], [1])
    assert (rex.B1_sparse.nrow, rex.B1_sparse.ncol) == (2, 1)
    assert rex.betti[0] == 1
    rex._ensure_vertex_count(3)
    assert (rex.B1_sparse.nrow, rex.B1_sparse.ncol) == (3, 1)
    assert rex.betti[0] == 2


def test_knowledge_labels_follow_declared_order_including_isolates():
    from agent.knowledge import Knowledge
    knowledge = Knowledge(
        entities={"isolated": ["isolated"], "b": ["b"], "a": ["a"]},
        edges=[("a", "relates", "b", "source")],
        labels={"isolated": "Isolated", "a": "A", "b": "B"}, report={},
    )
    construction = knowledge.edge_construction()
    assert construction.vertex_labels == ["Isolated", "B", "A"]
    assert construction.sources.tolist() == [2]
    assert construction.targets.tolist() == [1]
    rex = knowledge.rex(face_selection="none")
    assert rex.nV == 3
    assert rex.betti[0] == 2
    assert rex._agent_meta["vertex_labels"] == construction.vertex_labels
