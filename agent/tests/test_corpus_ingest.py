"""Corpus ingest: the parts that only go wrong at scale.

The build itself is covered by `test_document.py`. What is tested here is the driver:
that a resumed run does not duplicate, that a finished artifact stores identically to one
built in-process, that one bad file does not end a run, and that the record still
addresses its own prose.
"""
from __future__ import annotations

import os

import pytest

from agent.corpus.ingest import DOC_TAGS, doc_id_for, ingest_corpus, ingest_one, pending
from agent.rcdb import FileStore, MemoryStore
from rexgraph.corpus_profile import ENGLISH_GUTENBERG

_BOOK = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
         "CHAPTER I\n\nMr. Jim went to town. He bought a hat, a coat, and a pie.\n\n"
         "The rain fell hard. Nobody minded it at all.\n\n"
         "CHAPTER II\n\nShe read the letter twice. Then she burned it.\n\n"
         "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")


@pytest.fixture
def corpus(tmp_path):
    """Three readable books and one that is not a document at all."""
    for i in range(3):
        (tmp_path / f"pg{i}.txt").write_text(_BOOK, encoding="utf-8")
    (tmp_path / "broken.txt").write_bytes(b"\x00" * 8)
    return [str(tmp_path / f"pg{i}.txt") for i in range(3)]


#### the artifact a worker hands back ###########################################

def test_a_prepared_put_stores_what_an_ordinary_put_stores(corpus):
    """The whole point of `put_prepared` is that it is the same record by a cheaper
    route. If the two diverge, the ingest is writing something the rest of the system
    did not agree to."""
    from rexgraph.document import build_document, read_document

    raw, _exact = read_document(corpus[0])
    rex, _info = build_document(raw, profile=ENGLISH_GUTENBERG)

    direct = MemoryStore()
    direct.put("d", rex, {"input_type": "document"}, list(DOC_TAGS), analytics=False)

    r = ingest_one(corpus[0], profile=ENGLISH_GUTENBERG)
    assert r["ok"], r.get("error")
    viaworker = MemoryStore()
    viaworker.put_prepared("d", r["blob"], r["sig"], r["meta"], list(DOC_TAGS))

    a, b = direct.get("d"), viaworker.get("d")
    assert int(a.nV) == int(b.nV) and int(a.nE) == int(b.nE)
    for k in ("nV", "nE", "betti1", "merkle_root", "sectionings", "chain_valid"):
        assert direct.get_record("d").signature.get(k) == r["sig"].get(k), k


def test_betti_is_in_the_signature_a_worker_produces(corpus):
    """It is queried (`min_betti1`/`max_betti1`), so a corpus written without it cannot
    answer the question the column exists for."""
    r = ingest_one(corpus[0], profile=ENGLISH_GUTENBERG)
    assert r["sig"]["betti"] is not None
    assert "betti1" in r["sig"]


def test_the_analytics_columns_are_off_by_default_and_betti_is_not(corpus):
    r = ingest_one(corpus[0], profile=ENGLISH_GUTENBERG)
    assert r["sig"].get("kappa_mean") is None
    assert r["sig"]["betti1"] >= 0


#### resume ####################################################################

def test_a_second_run_over_the_same_paths_writes_nothing(tmp_path, corpus):
    store = FileStore(str(tmp_path / "store"))
    first = ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    assert first["written"] == 3 and first["failed"] == 0

    second = ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    assert second["total"] == 0 and second["written"] == 0
    for p in corpus:
        assert len(store.history(doc_id_for(p))) == 1, "a resume must not add a version"


def test_a_resumed_run_finishes_only_the_remainder(tmp_path, corpus):
    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus[:1], store, profile=ENGLISH_GUTENBERG, workers=2)
    assert pending(store, corpus) == corpus[1:]
    rest = ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    assert rest["written"] == 2
    assert all(len(store.history(doc_id_for(p))) == 1 for p in corpus)


def test_resume_off_is_how_a_real_new_version_is_written(tmp_path, corpus):
    """Appending a version is correct when it is MEANT: the resume filter exists so it
    is never an accident."""
    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus[:1], store, profile=ENGLISH_GUTENBERG, workers=2)
    ingest_corpus(corpus[:1], store, profile=ENGLISH_GUTENBERG, workers=2, resume=False)
    assert len(store.history(doc_id_for(corpus[0]))) == 2


#### one bad file ##############################################################

def test_a_file_that_fails_is_recorded_and_the_run_continues(tmp_path):
    """61,354 documents means the run must survive whatever is in the corpus."""
    good = tmp_path / "good.txt"
    good.write_text(_BOOK, encoding="utf-8")
    missing = str(tmp_path / "not-here.txt")
    store = FileStore(str(tmp_path / "store"))
    out = ingest_corpus([str(good), missing], store, profile=ENGLISH_GUTENBERG,
                        workers=2)
    assert out["written"] == 1
    assert out["failed"] == 1
    assert out["failures"][0]["path"] == missing
    assert out["failures"][0]["error"]
    assert store.get(doc_id_for(str(good))) is not None


def test_an_empty_corpus_is_not_an_error(tmp_path):
    store = FileStore(str(tmp_path / "store"))
    assert ingest_corpus([], store, profile=ENGLISH_GUTENBERG)["total"] == 0


#### what the record carries ###################################################

def test_the_layers_survive_into_the_store(tmp_path, corpus):
    """A stored document that lost its sectionings is a bag of relations: the layers are
    what make a section addressable."""
    from rexgraph.sectioning import sectionings_of

    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    rex = store.get(doc_id_for(corpus[0]))
    got = sectionings_of(rex)
    assert "chapter" in got and "paragraph" in got
    assert got["chapter"].spans is not None, "a section must still address bytes"


def test_the_heap_pointer_resolves_to_the_documents_own_prose(tmp_path, corpus):
    """The text is not stored; the record points at it. A pointer that does not seek to
    the right bytes is worse than no pointer."""
    from rexgraph.document import section_text
    from rexgraph.sectioning import sectionings_of

    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    rec = store.get_record(doc_id_for(corpus[0]))
    assert rec.meta["encoding_exact"] is True
    assert os.path.exists(rec.meta["heap"])
    rex = store.get(rec.id)
    # the pointer's whole purpose: one seek into the file, no stored copy, no re-parse
    text = section_text(rex, "chapter", 0, path=rec.meta["heap"])
    assert text.strip(), "a chapter span must resolve to prose"
    with open(rec.meta["heap"], "rb") as fh:
        raw = fh.read().decode("utf-8")
    assert text in raw, "the span must address THIS document's bytes"
    assert sectionings_of(rex)["chapter"].spans is not None


def test_the_heap_pointer_is_withheld_when_spans_cannot_address_the_file(tmp_path):
    """`read_document` decides this. The ingest's job is to not publish a pointer it was
    told is unusable."""
    p = tmp_path / "cp1252.txt"
    p.write_bytes(_BOOK.replace("hat", "h—t").encode("cp1252"))
    r = ingest_one(str(p), profile=ENGLISH_GUTENBERG)
    assert r["ok"]
    assert r["meta"]["encoding_exact"] is False
    assert r["meta"]["heap"] == ""


def test_every_record_is_tagged_so_the_corpus_is_separable(tmp_path, corpus):
    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    rec = store.get_record(doc_id_for(corpus[0]))
    assert set(DOC_TAGS) <= set(rec.signature.get("tags", []))


def test_the_id_is_stable_across_runs_and_directories():
    assert doc_id_for("/a/b/pg102.txt") == "pg102"
    assert doc_id_for("/other/pg102.txt") == doc_id_for("/a/b/pg102.txt")


#### the blob is framed and compressed ##########################################

def test_a_blob_round_trips_through_the_frame():
    from agent.rcdb import compress_blob, decompress_blob
    raw = b"boundary" * 4096
    c = compress_blob(raw)
    assert c[:4] == b"RXZ1"
    assert len(c) < len(raw)
    assert decompress_blob(c) == raw


def test_an_uncompressed_blob_written_before_this_still_reads():
    """Stores exist. A raw safetensors file opens with a little-endian u64 header
    length, so its first four bytes would have to read an 823 MB header to collide with
    the magic, and the format caps a header at 100 MB. The two are distinguishable
    exactly, so a legacy blob passes through rather than being guessed at."""
    from agent.rcdb import decompress_blob
    legacy = (280).to_bytes(8, "little") + b'{"__metadata__":{}}' + b"\x00" * 261
    assert decompress_blob(legacy) == legacy


def test_the_complex_survives_compression(corpus):
    from agent.rcdb import deserialize_complex, serialize_complex
    from rexgraph.document import build_document, read_document
    from rexgraph.sectioning import sectionings_of

    raw, _e = read_document(corpus[0])
    rex, _i = build_document(raw, profile=ENGLISH_GUTENBERG)
    back = deserialize_complex(serialize_complex(rex))
    assert int(back.nV) == int(rex.nV) and int(back.nE) == int(rex.nE)
    a, b = sectionings_of(rex), sectionings_of(back)
    assert sorted(a) == sorted(b)
    for k in a:
        assert (a[k].spans is None) == (b[k].spans is None)


def test_digests_are_derived_so_compression_actually_reaches_the_blob(corpus):
    """A digest is at full entropy and survives any compressor whole. When they were
    stored they were 35% of a document blob, so they dominated the COMPRESSED size far
    more than the raw one: dropping them is what makes compressing worth doing."""
    import zlib

    from agent.rcdb import serialize_complex
    from rexgraph.document import build_document, read_document

    raw, _e = read_document(corpus[0])
    rex, _i = build_document(raw, profile=ENGLISH_GUTENBERG)
    blob = serialize_complex(rex)
    # a compressed blob must not itself compress further by much: if it did, something
    # incompressible-and-derivable is still riding along
    assert len(zlib.compress(blob, 9)) > 0.9 * len(blob)


#### migrating a store written before compression ###############################

def _write_legacy_blobs(store):
    """Rewrite every blob as raw, uncompressed bytes: a store from before framing."""
    from agent.rcdb import decompress_blob
    n = 0
    for id_ in list(store._idx):
        for rec in store.history(id_):
            p = store._blob_path(rec.id, rec.version)
            raw = open(p, "rb").read()
            with open(p, "wb") as fh:
                fh.write(decompress_blob(raw))
            n += 1
    return n


def test_recompress_shrinks_a_legacy_store_and_keeps_every_complex(tmp_path, corpus):
    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    before = {r: (int(store.get(r).nV), int(store.get(r).nE)) for r in list(store._idx)}
    assert _write_legacy_blobs(store) == 3

    out = store.recompress()
    assert out["rewritten"] == 3 and out["failed"] == 0
    assert out["after"] < out["before"]
    for r, shape in before.items():
        rex = store.get(r)
        assert (int(rex.nV), int(rex.nE)) == shape, "the complex must survive the rewrite"


def test_recompress_is_idempotent(tmp_path, corpus):
    """An already-framed blob is skipped, so running it twice costs nothing and a store
    is never double-compressed."""
    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    first = store.recompress()
    assert first["rewritten"] == 0 and first["skipped"] == 3
    second = store.recompress()
    assert second == first


def test_a_blob_that_cannot_be_read_is_left_alone(tmp_path, corpus):
    """Verify then replace. A failure must leave the original byte-for-byte, because a
    migration that damages what it cannot convert is worse than one that refuses."""
    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    _write_legacy_blobs(store)
    victim = store._blob_path(doc_id_for(corpus[0]), 1)
    with open(victim, "wb") as fh:
        fh.write(b"not a complex at all")

    out = store.recompress()
    assert out["failed"] == 1 and out["rewritten"] == 2
    assert open(victim, "rb").read() == b"not a complex at all"
    assert not os.path.exists(f"{victim}.tmp")


def test_recompress_gives_the_same_answer_on_a_worker_pool(tmp_path, corpus):
    """Only file CONTENTS change (no record, index or log entry is touched) so the
    parallel path must agree with the serial one exactly. Both arms have to start from
    byte-identical input, so the store is COPIED rather than ingested twice."""
    import shutil

    src = FileStore(str(tmp_path / "serial"))
    ingest_corpus(corpus, src, profile=ENGLISH_GUTENBERG, workers=2)
    _write_legacy_blobs(src)
    shutil.copytree(tmp_path / "serial", tmp_path / "pool")

    a = FileStore(str(tmp_path / "serial")).recompress(workers=1)
    b = FileStore(str(tmp_path / "pool")).recompress(workers=3)
    assert a["rewritten"] == b["rewritten"] == 3
    assert a["failed"] == b["failed"] == 0
    assert a["before"] == b["before"], "the two arms must start from the same bytes"
    # NOT byte equality of the output: safetensors writes its `__metadata__` map from a
    # Rust HashMap, whose iteration order is randomly seeded per process, so the three
    # keys land in a different order in each writer. Measured over six writes of one
    # complex: tensor entries kept a single order, `__metadata__` took two. The content
    # is identical and the sizes land within a byte; the FILE is simply not reproducible,
    # which is worth knowing before anyone tries to dedupe blobs by hashing them.
    assert abs(a["after"] - b["after"]) <= 8 * a["rewritten"]
    for name in ("serial", "pool"):
        store = FileStore(str(tmp_path / name))
        for r in list(store._idx):
            assert store.get(r) is not None


def test_force_rewrites_an_already_framed_blob(tmp_path, corpus):
    """The magic says a blob is compressed, not which rex-state format_version it holds.
    When the format moves, a store that is already framed still needs rewriting."""
    store = FileStore(str(tmp_path / "store"))
    ingest_corpus(corpus, store, profile=ENGLISH_GUTENBERG, workers=2)
    assert store.recompress()["rewritten"] == 0, "framed, so skipped"
    out = store.recompress(force=True)
    assert out["rewritten"] == 3 and out["failed"] == 0
    for r in list(store._idx):
        assert store.get(r) is not None
