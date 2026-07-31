"""Scale: retrieval must not read every record in the store.

retrieve_from_store called store.list(limit=10**6) and filtered in Python, so every
query was O(all records) and every candidate's signature crossed the process
boundary. Fine for five documents; fatal for a consortium.

The fix is a label predicate the store can answer itself: SQLStore resolves it in an
indexed table and returns only matching rows, while Memory/File match the full
vocabulary already carried in record meta. The retrieval prefilter stops being a
Python scan over everything.
"""

import numpy as np
import pytest

from agent import rcdb
from agent import query_engine as qe
from agent.corpus import CorpusBuilder


DOCS = {
    "boundary": "The boundary map sends an edge to its endpoints. Composing two "
                "boundary maps gives zero. Orientation decides every sign.",
    "channels": "Each typed channel weighs the edges differently. The overlap channel "
                "counts shared endpoints. Frustration tracks sign conflict.",
    "storage": "Records persist as binary blobs beside a queryable signature. A "
               "signature carries structure without the payload. Blobs stay closed "
               "until a candidate needs scoring, so the database answers a prefilter "
               "from its index while the payload remains untouched on disk.",
}


def _corpus():
    c = CorpusBuilder()
    for did, text in DOCS.items():
        c.add_text(text, doc_id=did)
    c.build(depth="quick")
    return c


@pytest.fixture(params=["memory", "file", "sql"])
def store(request, tmp_path):
    if request.param == "memory":
        st = rcdb.MemoryStore()
    elif request.param == "file":
        st = rcdb.FileStore(str(tmp_path / "store"))
    else:
        st = rcdb.SQLStore(f"sqlite:///{tmp_path / 'rc.sqlite'}")
    _corpus().persist(st)
    return st


def test_labels_any_narrows_to_records_sharing_a_token(store):
    hits = store.query(labels_any=["frustration"], limit=50)
    assert {r.id for r in hits} == {"channels"}, [r.id for r in hits]


def test_labels_any_matches_the_full_vocabulary_not_a_12_label_sample(store):
    """labels_sample is a sample. A prefilter built on it silently misses documents
    whose matching term happens to fall outside the first twelve."""
    doc = next(d for d in _corpus().documents if d.doc_id == "storage")
    assert len(doc.vertex_labels) > 12, "fixture too small to prove the point"
    late = doc.vertex_labels[-1]
    hits = store.query(labels_any=[late], limit=50)
    assert "storage" in {r.id for r in hits}, f"{late!r} missed by the prefilter"


def test_labels_any_is_case_insensitive(store):
    assert {r.id for r in store.query(labels_any=["FRUSTRATION"], limit=50)} == {"channels"}


def test_labels_any_with_no_match_returns_nothing(store):
    assert store.query(labels_any=["xylophone"], limit=50) == []


def test_labels_any_unions_across_tokens(store):
    hits = store.query(labels_any=["frustration", "blobs"], limit=50)
    assert {r.id for r in hits} == {"channels", "storage"}


def test_label_predicate_composes_with_structural_ones(store):
    assert store.query(labels_any=["frustration"], min_nE=10 ** 9, limit=50) == []
    assert store.query(labels_any=["frustration"], min_nE=1, limit=50)


def test_limit_is_honoured(store):
    assert len(store.query(labels_any=["boundary", "frustration", "blobs"],
                           limit=1)) == 1


def test_retrieval_delegates_the_prefilter_to_the_store(store):
    """The point of the whole exercise. Retrieval used to pull every record with
    store.list(limit=10**6) and filter in Python; it must now hand the store a
    vocabulary predicate and a bounded limit, and let the store answer it. Whether a
    given backend has an index for that is the backend's business -- Memory and File
    scan by nature, SQLStore pushes it into the database (tested separately)."""
    asked = []
    real_query = store.query
    store.query = lambda **kw: (asked.append(kw), real_query(**kw))[1]
    try:
        sections, _ = qe.retrieve_sections("frustration sign conflict channel",
                                           top_k=3, store=store)
    finally:
        store.query = real_query

    assert sections and sections[0]["doc_id"] == "channels"
    assert asked, "retrieval never queried the store"
    assert asked[0].get("labels_any"), "no vocabulary predicate was pushed"
    assert asked[0].get("limit") and asked[0]["limit"] < 10 ** 6, (
        f"unbounded prefilter: limit={asked[0].get('limit')}")
    # and it must not fall back to a full enumeration afterwards. Comments are
    # stripped: the docstring explains the old store.list path on purpose.
    import inspect

    code = [ln.split("#", 1)[0] for ln in
            inspect.getsource(qe.retrieve_from_store).splitlines()]
    assert not any("store.list(" in ln for ln in code), (
        "retrieval still enumerates the store")


def test_retrieval_reports_how_much_it_narrowed(store):
    _, rel = qe.retrieve_sections("frustration sign conflict channel", top_k=3,
                                  store=store)
    assert rel["n_records"] <= 3
    assert rel["n_opened"] <= rel["n_records"]


def test_retrieval_still_works_when_nothing_shares_a_token(store):
    sections, rel = qe.retrieve_sections("xylophone quasar", top_k=3, store=store)
    assert sections == []


def test_a_prefix_still_scopes_retrieval(store):
    _corpus().persist(store, prefix="other/")
    sections, _ = qe.retrieve_sections("frustration sign conflict", top_k=5,
                                       store=store, prefix="other/")
    assert sections and all(s["doc_id"].startswith("other/") for s in sections)


def test_sql_pushes_the_label_predicate_into_the_database(tmp_path):
    """Not just the same answer: SQLStore must resolve labels in SQL rather than
    reading every row back and filtering in Python."""
    st = rcdb.SQLStore(f"sqlite:///{tmp_path / 'rc.sqlite'}")
    _corpus().persist(st)

    seen = []
    real_row_to_record = st._row_to_record

    def counting(row):
        seen.append(1)
        return real_row_to_record(row)

    st._row_to_record = counting
    hits = st.query(labels_any=["frustration"], limit=50)
    assert {r.id for r in hits} == {"channels"}
    assert len(seen) == 1, f"materialized {len(seen)} rows for a 1-record answer"
