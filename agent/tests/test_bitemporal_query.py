"""Bitemporal reads for the COLLECTION accessors, not just single-record get.

get/get_record/get_version took as_of/valid_at from the start, so one record could
always be read as it stood. list and query could not, so "the corpus as it stood at
T" was unanswerable, and once retrieval gained a label prefilter, that became a
silent false negative: the prefilter matched CURRENT labels, so a document whose
vocabulary had since been replaced was dropped before anything read its as_of blob.

A time-travelling query that quietly omits what was relevant at the time is worse
than one that is slow, and worse than one that refuses.
"""

import time

import pytest
from agent.corpus import CorpusBuilder

from agent import query_engine as qe
from agent import rcdb

OLD = ("Frustration tracks sign conflict between neighbouring edges. The overlap "
       "channel counts shared endpoints across the complex.")
NEW = ("Persistence diagrams record birth and death of cycles. Bottleneck distance "
       "compares two diagrams across filtration values.")


def _put(store, doc_id, text):
    c = CorpusBuilder()
    c.add_text(text, doc_id=doc_id)
    c.build(depth="quick")
    c.persist(store)


@pytest.fixture(params=["memory", "file", "sql"])
def revised(request, tmp_path):
    """One id whose vocabulary is entirely replaced, plus the instant between."""
    if request.param == "memory":
        st = rcdb.MemoryStore()
    elif request.param == "file":
        st = rcdb.FileStore(str(tmp_path / "store"))
    else:
        st = rcdb.SQLStore(f"sqlite:///{tmp_path / 'rc.sqlite'}")
    _put(st, "doc", OLD)
    time.sleep(0.02)
    t_mid = time.time()
    time.sleep(0.02)
    _put(st, "doc", NEW)
    return st, t_mid


def test_the_fixture_really_replaced_the_vocabulary(revised):
    store, t_mid = revised
    assert [r.version for r in store.history("doc")] == [1, 2]
    old_labels = (store.get("doc", as_of=t_mid)._agent_meta or {})["vertex_labels"]
    now_labels = (store.get("doc")._agent_meta or {})["vertex_labels"]
    assert "frustration" in old_labels and "frustration" not in now_labels


def test_query_reads_the_vocabulary_of_the_as_of_version(revised):
    store, t_mid = revised
    assert {r.id for r in store.query(labels_any=["frustration"], limit=10,
                                      as_of=t_mid)} == {"doc"}
    assert store.query(labels_any=["frustration"], limit=10) == []


def test_query_as_of_returns_the_version_that_was_current_then(revised):
    store, t_mid = revised
    hits = store.query(labels_any=["frustration"], limit=10, as_of=t_mid)
    assert [r.version for r in hits] == [1]
    assert [r.version for r in store.query(labels_any=["persistence"], limit=10)] == [2]


def test_list_as_of_sees_the_store_as_it_stood(revised):
    store, t_mid = revised
    assert [r.version for r in store.list(limit=10, as_of=t_mid)] == [1]
    assert [r.version for r in store.list(limit=10)] == [2]


def test_list_before_anything_existed_is_empty(revised):
    store, _ = revised
    assert store.list(limit=10, as_of=1.0) == []


def test_retrieval_as_of_finds_what_was_relevant_then(revised):
    """The regression that motivated all of this. The prefilter matched current
    labels, so time-travel retrieval silently missed the document."""
    store, t_mid = revised
    sections, _ = qe.retrieve_sections("frustration sign conflict", top_k=3,
                                       store=store, as_of=t_mid)
    assert [s["doc_id"] for s in sections] == ["doc"], "the as_of document was missed"


def test_retrieval_now_does_not_see_the_superseded_vocabulary(revised):
    store, _ = revised
    sections, _ = qe.retrieve_sections("frustration sign conflict", top_k=3,
                                       store=store)
    assert sections == []


def test_retrieval_as_of_still_answers_for_the_current_vocabulary(revised):
    store, _ = revised
    sections, _ = qe.retrieve_sections("persistence bottleneck diagrams", top_k=3,
                                       store=store)
    assert [s["doc_id"] for s in sections] == ["doc"]


def test_valid_at_is_honoured_by_the_collection_accessors(tmp_path):
    """tx time is when we recorded it; valid time is when it was true. They are
    different questions and both have to reach list/query."""
    st = rcdb.SQLStore(f"sqlite:///{tmp_path / 'rc.sqlite'}")
    c = CorpusBuilder()
    c.add_text(OLD, doc_id="doc")
    c.build(depth="quick")
    c.persist(st, valid_from=100.0)

    assert [r.id for r in st.list(limit=10, valid_at=150.0)] == ["doc"]
    assert st.list(limit=10, valid_at=50.0) == []
    assert {r.id for r in st.query(labels_any=["frustration"], limit=10,
                                   valid_at=150.0)} == {"doc"}
    assert st.query(labels_any=["frustration"], limit=10, valid_at=50.0) == []
