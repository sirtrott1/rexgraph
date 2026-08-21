"""A query must be the same KIND of object as the document it is scored against.

`interfacing_score` compares a query complex's vocabulary and a document's field. If the
query is a windowed co-occurrence graph and the document is a field of branching
relations, the comparison is between two constructions rather than two texts, and no
score function repairs that.
"""
from __future__ import annotations

import pytest

from agent.adapters.text import TextAdapter
from agent.auto import build_rex_from_edges
from agent.query_engine import build_query_rex

_TEXT = ("The cat sat quietly on the woven mat. "
         "A large dog chased that cat around the walled garden. "
         "Nothing else of interest happened afterwards.")


#### the branching construction #################################################

def test_branching_mode_carries_sentences_as_relations_not_pairs():
    ec = TextAdapter().build(_TEXT, relation_mode="branching")
    assert len(ec.sources) == 0 and len(ec.targets) == 0, "no pairs are enumerated"
    assert len(ec.branching) == 3, "one relation per sentence"
    assert all(len(s) >= 2 for s in ec.branching)


def test_pairwise_mode_is_unchanged_and_is_still_the_default():
    """37 call sites read `sources`/`targets`; the default must not move under them."""
    default = TextAdapter().build(_TEXT)
    explicit = TextAdapter().build(_TEXT, relation_mode="pairwise")
    assert len(default.sources) == len(explicit.sources) > 0
    assert not default.branching


def test_nE_counts_relations_at_any_arity():
    """`len(sources)` alone reported zero for a branching construction, so every caller
    reading nE as "is there anything here" concluded the text was empty."""
    ec = TextAdapter().build(_TEXT, relation_mode="branching")
    assert ec.nE == len(ec.branching) == 3


def test_the_complex_builds_from_the_branching_field():
    ec = TextAdapter().build(_TEXT, relation_mode="branching")
    rex = build_rex_from_edges(ec)
    assert int(rex.nE) == 3
    assert int(rex.nV) == len(ec.vertex_labels)


def test_a_branching_column_is_zero_sum_at_its_own_arity():
    import numpy as np

    from rexgraph.core._sparse import to_scipy_csr
    ec = TextAdapter().build(_TEXT, relation_mode="branching")
    B = to_scipy_csr(build_rex_from_edges(ec)._B1_dual).tocsc()
    assert np.allclose(np.asarray(B.sum(axis=0)).ravel(), 0.0)
    for c in range(B.shape[1]):
        col = B.data[B.indptr[c]:B.indptr[c + 1]]
        assert (col < 0).sum() == 1, "one head carries the -1"
        assert np.allclose(col[col > 0], 1.0 / (len(col) - 1)), "share is 1/(k-1)"


def test_an_unknown_relation_mode_is_refused():
    with pytest.raises(ValueError, match="relation_mode must be"):
        TextAdapter().build(_TEXT, relation_mode="clique")


#### the query path #############################################################

def test_build_query_rex_is_branching_by_default():
    rex, ec = build_query_rex("the cat chased the dog around the garden")
    assert ec.branching and len(ec.sources) == 0
    assert rex is not None and int(rex.nE) == len(ec.branching)


def test_the_query_and_the_document_tokenise_identically():
    """The actual goal. Alignment is by LABEL, so a query that tokenises differently
    aligns on a vocabulary neither text has: the pairwise path drops stopwords and the
    document path does not, so 'the' is shared under one construction and absent under
    the other."""
    from rexgraph.document import build_document
    _drex, dinfo = build_document(
        "The cat sat quietly on the woven mat.\n\n"
        "A large dog chased that cat around the walled garden.\n")
    doc_vocab = {w.lower() for w in dinfo["vocab"]}

    _r, branching = build_query_rex("the cat chased the dog", relation_mode="branching")
    _r2, pairwise = build_query_rex("the cat chased the dog", relation_mode="pairwise")
    b = {w.lower() for w in branching.vertex_labels}
    p = {w.lower() for w in pairwise.vertex_labels}

    assert b - doc_vocab == set(), "every branching query term is a document term"
    assert "the" in b and "the" in doc_vocab
    assert len(b & doc_vocab) > len(p & doc_vocab), (
        "the shared vocabulary must not depend on which construction built the query")


def test_a_one_word_query_is_a_witness_not_an_absence():
    """A one-word query is a true reading of the input, and the reading is a witness:
    `(+1)`, sum one, exists and bounds nothing."""
    rex, ec = build_query_rex("cat")
    assert ec is not None and ec.vertex_labels == ["cat"]
    assert rex is not None and int(rex.nE) == 1
    assert int(rex.edge_types[0]) == 3, "EdgeType.WITNESS"


@pytest.mark.parametrize("q", ["", "   "])
def test_an_empty_query_has_no_complex_and_raises_nothing(q):
    rex, ec = build_query_rex(q)
    assert rex is None


def test_no_faces_are_requested_in_branching_mode():
    """FACE_RULE fills cycles in a pairwise complex. A branching document has no faces
    added, so asking for them here would reintroduce the asymmetry this mode removes."""
    rex, _ec = build_query_rex("the cat chased the dog around the garden")
    assert int(rex.nF) == 0
