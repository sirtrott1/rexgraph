"""The agent layer asks the library for what the library already computes.

Each case here was a hand-rolled reimplementation that materialized something dense
or scanned where an index existed. They are equivalence tests: the rewired path has
to return what the old one returned, and keep working past the size the old one
silently gave up at.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _graph(nv=18, ne=40, seed=5):
    rng = np.random.RandomState(seed)
    src = rng.randint(0, nv, ne).astype(np.int32)
    tgt = ((src + 1 + rng.randint(0, 4, ne)) % nv).astype(np.int32)
    return RexGraph(sources=src, targets=tgt)


def test_star_of_vertex_is_the_dense_b1_row(seed=5):
    """`star_of_vertex(v)[1]` is the set a dense B1 row scan produced, without the
    nV x nE array built to read one row of it."""
    rex = _graph()
    B1 = rex.B1
    for v in range(rex.nV):
        want = set(np.flatnonzero(np.abs(B1[v, :]) > 0).tolist())
        got = set(np.flatnonzero(np.asarray(rex.star_of_vertex(v)[1], bool)).tolist())
        assert got == want, f"vertex {v}: {got ^ want}"


def test_an_edge_between_two_vertices_is_in_both_stars():
    """The path scan looked for `|B1[src,e]| > 0 and |B1[tgt,e]| > 0` over every
    edge. That set is the intersection of the two stars."""
    rex = _graph()
    B1 = rex.B1
    for s in range(0, rex.nV, 3):
        for t in range(1, rex.nV, 4):
            want = set(np.flatnonzero((np.abs(B1[s, :]) > 0) & (np.abs(B1[t, :]) > 0)).tolist())
            got = set(np.flatnonzero(
                np.asarray(rex.star_of_vertex(s)[1], bool)
                & np.asarray(rex.star_of_vertex(t)[1], bool)).tolist())
            assert got == want, f"{s}->{t}: {got ^ want}"


def test_the_character_path_is_sparse_at_every_size():
    """`_use_sparse_character` gates the whole scale-free stack. It reads "the dense
    RL was not built", and the dense RL is no longer built at any size, so the sparse
    path is what runs rather than a fallback that only large inputs reach."""
    for nv, ne in ((6, 8), (40, 120), (400, 1200)):
        rex = _graph(nv, ne, seed=nv)
        assert rex.spectral_bundle.get("RL") is None, f"dense RL built at nE={rex.nE}"
        assert rex._use_sparse_character is True, f"dense character path at nE={rex.nE}"


def test_the_interfacing_bundle_is_the_librarys_own():
    """One implementation. The agent used to assemble this from the dense kernel
    with a different G operator, so its numbers were not these numbers."""
    rex = _graph()
    ti, tw = np.array([0, 3, 7], np.int32), np.ones(3)
    b = rex.interfacing_vector(ti, tw, None)
    assert set(b) >= {"rho", "psi", "scores", "coverage", "efficiency"}
    assert np.asarray(b["scores"]).shape == (3,)
    assert np.all(np.isfinite(np.asarray(b["scores"])))


def test_faces_come_from_the_face_solver():
    """`rexgraph.faces` solves B1 c = 0 and is arity-general; the agent path used to
    run a triangle-only rule of its own."""
    from rexgraph.faces import autoface, cycle_basis, face_support
    square = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                      targets=np.array([1, 2, 3, 0], np.int32))
    assert sorted(face_support(c) for c in cycle_basis(square)) == [4]
    assert autoface(square, k=3) == 0 and autoface(square, k=4) == 1


def test_a_document_complex_has_faces():
    """Faces are asked for rather than assumed, and the corpus asks.

    With none, curl is identically 0 and every loop a document contains reads as
    harmonic instead, which silently changes what the platform reports about every
    document it has ever built.
    """
    import uuid

    from agent.corpus import CorpusBuilder
    text = (f"Alpha{uuid.uuid4().hex[:6]} connects beta. Beta connects gamma. "
            "Gamma connects alpha. Delta connects alpha. Alpha connects epsilon. "
            "Epsilon connects delta.")
    c = CorpusBuilder()
    c.add_text(text, doc_id="d1")
    c.build(depth="standard")
    rex = c.documents[0].rex
    assert rex.nF > 0, "the document has no faces, so its curl is identically zero"
    h = rex.hodge_full(np.ones(rex.nE))
    assert float(h["pct_curl"]) > 0.0, h


def test_the_document_cache_is_keyed_on_the_face_rule():
    """The rule is part of what the complex IS. Keying on adapter_kwargs alone
    served a complex built under a different rule as a cache hit."""
    import uuid

    import agent.corpus as C
    text = (f"Mu{uuid.uuid4().hex[:6]} connects nu. Nu connects xi. Xi connects mu. "
            "Omicron connects mu. Mu connects pi. Pi connects omicron.")

    def build(rule):
        old, C.DOC_FACE_RULE = C.DOC_FACE_RULE, rule
        try:
            c = C.CorpusBuilder()
            c.add_text(text, doc_id="d")
            c.build(depth="standard")
            return c.documents[0].rex.nF
        finally:
            C.DOC_FACE_RULE = old

    first = build("typed")
    assert build("none") == 0, "a rule change was served from cache"
    assert build("auto") > first, "a rule change was served from cache"
    assert build("typed") == first, "the original rule no longer reproduces"
