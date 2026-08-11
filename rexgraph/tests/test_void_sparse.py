"""The void complex reads its boundaries sparsely, and eta needs no cycle space.

void_complex asked for the dense B1 (45.8 GB on a 40k-vertex complex) to recover two
endpoint rows per relation, scanned every row of a dense B2 to find three relations
per face, and built the whole harmonic basis to answer a question that is a face-space
projection. These pin the replacements against the paths they replaced.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from rexgraph.core import _void
from rexgraph.faces import find_cycles
from rexgraph.graph import RexGraph
from rexgraph.harmonic_sparse import harmonic_basis_from_boundaries


def _triangle():
    return RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 0], dtype=np.int32))


def _k4():
    return RexGraph(sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32))


def _boundaries(rex):
    d1 = np.asarray(rex.B1)
    d2 = np.asarray(rex.B2_hodge)
    return d1, sp.csc_matrix(d1), (d2 if d2.size else None), (sp.csc_matrix(d2) if d2.size else None)


def test_endpoint_rows_agree_dense_and_sparse():
    for mk in (_triangle, _k4):
        rex = mk()
        d1, s1, _, _ = _boundaries(rex)
        pd_, md_ = _void._endpoint_rows(d1, rex.nV, rex.nE)
        ps_, ms_ = _void._endpoint_rows(s1, rex.nV, rex.nE)
        assert np.array_equal(pd_, ps_)
        assert np.array_equal(md_, ms_)
        # and they are what argmax/argmin said
        assert np.array_equal(pd_, np.argmax(d1, axis=0).astype(np.int32))
        assert np.array_equal(md_, np.argmin(d1, axis=0).astype(np.int32))


def test_realized_face_keys_agree_dense_and_sparse():
    rex = _k4()
    rex.add_faces(np.array([[0, 3, 1], [0, 4, 2]], dtype=np.int32))
    d1, s1, d2, s2 = _boundaries(rex)
    assert np.array_equal(_void._realized_face_keys(d2, rex.nE),
                          _void._realized_face_keys(s2, rex.nE))
    assert _void._realized_face_keys(None, rex.nE).size == 0


def test_a_filled_triangle_has_no_void():
    rex = _triangle()
    assert rex.void_complex["n_voids"] == 1
    rex2 = _triangle()
    rex2.add_faces(np.array([[0, 1, 2]], dtype=np.int32))
    assert rex2.void_complex["n_voids"] == 0


def test_void_columns_are_cycles():
    """B1 @ Bvoid = 0 by construction; the eta shortcut depends on it."""
    for mk in (_triangle, _k4):
        rex = mk()
        vc = rex.void_complex
        bv = vc["Bvoid"]
        if bv is None or bv.shape[1] == 0:
            continue
        B1s = sp.csc_matrix(np.asarray(rex.B1))
        assert float(abs(B1s @ bv).max()) == pytest.approx(0.0, abs=1e-12)


def test_eta_is_one_when_there_are_no_faces():
    """With no faces every cycle is harmonic, so eta needs no computation at all."""
    rex = _k4()
    vc = rex.void_complex
    assert vc["n_voids"] == 4
    assert np.allclose(vc["eta"], 1.0)


def _eta_via_harmonic_basis(B1s, B2s, Bv, n_voids):
    """The computation the projection replaced, kept here as the oracle."""
    H = harmonic_basis_from_boundaries(B1s.tocsc(), B2s.tocsr() if B2s is not None else None)
    bvn = np.asarray(Bv.multiply(Bv).sum(axis=0)).ravel()
    out = np.zeros(n_voids)
    k = H.shape[1]
    if k == 0:
        return out
    Hs = H.tocsr()
    G = np.asarray((Hs.T @ Bv).todense())
    HtH = (Hs.T @ Hs).tocsc()
    try:
        Y = sla.splu(HtH).solve(G)
    except Exception:
        Y = np.linalg.solve(np.asarray(HtH.todense()), G)
    pn = np.einsum("ij,ij->j", G, np.asarray(Y).reshape(k, n_voids))
    nz = bvn > 1e-15
    out[nz] = pn[nz] / bvn[nz]
    return out


def test_eta_matches_the_harmonic_basis_with_faces_present():
    rex = _k4()
    rex.add_faces(np.array([[0, 3, 1]], dtype=np.int32))
    _, s1, _, s2 = _boundaries(rex)
    ap, ai, ae = rex._adjacency_bundle
    tri, nT = _void.find_potential_triangles(ap, ai, ae, rex.nV, rex.nE)
    bv, _vi, nvo = _void.build_void_boundary(s1, s2, tri, nT, rex.nV, rex.nE)
    assert nvo == 3
    got = _void.harmonic_content_all_sparse(s1, s2, bv, nvo, rex.nE)
    want = _eta_via_harmonic_basis(s1, s2, bv, nvo)
    assert np.allclose(got, want, atol=1e-9)


def test_build_void_complex_identical_for_dense_and_sparse_inputs():
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(40):
        nv = int(rng.integers(5, 16))
        m = int(rng.integers(nv, nv * 3))
        s = rng.integers(0, nv, m).astype(np.int32)
        t = rng.integers(0, nv, m).astype(np.int32)
        keep = s != t
        if keep.sum() < 3:
            continue
        rex = RexGraph(sources=s[keep].copy(), targets=t[keep].copy())
        try:
            rex._ensure_clean()
            cyc = find_cycles(rex, 3)
            if len(cyc):
                rex.add_faces(np.asarray(cyc[:2], dtype=np.int32))
        except Exception:
            continue
        d1, s1, d2, s2 = _boundaries(rex)
        ap, ai, ae = rex._adjacency_bundle
        rest = (ap, ai, ae, rex.nV, rex.nE, None, None, 0, None, None)
        a = _void.build_void_complex(d1, d2, *rest)
        b = _void.build_void_complex(s1, s2, *rest)
        assert a["n_voids"] == b["n_voids"]
        assert a["n_potential"] == b["n_potential"]
        assert a["void_strain"] == pytest.approx(b["void_strain"])
        assert np.allclose(a["eta"], b["eta"], atol=1e-9)
        checked += 1
    assert checked >= 10, f"only exercised {checked} complexes"
