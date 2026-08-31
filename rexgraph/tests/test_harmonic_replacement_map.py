"""What replaced the dense harmonic module, function by function.

`rexgraph/core/_harmonic.pyx` builds a dense nE x nE L1 and eigendecomposes it
against a hardcoded cutoff. Every part of it that had a live purpose has an
eigen-free replacement, and this file is the mapping, executable so it cannot
drift:

    _harmonic.harmonic_basis        harmonic_sparse.harmonic_basis
    _harmonic.harmonic_projectors   harmonic_sparse.harmonic_projection
    _harmonic.harmonic_encode       hodge_coords.from_harmonic_coords
    _harmonic.harmonic_decode       hodge_coords.harmonic_coords
    _harmonic.harmonic_leakage      RexGraph.hodge / hodge_full
    _harmonic.harmonic_product_table hodge_coords.harmonic_closure

The dense module survives as the ORACLE these are checked against, which is what
`test_harmonic_sparse.py` uses it for, and as the dependency of the unreferenced
`rexgraph/harmonic.py`.

Two of its functions have no replacement and want none. The prime-coupling
experiments have no equivalent anywhere. And `harmonic_channel_character`
averages over the chosen basis vectors and then thresholds `np.std(chi_H) < 0.02`,
so it is basis-dependent and decides from a statistic against a magic number,
which is two standing directives at once.
"""

import itertools

import numpy as np

from rexgraph.core import _harmonic as densemod
from rexgraph.graph import RexGraph
from rexgraph.harmonic_sparse import harmonic_basis, harmonic_projection
from rexgraph.hodge_coords import (
    from_harmonic_coords,
    harmonic_closure,
    harmonic_coords,
    harmonic_frame,
)


def _k(n):
    e = list(itertools.combinations(range(n), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    r._ensure_clean()
    return r


def _dense(r):
    B1 = np.asarray(r.B1_dense, float)
    B2 = np.asarray(r.B2_dense, float)
    return B1, B2, densemod.harmonic_projectors(B1, B2)


def test_the_sparse_frame_spans_what_the_eigendecomposition_returns():
    """Different bases, one space. That is all that can be asked, and it is what
    matters: the projector is basis-free."""
    for n in (4, 5, 6):
        r = _k(n)
        _, _, projs = _dense(r)
        H = np.asarray(harmonic_frame(r).todense())
        P = H @ np.linalg.solve(H.T @ H, H.T)
        assert np.allclose(projs["P_harm"], P, atol=1e-9), n


def test_the_low_rank_projection_replaces_the_dense_projector():
    r = _k(5)
    _, _, projs = _dense(r)
    f = np.random.default_rng(0).normal(size=r.nE)
    assert np.allclose(projs["P_harm"] @ f, harmonic_projection(harmonic_basis(r), f),
                       atol=1e-9)


def test_encode_is_from_harmonic_coords():
    """`harmonic_encode(c, U) = U @ c` is exactly rebuilding a signal from its
    coordinates."""
    r = _k(5)
    H = harmonic_frame(r)
    f = np.random.default_rng(1).normal(size=r.nE)
    c = harmonic_coords(r, f, frame=H)
    assert np.allclose(from_harmonic_coords(r, c, frame=H),
                       harmonic_projection(H, f), atol=1e-9)


def test_decode_is_the_gram_solve_and_the_dense_form_is_the_special_case():
    """`harmonic_decode(x, U) = U^T x` is the coordinate formula for an ORTHONORMAL
    basis only. The frame is not orthonormal, so the general reading is the Gram
    solve; the two land on the same vector."""
    r = _k(5)
    B1, B2, projs = _dense(r)
    U, _ = densemod.harmonic_basis(B1, B2)
    H = harmonic_frame(r)
    f = np.random.default_rng(2).normal(size=r.nE)
    dense_vec = U @ densemod.harmonic_decode(f, U)
    frame_vec = from_harmonic_coords(r, harmonic_coords(r, f, frame=H), frame=H)
    assert np.allclose(dense_vec, frame_vec, atol=1e-9)
    # and the naive U^T on the non-orthonormal frame is NOT the coordinate
    Hd = np.asarray(H.todense())
    assert not np.allclose(Hd.T @ f, harmonic_coords(r, f, frame=H), atol=1e-6)


def test_leakage_is_the_hodge_split_without_three_dense_projectors():
    """`harmonic_leakage` takes P_harm, P_grad and P_curl, three nE x nE matrices.
    The split itself needs none of them."""
    r = _k(5)
    _, _, projs = _dense(r)
    f = np.random.default_rng(3).normal(size=r.nE)
    grad, curl, harm = r.hodge(f)
    assert np.allclose(projs["P_harm"] @ f, harm, atol=1e-8)
    assert np.allclose(projs["P_grad"] @ f, grad, atol=1e-8)
    assert np.allclose(projs["P_curl"] @ f, curl, atol=1e-8)


def test_the_product_table_is_harmonic_closure():
    """Same object. The closure entries are read against whichever frame is used,
    so the shapes match and the numbers are frame-relative; the projector test
    above is what pins the space they both live in."""
    r = _k(5)
    B1, B2, _ = _dense(r)
    U, _ = densemod.harmonic_basis(B1, B2)
    table = densemod.harmonic_product_table(U)
    C = harmonic_closure(r)
    assert C.shape == table["closure"].shape
    assert (C >= -1e-12).all() and (C <= 1.0 + 1e-9).all()


def test_only_the_dead_module_still_imports_the_dense_one():
    """If this fails, something new took a dependency on the eigen path."""
    import pathlib

    # Anchored to this file, not the caller's cwd. Relative roots plus a skip-if-absent
    # meant that running the suite from anywhere but the repository root scanned nothing
    # and passed, so the guard stopped guarding without ever failing.
    repo = pathlib.Path(__file__).resolve().parents[2]
    roots = [repo / "rexgraph", repo / "agent" / "agent"]
    scanned = 0
    offenders = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            scanned += 1
            if "tests" in p.parts or p.name == "harmonic.py":
                continue
            txt = p.read_text(errors="ignore")
            if "core._harmonic import" in txt or "from .core._harmonic" in txt:
                offenders.append(str(p))
    assert scanned, f"scanned no files under {[str(r) for r in roots]}; the guard is inert"
    assert offenders == [], offenders
