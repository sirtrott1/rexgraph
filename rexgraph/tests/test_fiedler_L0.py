"""The L0 Fiedler pair: kernel by count, preconditioner by residual.

The shipped path asked ARPACK for which='SM', which is the end of the spectrum Lanczos
resolves last. On a hub-heavy complex it did not converge at all and the except-branch
returned fiedler_val_L0 = 0.0 with a constant eigenvector, which four call sites read
as a real answer. These pin the replacement against dense ground truth in both
convergence regimes.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.fiedler import fiedler_L0, kernel_basis
from rexgraph.graph import RexGraph


def _laplacian(src, tgt, nV):
    A = sp.coo_matrix((np.ones(len(src)), (src, tgt)), shape=(nV, nV)).tocsr()
    A = A + A.T
    return (sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A).tocsr()


def _path_L(n):
    return _laplacian(np.arange(n - 1), np.arange(1, n), n)


def _random_L(nE, seed, avg_degree=10):
    rng = np.random.RandomState(seed)
    nV = nE // avg_degree
    src = rng.randint(0, nV, nE)
    tgt = rng.randint(0, nV, nE)
    m = src == tgt
    tgt[m] = (tgt[m] + 1) % nV
    return _laplacian(src, tgt, nV)


def test_kernel_basis_is_the_component_indicators():
    """ker(L0) is combinatorial, not spectral: one normalised indicator per component."""
    L = sp.block_diag([_path_L(5), _path_L(4)]).tocsr()
    U, ncomp = kernel_basis(L)
    assert ncomp == 2
    assert U.shape == (9, 2)
    # U is sparse now: a component-indicator matrix with one nonzero per row
    assert np.allclose(np.asarray((U.T @ U).todense()), np.eye(2))  # orthonormal
    assert np.allclose(np.asarray((L @ U).todense()), 0.0, atol=1e-12)  # the kernel


def test_path_matches_the_closed_form():
    """A path's Fiedler value is 4 sin^2(pi/2n) exactly. n=2500 takes the sparse path
    and has a tiny eigenvalue, which is where an absolute residual test fails."""
    n = 2500
    fv, fvec, _evals, _evecs = fiedler_L0(_path_L(n))
    assert fv == pytest.approx(4 * np.sin(np.pi / (2 * n)) ** 2, rel=1e-9)
    assert fvec.shape == (n,)


def test_cycle_matches_the_closed_form():
    n = 2500
    L = _laplacian(np.arange(n), np.roll(np.arange(n), -1), n)
    fv, _f, _e, _v = fiedler_L0(L)
    assert fv == pytest.approx(4 * np.sin(np.pi / n) ** 2, rel=1e-9)


@pytest.mark.parametrize("nE,seed", [(100000, 50), (500000, 60)])
def test_random_graphs_converge_without_a_factorization(nE, seed):
    """The regime a complete factorization chokes on: splu turns 209796 nonzeros into
    82 million here. The value must still be right."""
    L = _random_L(nE, seed)
    fv, fvec, _e, _v = fiedler_L0(L)
    assert fv > 0.0
    assert fvec.shape == (L.shape[0],)
    # certify it directly: the Rayleigh quotient of the returned vector
    rq = float(fvec @ (L @ fvec) / (fvec @ fvec))
    assert rq == pytest.approx(fv, rel=1e-6)


def test_more_components_than_k_still_finds_the_fiedler():
    """With more components than the block size the eigenvalue prefix is all zeros, so
    the Fiedler cannot be recovered by scanning it. It comes from the deflated solve."""
    n = 2500
    src = np.concatenate([np.arange(1200), np.arange(1300, 2499)])
    tgt = np.concatenate([np.arange(1, 1201), np.arange(1301, 2500)])
    L = _laplacian(src, tgt, n)
    _U, ncomp = kernel_basis(L)
    assert ncomp > 6                                    # more components than k
    dense = np.sort(np.linalg.eigvalsh(np.asarray(L.todense())))
    fv, _f, evals, _v = fiedler_L0(L)
    assert fv == pytest.approx(float(dense[ncomp]), rel=1e-6)
    assert fv > 0.0
    assert np.allclose(evals[:6], 0.0)                  # the prefix really is all zeros


def test_bundle_reports_a_real_eigenvector_not_a_constant():
    """The failure that motivated this: a fabricated constant vector of shape (nV, 1)."""
    n = 2500
    rex = RexGraph(sources=np.arange(n - 1, dtype=np.int32),
                   targets=np.arange(1, n, dtype=np.int32))
    sb = rex.spectral_bundle
    assert sb['fiedler_val_L0'] > 0.0
    evecs = np.asarray(sb['evecs_L0'])
    assert evecs.shape[0] == n
    assert np.unique(np.round(evecs, 9)).size > 2


def test_spectral_bundle_defers_the_solve():
    """Betti and alpha_G must not pay for the Fiedler."""
    n = 2500
    rex = RexGraph(sources=np.arange(n - 1, dtype=np.int32),
                   targets=np.arange(1, n, dtype=np.int32))
    sb = rex.spectral_bundle
    assert sb._filled is False
    _ = sb['beta0'], sb['alpha_G']
    assert sb._filled is False, "reading betti/alpha_G triggered the eigensolve"
    _ = sb['fiedler_val_L0']
    assert sb._filled is True


@pytest.mark.parametrize("route", ["dict", "unpack", "copy"])
def test_copying_the_bundle_does_not_hand_back_none(route):
    """dict(bundle) and {**bundle} take CPython's dict-to-dict fast path, which would
    copy the unresolved slots straight out and report None for a value that exists."""
    n = 2500
    rex = RexGraph(sources=np.arange(n - 1, dtype=np.int32),
                   targets=np.arange(1, n, dtype=np.int32))
    sb = rex.spectral_bundle
    out = {"dict": lambda: dict(sb), "unpack": lambda: {**sb}, "copy": sb.copy}[route]()
    assert out['fiedler_val_L0'] is not None
    assert out['fiedler_val_L0'] > 0.0
