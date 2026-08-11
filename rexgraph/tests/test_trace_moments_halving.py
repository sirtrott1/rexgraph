"""The trace moments come off one halved power walk, and they are exact.

tr(AB) = sum(A ⊙ Bᵀ), and every power of a symmetric X is symmetric, so
tr(X^k) = sum(X^p ⊙ X^q) for any p+q=k. Splitting in half means the walk climbs only
to X^⌈a_max/2⌉ and the trace never forms a product just to read its diagonal.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.scale_propagator import (
    renyi_entropy,
    renyi_from_moments,
    trace_moments,
    trace_power,
)


def _sym(n, density=0.3, seed=0):
    A = sp.random(n, n, density=density, random_state=seed)
    return (A + A.T).tocsr()


def _psd(n, density=0.3, seed=0):
    A = sp.random(n, n, density=density, random_state=seed)
    return (A @ A.T).tocsr()


@pytest.mark.parametrize("a", [1, 2, 3, 4, 5, 6, 7])
def test_trace_power_matches_dense(a):
    X = _sym(30, seed=a)
    want = float(np.trace(np.linalg.matrix_power(X.toarray(), a)))
    assert trace_power(X, a) == pytest.approx(want, rel=1e-10, abs=1e-10)


def test_trace_moments_matches_dense_at_every_order():
    X = _sym(40, seed=3)
    D = X.toarray()
    tr = trace_moments(X, 7)
    assert len(tr) == 7
    for k in range(1, 8):
        want = float(np.trace(np.linalg.matrix_power(D, k)))
        assert tr[k - 1] == pytest.approx(want, rel=1e-10, abs=1e-10)


def test_a_max_one_returns_just_the_trace():
    X = _sym(12, seed=5)
    assert trace_moments(X, 1) == [pytest.approx(float(X.diagonal().sum()))]


def test_walk_climbs_only_to_half(monkeypatch):
    """The point of the halving: a sweep to order 5 builds X² and X³ and stops.
    Counted at the matmul, so a regression to the full walk fails here."""
    X = _sym(25, seed=7)
    calls = {"n": 0}
    orig = sp.csr_matrix.__matmul__

    def counting(self, other):
        calls["n"] += 1
        return orig(self, other)

    monkeypatch.setattr(sp.csr_matrix, "__matmul__", counting)
    trace_moments(X, 5)
    assert calls["n"] == 2, f"expected ceil(5/2)-1 = 2 matmuls, got {calls['n']}"
    calls["n"] = 0
    trace_moments(X, 2)
    assert calls["n"] == 0, "tr(X²) is the Frobenius norm and needs no matmul"


def test_renyi_matches_the_eigendecomposition():
    X = _psd(35, seed=11)
    ev = np.linalg.eigvalsh(X.toarray())
    ev = ev[ev > 1e-12]
    p = ev / ev.sum()
    for a in (2, 3, 4, 5):
        want = float(np.log((p ** a).sum()) / (1 - a))
        assert renyi_entropy(X, a) == pytest.approx(want, abs=1e-8)


def test_renyi_from_moments_agrees_with_renyi_entropy():
    X = _psd(30, seed=13)
    tr = trace_moments(X, 5)
    for a in (2, 3, 4, 5):
        assert renyi_from_moments(tr, a) == pytest.approx(renyi_entropy(X, a), abs=1e-12)


def test_character_varentropy_uses_one_walk(monkeypatch):
    """H2 and H3 share a walk: asking twice would rebuild X² for nothing."""
    from rexgraph.graph import RexGraph
    rex = RexGraph(sources=np.arange(19, dtype=np.int32),
                   targets=np.arange(1, 20, dtype=np.int32))
    import rexgraph.scale_propagator as spg
    calls = {"n": 0}
    orig = spg.trace_moments
    monkeypatch.setattr(spg, "trace_moments", lambda X, a: (calls.__setitem__("n", calls["n"] + 1),
                                                            orig(X, a))[1])
    cv = rex.character_varentropy
    assert calls["n"] == 1, f"expected one shared walk, got {calls['n']}"
    assert cv["gap"] >= 0.0
