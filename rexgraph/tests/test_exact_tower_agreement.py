"""The exact rational tower must agree with the float one, at every arity.

`exact_channel_diagonals` RECONSTRUCTS each B1 column from its support rather than
reading the float boundary, which is deliberate: reading it would put the exact
tower on the exact value of a double. The cost is that the reconstruction encodes a
convention, and a convention can drift away from the construction silently.

It did. A witness (arity 1) is emitted by the construction as (+1) so that
L0 u = u, and the reconstruction assumed the head rule and emitted (-1). Only F
moved, since the diagonal squares the sign away and C takes absolute values, so
three of four channels agreed and the tower looked healthy.
"""
import itertools
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.harmonic_sparse import _b1_csc
from rexgraph.rational_trig import exact_channel_diagonals
from rexgraph.sparse_character import build_sparse_channels, channel_diagonals

CHANNELS = ["L1_down", "L_O", "L_SG", "L_C"]


def _g(src, tgt, **kw):
    r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32), **kw)
    r._ensure_clean()
    return r


def _hg(ptr, idx):
    r = RexGraph.from_hypergraph(np.array(ptr, np.int64), np.array(idx, np.int64))
    r._ensure_clean()
    return r


def _kn(k):
    a, b = zip(*itertools.combinations(range(k), 2), strict=False)
    return _g(list(a), list(b))


ZOO = {
    "K4": _kn(4),
    "K5": _kn(5),
    "C6": _g([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]),
    "tree": _g([0, 0, 1, 1], [1, 2, 3, 4]),
    "disconnected": _g([0, 1, 2, 5, 6, 7], [1, 2, 0, 6, 7, 5]),
    "parallel": _g([0, 0, 1, 2, 0], [1, 1, 2, 0, 2]),
    "weighted": _g([0, 1, 2, 0], [1, 2, 0, 2], w_E=np.array([2.0, 3.0, 5.0, 7.0])),
    "branching": _hg([0, 3, 6, 9], [0, 1, 2, 1, 2, 3, 0, 2, 3]),
    "witness": _hg([0, 1, 2, 4, 6], [0, 1, 0, 1, 1, 2]),
    "mixed-arity": _hg([0, 1, 3, 6, 10], [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]),
}


def _float_diags(rex):
    """What actually runs: the closed form when it applies, else the assembled one."""
    d = channel_diagonals(rex)
    if d is None:
        d = {n: L.diagonal() for n, L in dict(build_sparse_channels(rex)).items()}
    return d


def _exact_diags(rex):
    ex = exact_channel_diagonals(rex)
    return ex[0] if isinstance(ex, tuple) else ex


@pytest.mark.parametrize("name", list(ZOO))
@pytest.mark.parametrize("channel", CHANNELS)
def test_the_exact_tower_matches_the_float_tower(name, channel):
    rex = ZOO[name]
    a = np.array([float(x) for x in _exact_diags(rex)[channel]])
    b = np.asarray(_float_diags(rex)[channel], dtype=float)
    assert a.shape == b.shape
    assert np.abs(a - b).max() < 1e-9, (name, channel, a, b)


@pytest.mark.parametrize("name", list(ZOO))
def test_the_reconstructed_column_is_the_column_that_exists(name):
    """The guard that would have caught it directly. Whatever convention the exact
    tower reconstructs must be the one the construction actually emitted, so the
    reconstruction is checked against the float boundary itself."""
    rex = ZOO[name]
    B = np.asarray(_b1_csc(rex).todense())
    bp = np.asarray(rex._boundary_ptr)
    bi = np.asarray(rex._boundary_idx)
    for e in range(int(rex.nE)):
        support = [int(v) for v in bi[bp[e]:bp[e + 1]]]
        k = len(support)
        if k == 0:
            continue
        if k == 1:
            want = {support[0]: Fraction(1)}
        else:
            want = {support[0]: Fraction(-1)}
            for v in support[1:]:
                want[v] = want.get(v, Fraction(0)) + Fraction(1, k - 1)
        for v, val in want.items():
            assert abs(float(val) - B[v, e]) < 1e-12, (name, e, v, float(val), B[v, e])


def test_a_witness_column_is_plus_one_and_not_the_head_rule():
    """Named on its own because it is the exception that broke the tower: arity 1
    has no second vertex, so the zero-sum condition does not constrain it and the
    construction emits (+1), not the head's (-1)."""
    rex = ZOO["witness"]
    B = np.asarray(_b1_csc(rex).todense())
    arity = np.diff(_b1_csc(rex).indptr)
    ones = np.flatnonzero(arity == 1)
    assert ones.size >= 1
    for e in ones:
        col = B[:, e]
        assert np.count_nonzero(col) == 1
        assert float(col[np.flatnonzero(col)[0]]) == 1.0


def test_frustration_exact_matches_the_definition_at_every_arity():
    """F[e,e] = sum_{f != e} |T[e,f] - G[e,f]|, computed straight from the boundary.
    This is the reading that stayed right while the reconstruction drifted."""
    for name, rex in ZOO.items():
        B = np.asarray(_b1_csc(rex).todense())
        w = getattr(rex, "w_E", None)
        w = (np.asarray(w, float) if w is not None and np.size(w) == B.shape[1]
             else np.ones(B.shape[1]))
        Bw = B * w[None, :]
        T = Bw.T @ Bw
        G = np.abs(Bw).T @ np.abs(Bw)
        off = T - G
        np.fill_diagonal(off, 0.0)
        want = np.abs(off).sum(axis=1)
        got = np.asarray(rex.frustration_exact.diagonal(), dtype=float)
        assert np.abs(got - want).max() < 1e-9, name
