"""The cheap character reads diagonals, so it should not assemble the operators.

build_sparse_character_cheap says it works "from DIAGONALS only" and then built the
four edge x edge channel operators to read their diagonals off. That assembly costs
sum_v deg(v)^2 nonzeros, which one hub detonates: on a GO-shaped complex (max degree
24256) it was 112 s for four arrays of length nE.

The closed forms below are exact for a signed pairwise unweighted complex and are
gated to exactly that case; anything with arity or weighting still assembles.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.graph import RexGraph
from rexgraph.sparse_character import (
    build_sparse_channels,
    channel_diagonals,
    closed_form_applies,
)


def _triangle():
    return RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 0], dtype=np.int32))


def _random(nv, m, seed):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, nv, m).astype(np.int32)
    t = rng.integers(0, nv, m).astype(np.int32)
    keep = s != t
    rex = RexGraph(sources=s[keep].copy(), targets=t[keep].copy())
    rex._ensure_clean()
    return rex


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_closed_form_matches_the_assembled_diagonal(seed):
    """Each channel, against the operator it replaces."""
    rex = _random(12, 30, seed)
    assert closed_form_applies(rex)
    got = channel_diagonals(rex)
    built = dict(build_sparse_channels(rex))
    for name, L in built.items():
        assert name in got, name
        assert np.allclose(got[name], L.diagonal(), atol=1e-12), name


def test_the_four_readings_are_what_the_model_says():
    """T is the boundary column's squared norm, G shares T's diagonal because squaring
    kills the sign, C is the line-graph degree, F is twice the disagreement count."""
    rex = _triangle()
    d = channel_diagonals(rex)
    assert np.allclose(d['L1_down'], 2.0)          # every 2-ary column has norm^2 = 2
    assert np.allclose(d['L_O'], d['L1_down'])     # identical diagonals
    assert np.allclose(d['L_C'], 2.0)              # each edge meets 2 others
    assert np.all(d['L_SG'] >= 0.0)


def test_gate_rejects_weighting():
    """Under weighting diag(G) = sum_v w_v B1[v,e]^2 stops equalling diag(T)."""
    rex = _triangle()
    assert closed_form_applies(rex)

    class _Weighted:
        nE = 3
        _B1_dual = rex._B1_dual
        w_E = np.array([1.0, 2.0, 1.0])
    assert not closed_form_applies(_Weighted())
    assert channel_diagonals(_Weighted()) is None


def test_gate_rejects_non_binary_relations():
    """A branching column carries -1 and 1/(k-1), so the sign-count reading of F and
    the T/G diagonal identity both stop holding."""
    B1 = sp.csc_matrix(np.array([[-1.0, 0.0], [0.5, -1.0], [0.5, 1.0]]))

    class _Fake:
        nE = 2
        _B1_dual = B1
        w_E = None
    assert not closed_form_applies(_Fake())


def test_character_is_unchanged_by_the_closed_form():
    """chi / chi_star / traces / rl_diag against the assemble-then-read computation."""
    for seed in range(6):
        rex = _random(14, 36, seed)
        cheap = rex._sparse_character
        built = dict(build_sparse_channels(rex))
        names, traces, diags = [], [], []
        for name in ('L1_down', 'L_O', 'L_SG', 'L_C'):
            if name not in built:
                continue
            d = built[name].diagonal()
            tr = float(d.sum())
            if tr > 1e-15:
                names.append(name)
                traces.append(tr)
                diags.append(d / tr)
        assert list(cheap['hat_names']) == names
        assert np.allclose(cheap['trace_values'], traces)
        hd = np.stack(diags, axis=1)
        rl = hd.sum(axis=1)
        assert np.allclose(cheap['rl_diag'], rl, atol=1e-12)
        good = rl > 1e-15
        assert np.allclose(cheap['chi'][good], hd[good] / rl[good, None], atol=1e-12)


def test_operators_are_built_only_when_asked():
    rex = _random(12, 30, 9)
    cheap = rex._sparse_character
    assert cheap._filled is False
    _ = cheap['chi'], cheap['chi_star'], cheap['rl_diag'], cheap['nhats']
    assert cheap._filled is False, "reading a diagonal assembled the operators"
    assert cheap['RL'] is not None
    assert cheap._filled is True


@pytest.mark.parametrize("route", ["dict", "unpack", "copy"])
def test_copying_resolves_the_operators(route):
    rex = _random(12, 30, 10)
    cheap = rex._sparse_character
    out = {"dict": lambda: dict(cheap), "unpack": lambda: {**cheap},
           "copy": cheap.copy}[route]()
    assert out['RL'] is not None
    assert isinstance(out['hats'], list)
