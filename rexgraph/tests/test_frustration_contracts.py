"""Raw channel contracts, derived here rather than imported.

Dense arrays below are tiny independent oracles; the evaluated diagonal path is
guarded.
"""
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph import compute
from rexgraph.graph import RexGraph
from rexgraph.rational_trig import exact_channel_diagonals
from rexgraph.sparse_character import (
    build_sparse_channels, channel_diagonals, channel_diagonals_integer,
)


NAMES = ("L1_down", "L_O", "L_SG", "L_C")


def _rex(cells, weights, reading="share"):
    ptr = np.array([0, *np.cumsum([len(cell) for cell in cells])], dtype=np.int32)
    idx = np.array([v for cell in cells for v in cell], dtype=np.int32)
    return RexGraph(boundary_ptr=ptr, boundary_idx=idx,
                    w_E=np.array(weights, dtype=object), c_channel=reading)


def _forbidden(*args, **kwargs):
    raise AssertionError("diagonal reader assembled an operator or used an eigen oracle")


@pytest.mark.parametrize("reading", ["share", "count"])
@pytest.mark.parametrize("weights", [
    [1, -1, 1], [0, -1, 1], [-1, -1, -1],
    [Fraction(-2, 3), Fraction(3, 5), Fraction(7, 4)],
])
def test_signed_weight_diagonals_match_both_oracles_without_assembly(monkeypatch, reading, weights):
    r = _rex([(0, 1), (1, 2), (2, 0)], weights, reading)
    exact, _ = exact_channel_diagonals(r)
    assembled = {name: matrix.diagonal() for name, matrix in build_sparse_channels(r)}
    integers, scale = channel_diagonals_integer(r)
    for name in NAMES:
        assert [Fraction(int(x), scale) for x in integers[name]] == exact[name]
    from rexgraph import sparse_character
    monkeypatch.setattr(sparse_character, "build_sparse_channels", _forbidden)
    monkeypatch.setattr(RexGraph, "overlap_gramian_sparse", property(_forbidden))
    for cls in (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix):
        monkeypatch.setattr(cls, "toarray", _forbidden)
    monkeypatch.setattr(np.linalg, "eigh", _forbidden)
    monkeypatch.setattr(sp.linalg, "eigsh", _forbidden)
    actual = channel_diagonals(r)
    for name in NAMES:
        np.testing.assert_allclose(actual[name], [float(x) for x in exact[name]])
        np.testing.assert_allclose(actual[name], assembled[name])
    assert np.all(actual["L_SG"] >= 0)
    assert list(r.w_E) == weights


@pytest.mark.parametrize("lane", ["cpu", "openmp"])
@pytest.mark.parametrize("weights", [
    [-2., 3., -0.5, 0., 1.25],
    [Fraction(-2, 3), Fraction(3, 5), Fraction(7, 4), 0, -2],
])
def test_signed_weights_in_registered_lanes_at_mixed_arity(lane, weights):
    r = _rex([(0,), (0, 1), (1, 0, 2), (2, 0, 1, 3), (3, 2)], weights)
    exact, _ = exact_channel_diagonals(r)
    w = np.asarray(r.w_E, dtype=float)
    saved = w.copy()
    out = compute.dispatch("channel_tower", r._boundary_ptr, r._boundary_idx,
                           int(r.nV), w, prefer=lane)
    for actual, name in zip(out, NAMES, strict=True):
        np.testing.assert_allclose(actual, [float(x) for x in exact[name]])
    np.testing.assert_array_equal(w, saved)


@pytest.mark.parametrize("threads", [1, 4])
def test_compiled_kernel_accepts_signed_weights_and_reused_transpose(threads):
    from rexgraph.core._channel_tower import channel_diagonals_any_arity, transpose_incidence
    r = _rex([(0,), (0, 1), (1, 0, 2), (2, 0, 1, 3)], [-2, 3, -0.5, 0])
    exact, _ = exact_channel_diagonals(r)
    bp, bi, nV = r._boundary_ptr, r._boundary_idx, int(r.nV)
    transposed = transpose_incidence(bp, bi, nV, threads)
    w = np.asarray(r.w_E, dtype=float)
    saved = w.copy()
    out = channel_diagonals_any_arity(bp, bi, nV, w, threads, transposed)
    for actual, name in zip(out, NAMES, strict=True):
        np.testing.assert_allclose(actual, [float(x) for x in exact[name]])
    np.testing.assert_array_equal(w, saved)


@pytest.mark.parametrize("cells,weights,total", [
    ([(0, 1), (1, 2)], [1, 1], Fraction(4)),
    ([(0, 1, 2), (1, 0)], [1, 1], Fraction(6)),
    ([(0, 1, 2, 3), (1, 0)], [1, 1], Fraction(16, 3)),
    ([(0, 1, 2, 3, 4), (1, 0)], [1, 1], Fraction(5)),
    ([(0, 1, 2), (1, 0)], [2, 3], Fraction(36)),
    ([(0, 1, 2, 3), (0, 1)], [1, 1], Fraction(0)),
    ([(0, 1, 2, 3), (1, 0)], [-2, 3], Fraction(32)),
    ([(0, 1), (1, 2)], [1, Fraction(-1, 2**44)], Fraction(1, 2**42)),
])
def test_factor_two_and_arity_shares(cells, weights, total):
    # Independently derived here.
    r = _rex(cells, weights)
    exact, _ = exact_channel_diagonals(r)
    assert sum(exact["L_SG"]) == total
    assert sum(channel_diagonals(r)["L_SG"]) == pytest.approx(float(total), rel=1e-12, abs=0)


def test_weight_signs_conjugate_full_F_but_leave_its_diagonal_unchanged():
    cells = [(0, 1), (1, 2), (2, 0)]
    positive = dict(build_sparse_channels(_rex(cells, [1, 1, 1])))
    signed = dict(build_sparse_channels(_rex(cells, [1, -1, 1])))
    S = sp.diags([1., -1., 1.])
    for name in ("L1_down", "L_O", "L_SG"):
        np.testing.assert_allclose(signed[name].toarray(), (S @ positive[name] @ S).toarray())
    assert np.linalg.eigvalsh(signed["L_SG"].toarray()).min() >= -1e-12
    np.testing.assert_allclose(signed["L_SG"] @ np.ones(3), [4, 8, 4])
    np.testing.assert_allclose(signed["L_SG"] @ np.array([1, -1, 1]), 0)


def test_zero_weight_can_mask_an_orientation_disagreement():
    r = _rex([(0, 1), (1, 2)], [0, 1])
    assert exact_channel_diagonals(r)[0]["L_SG"] == [0, 0]
    np.testing.assert_array_equal(channel_diagonals(r)["L_SG"], [0, 0])


@pytest.mark.parametrize("cells", [
    [(0, 1), (1, 2), (2, 0)],
    [(3, 0, 2, 1), (2, 1), (0,)],
    [(2, 2), (1, 2), (3,)],
])
def test_count_overlap_does_not_sort_or_compact_primary_incidence(cells):
    r = _rex(cells, [1] * len(cells), "count")
    bp, bi = r._boundary_ptr.copy(), r._boundary_idx.copy()
    before, _ = exact_channel_diagonals(r)
    count = r.overlap_count_sparse
    wanted = [[len(set(a) & set(b)) for b in cells] for a in cells]
    np.testing.assert_array_equal(count.toarray(), wanted)
    np.testing.assert_array_equal(r._boundary_ptr, bp)
    np.testing.assert_array_equal(r._boundary_idx, bi)
    assert exact_channel_diagonals(r)[0] == before


def test_hip_signed_weights_when_device_is_available():
    from rexgraph import hip_ternary
    if not hip_ternary.available():
        pytest.skip("HIP channel-tower device unavailable")
    r = _rex([(0,), (0, 1), (1, 0, 2), (2, 0, 1, 3)], [-2, 3, -0.5, 0])
    exact, _ = exact_channel_diagonals(r)
    out = hip_ternary.channel_tower(r._boundary_ptr, r._boundary_idx, int(r.nV),
                                   np.asarray(r.w_E, dtype=float))
    for actual, name in zip(out, NAMES, strict=True):
        np.testing.assert_allclose(actual, [float(x) for x in exact[name]])
