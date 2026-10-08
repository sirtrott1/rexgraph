"""Source arithmetic, channel actions and numerical reading domains."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.channel_operator import channel_operator
from rexgraph.core import _character, _curvature, _holomorphic, _wave
from rexgraph.graph import RexGraph
from rexgraph.rational_trig import rational_reconstruct
from rexgraph.sparse_character import (
    build_sparse_character_cheap, build_sparse_rl, channel_diagonals,
    closed_form_applies, compute_sparse_phi, factored_channel_actions,
)
from rexgraph.tower import closure_at


@pytest.mark.parametrize("g_channel", ["raw", "normalized"])
@pytest.mark.parametrize("c_channel", ["share", "count"])
@pytest.mark.parametrize("weights", [None, [0, 2, 3, 1]])
def test_selected_factored_hats_match_declared_channels(g_channel, c_channel, weights):
    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10], np.int32),
        np.array([0, 1, 2, 3, 1, 0, 2, 1, 3, 2], np.int32),
        g_channel=g_channel, c_channel=c_channel, w_E=weights,
    )
    rl, hats, names, traces = build_sparse_rl(rex)
    assert not closed_form_applies(rex)
    diagonals = channel_diagonals(rex)
    for name, hat, trace in zip(names, hats, traces, strict=True):
        np.testing.assert_allclose(diagonals[name], hat.diagonal()*trace, atol=1e-14)
    cheap = build_sparse_character_cheap(rex)
    np.testing.assert_allclose(cheap['trace_values'], traces, atol=1e-14)
    np.testing.assert_allclose(cheap['rl_diag'], rl.diagonal(), atol=1e-14)
    np.testing.assert_allclose(cheap['RL'].toarray(), rl.toarray(), atol=1e-14)
    apply_rl, apply_hat = factored_channel_actions(rex, names, traces)
    block = np.array([[1., 0.], [-2., 1.], [3., -1.], [0., 2.]])
    for values in (block[:, 0], block):
        np.testing.assert_allclose(apply_rl(values), rl @ values, atol=1e-14)
        for name, hat in zip(names, hats, strict=True):
            np.testing.assert_allclose(apply_hat(name, values), hat @ values, atol=1e-14)


@pytest.mark.parametrize("g_channel", ["raw", "normalized"])
@pytest.mark.parametrize("c_channel", ["share", "count"])
@pytest.mark.parametrize("weights", [None, [0, 2, 3, 1], [1, -2, 3, 0]])
def test_numerical_repeated_slot_actions_preserve_the_source(g_channel, c_channel, weights):
    if g_channel == "normalized" and weights is not None and min(weights) < 0:
        with pytest.raises(ValueError, match="nonnegative relation weights"):
            factored_channel_actions(RexGraph.from_graph(
                [0, 0, 1, 1], [0, 1, 0, 1], g_channel=g_channel,
                c_channel=c_channel, w_E=weights), [], [])
        return
    rex = RexGraph.from_graph([0, 0, 1, 1], [0, 1, 0, 1],
        g_channel=g_channel, c_channel=c_channel, w_E=weights)
    supports = rex.relation_supports()
    boundary = rex._B1_dual
    stored = tuple(a.copy() for a in (boundary.row_ptr, boundary.col_idx, boundary.vals))
    rl, hats, names, traces = build_sparse_rl(rex)
    assert not closed_form_applies(rex)
    if g_channel == 'normalized':
        with pytest.raises(ValueError, match='distinct participants'):
            channel_diagonals(rex)
    else:
        diagonals = channel_diagonals(rex)
        for name, hat, trace in zip(names, hats, traces, strict=True):
            np.testing.assert_allclose(diagonals[name], hat.diagonal()*trace, atol=1e-14)
        cheap = build_sparse_character_cheap(rex)
        np.testing.assert_allclose(cheap['trace_values'], traces, atol=1e-14)
        np.testing.assert_allclose(cheap['rl_diag'], rl.diagonal(), atol=1e-14)
        np.testing.assert_allclose(cheap['RL'].toarray(), rl.toarray(), atol=1e-14)
    apply_rl, apply_hat = factored_channel_actions(rex, names, traces)
    block = np.array([[1., 0.], [-2., 1.], [3., -1.], [0., 2.]])
    for values in (block[:, 0], block):
        np.testing.assert_allclose(apply_rl(values), rl @ values, atol=1e-14)
        for name, hat in zip(names, hats, strict=True):
            np.testing.assert_allclose(apply_hat(name, values), hat @ values, atol=1e-14)
    assert rex.relation_supports() == supports
    for current, original in zip((boundary.row_ptr, boundary.col_idx, boundary.vals), stored, strict=True):
        np.testing.assert_array_equal(current, original)
    with pytest.raises(ValueError, match="distinct participants"):
        channel_operator(rex, "F")


def test_zero_channel_mass_has_a_finite_green_preconditioner():
    rex = RexGraph.from_graph([0], [0], w_E=[0])
    cheap = build_sparse_character_cheap(rex)
    with np.errstate(divide='raise', invalid='raise'):
        result = compute_sparse_phi(rex, cheap, backend='cpu')
    np.testing.assert_array_equal(result['phi'], [[.25, .25, .25, .25]])
    np.testing.assert_array_equal(result['kappa'], [1.])


def test_normalized_frustration_retains_raw_g_energy():
    raw = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    normalized = RexGraph.from_graph([0, 1, 2], [1, 2, 0], g_channel="normalized")
    x = np.array([Q(1), Q(2), Q(-1)], dtype=object)
    expected = Q(2) * sum((x[i] - x[j])**2 for i, j in ((0, 1), (1, 2), (0, 2)))
    for rex in (raw, normalized):
        f = channel_operator(rex, "F").apply(x, exact=True)
        assert sum(x * f) == expected
    np.testing.assert_array_equal(
        channel_operator(raw, "F").apply(x, exact=True),
        channel_operator(normalized, "F").apply(x, exact=True),
    )


@pytest.mark.parametrize("weights", [[Q(2), Q(3), Q(0)], [Q(-2), Q(3), Q(1)]])
def test_exact_frustration_energy_with_signed_and_zero_weights(weights):
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0], w_E=weights)
    x = np.array([Q(2), Q(-1, 3), Q(7, 5)], dtype=object)
    off = [Q(-2)*weights[i]*weights[j] for i, j in ((0, 1), (1, 2), (0, 2))]
    expected = sum(
        abs(d)*(x[i] + (1 if d > 0 else -1)*x[j])**2
        for d, (i, j) in zip(off, ((0, 1), (1, 2), (0, 2)), strict=True)
    )
    fx = channel_operator(rex, "F").apply(x, exact=True)
    assert sum(x * fx) == expected >= 0


def test_channel_actions_share_incidence_passes(monkeypatch):
    from rexgraph.native_sparse import NativeSparse
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    _, _, names, traces = build_sparse_rl(rex)
    apply_rl, _ = factored_channel_actions(rex, names, traces)
    calls = []
    original = NativeSparse.apply

    def counted(self, values, *, transpose=False):
        calls.append(transpose)
        return original(self, values, transpose=transpose)

    monkeypatch.setattr(NativeSparse, "apply", counted)
    apply_rl(np.eye(3))
    assert calls.count(False) == calls.count(True) == 3


@pytest.mark.parametrize("g_channel", ["raw", "normalized"])
def test_phi_solves_the_selected_channel_sum(g_channel):
    from rexgraph.sparse_character import build_sparse_character_cheap, compute_sparse_phi
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 3], g_channel=g_channel)
    rl, hats, _, _ = build_sparse_rl(rex)
    rhs = np.asarray(rex.B1).T
    response = np.linalg.solve(rl.toarray(), rhs)
    denominator = np.sum(rhs*response, axis=0)
    expected = np.array([np.sum(response*(h @ response), axis=0)/denominator for h in hats]).T
    actual = compute_sparse_phi(rex, build_sparse_character_cheap(rex), chunk=2, backend="cpu")
    np.testing.assert_allclose(actual["phi"], expected, atol=1e-9)


@pytest.mark.parametrize("shape", [(2,), (3, 1, 1)])
def test_factored_actions_reject_wrong_carriers(shape):
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    _, _, names, traces = build_sparse_rl(rex)
    apply_rl, apply_hat = factored_channel_actions(rex, names, traces)
    with pytest.raises(ValueError, match="vector or block"):
        apply_rl(np.zeros(shape))
    with pytest.raises(ValueError, match="vector or block"):
        apply_hat(names[0], np.zeros(shape))


def test_fractional_boundaries_do_not_claim_integer_curvature():
    b1 = np.array([[-1., -1, -1, -1], [1/3, 1, 0, 0],
                   [1/3, 0, 1, 0], [1/3, 0, 0, 1]])
    b2 = np.array([[-1.], [1/3], [1/3], [1/3]])
    np.testing.assert_allclose(b1 @ b2, 0, atol=1e-15)
    out = _curvature.lagrangian_curvature(b1, b2)
    assert out["c2"] == pytest.approx(float(Q(137, 242)))
    assert "c2_exact" not in out
    assert out["L_T_trace"] == pytest.approx(float(Q(274, 9)))


@pytest.mark.parametrize("scale,certified", [(1., True), (2**20, False)])
def test_integer_curvature_certificate_has_a_reduction_bound(scale, certified):
    b1 = scale*np.array([[-1., 0, 1], [1., -1, 0], [0., 1, -1]])
    b2 = scale*np.ones((3, 1))
    out = _curvature.lagrangian_curvature(b1, b2)
    assert out["c2"] == pytest.approx(.5)
    assert ("c2_exact" in out) == certified
    if certified:
        assert Q(out["c2_exact"]) == Q(1, 2)


def test_commuting_hats_can_have_nonzero_product_imbalance():
    _, hats, _, _ = build_sparse_rl(RexGraph.from_graph([0, 1, 2], [1, 2, 0]))
    hats = [hat.toarray() for hat in hats]
    time, space = hats[0], sum(hats[1:])
    np.testing.assert_allclose(time @ space, space @ time, atol=1e-15)
    result = _holomorphic.relational_cr(hats)
    np.testing.assert_allclose(result["cr"], float(Q(7, 9)))


def test_empty_and_mismatched_hat_carriers():
    hats = [np.empty((0, 0)) for _ in range(4)]
    result = _holomorphic.relational_cr(hats)
    assert result["cr_mean"] == result["cr_std"] == 0
    with pytest.raises(ValueError, match="one shape"):
        _holomorphic.relational_cr([np.eye(2), np.eye(3), np.eye(2), np.eye(2)])


def test_dipole_measures_columns_not_their_span():
    psi = np.ones(3)
    face = np.ones((3, 1))
    first = _character.face_void_dipole(psi, face, face, 3, 1)
    scaled = _character.face_void_dipole(psi, 2*face, face, 3, 1)
    assert first["face_affinity"] == first["void_affinity"] == 3
    assert first["dipole_ratio"] == 0
    assert scaled["face_affinity"] == 12
    assert scaled["dipole_ratio"] == pytest.approx(float(Q(3, 5)))
    assert _character.face_void_dipole(psi, face, np.empty((3, 0)), 3, 1)["void_affinity"] == 0
    empty = _character.face_void_dipole(np.empty(0), np.empty((0, 0)), None, 0, 0)
    assert empty["total_projection"] == 0
    with pytest.raises(ValueError, match="shapes"):
        _character.face_void_dipole(psi, np.ones((2, 1)), None, 3, 1)


def test_two_cofaces_do_not_require_equal_coefficient_mass():
    src = np.array([0, 0, 0, 1, 1, 2], np.int32)
    dst = np.array([1, 2, 3, 2, 3, 3], np.int32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    original = RexGraph.from_simplicial(src, dst, faces)
    scaled = RexGraph(sources=src, targets=dst,
        B2_col_ptr=original._B2_col_ptr.copy(), B2_row_idx=original._B2_row_idx.copy(),
        B2_vals=2*original._B2_vals)
    assert scaled.chain_valid
    result = closure_at(scaled)
    assert result["every_two"] and result["closed"]
    assert not result["mass_equal"]


def _pair_lagrangian(dt, weight=1.):
    return _wave.lagrangian_step(np.array([1., 2.]), np.zeros(2),
        np.array([0], np.int32), np.array([1], np.int32), np.array([weight]), 2, 1, dt)


def test_pair_lagrangian_uses_declared_timestep_and_weights():
    first, weighted = _pair_lagrangian(.0001), _pair_lagrangian(.0001, 99.)
    assert first["T"] == weighted["T"] == 250000000
    assert first["V"] == -2 and weighted["V"] == -198
    assert first["L"] == first["T"] - first["V"]


@pytest.mark.parametrize("dt", [0., -1., float("nan"), float("inf")])
def test_pair_lagrangian_rejects_invalid_timestep(dt):
    with pytest.raises(ValueError, match="positive and finite"):
        _pair_lagrangian(dt)


def test_float_rational_candidate_does_not_certify_source():
    tiny = Q(1, 10**20)
    assert rational_reconstruct([float(tiny)]) == [Q(0)] != [tiny]
    assert rational_reconstruct([float("inf")]) is None
    with pytest.raises(ValueError, match="positive"):
        rational_reconstruct([.5], max_denominator=0)
    with pytest.raises(TypeError, match="integer"):
        rational_reconstruct([.5], max_denominator=True)
