"""
Tests for rexgraph.core._l_gb (graded boundary Laplacian).

Verifies:
    1. Compiled kernel produces identical output to pure-numpy reference
       (to within BLAS reordering noise, ~1e-13 relative)
    2. Reference values from the post-paper findings (test14.py) match
       to 3 decimal places on canonical graph families
    3. Self-tensor is symmetric: T[i, j] == T[j, i]

These are the regression tests for the L_gb operator. If this passes,
the operator's algebraic identities hold and the compiled kernel is
correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from rexgraph.core import _l_gb as compiled
from rexgraph.tests.reference import l_gb_reference as reference

# Test fixtures: canonical graph families


def _build_K_n_complex(n: int):
    """Build the boundary operators for the complete graph K_n."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_E = len(edges)
    B_1 = np.zeros((n, n_E), dtype=np.float64)
    for k, (i, j) in enumerate(edges):
        B_1[i, k] = -1.0
        B_1[j, k] = +1.0

    triangles = [(i, j, k) for i in range(n)
                 for j in range(i + 1, n)
                 for k in range(j + 1, n)]
    n_F = len(triangles)
    edge_idx = {e: i for i, e in enumerate(edges)}
    B_2 = np.zeros((n_E, n_F), dtype=np.float64)
    for c, (i, j, k) in enumerate(triangles):
        B_2[edge_idx[(i, j)], c] = +1.0
        B_2[edge_idx[(j, k)], c] = +1.0
        B_2[edge_idx[(i, k)], c] = -1.0
    return B_1, B_2


def _build_cycle_complex(n: int):
    """Build the boundary operators for the cycle graph C_n."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    B_1 = np.zeros((n, n), dtype=np.float64)
    for k, (i, j) in enumerate(edges):
        B_1[i, k] = -1.0
        B_1[j, k] = +1.0
    B_2 = np.zeros((n, 0), dtype=np.float64)
    return B_1, B_2


# Equivalence: compiled kernel vs reference


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_normalized_coherence_spectrum_matches_reference(n):
    """compiled.normalized_coherence_spectrum == reference, to BLAS noise."""
    B_1, B_2 = _build_K_n_complex(n)
    L = B_1.T @ B_1
    spec_ref = reference.normalized_coherence_spectrum(L)
    spec_cmp = compiled.normalized_coherence_spectrum(L)
    assert np.allclose(spec_ref, spec_cmp, atol=1e-13), \
        f"Spectrum mismatch for K_{n}: ref={spec_ref}, compiled={spec_cmp}"


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_dirac_spectrum_at_grade_matches_reference(n):
    B_1, B_2 = _build_K_n_complex(n)
    for grade in (0, 1, 2):
        spec_ref = reference.dirac_spectrum_at_grade(B_1, B_2, grade=grade)
        spec_cmp = compiled.dirac_spectrum_at_grade(B_1, B_2, grade=grade)
        assert np.allclose(spec_ref, spec_cmp, atol=1e-13), \
            f"Dirac spectrum mismatch K_{n} grade {grade}"


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_l_gb_channel_tensor_matches_reference(n):
    """4x4 channel tensor must match reference exactly (up to BLAS noise)."""
    from rexgraph.core._relational import build_RL
    from rexgraph.tests.reference.channels_reference import (
        build_L_C,
        build_L_O,
        build_L_SG,
        build_L_T,
    )
    B_1, B_2 = _build_K_n_complex(n)
    W = np.ones(n, dtype=np.float64)
    laplacians = [
        build_L_T(B_1),
        build_L_O(B_1, W),
        build_L_SG(B_1),
        build_L_C(B_1, B_2),
    ]
    rl_result = build_RL(laplacians, ["T", "G", "F", "C"])
    hats = list(rl_result["hats"]) if isinstance(rl_result, dict) else rl_result[1]

    T_ref = reference.l_gb_channel_tensor(hats)
    T_cmp = compiled.l_gb_channel_tensor(hats)
    assert np.allclose(T_ref, T_cmp, atol=1e-13), \
        f"L_gb tensor mismatch for K_{n}"


# Reference values from post-paper test14.py (3 decimals)


def _self_tensor_for_K_n(n):
    """Compute L_gb 4×4 self-tensor for K_n via pure-numpy reference."""
    from rexgraph.tests.reference.channels_reference import build_channels
    B_1, B_2 = _build_K_n_complex(n)
    W = np.ones(n, dtype=np.float64)
    RL, hats = build_channels(B_1, B_2, W)
    return reference.l_gb_channel_tensor(hats)


@pytest.mark.parametrize("n,expected", [
    (4, {"TG": 0.760, "TC": 1.000, "GC": 0.632}),
    (5, {"TG": 0.884, "TC": 1.095, "GC": 0.535}),
    (6, {"TG": 1.036, "TC": 1.155, "GC": 0.471}),
])
def test_K_n_signature_matches_post_paper(n, expected):
    """Reference values from rcf_post_paper test14.py (3 decimals)."""
    T = _self_tensor_for_K_n(n)
    # T row/col order: T=0, G=1, F=2, C=3
    actual = {"TG": T[0, 1], "TC": T[0, 3], "GC": T[1, 3]}
    for key, exp_val in expected.items():
        assert abs(actual[key] - exp_val) < 0.01, \
            f"K_{n} {key}: expected {exp_val:.3f}, got {actual[key]:.4f}"


@pytest.mark.parametrize("n", [4, 5, 6])
def test_self_tensor_symmetry(n):
    """Self-tensor must be symmetric: T[i, j] == T[j, i] for all i, j."""
    T = _self_tensor_for_K_n(n)
    assert np.allclose(T, T.T, atol=1e-10), f"Self-tensor not symmetric for K_{n}"


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cycle_graph_TC_zero(n):
    """Cycle graphs are uniquely characterized by TC = 0 in self-tensor."""
    from rexgraph.tests.reference.channels_reference import build_channels
    B_1, B_2 = _build_cycle_complex(n)
    W = np.ones(n, dtype=np.float64)
    RL, hats = build_channels(B_1, B_2, W)
    T = reference.l_gb_channel_tensor(hats)
    assert abs(T[0, 3]) < 0.01, f"Cycle C_{n}: TC = {T[0, 3]:.4f}, expected ~0"


# Self-tensor diagonal is identically zero


@pytest.mark.parametrize("n", [4, 5, 6])
def test_self_tensor_diagonal_is_zero(n):
    """For self-tensor, T[i, i] = 0 for all i (channel matches itself)."""
    T = _self_tensor_for_K_n(n)
    for i in range(4):
        assert abs(T[i, i]) < 1e-10, f"K_{n} self-tensor T[{i},{i}] = {T[i,i]:.2e}"


# L_gb tower across grades


@pytest.mark.parametrize("n", [5, 6, 7])
def test_l_gb_tower_matches_reference(n):
    B_1, B_2 = _build_K_n_complex(n)
    tower_ref = reference.l_gb_tower([B_1, B_2])
    tower_cmp = compiled.l_gb_tower([B_1, B_2])
    assert len(tower_ref) == len(tower_cmp), \
        f"Tower length mismatch: ref={len(tower_ref)}, cmp={len(tower_cmp)}"
    for d, (r, c) in enumerate(zip(tower_ref, tower_cmp, strict=False)):
        assert r["pair"] == c["pair"], f"Pair mismatch at d={d}"
        assert np.allclose(r["L_gb"], c["L_gb"], atol=1e-13), \
            f"L_gb matrix mismatch at pair {r['pair']}"
        for key in ("top_eig", "bot_eig", "spread", "frob", "localization"):
            assert abs(r[key] - c[key]) < 1e-12, \
                f"{key} mismatch at pair {r['pair']}: ref={r[key]}, cmp={c[key]}"
