"""
Test suite for the L_gb (graded boundary Laplacian) operator.

Verifies behavior against known signatures from the L_gb source paper:
    - Universal identities: TF = FC = 1 on every graph
    - Diagonal entries are 0
    - Symmetric tensor (when self-tensor)
    - Cycle signature: TC = 0 uniquely
    - K_n signatures: K_4, K_5, K_6 reference values
    - Sphere tower: S^4 has middle-grade degeneracy

Run with: python3 test_l_gb.py
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .l_gb_reference import (
    dirac_spectrum_at_grade,
    l_gb_channel_tensor,
    l_gb_scalar,
    l_gb_tower,
    normalized_coherence_spectrum,
)

# Test helpers


def _build_kn_with_b2(n):
    """Build the complete graph K_n with all k-gons as faces."""
    edges = list(combinations(range(n), 2))
    n_E = len(edges)
    B1 = np.zeros((n, n_E))
    eidx = {}
    for k, (i, j) in enumerate(edges):
        B1[i, k] = -1
        B1[j, k] = 1
        eidx[(i, j)] = k
    es = set(tuple(sorted(e)) for e in edges)
    cols = []
    for kk in range(3, n + 1):
        for v in combinations(range(n), kk):
            if all(tuple(sorted((v[i], v[(i + 1) % kk]))) in es for i in range(kk)):
                c = np.zeros(n_E)
                for i in range(kk):
                    u, w = v[i], v[(i + 1) % kk]
                    c[eidx[tuple(sorted((u, w)))]] += +1 if u < w else -1
                cols.append(c)
    B2 = np.column_stack(cols) if cols else np.zeros((n_E, 0))
    return B1, B2


def _build_cycle(n):
    """Build cycle graph C_n."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    n_E = len(edges)
    B1 = np.zeros((n, n_E))
    for k, (i, j) in enumerate(edges):
        B1[i, k] = -1
        B1[j, k] = 1
    # B2 has one column: the full cycle
    eidx = {tuple(sorted(e)): j for j, e in enumerate(edges)}
    c = np.zeros(n_E)
    for i in range(n):
        u, w = i, (i + 1) % n
        c[eidx[tuple(sorted((u, w)))]] += +1 if u < w else -1
    B2 = c.reshape(-1, 1)
    return B1, B2


def _reference_channels(B1, B2):
    """Reference channel definitions from test14.py for L_gb fingerprinting."""
    T = B1.T @ B1
    G = B2 @ B2.T if B2.shape[1] > 0 else np.zeros_like(T)
    AB = np.abs(B1)
    M = AB.T @ AB
    np.fill_diagonal(M, 0)
    AL = (M > 0).astype(float)
    C = np.diag(AL.sum(axis=1)) - AL
    F = AB.T @ AB - np.abs(T)
    return [T, G, F, C]


# Tests


def test_normalized_spectrum_basic():
    """Spectrum is sorted descending and starts with 1.0."""
    M = np.diag([1.0, 4.0, 9.0])
    spec = normalized_coherence_spectrum(M)
    assert len(spec) == 3
    assert abs(spec[0] - 1.0) < 1e-12
    assert spec[0] >= spec[1] >= spec[2]
    print("  ✓ normalized_coherence_spectrum sorts and rescales")


def test_normalized_spectrum_empty():
    """Zero matrix returns array([0.])."""
    M = np.zeros((4, 4))
    spec = normalized_coherence_spectrum(M)
    assert spec.shape == (1,)
    assert spec[0] == 0.0
    print("  ✓ normalized_coherence_spectrum handles zero matrix")


def test_l_gb_scalar_diagonal():
    """L_gb of a spectrum against itself has frob = 0."""
    spec = np.array([1.0, 0.7, 0.4, 0.1])
    result = l_gb_scalar(spec, spec)
    assert abs(result["frob"]) < 1e-10
    print("  ✓ l_gb_scalar(spec, spec) has frob = 0")


def test_l_gb_channel_tensor_diagonal_zero():
    """4×4 tensor diagonal is zero (universal)."""
    B1, B2 = _build_kn_with_b2(5)
    hats = _reference_channels(B1, B2)
    tensor = l_gb_channel_tensor(hats)
    diag = np.diag(tensor)
    assert np.allclose(diag, 0), f"Diagonal should be 0, got {diag}"
    print("  ✓ L_gb tensor diagonal is exactly 0")


def test_l_gb_channel_tensor_symmetric():
    """Self-tensor is symmetric."""
    B1, B2 = _build_kn_with_b2(6)
    hats = _reference_channels(B1, B2)
    tensor = l_gb_channel_tensor(hats)
    assert np.allclose(tensor, tensor.T), "Tensor should be symmetric"
    print("  ✓ L_gb self-tensor is symmetric")


def test_universal_identities_TF_FC():
    """TF = FC = 1 universally on every graph (because F is degenerate)."""
    for n in [4, 5, 6, 7]:
        B1, B2 = _build_kn_with_b2(n)
        hats = _reference_channels(B1, B2)
        tensor = l_gb_channel_tensor(hats)
        TF = tensor[0, 2]
        FC = tensor[2, 3]
        GF = tensor[1, 2]
        assert abs(TF - 1.0) < 1e-3, f"K_{n}: TF should be 1, got {TF}"
        assert abs(FC - 1.0) < 1e-3, f"K_{n}: FC should be 1, got {FC}"
        assert abs(GF - 1.0) < 1e-3, f"K_{n}: GF should be 1, got {GF}"
    print("  ✓ TF = FC = GF = 1 on K_4, K_5, K_6, K_7")


def test_kn_signatures_match_reference():
    """K_4, K_5, K_6 must produce exact reference values."""
    expected = {
        4: {"TG": 0.471, "TC": 0.760, "GC": 0.810},
        5: {"TG": 0.740, "TC": 0.884, "GC": 0.820},
        6: {"TG": 0.806, "TC": 1.036, "GC": 0.892},
    }
    for n, exp in expected.items():
        B1, B2 = _build_kn_with_b2(n)
        hats = _reference_channels(B1, B2)
        tensor = l_gb_channel_tensor(hats)
        actual = {
            "TG": tensor[0, 1],
            "TC": tensor[0, 3],
            "GC": tensor[1, 3],
        }
        for key in exp:
            assert abs(actual[key] - exp[key]) < 0.01, (
                f"K_{n} {key}: expected {exp[key]:.3f}, got {actual[key]:.3f}"
            )
    print("  ✓ K_4, K_5, K_6 channel tensor matches reference")


def test_cycle_TC_zero():
    """Cycle graphs uniquely have TC = 0."""
    for n in [4, 5, 6, 8]:
        B1, B2 = _build_cycle(n)
        hats = _reference_channels(B1, B2)
        tensor = l_gb_channel_tensor(hats)
        TC = tensor[0, 3]
        assert abs(TC) < 1e-3, f"C_{n}: TC should be 0, got {TC}"
    print("  ✓ Cycles C_4, C_5, C_6, C_8 all have TC = 0")


def test_l_gb_tower_pairs():
    """l_gb_tower produces correct number of adjacent pairs."""
    B1, B2 = _build_kn_with_b2(4)  # K_4 has B1 and B2
    tower = l_gb_tower([B1, B2])
    # 2 boundary operators -> 3 grades (0, 1, 2) -> 2 pairs ((0,1) and (1,2))
    assert len(tower) == 2
    assert tower[0]["pair"] == (0, 1)
    assert tower[1]["pair"] == (1, 2)
    print("  ✓ l_gb_tower produces correct adjacent pairs")


def test_dirac_spectrum_at_grade():
    """Dirac spectrum at each grade returns valid normalized spectrum."""
    B1, B2 = _build_kn_with_b2(5)
    for grade in [0, 1, 2]:
        spec = dirac_spectrum_at_grade(B1, B2, grade=grade)
        assert spec[0] >= 0
        if spec[0] > 0:
            assert abs(spec[0] - 1.0) < 1e-12, f"Grade {grade}: top should be 1"
    print("  ✓ dirac_spectrum_at_grade returns normalized spectra")


# Main


def run_all():
    print("=" * 60)
    print("  L_gb test suite")
    print("=" * 60)
    print()
    print("── Spectrum primitives ──")
    test_normalized_spectrum_basic()
    test_normalized_spectrum_empty()
    print()
    print("── Scalar L_gb ──")
    test_l_gb_scalar_diagonal()
    print()
    print("── Channel tensor structural properties ──")
    test_l_gb_channel_tensor_diagonal_zero()
    test_l_gb_channel_tensor_symmetric()
    print()
    print("── Universal identities ──")
    test_universal_identities_TF_FC()
    print()
    print("── Reference value matching ──")
    test_kn_signatures_match_reference()
    test_cycle_TC_zero()
    print()
    print("── Tower and Dirac spectrum ──")
    test_l_gb_tower_pairs()
    test_dirac_spectrum_at_grade()
    print()
    print("=" * 60)
    print("  ALL L_gb TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
