"""
Graded boundary Laplacian (L_gb).

A structural operator on the relational complex. Unlike the classical Hodge
Laplacian which lives at a single grade, L_gb measures structural coupling
between adjacent grades, or between channel pairs at the same grade.

Three variants:

1. l_gb_scalar(spec_d, spec_d1) - between adjacent grades d and d+1.
   Rank-2 by construction (difference of two rank-1 projections).
   One positive eigenvalue (dominant side), one negative (subdominant).

2. l_gb_channel_tensor(hats_A, hats_B) - 4×4 within a single rex.
   Entry [i,j] is the Frobenius norm of L_gb between channel i and channel j.
   Acts as a structural fingerprint distinguishing graph families.

3. l_gb_tower(B_list) - sweep across all adjacent grade pairs.
   Used for sphere fingerprinting: S^n has a distinctive tower signature.

Reference: the L_gb source paper sections 6-9.

This is a pure-NumPy implementation. When merged into the main rexgraph
repository, it should be reimplemented as a Cython module
`rexgraph/core/_l_gb.pyx` for ~100x speedup on large complexes.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# Spectrum extraction


def normalized_coherence_spectrum(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Sorted absolute eigenvalues, rescaled so max = 1.

    Used as the canonical spectrum representation for L_gb computations.
    Symmetrizes M before computing eigenvalues for numerical stability.

    Parameters
    ----------
    M : np.ndarray, square
        Symmetric (or near-symmetric) matrix.
    eps : float
        Eigenvalues with |λ| < eps are discarded as numerical noise.

    Returns
    -------
    np.ndarray of length k where k = number of nonzero eigenvalues.
    Sorted in decreasing order, with values[0] = 1.
    Returns array([0.]) if all eigenvalues are below eps.
    """
    if M.size == 0 or M.shape[0] == 0:
        return np.zeros(1)
    M_sym = 0.5 * (M + M.T)
    try:
        evals = np.linalg.eigvalsh(M_sym)
    except np.linalg.LinAlgError:
        return np.zeros(1)
    a = np.sort(np.abs(evals))[::-1]
    a = a[a > eps]
    if len(a) == 0:
        return np.zeros(1)
    return a / a[0]


def dirac_spectrum_at_grade(
    B1: np.ndarray,
    B2: Optional[np.ndarray] = None,
    grade: int = 1,
) -> np.ndarray:
    """Dirac coherence spectrum at a given grade.

    The Dirac operator D acts on the graded chain space. Its restriction
    to grade d is essentially the boundary block from grade d to grade d-1
    composed with its adjoint.

    Parameters
    ----------
    B1 : np.ndarray (n_V × n_E)
        Grade-1 boundary operator.
    B2 : np.ndarray (n_E × n_F), optional
        Grade-2 boundary operator. Required if grade >= 2.
    grade : int
        Which grade to extract the spectrum at (0, 1, or 2).

    Returns
    -------
    Normalized coherence spectrum (sorted abs eigenvalues, max = 1).
    """
    if grade == 0:
        # Grade 0: vertex Laplacian B1 @ B1^T
        L = B1 @ B1.T
    elif grade == 1:
        # Grade 1: full Hodge Laplacian L_1 = B1^T @ B1 + B2 @ B2^T
        L_down = B1.T @ B1
        if B2 is not None and B2.size > 0:
            L_up = B2 @ B2.T
            L = L_down + L_up
        else:
            L = L_down
    elif grade == 2:
        if B2 is None or B2.size == 0:
            return np.zeros(1)
        L = B2.T @ B2
    else:
        raise ValueError(f"Grade {grade} not supported (need 0, 1, or 2)")

    return normalized_coherence_spectrum(L)


# Scalar L_gb between adjacent grades


def l_gb_scalar(spec_d: np.ndarray, spec_d1: np.ndarray) -> dict:
    """Graded boundary Laplacian between adjacent grades d and d+1.

    L_gb^(d,d+1) = (k_d k_d^T) / ||k_d||²  -  (k_{d+1} k_{d+1}^T) / ||k_{d+1}||²

    where k_d is the normalized coherence spectrum at grade d. The result
    is rank-2 by construction with one positive and one negative eigenvalue.

    Parameters
    ----------
    spec_d : np.ndarray
        Coherence spectrum at grade d.
    spec_d1 : np.ndarray
        Coherence spectrum at grade d+1.

    Returns
    -------
    dict with:
        top_eig : float - top (positive) eigenvalue of L_gb
        bot_eig : float - bottom (negative) eigenvalue of L_gb
        spread : float - top_eig - bot_eig
        localization : float - sign-weighted mass on grade d vs grade d+1
                                 negative = grade d dominates
                                 positive = grade d+1 dominates
        frob : float - Frobenius norm
        L_gb : np.ndarray - the full L_gb matrix (padded to common size)
    """
    L = max(len(spec_d), len(spec_d1))
    a = np.pad(spec_d, (0, L - len(spec_d)))
    b = np.pad(spec_d1, (0, L - len(spec_d1)))

    # Match the reference convention from the L_gb source:
    # when one spectrum is zero, the projection collapses to just the other
    # rank-1 projection, whose Frobenius norm is 1. This produces the
    # universal TF=FC=1 identity on graphs where F is degenerate.
    nx = max(float(np.linalg.norm(a)), 1e-12)
    ny = max(float(np.linalg.norm(b)), 1e-12)
    PA = np.outer(a, a) / (nx * nx)
    PB = np.outer(b, b) / (ny * ny)
    L_gb = PA - PB
    na = nx
    nb = ny

    evals = np.linalg.eigvalsh(0.5 * (L_gb + L_gb.T))
    top = float(evals[-1])
    bot = float(evals[0])
    frob = float(np.linalg.norm(L_gb, "fro"))

    # Localization: sign-weighted mass differential
    # Take the top eigenvector of |L_gb|, project it onto a (grade d) vs b (grade d+1)
    abs_L = np.abs(L_gb)
    try:
        eigvals_abs, eigvecs_abs = np.linalg.eigh(0.5 * (abs_L + abs_L.T))
        v_top = eigvecs_abs[:, -1]
        # Inner products with normalized a and b
        a_norm = a / na
        b_norm = b / nb
        ma = float(np.dot(v_top, a_norm)) ** 2
        mb = float(np.dot(v_top, b_norm)) ** 2
        # Localization in [-1, 1]: positive = grade d+1, negative = grade d
        if ma + mb > 1e-15:
            localization = (mb - ma) / (mb + ma)
        else:
            localization = 0.0
    except np.linalg.LinAlgError:
        localization = 0.0

    return {
        "top_eig": top,
        "bot_eig": bot,
        "spread": top - bot,
        "localization": localization,
        "frob": frob,
        "L_gb": L_gb,
    }


# 4×4 channel L_gb tensor (within-grade fingerprint)


def l_gb_channel_tensor(
    hats_A: List[np.ndarray],
    hats_B: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """4×4 L_gb channel tensor.

    Entry [i, j] is the Frobenius norm of L_gb between the i-th channel of
    hats_A and the j-th channel of hats_B (or hats_A again if hats_B is None,
    giving the self-tensor).

    The order is always [T, G, F, C].

    Self-tensor properties (from the L_gb source paper section 7):
        - Diagonal entries are 0 (each channel against itself).
        - TF = FC = 1 universally on every graph.
        - Cycles: TC = 0 uniquely.
        - Bipartite: GF = 0, TG ≈ 1.
        - K_n: all nonzero, scaling with n. K_6 has TC > 1.
        - Petersen: GC ≈ 1.233 (only graph with GC > 1 besides K_6).

    Parameters
    ----------
    hats_A : list of 4 np.ndarray
        The four channels [T_hat, G_hat, F_hat, C_hat] of the first complex.
    hats_B : list of 4 np.ndarray, optional
        If provided, computes cross-tensor between A and B.
        If None, computes the self-tensor of A.

    Returns
    -------
    np.ndarray of shape (4, 4) with entries in [0, ~2].
    """
    if hats_B is None:
        hats_B = hats_A

    if len(hats_A) != 4 or len(hats_B) != 4:
        raise ValueError("Expected exactly 4 channels per side (T, G, F, C)")

    specs_A = [normalized_coherence_spectrum(h) for h in hats_A]
    specs_B = [normalized_coherence_spectrum(h) for h in hats_B]

    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue  # diagonal stays 0 by construction
            sA, sB = specs_A[i], specs_B[j]
            L = max(len(sA), len(sB))
            a = np.pad(sA, (0, L - len(sA)))
            b = np.pad(sB, (0, L - len(sB)))
            # Use the test14.py convention: max(norm, 1e-12) as denominator
            # so degenerate channels produce the universal identity (norm = 1)
            na = max(float(np.linalg.norm(a)), 1e-12)
            nb = max(float(np.linalg.norm(b)), 1e-12)
            PA = np.outer(a, a) / (na * na)
            PB = np.outer(b, b) / (nb * nb)
            T[i, j] = float(np.linalg.norm(PA - PB, "fro"))
    return T


# Tower L_gb across all adjacent grades


def l_gb_tower(B_list: List[np.ndarray]) -> List[dict]:
    """Sweep L_gb across all adjacent grade pairs.

    For a complex with boundary operators [B1, B2, B3, ...], computes the
    L_gb scalar at each adjacent pair: (1,2), (2,3), (3,4), ...

    Used for sphere fingerprinting (the L_gb source paper section 9):
        - S^2: single pair, top_eig = 0
        - S^3: two pairs, both top_eig ≈ 0.577
        - S^4: three pairs. Middle pair (2,3) has top_eig = 0 and POSITIVE
               localization (+0.28), distinguishing dimension 4.
        - S^5: four pairs, nested symmetric pattern

    Parameters
    ----------
    B_list : list of np.ndarray
        Boundary operators [B1, B2, B3, ...]. B_d has shape (n_{d-1}, n_d).
        Allowed to contain None or empty arrays for missing grades.

    Returns
    -------
    list of dicts, one per adjacent pair (d, d+1).
    Each dict has the same fields as l_gb_scalar, plus:
        pair : tuple (d, d+1)
    """
    # Compute spectrum at each grade
    n_grades = len(B_list)
    if n_grades == 0:
        return []

    # Use the highest-grade boundary for the spectrum at each grade
    # For grade d, we use the full Hodge Laplacian L_d = B_d^T B_d + B_{d+1} B_{d+1}^T
    specs = []
    for d in range(n_grades + 1):
        # Down part: B_d^T @ B_d (only if d >= 1)
        if d >= 1 and d - 1 < n_grades and B_list[d - 1] is not None and B_list[d - 1].size > 0:
            B_d = B_list[d - 1]  # boundary at grade d
            L_down = B_d.T @ B_d
        else:
            L_down = None
        # Up part: B_{d+1} @ B_{d+1}^T
        if d < n_grades and B_list[d] is not None and B_list[d].size > 0:
            B_dp1 = B_list[d]  # boundary at grade d+1
            L_up = B_dp1 @ B_dp1.T
        else:
            L_up = None

        if L_down is None and L_up is None:
            specs.append(np.zeros(1))
        elif L_down is None:
            specs.append(normalized_coherence_spectrum(L_up))
        elif L_up is None:
            specs.append(normalized_coherence_spectrum(L_down))
        else:
            # Match dimensions by zero-padding the smaller
            n = max(L_down.shape[0], L_up.shape[0])
            if L_down.shape[0] < n:
                pad = n - L_down.shape[0]
                L_down = np.pad(L_down, ((0, pad), (0, pad)))
            if L_up.shape[0] < n:
                pad = n - L_up.shape[0]
                L_up = np.pad(L_up, ((0, pad), (0, pad)))
            specs.append(normalized_coherence_spectrum(L_down + L_up))

    # Compute L_gb between each adjacent pair of grades
    results = []
    for d in range(len(specs) - 1):
        result = l_gb_scalar(specs[d], specs[d + 1])
        result["pair"] = (d, d + 1)
        results.append(result)

    return results
