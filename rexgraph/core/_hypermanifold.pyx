# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._hypermanifold - Filtered manifold sequence and dimensional analysis.

The hypermanifold is the filtered family M1 < M2 < ... < Mk where
each inclusion adds new cells, new Dirac eigenmodes, one Bianchi
identity, and curl content.

The harmonic shadow at dimension d is ker(L_d(d)) \\ ker(L_d(d+1)):
the cycles that become boundaries when (d+1)-cells are added. Its
dimension equals rank(B_{d+1}), predicting the number of (d+1)-cells
needed to kill all d-cycles.

Reference: RCFE Foundations, Sections 7, 8, 9.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
)

np.import_array()


def build_manifold_sequence(np.ndarray[f64, ndim=1] evals_L0,
                             np.ndarray[f64, ndim=1] evals_L1,
                             np.ndarray[f64, ndim=1] evals_L2,
                             int nV, int nE, int nF,
                             double tol=1e-8):
    """Build the filtered manifold sequence M1 < M2 < M3.

    For each truncation dimension d, compute:
    - N_d: total degrees of freedom (sum of cell counts up to dim d)
    - betti_d: Betti numbers truncated at dimension d
    - n_bianchi: number of Bianchi identities (d - 1)

    Parameters
    ----------
    evals_L0, evals_L1, evals_L2 : eigenvalues of Hodge Laplacians
    nV, nE, nF : cell counts
    tol : threshold for zero eigenvalue

    Returns
    -------
    dict with manifold sequence data
    """
    # Betti numbers at each truncation level
    cdef int beta0 = 0, beta1 = 0, beta2 = 0
    cdef int j

    for j in range(evals_L0.shape[0]):
        if fabs(evals_L0[j]) < tol:
            beta0 += 1

    for j in range(evals_L1.shape[0]):
        if fabs(evals_L1[j]) < tol:
            beta1 += 1

    for j in range(evals_L2.shape[0]):
        if fabs(evals_L2[j]) < tol:
            beta2 += 1

    # M1: vertices + edges only (1-rex)
    # beta_0(1) = beta0 (from L0, unchanged)
    # beta_1(1) = nE - rank(B1) - rank(B2) but at d=1, no B2 contribution
    # Actually beta_1(1) = dim ker(L1_down) since L1 = L1_down at d=1
    # But we only have the full L1 eigenvalues. We can compute beta_1(1)
    # as nE - rank(B1) = nE - (nV - beta0)
    cdef int rank_B1 = nV - beta0
    cdef int beta1_at_d1 = nE - rank_B1

    # M2: vertices + edges + faces (2-rex)
    # beta_1(2) = beta1 (from full L1 which includes B2 contribution)
    # beta_2(2) = beta2

    manifolds = []

    # d=1: 1-rex
    manifolds.append({
        'dimension': 1,
        'cells': [nV, nE],
        'N': nV + nE,
        'betti': [beta0, beta1_at_d1],
        'n_bianchi': 0,
    })

    # d=2: 2-rex
    if nF > 0:
        manifolds.append({
            'dimension': 2,
            'cells': [nV, nE, nF],
            'N': nV + nE + nF,
            'betti': [beta0, beta1, beta2],
            'n_bianchi': 1,
        })

    return {
        'manifolds': manifolds,
        'max_dimension': 2 if nF > 0 else 1,
        'total_N': nV + nE + nF,
    }


def harmonic_shadow(np.ndarray[f64, ndim=1] evals_Ld_at_d,
                     np.ndarray[f64, ndim=1] evals_Ld_at_d1,
                     double tol=1e-8):
    """Compute the harmonic shadow at dimension d.

    The harmonic shadow is ker(L_d(d)) \\ ker(L_d(d+1)):
    the harmonic forms at dimension d that become exact (non-harmonic)
    when (d+1)-cells are added.

    Its dimension equals rank(B_{d+1}).

    Parameters
    ----------
    evals_Ld_at_d : eigenvalues of L_d with only d-cells present
    evals_Ld_at_d1 : eigenvalues of L_d with (d+1)-cells present

    Returns
    -------
    shadow_dim : int - dimension of the harmonic shadow
    beta_d : int - beta_d at truncation d
    beta_d1 : int - beta_d at truncation d+1
    """
    cdef int beta_d = 0, beta_d1 = 0
    cdef int j

    for j in range(evals_Ld_at_d.shape[0]):
        if fabs(evals_Ld_at_d[j]) < tol:
            beta_d += 1

    for j in range(evals_Ld_at_d1.shape[0]):
        if fabs(evals_Ld_at_d1[j]) < tol:
            beta_d1 += 1

    cdef int shadow_dim = beta_d - beta_d1
    if shadow_dim < 0:
        shadow_dim = 0

    return shadow_dim, beta_d, beta_d1


def dimensional_subsumption(list betti_sequence):
    """Verify Theorem 8.1: beta_k(d+1) <= beta_k(d).

    Parameters
    ----------
    betti_sequence : list of lists, betti_sequence[d] = [beta_0, ..., beta_d]
        Betti numbers at each truncation level.

    Returns
    -------
    is_valid : bool - True if subsumption holds
    violations : list of (d, k, beta_k_d, beta_k_d1) tuples
    """
    violations = []
    for d in range(len(betti_sequence) - 1):
        betti_d = betti_sequence[d]
        betti_d1 = betti_sequence[d + 1]
        # Compare up to min dimension
        for k in range(min(len(betti_d), len(betti_d1))):
            if betti_d1[k] > betti_d[k]:
                violations.append((d, k, betti_d[k], betti_d1[k]))

    return len(violations) == 0, violations


def compute_betti_from_evals(np.ndarray[f64, ndim=1] evals, double tol=1e-8):
    """Count zero eigenvalues (Betti number from Hodge theory)."""
    cdef int count = 0
    cdef int j
    for j in range(evals.shape[0]):
        if fabs(evals[j]) < tol:
            count += 1
    return count
