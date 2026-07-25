"""
Character bundle: χ (per-edge), φ (per-vertex), χ* (averaged), κ (coherence).

Pure-NumPy reference implementations matching the corrected definitions
from `rexgraph.core._character`. Builds on the trace-normalized hat
operators from `channels.py`.

The framework's algebraic identities (these are debug oracles):

    χ(e, k) = hat_k[e, e] / RL[e, e]
        => sum_k χ(e, k) = 1 for every edge e (simplex-valued)

    φ(v, k) = x^T hat_k x / s_vv,  where x = RL^+ B_1[v, :], s_vv = B_1[v, :]^T x
        => sum_k φ(v, k) = 1 for every structural vertex (simplex-valued)
        => NO clipping or renormalization needed when hats are properly
          trace-normalized

    χ*(v, k) = mean over incident edges of χ(e, k)

    κ(v) = 1 - 0.5 * L1(φ(v) - χ*(v))
        => κ ∈ [0, 1] for every structural vertex
        => measures agreement between rigorous (φ) and naive (χ*) per-vertex character
        => low κ is the framework's built-in noise/incoherence indicator

If any of these identities fail to hold, something upstream is broken
(usually the channel construction). Do NOT clip or renormalize to "fix"
them - that masks the actual bug.

Reference: rcf_session_bundle/rex_phase_b_v2.py and rexgraph.core._character
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# χ (per-edge structural character)


def compute_chi(RL: np.ndarray, hats: List[np.ndarray]) -> np.ndarray:
    """Per-edge structural character: χ(e, k) = hat_k[e, e] / RL[e, e].

    Returns an (n_E, 4) array where each row is a probability distribution
    over the four channels (T, G, F, C).

    When RL[e, e] ≈ 0 (degenerate edge), defaults to uniform 0.25.

    Sum over k MUST equal 1 for every edge when hats are trace-normalized
    and RL = sum_k hat_k. If sums differ from 1 by more than 1e-10, the
    upstream channel construction is broken.
    """
    n_E = RL.shape[0]
    if n_E == 0:
        return np.zeros((0, 4))

    chi = np.zeros((n_E, 4))
    rl_diag = np.diag(RL)
    for e in range(n_E):
        if abs(rl_diag[e]) > 1e-15:
            for k in range(4):
                chi[e, k] = hats[k][e, e] / rl_diag[e]
        else:
            chi[e, :] = 0.25
    return chi


# φ (per-vertex structural character via pseudoinverse)


def compute_phi_pseudoinverse(
    B_1: np.ndarray,
    RL: np.ndarray,
    hats: List[np.ndarray],
) -> np.ndarray:
    """Per-vertex structural character via pseudoinverse.

    For each vertex v:
        x = RL^+ B_1[v, :]
        s_vv = B_1[v, :]^T x
        φ(v, k) = x^T hat_k x / s_vv

    Returns (n_V, 4). Defaults to uniform 0.25 for non-structural vertices
    (vertices with no incident edges) and degenerate cases.

    NOTE: With proper trace normalization of hats and RL = sum_k hat_k,
    sum_k φ(v, k) = 1 exactly for every structural vertex. If you have
    to clip/renormalize, the channels are wrong.
    """
    n_V, n_E = B_1.shape
    if n_E == 0:
        return np.full((n_V, 4), 0.25)

    RL_pinv = np.linalg.pinv(RL, rcond=1e-10)
    phi = np.zeros((n_V, 4))

    for v in range(n_V):
        b = B_1[v, :]
        if np.max(np.abs(b)) < 1e-15:
            # Non-structural vertex (no incident edges)
            phi[v, :] = 0.25
            continue
        x = RL_pinv @ b
        s_vv = float(b @ x)
        if abs(s_vv) < 1e-12:
            phi[v, :] = 0.25
            continue
        # φ(v, k) = x^T hat_k x / s_vv
        phi[v, :] = np.array([float(x @ hats[k] @ x) / s_vv for k in range(4)])

    return phi


# χ* (mean χ over incident edges, naive per-vertex character)


def compute_chi_star(B_1: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """Mean structural character over incident edges (naive per-vertex view).

    χ*(v, k) = mean over edges incident to v of χ(e, k).

    Defaults to uniform 0.25 for non-structural vertices. Compared against
    the rigorous φ to compute coherence κ.

    Returns (n_V, 4).
    """
    n_V, n_E = B_1.shape
    if n_E == 0:
        return np.full((n_V, 4), 0.25)

    incident_mask = np.abs(B_1) > 1e-10
    chi_star = np.zeros((n_V, 4))
    counts = np.zeros(n_V)

    for e in range(n_E):
        for v in np.where(incident_mask[:, e])[0]:
            chi_star[v] += chi[e]
            counts[v] += 1

    for v in range(n_V):
        if counts[v] == 0:
            chi_star[v, :] = 0.25
        else:
            chi_star[v] /= counts[v]

    return chi_star


# κ (coherence: agreement between φ and χ*)


def compute_kappa(phi: np.ndarray, chi_star: np.ndarray) -> np.ndarray:
    """Per-vertex coherence: κ(v) = 1 - 0.5 * L1(φ(v) - χ*(v)).

    Range [0, 1]. High κ means the rigorous pseudoinverse character
    agrees with the naive averaged-edge character (the vertex sits in
    a structurally coherent neighborhood). Low κ flags structural noise,
    OCR damage, mixed-register content, or other incoherence.

    The 0.5 factor turns L1 distance on the simplex into the total
    variation metric, which lives in [0, 1] for two probability vectors.

    Returns (n_V,) array of coherence values.
    """
    return 1.0 - 0.5 * np.sum(np.abs(phi - chi_star), axis=1)


# Hodge decomposition (gradient/curl/harmonic energies)


def hodge_decompose(
    B_1: np.ndarray,
    B_2: np.ndarray,
    flow: np.ndarray,
) -> Tuple[float, float, float]:
    """Decompose an edge signal into gradient + curl + harmonic energies.

    Solves:
        grad = B_1^T φ_pot  where L_0 φ_pot = B_1 flow
        curl = B_2 ψ        where L_2 ψ = B_2^T flow
        harm = flow - grad - curl

    Returns the energy fractions (pct_grad, pct_curl, pct_harm) summing to 1
    when the chain condition B_1 B_2 = 0 holds (which it does by construction).

    Energy interpretation:
        - gradient: directly retrievable from vertex potentials
        - curl: requires relational reasoning through cycles
        - harmonic: depends on global topology (Betti number content)

    Returns
    -------
    (pct_grad, pct_curl, pct_harm) : tuple of three floats summing to 1
    """
    n_V, n_E = B_1.shape
    if n_E == 0:
        return 0.0, 0.0, 0.0

    # Gradient component
    L0 = B_1 @ B_1.T
    rhs_g = B_1 @ flow
    phi_pot, *_ = np.linalg.lstsq(L0, rhs_g, rcond=1e-10)
    grad = B_1.T @ phi_pot

    # Curl component (only if B_2 has columns)
    if B_2.size > 0 and B_2.shape[1] > 0:
        L2 = B_2.T @ B_2
        rhs_c = B_2.T @ flow
        psi, *_ = np.linalg.lstsq(L2, rhs_c, rcond=1e-10)
        curl = B_2 @ psi
    else:
        curl = np.zeros(n_E)

    # Harmonic = remainder
    harm = flow - grad - curl

    eg = float(grad @ grad)
    ec = float(curl @ curl)
    eh = float(harm @ harm)
    total = eg + ec + eh

    if total < 1e-30:
        return 0.0, 0.0, 0.0
    return eg / total, ec / total, eh / total


# Verification (debug oracles for character bundle)


def verify_character_identities(
    chi: np.ndarray,
    phi: np.ndarray,
    kappa: np.ndarray,
    structural_vertex_mask: np.ndarray = None,
    tol: float = 1e-8,
) -> dict:
    """Verify the framework's character-bundle identities.

    These are debug oracles. If they fail, the channel construction is wrong.

    Identities:
        - sum_k χ(e, k) = 1 for every edge
        - sum_k φ(v, k) = 1 for every structural vertex (no clipping needed)
        - κ(v) ∈ [0, 1] for every structural vertex
    """
    results = {}

    # χ rows sum to 1
    if chi.size > 0:
        chi_sums = chi.sum(axis=1)
        chi_max_dev = float(np.max(np.abs(chi_sums - 1.0)))
        results["chi_simplex"] = (chi_max_dev, chi_max_dev < tol)

    # φ rows sum to 1 (on structural vertices)
    if phi.size > 0:
        if structural_vertex_mask is not None:
            phi_sub = phi[structural_vertex_mask]
        else:
            phi_sub = phi
        if phi_sub.size > 0:
            phi_sums = phi_sub.sum(axis=1)
            phi_max_dev = float(np.max(np.abs(phi_sums - 1.0)))
            results["phi_simplex"] = (phi_max_dev, phi_max_dev < tol)

    # κ in [0, 1]
    if kappa.size > 0:
        if structural_vertex_mask is not None:
            kappa_sub = kappa[structural_vertex_mask]
        else:
            kappa_sub = kappa
        if kappa_sub.size > 0:
            kmin = float(kappa_sub.min())
            kmax = float(kappa_sub.max())
            results["kappa_in_unit_interval"] = (
                (kmin, kmax),
                kmin >= -tol and kmax <= 1 + tol,
            )

    return results
