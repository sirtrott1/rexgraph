"""Dense NumPy fixtures for four channel operators and their normalized sum.

The functions construct the signed down Laplacian, normalized overlap operator,
frustration Laplacian and a copath operator containing both Hodge sectors.
Each nonzero channel is divided by its trace before summation. A zero channel
contributes the zero matrix, so the sum's trace is the number of nonzero channels.
Tests use this declared operator profile to compare selected compiled operations.
"""

from __future__ import annotations

import numpy as np

# Helpers


def _symmetrize(M: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix to remove numerical asymmetry from FP arithmetic."""
    return 0.5 * (M + M.T)


def trace_normalize(M: np.ndarray) -> np.ndarray:
    """Divide M by its trace. Returns zero matrix if trace is near zero.

    For valid PSD operators with positive trace, this scales so tr(M̂) = 1.
    For degenerate/empty operators, returns the zero matrix (so they
    contribute zero to RL without causing division errors).
    """
    if M.size == 0:
        return M
    tr = float(np.trace(M))
    if abs(tr) < 1e-15:
        return np.zeros_like(M)
    return M / tr


# Topological channel: T = B_1^T B_1


def build_L_T(B_1: np.ndarray) -> np.ndarray:
    """Topological (down) Laplacian: T = B_1^T @ B_1.

    Eigenvalues encode the cycle/cut structure of the underlying graph.
    By construction tr(T) = sum of squared L2 norms of B_1 columns = sum
    of edge weights times 2 (each B_1 column has two non zero entries).
    """
    n_V, n_E = B_1.shape
    if n_E == 0:
        return np.zeros((0, 0))
    return _symmetrize(B_1.T @ B_1)


# Geometric channel: L_O (overlap Laplacian)


def build_L_O(B_1: np.ndarray, vertex_weights: np.ndarray) -> np.ndarray:
    """Overlap Laplacian L_O = I - D_ov^(-1/2) K D_ov^(-1/2).

    Where:
        K[i, j] = sum_v W[v] * |B_1[v, i]| * |B_1[v, j]|
        d_ov[i] = sum_j K[i, j]                       <- ROW SUM (critical)
        D_ov    = diag(d_ov)

    Returns symmetric PSD with eigenvalues in [0, 1].

    The degree vector contains row sums of K, including off diagonal overlap.
    Using only the diagonal would define a different normalization.

    Parameters

    B_1 : np.ndarray (n_V, n_E)
        Grade-1 boundary operator.
    vertex_weights : np.ndarray (n_V,)
        Per-vertex weights (typically token counts in text application).

    Returns

    L_O : np.ndarray (n_E, n_E), symmetric PSD with eigs in [0, 1].
    """
    n_V, n_E = B_1.shape
    if n_E == 0:
        return np.zeros((0, 0))

    abs_B = np.abs(B_1)
    # K[i, j] = sum_v W[v] * |B_1[v, i]| * |B_1[v, j]|
    K = abs_B.T @ (vertex_weights[:, None] * abs_B)

    # ROW SUM, not diagonal
    d_ov = K.sum(axis=1)

    eps = 1e-12
    inv_sqrt = np.where(
        d_ov > eps,
        1.0 / np.sqrt(np.maximum(d_ov, eps)),
        0.0,
    )

    # Normalized matrix S, then L_O = I - S
    S = (inv_sqrt[:, None] * K) * inv_sqrt[None, :]
    L = np.eye(n_E) - S
    return _symmetrize(L)


# Frustration channel: L_SG


def build_L_SG(B_1: np.ndarray,
               vertex_weights: np.ndarray | None = None) -> np.ndarray:
    """Frustration Laplacian L_SG = D_{|K_off|} - K_off.

    Where:
        w(v) = 1 / log(deg(v) + e)        inverse-log-degree (default)
        K_s = B_1^T @ diag(w) @ B_1        signed weighted Gramian
        K_off = K_s with diagonal zeroed
        L_SG = D_{|K_off|} - K_off

    For unsigned graphs (all edge orientations consistent), K_s is dominated
    by the unsigned coupling and L_SG reduces to a degree weighted line graph
    Laplacian. The frustration channel measures sign disagreement among
    edges sharing vertices.

    Parameters

    B_1 : np.ndarray (n_V, n_E)
    vertex_weights : np.ndarray (n_V,), optional
        If None, uses inverse-log-degree weights (the default per the repo).

    Returns

    L_SG : np.ndarray (n_E, n_E), symmetric PSD.
    """
    n_V, n_E = B_1.shape
    if n_E == 0:
        return np.zeros((0, 0))

    if vertex_weights is None:
        # inverse log degree
        deg = (np.abs(B_1) > 1e-15).sum(axis=1).astype(float)
        vertex_weights = 1.0 / np.log(deg + np.e)
    else:
        vertex_weights = np.asarray(vertex_weights, dtype=np.float64)

    # K_s = B_1^T @ diag(w_v) @ B_1
    K_s = B_1.T @ (vertex_weights[:, None] * B_1)
    # Zero the diagonal
    K_off = K_s - np.diag(np.diag(K_s))
    # L_SG = D_{|K_off|} - K_off
    L_SG = np.diag(np.abs(K_off).sum(axis=1)) - K_off
    return _symmetrize(L_SG)


# Copath channel: L_C


def build_L_C(B_1: np.ndarray, B_2: np.ndarray) -> np.ndarray:
    """Copath Laplacian L_C = L1_down + B_2 @ B_2^T.

    This is the line graph Hodge edge Laplacian: it combines the down
    Laplacian (B_1^T B_1) with the up Laplacian (B_2 B_2^T). The result
    has the line graph structure of the rex.

    NOTE: this is NOT a copy of the unsigned line graph adjacency Laplacian.
    The B_2 contribution carries the face structure that distinguishes L_C
    from a pure adjacency based operator.

    Parameters

    B_1 : np.ndarray (n_V, n_E)
    B_2 : np.ndarray (n_E, n_F)
        May be (n_E, 0) if no faces.

    Returns

    L_C : np.ndarray (n_E, n_E), symmetric PSD.
    """
    n_V, n_E = B_1.shape
    if n_E == 0:
        return np.zeros((0, 0))

    L1_down = B_1.T @ B_1
    if B_2.size > 0 and B_2.shape[1] > 0:
        L1_up = B_2 @ B_2.T
        return _symmetrize(L1_down + L1_up)
    return _symmetrize(L1_down)


# Channel bundle: build all four trace normalized hats


def build_channels(
    B_1: np.ndarray,
    B_2: np.ndarray,
    vertex_weights: np.ndarray,
    frustration_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build all four trace normalized hat operators and the relational Laplacian.

    Returns (RL, [hat_T, hat_G, hat_F, hat_C]) where each hat has trace 1
    by construction (when the underlying operator is non degenerate) and
    tr(RL) = 4 (the framework identity).

    Parameters

    B_1 : np.ndarray (n_V, n_E)
    B_2 : np.ndarray (n_E, n_F)
    vertex_weights : np.ndarray (n_V,)
        For the L_O overlap Laplacian (typically token counts in text).
    frustration_weights : np.ndarray (n_V,), optional
        For L_SG. Defaults to inverse-log-degree if not provided.

    Returns

    RL : np.ndarray (n_E, n_E)
        The relational Laplacian RL = sum_k hat_k. tr(RL) = 4.
    hats : list of 4 np.ndarray
        [hat_T, hat_G, hat_F, hat_C], each with trace 1.
    """
    n_V, n_E = B_1.shape
    if n_E == 0:
        return np.zeros((0, 0)), [np.zeros((0, 0))] * 4

    # Build raw operators
    L_T = build_L_T(B_1)
    L_G = build_L_O(B_1, vertex_weights)
    L_F = build_L_SG(B_1, frustration_weights)
    L_C = build_L_C(B_1, B_2)

    # Trace normalize each: this is non negotiable
    hat_T = trace_normalize(L_T)
    hat_G = trace_normalize(L_G)
    hat_F = trace_normalize(L_F)
    hat_C = trace_normalize(L_C)

    RL = hat_T + hat_G + hat_F + hat_C
    return RL, [hat_T, hat_G, hat_F, hat_C]


# Verification helpers (debug oracles)


def verify_channel_identities(RL: np.ndarray,
                              hats: list[np.ndarray],
                              tol: float = 1e-10) -> dict:
    """Verify the framework's algebraic identities on the trace normalized hats.

    Returns a dict of identity name -> bool valid plus diagnostic values.

    Identities checked:
        - tr(hat_k) = 1 for each k where hat_k is non-zero
        - tr(RL) = number of non-zero hats (typically 4)
        - All hats are PSD
        - RL is PSD
    """
    results = {}
    n_nonzero = sum(1 for h in hats if np.any(h))

    # Per hat trace
    traces = [float(np.trace(h)) for h in hats]
    for _i, (name, tr) in enumerate(zip("TGFC", traces, strict=False)):
        if abs(tr) > 1e-15:  # only check non degenerate
            results[f"tr(hat_{name})"] = (tr, abs(tr - 1.0) < tol)

    # RL trace
    rl_trace = float(np.trace(RL))
    results["tr(RL)"] = (rl_trace, abs(rl_trace - n_nonzero) < tol)

    # PSD checks (smallest eigenvalue >= -tol)
    for name, h in zip("TGFC", hats, strict=False):
        if h.size > 0 and np.any(h):
            try:
                ev_min = float(np.linalg.eigvalsh(h).min())
                results[f"PSD(hat_{name})"] = (ev_min, ev_min >= -tol)
            except np.linalg.LinAlgError:
                results[f"PSD(hat_{name})"] = (None, False)

    if RL.size > 0:
        try:
            ev_min = float(np.linalg.eigvalsh(RL).min())
            results["PSD(RL)"] = (ev_min, ev_min >= -tol)
        except np.linalg.LinAlgError:
            results["PSD(RL)"] = (None, False)

    return results
