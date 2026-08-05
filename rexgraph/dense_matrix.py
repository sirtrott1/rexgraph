"""rexgraph.dense_matrix: dense materialization and dense-only linear algebra.

The RexGraph numerical core is sparse/exact first: Laplacians assemble as scipy
CSR (``rexgraph.core._laplacians``), the G-channel Gramian as sparse
``|B1|^T |B1|`` (``_overlap``), Lagrangian curvature via sparse trace identities
(``_curvature``), and the character / coherence Green's function via a single SPD
Cholesky solve of the relational Laplacian
(``_relational.build_green_cache_spd``) - no eigendecomposition and no dense
pseudoinverse on that path.

This module is the *modular home* for the places where a dense ``ndarray`` is
still the right (or the only) representation: materializing a sparse operator for
a dense consumer, and the genuinely spectral operations that have no cheaper
sparse form. Keeping them here, rather than scattered inline in ``graph.py``,
makes the dense path explicit and isolated, so it stays available as a fallback
and is easy to bypass, while the primary code paths remain on the sparse/exact
kernels.
"""
from __future__ import annotations

import numpy as np

_f64 = np.float64


def ensure_dense(M):
    """Materialize ``M`` as a dense float64 ndarray.

    Pass-through for existing ndarrays and ``None``; densifies scipy sparse /
    ``_sparse`` CSR objects via ``.toarray()``. This is the single densification
    chokepoint: call it only when a consumer genuinely needs a dense operator.
    """
    if isinstance(M, np.ndarray):
        return M
    if M is None:
        return None
    return np.asarray(M.toarray(), dtype=_f64)


def spectral_distance(A, B):
    """Sorted-eigenvalue (spectral) distance between two symmetric dense operators.

    ``||sort(eig A) - sort(eig B)||_2``. Isolated here because it genuinely needs
    the full spectrum of both operators, and there is no sparse shortcut for an
    all-eigenvalue comparison.
    """
    ea = np.sort(np.linalg.eigvalsh(A))
    eb = np.sort(np.linalg.eigvalsh(B))
    return float(np.linalg.norm(ea - eb))
