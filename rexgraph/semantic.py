"""Semantic significance as a FIELD, not a score.

`R_eff(e)` collapses a relation to one number. The number is the quadrance of a vector
that was there all along, and keeping the vector is what makes the reading usable by a
learner: significance stops being a weight to multiply by and becomes a geometry to
learn in.

    v_e = L0^+ b_e            in vertex space, one vector per relation
    <u, v>_L = u^T L0 v       the inner product it lives under
    Q(v_e)   = R_eff(e)       so the quadrance IS the significance (Theorem 16)
    s(v_i, v_j)               the spread, the semantic distance between two relations

`v_e` is the potential induced by pushing one unit of flow along `e`, so its support is
where that relation's influence actually reaches, and two relations are close when they
move the complex the same way. Both readings come from the same solve as `R_eff`, so
nothing here takes an eigendecomposition.

The embedding is `nV x nE`: a (0,1)-tensor over the complex, which is what a cochain
learner consumes.
"""
from __future__ import annotations

import numpy as np

__all__ = ["relation_field", "semantic_gram", "semantic_spread", "significance"]


def relation_field(rex, edges=None):
    """(V, Q) - the embedding and its quadrances.

    V is `nV x len(edges)`, column `i` being `L0^+ b_e` for relation `edges[i]`.
    Q[i] = <V[:,i], V[:,i]>_L = R_eff(edges[i]), summing to rank(B1) over everything.
    """

    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.fiedler import deflated_operator
    from rexgraph.sparse_character import _block_cg

    B1 = to_scipy_csr(rex._B1_dual).tocsc()
    nE = B1.shape[1]
    idx = np.arange(nE) if edges is None else np.asarray(edges, dtype=int).ravel()
    if idx.size == 0:
        return np.zeros((B1.shape[0], 0)), np.zeros(0)
    Bc = np.ascontiguousarray(np.asarray(B1[:, idx].todense(), dtype=np.float64))
    # L0 is never formed: _block_cg takes the operator as a callable, and B1 is 22x
    # smaller than B1 B1^T with a matvec 6.8x faster (see fiedler.deflated_operator).
    apply_A, dinv, _U, _nc = deflated_operator(B1)
    V = _block_cg(apply_A, Bc, dinv, tol=1e-12, maxit=500)
    return V, np.einsum("ve,ve->e", Bc, V)


def significance(rex, edges=None):
    """The quadrances alone, which is `R_eff`. Kept for callers that want the scalar
    and want it named for what it is."""
    return relation_field(rex, edges)[1]


def semantic_gram(rex, edges=None):
    """The Gram block of the field under the L-inner product, `G[i,j] = b_i^T L0^+ b_j`.

    Its diagonal is the significance and its off-diagonal is how much two relations
    move the complex together. PSD by construction, so it is a kernel over relations.
    """

    from rexgraph.core._sparse import to_scipy_csr
    B1 = to_scipy_csr(rex._B1_dual).tocsc()
    idx = (np.arange(B1.shape[1]) if edges is None
           else np.asarray(edges, dtype=int).ravel())
    V, _q = relation_field(rex, idx)
    Bc = np.asarray(B1[:, idx].todense(), dtype=np.float64)
    G = Bc.T @ V
    return 0.5 * (G + G.T)                      # symmetrise the solve's rounding


def semantic_spread(rex, edges=None):
    """The pairwise semantic distance: the spread of the field, `1 - G^2/(Q_i Q_j)`.

    Zero when two relations move the complex in the same direction, one when they are
    L-orthogonal. This is section 1's spread with the field as its vectors, so it is
    the Gram block over its own diagonal and inherits everything that says: no square
    root, rational whenever the entries are, and defined at any arity.
    """
    G = semantic_gram(rex, edges)
    q = np.diag(G).copy()
    safe = np.where(np.abs(q) > 1e-300, q, 1.0)
    S = 1.0 - (G ** 2) / np.outer(safe, safe)
    S[np.abs(q) <= 1e-300, :] = 0.0
    S[:, np.abs(q) <= 1e-300] = 0.0
    np.fill_diagonal(S, 0.0)
    return np.clip(0.5 * (S + S.T), 0.0, 1.0)
