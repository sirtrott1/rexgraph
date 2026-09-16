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

    from rexgraph.green import vertex_green
    Bc = _relation_sources(rex, edges)
    V = vertex_green(rex).solve(Bc)
    return V, np.einsum("ve,ve->e", Bc, V)


def _relation_sources(rex, edges):
    """Read only the requested boundary columns into the output source block."""
    from numbers import Integral
    from rexgraph.native_sparse import NativeSparse
    rex._ensure_clean()
    boundary = NativeSparse(rex._B1_dual)
    idx = list(range(boundary.shape[1])) if edges is None else list(edges)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in idx):
        raise TypeError("relation indices must be integers")
    if any(i < 0 or i >= boundary.shape[1] for i in idx):
        raise ValueError("relation index is outside the boundary")
    out = np.zeros((boundary.shape[0], len(idx)))
    for j, edge in enumerate(idx):
        lo, hi = boundary.dual.col_ptr[edge:edge + 2]
        np.add.at(out[:, j], boundary.dual.row_idx[lo:hi], boundary.dual.vals_csc[lo:hi])
    return out


def significance(rex, edges=None):
    """The quadrances alone, which is `R_eff`. Kept for callers that want the scalar
    and want it named for what it is."""
    return relation_field(rex, edges)[1]


def semantic_gram(rex, edges=None):
    """The Gram block of the field under the L-inner product, `G[i,j] = b_i^T L0^+ b_j`.

    Its diagonal is the significance and its off diagonal is how much two relations
    move the complex together. PSD by construction, so it is a kernel over relations.
    """

    from rexgraph.green import vertex_green
    Bc = _relation_sources(rex, edges)
    V = vertex_green(rex).solve(Bc)
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
