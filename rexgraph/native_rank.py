"""Exact boundary rank/homology from native incidence, without SciPy or spectra.

No new rank algorithm: adapters feed graded_boundary's guarded structural
shortcuts and memoized sparsest first integer elimination. Rank clears C1's
declared denominators; chain composition uses the ORIGINAL rational columns.
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm
from numbers import Integral

import numpy as np

from rexgraph.graded_boundary import (
    _exact_composition_residual,
    _integer_columns,
    _rank_integer_columns,
)
from rexgraph.native_sparse import boundary_carriers


def primary_columns(rex, *, integer=False):
    """Read primary slots with their original arity, including repeated slots.

    Integer form scales a non witness column by k-1 BEFORE coalescing. Rational
    form uses that same integer column divided by k-1, never float recovery.
    Relation metrics and declared signs do not change this boundary map.
    """
    rex._ensure_clean()
    columns = []
    ptr, indices = rex._boundary_ptr, rex._boundary_idx
    if ptr is None:
        supports = zip(*rex._ensure_src_tgt(), strict=True)
    else:
        supports = (indices[int(ptr[j]):int(ptr[j + 1])] for j in range(int(rex.nE)))
    for support in supports:
        k = len(support)
        scale = max(1, k - 1)
        col = {}
        for slot, vertex in enumerate(support):
            vertex = int(vertex)
            col[vertex] = col.get(vertex, 0) + (-scale if k > 1 and slot == 0 else 1)
        columns.append({v: c if integer else Fraction(c, scale) for v, c in col.items() if c})
    return columns


def clear_column_denominators(columns):
    """An exact rank representative, NOT a replacement boundary for composition."""
    result = []
    for col in columns:
        scale = 1
        for value in col.values():
            scale = lcm(scale, value.denominator)
        result.append({row: int(value * scale) for row, value in col.items() if value})
    return result


def boundary_columns(rex, grade, *, integer=False, carriers=None):
    """Return shape and exact columns of one carried boundary; reject non-Z upper maps."""
    if isinstance(grade, (bool, np.bool_)) or not isinstance(grade, Integral):
        raise TypeError("boundary grade must be an integer")
    if grade == 1 and carriers is None:
        columns = primary_columns(rex, integer=integer)
        return (int(rex.nV), int(rex.nE)), columns
    maps = boundary_carriers(rex) if carriers is None else carriers
    if not 1 <= grade <= len(maps):
        raise ValueError(f"boundary grade {grade} is not present")
    matrix = maps[grade - 1]
    columns = primary_columns(rex, integer=integer) if grade == 1 else _integer_columns(matrix)
    if columns is None:
        raise ValueError("exact boundary rank requires integer higher boundary coefficients")
    if len(columns) != matrix.shape[1]:
        raise ValueError("primary boundary slots differ from the carried cell axis")
    return matrix.shape, columns


def boundary_rank(rex, grade, *, return_info=False):
    shape, columns = boundary_columns(rex, grade, integer=True)
    result = _rank_integer_columns(shape, columns)
    return result if return_info else result[0]


def exact_tower(rex):
    """Original rational/integer tower, checking grade axes but not reducing ranks."""
    maps = boundary_carriers(rex)
    shapes, columns = [], []
    for grade in range(1, len(maps) + 1):
        shape, col = boundary_columns(rex, grade, carriers=maps)
        if shapes and shapes[-1][1] != shape[0]:
            raise ValueError("boundary tower has incompatible consecutive grade axes")
        shapes.append(shape)
        columns.append(col)
    return shapes, columns


def tower_chain_residual(shapes, columns):
    """Composition of ORIGINAL columns, not independently rescaled rank carriers."""
    return max((_exact_composition_residual(columns[k], columns[k + 1])
                for k in range(len(shapes) - 1)), default=Fraction(0))


def betti_from_rex(rex, *, return_info=False):
    """Full carried homology over Q, with exact chain law certification.

    Decorative nonbounding B2 columns are excluded by the existing Hodge face
    policy. Every remaining consecutive map must compose to exact zero. Real
    measured higher coefficients are refused, never classified by an SVD.
    """
    shapes, columns = exact_tower(rex)
    residual = tower_chain_residual(shapes, columns)
    if residual:
        raise ValueError(f"Betti requires the chain condition; exact residual = {residual}")
    sizes = [shapes[0][0]] + [shape[1] for shape in shapes]
    readings = [_rank_integer_columns(shape, clear_column_denominators(col))
                for shape, col in zip(shapes, columns, strict=True)]
    ranks = [0] + [rank for rank, _ in readings] + [0]
    betti = tuple(n - ranks[g] - ranks[g + 1] for g, n in enumerate(sizes))
    return (betti, tuple(method for _, method in readings)) if return_info else betti


def harmonic_shadow(rex):
    """C1 cycles killed by the Hodge eligible C2 columns, counted exactly.

    Uses boundary ranks, including witness columns, rather than the graph
    incidence identity rank(B1)=nV minus components. No basis is selected.
    """
    shapes, columns = exact_tower(rex)
    residual = tower_chain_residual(shapes, columns)
    if residual:
        raise ValueError(f"harmonic shadow requires the chain condition: {residual}")
    rank1 = _rank_integer_columns(shapes[0], clear_column_denominators(columns[0]))[0]
    rank2 = _rank_integer_columns(shapes[1], columns[1])[0] if len(shapes) > 1 else 0
    cycles = shapes[0][1] - rank1
    return {"shadow_dim": rank2, "beta_1_at_d1": cycles, "beta_1_at_d2": cycles - rank2}
