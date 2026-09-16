"""Exact down and up correspondence defects in identity coordinate metrics."""
from fractions import Fraction

import numpy as np

from rexgraph.chain_map import ChainMap, GradedMap, _chain_residual
from rexgraph.cochain import Chain
from rexgraph.graded_metric import _diagonal_contraction, _fraction
from rexgraph.type_accession import _action


def validate_correspondence(value):
    value = value.declaration if isinstance(value, ChainMap) else value
    if not isinstance(value, GradedMap):
        raise TypeError("temporal correspondence requires an explicit GradedMap")
    value.check_state()
    if _chain_residual(value.domain) or _chain_residual(value.codomain):
        raise ValueError("correspondence endpoints must satisfy the exact chain law")
    return value


def _defect_fields(field, correspondence):
    """Apply B'J-JB and (B') transpose J-J B transpose, over Q.

    Endpoint metrics are explicitly identity. A supplied map need not preserve
    either square, and no equality of endpoint cell counts is assumed. The
    returned arrays live on named target coordinates, not the source basis.
    """
    mapping = validate_correspondence(correspondence)
    if not isinstance(field, Chain) or field.source is not mapping.domain.source or field.source is None:
        raise TypeError("field delta requires a Chain bound to the correspondence domain Rex")
    if field.cell_keys is not None:
        raise ValueError("field delta requires canonical source cell order")
    k = field.grade
    left, right = mapping.domain, mapping.codomain
    values = field.numpy()
    if not 0 <= k < len(left.sizes) or values.ndim not in (1, 2) or values.shape[0] != left.sizes[k]:
        raise ValueError("field delta requires a carried grade and matching vector or block")
    # Validate even components annihilated by a zero correspondence.
    values = np.asarray([_fraction(v) for v in values.flat], dtype=object).reshape(values.shape)

    def action(entries, rows, cols, vector, transpose=False):
        if transpose:
            entries = tuple((j, i, v) for i, j, v in entries)
            rows, cols = cols, rows
        return _action(entries, (rows, cols), vector, exact=True)

    mapped = action(mapping.components[k], right.sizes[k], left.sizes[k], values)
    down = np.full((0, *values.shape[1:]), Fraction(0), dtype=object)
    up = down.copy()
    if k:
        before = action(left.boundaries[k-1], left.sizes[k-1], left.sizes[k], values)
        down = (action(right.boundaries[k-1], right.sizes[k-1], right.sizes[k], mapped)
                - action(mapping.components[k-1], right.sizes[k-1], left.sizes[k-1], before))
    if k + 1 < len(left.sizes):
        before = action(left.boundaries[k], left.sizes[k], left.sizes[k+1], values, True)
        up = (action(right.boundaries[k], right.sizes[k], right.sizes[k+1], mapped, True)
              - action(mapping.components[k+1], right.sizes[k+1], left.sizes[k+1], before))
    return mapping, k, down, up


def _quadrances(down, up):
    return tuple(_diagonal_contraction(v, v, (Fraction(1),)*len(v), exact=True) for v in (down, up))


def field_delta(field, correspondence):
    """Return exact defects with their target coordinate names and quadrances."""
    mapping, k, down, up = _defect_fields(field, correspondence)
    left, right = mapping.domain, mapping.codomain

    def describe(vector, grade):
        space = right.spaces[grade] if 0 <= grade < len(right.spaces) else None
        return {"grade": grade, "name": None if space is None else space.name,
                "keys": () if space is None else space.keys, "shape": vector.shape,
                "values": tuple(vector.tolist()), "implicit_zero": 0}

    qdown, qup = _quadrances(down, up)
    return {"down": describe(down, k-1), "up": describe(up, k+1),
            "down_quadrance": qdown, "up_quadrance": qup,
            "moment": qdown + qup, "oriented_moment": qdown - qup,
            "grade": k, "metrics": "identity", "coefficient_domain": "Q",
            "correspondence_digest": mapping.coefficient_digest,
            "source_boundary_digest": left.coefficient_digest,
            "target_boundary_digest": right.coefficient_digest}


def field_delta_moment(field, correspondence, *, oriented=False):
    """Contract the defect arrays without constructing diagnostic coordinate records."""
    if not isinstance(oriented, bool):
        raise TypeError("oriented must be a boolean")
    _, _, down, up = _defect_fields(field, correspondence)
    qdown, qup = _quadrances(down, up)
    return qdown - qup if oriented else qdown + qup
