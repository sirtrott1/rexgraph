"""Exact validation of declared higher cell boundaries, without attachment."""
from fractions import Fraction
from numbers import Integral

import numpy as np

from rexgraph.chain_map import _columns, _exact_entries
from rexgraph.graded_boundary import _exact_compose_columns
from rexgraph.io.partition_state import partition_tower

__all__ = ["validate_relations"]


def _proposals(rex, candidates, grade):
    """Check axes and exact declarations without composing candidate boundaries."""
    if isinstance(grade, (bool, np.bool_)) or not isinstance(grade, Integral):
        raise TypeError("candidate grade must be an integer")
    boundaries, tower = partition_tower(rex)
    if not 2 <= grade <= len(tower) + 1:
        raise ValueError("candidate grade must be 2 through one above the carried tower")
    if not isinstance(candidates, (list, tuple)):
        raise TypeError("candidates must be a sequence of sparse boundary columns")
    triples = []
    for j, column in enumerate(candidates):
        if not isinstance(column, (list, tuple)):
            raise TypeError("each candidate must be a sequence of (index, coefficient) pairs")
        for entry in column:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise TypeError("candidate entries must be (index, coefficient) pairs")
            triples.append((entry[0], j, entry[1]))
    entries = _exact_entries(triples, (boundaries[grade - 2].shape[1], len(candidates)))
    return tower[grade - 2], _columns(entries, len(candidates))


def _stored_integer(value):
    """Whether an exact coefficient survives the current upper boundary carrier."""
    if value.denominator != 1:
        return False
    try:
        return Fraction(float(value)) == value
    except (OverflowError, ValueError):
        return False


def validate_relations(rex, candidates, *, grade=2):
    """Read each declared column against the original rational boundary.

    Candidates are sparse sequences of (lower cell index, integer or Fraction)
    pairs on the bound source's canonical grade ``grade - 1`` axis. Repeated
    coordinates add before validation. Zero columns are not attaching cells.
    Grades start at two: primary relations use their separate incidence grammar.

    Closure, integrality and storage compatibility are distinct readings. A
    valid candidate is nonzero, closed and exactly storable as an integral upper
    boundary. This is not a novelty, independence, identity or model score test.
    No coefficients are rescaled and no source or store is changed.
    """
    lower, columns = _proposals(rex, candidates, grade)
    residuals = tuple(tuple(sorted(column.items()))
                      for column in _exact_compose_columns(lower, columns))
    nonzero = tuple(bool(column) for column in columns)
    closed = tuple(not residual for residual in residuals)
    integral = tuple(all(value.denominator == 1 for value in column.values()) for column in columns)
    storable = tuple(all(_stored_integer(value) for value in column.values()) for column in columns)
    valid = tuple(a and b and c for a, b, c in zip(nonzero, closed, storable, strict=True))
    return {"grade": int(grade), "valid": valid, "closed": closed, "nonzero": nonzero,
            "integral": integral, "storable": storable, "residuals": residuals,
            "accepted": tuple(i for i, accepted in enumerate(valid) if accepted)}
