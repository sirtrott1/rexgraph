"""The grade 1 column: existence, orientation, and share, in one place.

A primary relation's column is the one head constructor ``b = s - h``: the
distinguished participant carries ``-1`` and the tails carry positive rational shares
summing to one. The EQUAL share ``s_i = 1/(k-1)`` is the specialisation every relation
takes unless it declares otherwise, and the arity one witness is the separate
constructor ``b = e_i`` whose column sums to one rather than zero.

Two components of that column had nowhere to live. The head was a POSITION, the first
slot of the stored support, and the share was recomputed from the arity at each of the
twenty or so places that needed it. So a relation could not say "my head is this
participant" or "these tails are not equal". Neither is an extension of the model: the
equal share is the specialisation, and it was the only one that could be written down.

This module is the one reader. `slot_coefficients` answers the float column for every
incidence in CSR order and `exact_slot_coefficients` the rational one, both honouring a
declared head and a declared share vector and both returning the canonical column when
nothing is declared. A caller that needs the column asks here rather than writing
``-1`` at slot zero and ``1/(k-1)`` on the rest, which is the shape that let a
declaration be honoured in one place and silently dropped in another.

Declarations are canonicalized at arity two: a pairwise relation declaring its second
slot as the head is stored with its support swapped, which is the same column, so every
pairwise path keeps reading exactly what it read before. A complex whose relations are
all pairwise therefore never carries a declaration.
"""
from __future__ import annotations

from fractions import Fraction
from numbers import Integral

import numpy as np

__all__ = [
    "ColumnDeclaration",
    "canonical_column",
    "declaration_of",
    "require_canonical",
    "declaration_from_entries",
    "exact_slot_coefficients",
    "slot_coefficients",
    "validate_declaration",
]

_i32 = np.int32
_i64 = np.int64
_f64 = np.float64


class ColumnDeclaration:
    """The declared head slot and share vector of every relation, or None per relation.

    ``head_slot`` is one index per relation into its own support, and ``share_num`` and
    ``share_den`` are one exact rational per INCIDENCE, in the same CSR order as
    ``boundary_idx``, carrying ``s`` (the head's entry is zero, not its ``-1``). A
    relation with no declared share has ``share_den`` zero across its slots, which is
    how "declared nothing" is told apart from a declared zero.
    """

    __slots__ = ("head_slot", "share_num", "share_den")

    def __init__(self, head_slot, share_num, share_den):
        self.head_slot = head_slot
        self.share_num = share_num
        self.share_den = share_den

    def __bool__(self):
        return bool(np.any(self.head_slot) or np.any(self.share_den))


def declaration_of(source):
    """The declaration a complex carries, or None.

    Read through an attribute lookup rather than the slot, because the readers are also
    handed carriers that are not a RexGraph: a partition view, a test double, a snapshot
    rebuilt from arrays. Absent means canonical, which is what those all are.
    """
    return getattr(source, "_declaration", None)


def require_canonical(operation: str, *sources) -> None:
    """Decline `operation` where any source declares a head or a share.

    The carriers that ask for this identify a relation by its support and derive the
    share from the arity, so a declared column would be read as the canonical one. Asked
    through the attribute rather than the type, because these functions also take label
    lists and other non complexes, which declare nothing.
    """
    for source in sources:
        require = getattr(source, "_require_canonical_columns", None)
        if require is not None:
            require(operation)


def canonical_column(arity: int) -> list[Fraction]:
    """The column every relation takes unless it declares otherwise."""
    if arity <= 0:
        return []
    if arity == 1:
        return [Fraction(1)]                       # the witness, sum one
    share = Fraction(1, arity - 1)
    return [Fraction(-1)] + [share] * (arity - 1)


def _rational(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, tuple) and len(value) == 2:
        return Fraction(int(value[0]), int(value[1]))
    if isinstance(value, (np.integer,)):
        return Fraction(int(value))
    raise TypeError("a declared share must be an integer, a Fraction, or (numerator, "
                    f"denominator); got {type(value).__name__}")


def validate_declaration(boundary_ptr, boundary_idx, head_slot=None, shares=None):
    """Check a declaration against its complex and return it, or None when canonical.

    Each refusal names the constraint it failed: a supported head, ``s_h = 0``,
    ``s_i > 0`` on the tails, and ``1^T s = 1``. An
    arity one witness declares neither, because its column is the separate constructor.
    """
    ptr = np.asarray(boundary_ptr, dtype=_i64)
    idx = np.asarray(boundary_idx, dtype=_i64)
    nE = int(ptr.shape[0]) - 1 if ptr.shape[0] else 0
    nnz = int(idx.shape[0])
    if head_slot is None and shares is None:
        return None, None

    heads = (np.zeros(nE, dtype=_i32) if head_slot is None
             else np.asarray(head_slot, dtype=_i32).copy())
    if heads.shape != (nE,):
        raise ValueError(f"a declared head needs one slot per relation ({nE})")
    num = np.zeros(nnz, dtype=_i64)
    den = np.zeros(nnz, dtype=_i64)
    if shares is not None:
        values = list(shares)
        if len(values) != nnz:
            raise ValueError(f"a declared share needs one entry per incidence ({nnz})")
        for position, value in enumerate(values):
            if value is None:
                continue
            q = _rational(value)
            num[position], den[position] = q.numerator, q.denominator

    for e in range(nE):
        lo, hi = int(ptr[e]), int(ptr[e + 1])
        arity = hi - lo
        head = int(heads[e])
        declared = bool(np.any(den[lo:hi]))
        if arity <= 0:
            continue
        if not 0 <= head < arity:
            raise ValueError(f"relation {e} declares head slot {head}, which is not one "
                             f"of its {arity} participants (a supported head is essential)")
        if arity == 1:
            if head or declared:
                raise ValueError(f"relation {e} is an arity-one witness: its column is the "
                                 "separate constructor b = e_i and declares no head or share")
            continue
        if arity == 2 and idx[lo] == idx[lo + 1]:
            if head or declared:
                raise ValueError(f"relation {e} is a self loop, whose column is zero; it "
                                 "carries no declared head or share")
            continue
        if declared:
            total = Fraction(0)
            for slot in range(arity):
                position = lo + slot
                if den[position] <= 0:
                    raise ValueError(f"relation {e} declares a share for some slots and not "
                                     "for others; declare the whole vector, including the "
                                     "head's zero")
                q = Fraction(int(num[position]), int(den[position]))
                if slot == head:
                    if q != 0:
                        raise ValueError(f"relation {e} declares share {q} at its head: "
                                         "s_h = 0, because the head carries the -1")
                elif q <= 0:
                    raise ValueError(f"relation {e} declares share {q} at slot {slot}: "
                                     "s_i > 0 on every tail")
                else:
                    total += q
            if total != 1:
                raise ValueError(f"relation {e} declares tail shares summing to {total}: "
                                 "1^T s = 1")

    # Arity two is the ordinary edge: a declared head there is the same column with its
    # support swapped, and the only admissible share vector is the forced (0, 1). Storing
    # it as the swapped support keeps every pairwise reader exactly as it was, and means a
    # complex whose relations are all pairwise never carries a declaration at all.
    support = idx.copy()
    for e in range(nE):
        lo, hi = int(ptr[e]), int(ptr[e + 1])
        arity = hi - lo
        if arity == 2:
            if heads[e] == 1:
                support[lo], support[lo + 1] = idx[lo + 1], idx[lo]
            heads[e] = 0
            num[lo:hi] = 0
            den[lo:hi] = 0
        elif arity > 2 and heads[e] and not np.any(den[lo:hi]):
            # A head declared over EQUAL shares is the same column with that participant
            # first, because the tails are interchangeable there. Storing it that way is
            # what `from_cells` has always done, and it keeps a complex that declares
            # nothing unequal out of the declared path entirely.
            head = int(heads[e])
            support[lo:hi] = np.concatenate((idx[lo + head:lo + head + 1],
                                             np.delete(idx[lo:hi], head)))
            heads[e] = 0
    declaration = ColumnDeclaration(heads, num, den)
    return (declaration if declaration else None), support


def _arities(ptr):
    return np.diff(np.asarray(ptr, dtype=_i64))


def slot_coefficients(boundary_ptr, boundary_idx, declaration=None) -> np.ndarray:
    """The float column entry of every incidence, in CSR order.

    Vectorised over the canonical case, which is every relation in a complex that
    declares nothing, so the common path costs one pass and no Python loop.
    """
    ptr = np.asarray(boundary_ptr, dtype=_i64)
    idx = np.asarray(boundary_idx, dtype=_i64)
    nE = int(ptr.shape[0]) - 1 if ptr.shape[0] else 0
    if nE <= 0 or idx.size == 0:
        return np.zeros(0, dtype=_f64)
    k = _arities(ptr)
    owner = np.repeat(np.arange(nE, dtype=_i64), k)
    is_head = np.zeros(idx.size, dtype=bool)
    is_head[ptr[:-1][k > 0]] = True
    share = np.zeros(nE, dtype=_f64)
    wide = k >= 2
    share[wide] = 1.0 / (k[wide] - 1)               # 1 at k == 2: the plain edge
    values = np.where(is_head, -1.0, share[owner])
    one = k == 1
    if one.any():                                   # the witness: a single +1, no head
        values[np.repeat(one, k)] = 1.0
    if declaration is None:
        return values
    # A declared head moves the -1 off slot zero; the tails keep the equal share unless a
    # share vector is declared too, and then the head's own -1 replaces its declared zero.
    heads, num, den = declaration.head_slot, declaration.share_num, declaration.share_den
    declared_share = np.flatnonzero(np.asarray(den) > 0)
    if declared_share.size:
        values[declared_share] = (num[declared_share].astype(_f64)
                                  / den[declared_share].astype(_f64))
    for e in np.flatnonzero(np.asarray(heads) != 0):
        lo, hi = int(ptr[e]), int(ptr[e + 1])
        head = lo + int(heads[e])
        if not np.any(den[lo:hi]):
            values[lo:hi] = 1.0 / (hi - lo - 1)
        values[head] = -1.0
    for e in np.unique(owner[declared_share]):
        values[int(ptr[e]) + int(heads[e])] = -1.0
    return values


def exact_slot_coefficients(boundary_ptr, boundary_idx, declaration=None) -> list:
    """The exact column entry of every incidence, in CSR order, as Fractions."""
    ptr = np.asarray(boundary_ptr, dtype=_i64)
    nE = int(ptr.shape[0]) - 1 if ptr.shape[0] else 0
    out: list[Fraction] = []
    for e in range(nE):
        lo, hi = int(ptr[e]), int(ptr[e + 1])
        arity = hi - lo
        head = 0 if declaration is None else int(declaration.head_slot[e])
        declared = (declaration is not None and arity > 1
                    and bool(np.any(declaration.share_den[lo:hi])))
        if declared:
            column = [Fraction(int(declaration.share_num[lo + slot]),
                               int(declaration.share_den[lo + slot]))
                      for slot in range(arity)]
            column[head] = Fraction(-1)
        else:
            column = canonical_column(arity)
            if head and arity > 1:
                column[head], column[0] = column[0], column[head]
        out.extend(column)
    return out


def declaration_from_entries(indices, coefficients):
    """Read one declared column into (head slot, share vector), or None when canonical.

    ``coefficients`` is the column itself, ``b = s - h``: exactly one entry equal to -1
    and positive rational tails summing to one. Returns ``(head, shares)`` with shares in
    slot order and the head's entry zero.
    """
    values = [_rational(c) for c in coefficients]
    negative = [i for i, c in enumerate(values) if c < 0]
    if len(values) == 1:
        if values[0] != 1:
            raise ValueError("an arity-one witness column is b = e_i, coefficient 1")
        return None
    if len(negative) != 1 or values[negative[0]] != -1:
        raise ValueError("a grade-1 column is b = s - h: one negative distinguished "
                         "participant carrying -1, and positive shares on the rest")
    head = negative[0]
    tails = [c for i, c in enumerate(values) if i != head]
    if any(c <= 0 for c in tails):
        raise ValueError("s_i > 0 on every tail of a grade-1 column")
    # Unit tails are ORIENTATION SIGNS, the positional form every plain support declares:
    # they choose the head and the column is then materialised at the equal share. Tails
    # summing to one are the SHARE VECTOR itself. The two readings agree at arity two,
    # where the only admissible share is 1.
    if all(c == 1 for c in tails):
        return None if head == 0 else (head, None)
    if sum(tails) != 1:
        raise ValueError(f"the tail shares of a grade-1 column sum to {sum(tails)}: they "
                         "are either unit orientation signs or a share vector summing to 1")
    if all(c == Fraction(1, len(values) - 1) for c in tails):
        return None if head == 0 else (head, None)  # the equal share, written out
    shares = [Fraction(0) if i == head else c for i, c in enumerate(values)]
    return head, shares
