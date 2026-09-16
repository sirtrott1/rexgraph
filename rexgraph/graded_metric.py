"""Positive diagonal forms on declared cell bases, without square root coordinates.

These metrics are explicit inputs to contractions. They do not reinterpret the
signed channel weights, replace a boundary metric, or define a type accession.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from numbers import Integral, Real

import numpy as np

from rexgraph.cells import cell_count
from rexgraph.cochain import Chain, Cochain, Field


def _fraction(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("exact coefficients must be integers or Fractions, not booleans")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    raise TypeError("exact coefficients must be integers or Fractions")


def positive_diagonal(values):
    """Validate source coefficients without casting exact magnitudes to float."""
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError("metric requires one diagonal weight per cell")
    result = []
    for value in array:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Fraction)):
            raise TypeError("metric weights must be real scalars, not complex or boolean")
        if isinstance(value, (Integral, Fraction)):
            value = _fraction(value)
        else:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("metric weights must be finite")
        if value <= 0:
            raise ValueError("metric weights must be strictly positive; semidefinite forms need a separate contract")
        result.append(value)
    return tuple(result)


def _diagonal_contraction(av, bv, weights, *, exact=False):
    """Coefficient kernel shared by cell metrics and explicit type cross pairings.

    The caller establishes space/variance and the meaning of the diagonal.
    """
    av, bv = np.asarray(av), np.asarray(bv)
    if av.ndim not in (1, 2) or av.shape != bv.shape or av.shape[0] != len(weights):
        raise ValueError("metric contraction requires matching vector/block shapes")
    if exact:
        if not all(isinstance(w, Fraction) for w in weights):
            raise TypeError("exact contraction requires an integer/rational metric")
        # Validate even coefficients annihilated by a zero sparse action.
        aa = tuple(_fraction(v) for v in av.flat)
        bb = tuple(_fraction(v) for v in bv.flat)
        columns = av.shape[1] if av.ndim == 2 else 1
        return sum((w * aa[i*columns+j] * bb[i*columns+j]
                    for i, w in enumerate(weights) for j in range(columns)), Fraction(0))
    dtype = complex if np.iscomplexobj(av) or np.iscomplexobj(bv) else float
    try:
        aa, bb, diagonal = np.asarray(av, dtype=dtype), np.asarray(bv, dtype=dtype), np.asarray(weights, dtype=float)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError("metric contraction inputs are not representable numerically") from exc
    if not all(np.all(np.isfinite(x)) for x in (aa, bb, diagonal)) or np.any(diagonal == 0):
        raise FloatingPointError("metric contraction inputs are not representable numerically")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.vdot(aa, (diagonal[:, None] if aa.ndim == 2 else diagonal) * bb).item()
    if not np.isfinite(result):
        raise FloatingPointError("metric contraction is outside numerical range")
    return result


@dataclass(frozen=True)
class DiagonalMetric:
    """Immutable coefficients beside source, grade and ordered basis identity."""

    source: object
    grade: int
    weights: tuple
    cell_keys: tuple | None = None

    def __post_init__(self):
        if isinstance(self.grade, (bool, np.bool_)) or not isinstance(self.grade, Integral):
            raise TypeError("metric grade must be an integer")
        grade = int(self.grade)
        values = positive_diagonal(self.weights)
        if len(values) != cell_count(self.source, grade, allow_empty_upper=True):
            raise ValueError("metric diagonal must match its source grade population")
        keys = None if self.cell_keys is None else tuple(self.cell_keys)
        if keys is not None and len(keys) != len(values):
            raise ValueError("metric basis must match its diagonal")
        object.__setattr__(self, "grade", grade)
        object.__setattr__(self, "weights", values)
        object.__setattr__(self, "cell_keys", keys)

    @property
    def exact(self):
        return all(isinstance(w, Fraction) for w in self.weights)

    @property
    def coefficient_digest(self):
        # Basis/source are separate descriptor fields. Values are not printed in plans.
        parts = [f"q:{w.numerator}/{w.denominator}" if isinstance(w, Fraction)
                 else "f:" + w.hex() for w in self.weights]
        return sha256(("diagonal-v1|" + "|".join(parts)).encode()).hexdigest()

    def _carrier(self, value):
        value = value.cochain if isinstance(value, Field) else value
        if not isinstance(value, (Chain, Cochain)):
            raise TypeError("metric contraction requires a Chain or Cochain")
        if value.source is not self.source or value.grade != self.grade:
            raise ValueError("metric contraction requires the same source and grade")
        keys = None if value.cell_keys is None else tuple(value.cell_keys)
        if keys != self.cell_keys:
            raise ValueError("metric contraction requires the same ordered basis")
        array = np.asarray(value.values)
        if array.ndim not in (1, 2) or array.shape[0] != len(self.weights):
            raise ValueError("metric contraction requires a matching vector or block cell axis")
        if cell_count(self.source, self.grade, allow_empty_upper=True) != len(self.weights):
            raise ValueError("metric source population changed; bind a fresh metric")
        return value, array

    def moment(self, left, right, *, exact=False):
        """sum_columns left* M right; a signed/Hermitian Frobenius contraction."""
        a, av = self._carrier(left)
        b, bv = self._carrier(right)
        if type(a) is not type(b) or av.shape != bv.shape:
            raise ValueError("metric contraction requires matching variance and shapes")
        return _diagonal_contraction(av, bv, self.weights, exact=exact)


def diagonal_metric(rex, grade, weights=None):
    """Identity if omitted; otherwise an explicitly supplied positive Cochain.

    No channel or relation weight is silently chosen as a contraction metric.
    """
    if weights is None:
        return DiagonalMetric(rex, grade, (Fraction(1),) * cell_count(rex, grade, allow_empty_upper=True))
    if not isinstance(weights, Cochain) or weights.source is not rex or weights.grade != grade:
        raise TypeError("metric weights require a Cochain at the same source and grade")
    return DiagonalMetric(rex, grade, weights.values, weights.cell_keys)


def integrate(cochain, chain, *, exact=False):
    """Canonical dual pairing sum_i omega_i c_i, summed over block columns.

    This is bilinear, including over complex coefficients: a cochain is already
    the dual functional. It is not the Hermitian Riesz/metric pairing ``moment``.
    With d = B^T it satisfies (d omega)(c) = omega(B c). No metric, quadrature,
    Gram matrix or eigensystem is constructed. Exact mode accepts Q only.
    """
    if not isinstance(exact, (bool, np.bool_)):
        raise TypeError("exact must be a boolean")
    if not isinstance(cochain, Cochain) or not isinstance(chain, Chain):
        raise TypeError("integration requires a Cochain followed by a Chain")
    if cochain.source is None or chain.source is not cochain.source or cochain.grade != chain.grade:
        raise ValueError("integration requires the same bound source and grade")
    # Reuse the existing source, basis, population and vector/block validation.
    identity = DiagonalMetric(cochain.source, cochain.grade,
                              (Fraction(1),) * cochain.n_cells, cochain.cell_keys)
    _, av = identity._carrier(cochain)
    _, bv = identity._carrier(chain)
    if av.shape != bv.shape:
        raise ValueError("integration requires matching vector/block shapes")
    if not exact:
        for values in (av, bv):
            if values.dtype.kind not in "iufcO" or (values.dtype.kind == "O" and any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, complex, np.complexfloating, Fraction))
                for v in values.flat
            )):
                raise TypeError("integration requires numeric coefficients, not booleans or strings")
        # The shared kernel conjugates its first operand. Undo that only for
        # this dual evaluation; MOMENT keeps its Hermitian semantics unchanged.
        if np.iscomplexobj(av):
            av = av.conjugate()
        elif av.dtype.kind == "O" and any(isinstance(v, (complex, np.complexfloating)) for v in av.flat):
            av = np.asarray(av, dtype=complex).conjugate()
        if bv.dtype.kind == "O" and any(isinstance(v, (complex, np.complexfloating)) for v in bv.flat):
            bv = np.asarray(bv, dtype=complex)
    return _diagonal_contraction(av, bv, identity.weights, exact=exact)
