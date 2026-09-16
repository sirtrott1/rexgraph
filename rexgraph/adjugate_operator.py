"""Exact adjugate actions from streamed power traces and a scalar polynomial.

No inverse, eigenbasis, dense matrix or numerical determinant is used. This is
a global algebraic reading, not a linear cost local query: an n cell action
requires n(n-1) exact primal actions to obtain its coefficients. Coefficients
are recomputed per application so they cannot outlive their primal action.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from rexgraph.cells import cell_count
from rexgraph.linear_operator import RexOperator, _exact_array, _numeric_array


class AdjugateOperator(RexOperator):
    """Polynomial adj(A), including singular A; inherits its grade and variance."""

    def __init__(self, primal):
        if (not isinstance(primal, RexOperator) or primal.source is None or primal.exact_matvec is None
                or primal.domain_grade != primal.codomain_grade or primal.shape[0] != primal.shape[1]):
            raise TypeError("adjugate requires a bound square operator with a certified Q action")
        n = primal.shape[0]

        def check():
            if cell_count(primal.source, primal.domain_grade, allow_empty_upper=True) != n:
                raise ValueError("adjugate source population changed; bind a fresh operator")

        def coefficients():
            check()
            # p(t) = t^n + c1 t^(n-1) + ... + cn. Only c1..c(n-1)
            # enter adj(A), so neither inversion nor det(A) is needed.
            traces = [Fraction(0)] * n
            for i in range(n):
                vector = np.full(n, Fraction(0), dtype=object)
                vector[i] = Fraction(1)
                for power in range(1, n):
                    vector = _exact_array(primal.apply(vector, exact=True))
                    if vector.shape != (n,):
                        raise ValueError("adjugate primal must preserve its declared cell axis")
                    traces[power] += vector[i]
            result = [Fraction(1)]
            for k in range(1, n):
                result.append(-sum((result[k-j] * traces[j] for j in range(1, k+1)), Fraction(0)) / k)
            return tuple(result)

        def action(values, exact=False, transpose=False):
            check()
            values = _exact_array(values) if exact else _numeric_array(values, operation="adjugate")
            if not n:
                return values.copy()
            coeff = coefficients()
            if not exact:
                coeff = _numeric_array(np.array(coeff, dtype=object), operation="adjugate coefficients")
            apply = primal.transpose_apply if transpose else primal.apply
            out = values.copy()
            for c in coeff[1:]:
                out = apply(out, exact=exact) + c * values
                if out.shape != values.shape:
                    raise ValueError("adjugate primal changed the vector or block shape")
                out = _exact_array(out) if exact else _numeric_array(out, operation="adjugate result")
            return (-1 if n % 2 == 0 else 1) * out

        check()
        super().__init__("ADJUGATE", primal.shape, primal.domain_grade, primal.codomain_grade,
                         action, source=primal.source, variance=primal.variance, symmetric=primal.symmetric,
                         construction="exact-adjugate", exact_matvec=lambda x: action(x, True),
                         transpose_matvec=(lambda x: action(x, transpose=True)) if primal.has_transpose else None,
                         exact_transpose_matvec=(lambda x: action(x, True, True))
                             if primal.exact_transpose_matvec is not None else None,
                         parameters=(("coefficient_method", "exact-streamed-power-traces"),
                                     ("coefficient_actions", n * max(0, n-1))))
        object.__setattr__(self, "primal", primal)
        object.__setattr__(self, "coefficients", coefficients)
