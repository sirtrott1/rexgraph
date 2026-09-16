"""Factored rational Cayley transforms and certified partial complex actions.

These actions use the Euclidean form on one canonical grade. They do not
choose spectral planes or replace the separate SVD diagnostic in hodge_coords.
"""
from fractions import Fraction as Q
from math import isqrt

import numpy as np

from rexgraph.cells import cell_count
from rexgraph.graded_metric import _fraction
from rexgraph.green import GreenOperator
from rexgraph.io.catalog import object_digest
from rexgraph.linear_operator import RexOperator, _exact_array, _numeric_array
from rexgraph.operator_bracket import OperatorBracket


def skew_generator(generator):
    if (not isinstance(generator, (OperatorBracket, RationalOperator))
            or getattr(generator, "euclidean_skew_adjoint", None) is not True):
        raise TypeError("generator requires a native Euclidean skew adjoint certificate")
    if (generator.source is None or generator.domain_grade != generator.codomain_grade
            or generator.shape[0] != generator.shape[1]
            or generator.exact_matvec is None or generator.exact_transpose_matvec is None):
        raise TypeError("generator requires a bound square rational action and transpose")
    if generator.shape[0] != cell_count(generator.source, generator.domain_grade, allow_empty_upper=True):
        raise ValueError("generator population differs from its source")
    if isinstance(generator, RationalOperator):
        generator.check_state()
    return generator


def rotation_triple(a, b, c):
    a, b, c = map(_fraction, (a, b, c))
    if c <= 0 or a*a + b*b != c*c:
        raise ValueError("rotation requires a*a+b*b=c*c and c>0 over Q")
    return a/c, b/c


class RationalOperator(RexOperator):
    def __init__(self, generator, name, action, *, parameters=(), skew=False):
        digest = object_digest(generator.source)

        def checked(values, exact=False, transpose=False):
            self.check_state()
            x = _exact_array(values) if exact else _numeric_array(values, operation=name)
            result = action(x, exact=exact, transpose=transpose)
            self.check_state()
            if result.shape != x.shape:
                raise ValueError("rational action changed its vector or block axis")
            return result if exact else _numeric_array(result, operation=name)

        super().__init__(name, generator.shape, generator.domain_grade, generator.codomain_grade,
                         checked, source=generator.source, variance=generator.variance,
                         construction="rational-operator", exact_matvec=lambda x: checked(x, True),
                         transpose_matvec=lambda x: checked(x, transpose=True),
                         exact_transpose_matvec=lambda x: checked(x, True, True), parameters=(("kind", name), *parameters))
        object.__setattr__(self, "generator", generator)
        object.__setattr__(self, "state_digest", digest)
        object.__setattr__(self, "euclidean_skew_adjoint", True if skew else None)

    def check_state(self):
        if object_digest(self.source) != self.state_digest:
            raise ValueError("rational operator source state changed; bind a fresh action")


def _exact_positive_solve(action, values):
    """Finite Q conjugate directions for a certified positive definite action.

    No tolerance decides termination. Each nonzero right hand side needs at
    most n steps in exact arithmetic; exhaustion or a nonpositive pivot refuses.
    Only vectors are retained, not a matrix or a Krylov basis.
    """
    block = values[:, None] if values.ndim == 1 else values
    result = np.full(block.shape, Q(0), dtype=object)
    n = block.shape[0]
    for j in range(block.shape[1]):
        r = block[:, j].copy()
        p = r.copy()
        norm = sum((v*v for v in r), Q(0))
        for _ in range(n):
            if norm == 0:
                break
            ap = action(p)
            pivot = sum((a*b for a, b in zip(p, ap, strict=True)), Q(0))
            if pivot <= 0:
                raise ValueError("exact positive solve encountered a nonpositive pivot")
            step = norm/pivot
            result[:, j] += step*p
            r -= step*ap
            next_norm = sum((v*v for v in r), Q(0))
            p = r + (next_norm/norm)*p
            norm = next_norm
        if norm != 0 or any(action(result[:, j]) != block[:, j]):
            raise ArithmeticError("exact positive solve did not certify its residual")
    return result[:, 0] if values.ndim == 1 else result


def cayley(generator, parameter, *, tol=1e-10, maxiter=1000):
    """(I-tA)^-1(I+tA), using the positive action I-t^2 A^2.

    Q conjugate directions supply exact solves. Numerical solves reuse Core's
    existing factored resolvent and its convergence checks. This is not exp(tA).
    """
    generator = skew_generator(generator)
    t = _fraction(parameter)
    square = RexOperator("negative-skew-square", generator.shape, generator.domain_grade,
        generator.codomain_grade, lambda x: -generator.apply(generator.apply(x)),
        source=generator.source, variance=generator.variance, symmetric=True, psd=True)
    # Validation and the numerical solve policy belong to the existing Core API.
    GreenOperator.resolvent(square, 0, tol=tol, maxiter=maxiter)
    numerical = []

    def action(x, *, exact, transpose):
        if not t:
            return x.copy()
        coefficient = (-t if transpose else t) if exact else float(-t if transpose else t)
        a = lambda v: generator.apply(v, exact=exact)
        ax = a(x)
        rhs = x + 2*coefficient*ax + coefficient*coefficient*a(ax)
        if exact:
            return _exact_positive_solve(lambda v: v-t*t*a(a(v)), rhs)
        if not numerical:
            numerical.append(GreenOperator.resolvent(square, t*t, tol=tol, maxiter=maxiter))
        # The generator is real; complex coefficients use the same real solver.
        solution = (numerical[0].solve(rhs.real) + 1j*numerical[0].solve(rhs.imag)
                    if np.iscomplexobj(rhs) else numerical[0].solve(rhs))
        target = x + coefficient*ax
        error = solution - coefficient*a(solution) - target
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(error)):
            raise RuntimeError("Cayley equation returned a nonfinite residual")
        # The positive solve tolerance alone need not bound the original equation.
        targets = target[:, None] if x.ndim == 1 else target
        errors = error[:, None] if x.ndim == 1 else error
        for wanted, residual in zip(targets.T, errors.T, strict=True):
            if not len(wanted):
                continue
            scale = max(np.max(np.abs(wanted)), np.max(np.abs(residual)))
            if not np.isfinite(scale):
                raise RuntimeError("Cayley equation returned a nonfinite residual")
            if scale and np.linalg.norm(residual/scale) > tol*np.linalg.norm(wanted/scale):
                raise RuntimeError("Cayley solve did not meet the original equation tolerance")
        return solution

    return RationalOperator(generator, "CAYLEY", action,
                            parameters=(("parameter", t), ("tol", tol), ("maxiter", maxiter)))


def complex_structure(generator, scale=None):
    """J=A/s on a certified support, with J^3=-J and P=-J^2.

    A^3=-qA is checked by streaming basis vectors. An inferred s must be an
    exact rational square root of q. Distinct frequencies or a required
    irrational normalization are refused, not approximated or truncated.
    The zero generator has J=P=0 and conventional scale one.
    """
    generator = skew_generator(generator)
    given = None if scale is None else _fraction(scale)
    if given is not None and given <= 0:
        raise ValueError("complex structure scale must be positive")
    n = generator.shape[0]
    q = given*given if given is not None else None
    for i in range(n):
        e = np.full(n, Q(0), dtype=object)
        e[i] = Q(1)
        first = generator.apply(e, exact=True)
        third = generator.apply(generator.apply(first, exact=True), exact=True)
        if q is None:
            for a, b in zip(first, third, strict=True):
                if a:
                    q = -b/a
                    break
        if q is not None and (q <= 0 or any(third != -q*first)):
            raise ValueError("generator does not satisfy a single positive rational frequency square")
    q = Q(1) if q is None else Q(q)
    p, d = isqrt(q.numerator), isqrt(q.denominator)
    if p*p != q.numerator or d*d != q.denominator:
        raise ValueError("complex structure normalization leaves Q; supply a rational invariant generator")
    scale = Q(p, d)

    def action(x, *, exact, transpose):
        factor = (-1 if transpose else 1)/scale
        return generator.apply(x, exact=exact)*(factor if exact else float(factor))
    return RationalOperator(generator, "COMPLEX_STRUCTURE", action,
                            parameters=(("scale", scale), ("certificate", "J^3=-J; P=-J^2")), skew=True)


def rational_rotation(structure, a, b, c):
    """I+(a/c-1)P+(b/c)J, with identity on the kernel of the certified J."""
    if not isinstance(structure, RationalOperator) or structure.name != "COMPLEX_STRUCTURE":
        raise TypeError("rotation requires an explicit certified COMPLEX_STRUCTURE")
    structure.check_state()
    cosine, sine = rotation_triple(a, b, c)

    def action(x, *, exact, transpose):
        cs = cosine if exact else float(cosine)
        sn = (-sine if transpose else sine) if exact else float(-sine if transpose else sine)
        jx = structure.apply(x, exact=exact)
        projection = -structure.apply(jx, exact=exact)
        return x + (cs-1)*projection + sn*jx
    return RationalOperator(structure, "RATIONAL_ROTATION", action,
                            parameters=(("cosine", cosine), ("sine", sine)))


def resolvent_word(parameters, word):
    """Validate rational generator scales and reduce adjacent inverse letters."""
    if not isinstance(parameters, (tuple, list)) or not parameters:
        raise ValueError("resolvent group requires a nonempty parameter sequence")
    scales = tuple(_fraction(t) for t in parameters)
    if any(t < 0 for t in scales):
        raise ValueError("resolvent group parameters must be nonnegative")
    return scales, generator_word(len(scales), word, identity={i+1 for i, t in enumerate(scales) if not t})


def generator_word(count, word, *, identity=()):
    """Reduce a finite word by declared identities and adjacent inverse pairs.

    This is a syntactic reduction, not a group membership or equality solver.
    """
    from numbers import Integral
    if not isinstance(word, (tuple, list)):
        raise TypeError("group word requires a sequence of signed generator indices")
    reduced = []
    for letter in word:
        if isinstance(letter, (bool, np.bool_)) or not isinstance(letter, Integral):
            raise TypeError("group word letters must be integers, not booleans")
        letter = int(letter)
        if not 1 <= abs(letter) <= count:
            raise ValueError("group word indices start at one and must name a generator")
        if abs(letter) in identity:
            continue
        if reduced and reduced[-1] == -letter:
            reduced.pop()
        else:
            reduced.append(letter)
    return tuple(reduced)


class ResolventGroup(RationalOperator):
    """A generated operator subgroup with one selected word as its apply action.

    A positive letter names (I+tL)^-1; a negative letter names I+tL.
    Products act from right to left. No commutation or finite group order is
    inferred. The empty word is identity; word reduction is not an equality test.
    """

    def __init__(self, operators, parameters, word=(), *, tol=1e-10, maxiter=1000):
        if not isinstance(operators, (tuple, list)) or not operators:
            raise ValueError("resolvent group requires a nonempty operator sequence")
        operators = tuple(operators)
        scales, reduced = resolvent_word(parameters, word)
        if len(operators) != len(scales):
            raise ValueError("one parameter is required per resolvent generator")
        first = operators[0]
        for op in operators:
            if (not isinstance(op, RexOperator) or op.source is None or not op.symmetric or not op.psd
                    or op.domain_grade != op.codomain_grade or op.shape[0] != op.shape[1]
                    or op.exact_matvec is None or op.exact_transpose_matvec is None):
                raise TypeError("resolvent generators require bound Euclidean PSD square Q actions and transposes")
            if (op.source is not first.source or op.shape != first.shape
                    or op.domain_grade != first.domain_grade or op.variance != first.variance):
                raise ValueError("resolvent generators require one source, grade, shape and variance")
            if hasattr(op, "check_state"):
                op.check_state()
        if first.shape[0] != cell_count(first.source, first.domain_grade, allow_empty_upper=True):
            raise ValueError("resolvent generator population differs from its source")
        # Validate numerical controls without converting exact generator scales.
        GreenOperator.resolvent(first, 0, tol=tol, maxiter=maxiter)
        numerical = {}

        def action(x, *, exact, transpose):
            out = x.copy()
            for letter in (reduced if transpose else reversed(reduced)):
                i = abs(letter)-1
                op, t = operators[i], scales[i]
                if letter < 0:
                    coefficient = t if exact else float(t)
                    out = out + coefficient*op.apply(out, exact=exact)
                elif exact:
                    out = _exact_positive_solve(lambda v: v+t*op.apply(v, exact=True), out)
                else:
                    if i not in numerical:
                        numerical[i] = GreenOperator.resolvent(op, t, tol=tol, maxiter=maxiter)
                    solve = numerical[i].solve
                    out = solve(out.real) + 1j*solve(out.imag) if np.iscomplexobj(out) else solve(out)
                if not exact:
                    out = _numeric_array(out, operation="resolvent word")
            return out

        super().__init__(first, "RESOLVENT_GROUP", action,
            parameters=(("scales", scales), ("word", reduced), ("tol", tol), ("maxiter", maxiter)))
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "word", reduced)
        object.__setattr__(self, "tol", tol)
        object.__setattr__(self, "maxiter", maxiter)

    def element(self, word):
        self.check_state()
        return ResolventGroup(self.operators, self.scales, word, tol=self.tol, maxiter=self.maxiter)

    @property
    def inverse(self):
        return self.element(tuple(-letter for letter in reversed(self.word)))

    @property
    def identity(self):
        return self.element(())
