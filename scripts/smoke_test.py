#!/usr/bin/env python3
"""What a built wheel has to do before it is worth shipping.

Run by cibuildwheel against each wheel, from a temporary directory so the source tree
is not on sys.path. Kept as a file rather than a -c one liner because it grew: the
one liner version checked betti and nothing else, passed on four manylinux wheels whose
_laplacians could not import, and shipped them.

The rule it now follows: touch every compiled module the package actually needs, and
import them individually so a failure names the module rather than the symptom.
"""

import sys
from fractions import Fraction

import numpy as np

import rexgraph
from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.channel_operator import channel_operator
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import DiagonalMetric, integrate
from rexgraph.graph import RexGraph
from rexgraph.green import GreenOperator
from rexgraph.linear_operator import boundary_operator, metric_adjoint
from rexgraph.operator_bracket import operator_bracket
from rexgraph.sheaf import ExactSheaf
from rexgraph.type_accession import TypeAccession
from rexgraph.weighted_dirac import GradedChain, weighted_dirac
from rexgraph.weighted_hodge import weighted_hodge

print(f"  python   {sys.version.split()[0]}")
print(f"  rexgraph {rexgraph.__version__} from {rexgraph.__file__}")

# every kernel, by name. A missing BLAS takes out _laplacians and _linalg while leaving
# _boundary fine, so "the package imports" is not the same as "the package works".
missing = []
for name in ("_boundary", "_channels", "_character", "_cycles", "_dirac", "_faces",
             "_field", "_signal", "_harmonic", "_hodge", "_laplacians", "_linalg", "_overlap",
             "_persistence", "_query", "_sparse", "_spectral", "_state"):
    try:
        __import__(f"rexgraph.core.{name}")
    except Exception as exc:                      # noqa: BLE001 - reporting, not handling
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
if missing:
    print("  compiled modules that failed to import:")
    for line in missing:
        print(f"    {line}")
    raise SystemExit(1)
print("  all 18 probed kernels import")

# topology
r = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
             targets=np.array([1, 2, 3, 0], np.int32))
r._ensure_clean()
assert tuple(r.betti) == (1, 1, 0), r.betti

# the character, which is what goes through _laplacians and therefore through BLAS
chi = np.asarray(r.structural_character)
assert chi.shape[0] == 4, chi.shape
assert np.allclose(chi.sum(axis=1), 1.0), chi.sum(axis=1)

# a branching relation, since arity above two is the whole point
h = RexGraph.from_hypergraph(np.array([0, 4, 7], np.int32),
                             np.array([0, 1, 2, 3, 3, 4, 5], np.int32))
h._ensure_clean()
assert int(h.nE) == 2 and int(h.nV) == 6, (h.nE, h.nV)

print(f"  4-cycle betti {tuple(r.betti)}, chi {chi.shape} rows summing to 1")
print(f"  branching nV={int(h.nV)} nE={int(h.nE)} betti {tuple(h.betti)}")

# Pure Python native math is just as necessary as the compiled kernels. These
# imports/actions must succeed in the wheel without source checkout fallbacks.
f = channel_operator(r, "F")
assert f.diagonal(exact=True).tolist() == [Fraction(4)] * 4
assert np.allclose(f.diagonal(), [4] * 4)
assert f.apply(np.array([1, 0, 0, 0]), exact=True).tolist() == [Fraction(4), Fraction(-2), Fraction(0), Fraction(-2)]
assert channel_operator(h, "G").apply(np.array([1, 0]), exact=True).tolist() == [Fraction(4, 3), Fraction(1, 3)]
x = Cochain(1, np.ones(4, dtype=int), source=r)
metric = DiagonalMetric(r, 1, (2,) * 4)
assert metric.moment(x, x, exact=True) == Fraction(8)
assert integrate(x, Chain(1, np.ones(4, dtype=int), source=r), exact=True) == Fraction(4)
adjoint = metric_adjoint(boundary_operator(h, 1), DiagonalMetric(h, 1, (2, 3)))
assert adjoint.apply(np.array([1, 2, 3, 4, 5, 6]), exact=True).tolist() == [Fraction(1), Fraction(1, 2)]
weighted = weighted_hodge(h, 1, metric=DiagonalMetric(h, 1, (2, 3)))
rhs = np.array([3., 2.])
solved = GreenOperator.resolvent(weighted).solve(rhs)
np.testing.assert_allclose(solved + weighted.apply(solved), rhs)
assert weighted.apply(np.array([1, 0]), exact=True).tolist() == [Fraction(2, 3), Fraction(-1, 9)]
dirac = weighted_dirac(h, metrics=[DiagonalMetric(h, 1, (2, 3))])
seed = GradedChain(h, [Chain(1, np.array([1, 0]), source=h)])
square = dirac.apply(dirac.apply(seed, exact=True), exact=True)
assert square.component(1).values.tolist() == [Fraction(2, 3), Fraction(-1, 9)]
anti_dirac = weighted_dirac(h, metrics=[DiagonalMetric(h, 1, (2, 3))], anti=True)
bracket = operator_bracket(dirac, anti_dirac).apply(seed, exact=True)
assert bracket.component(1).values.tolist() == [Fraction(4, 3), Fraction(-2, 9)]
a = TypeAccession(r, 1, "all", tuple((i, i, 2) for i in range(4)))
assert a.apply(x, exact=True).values.tolist() == [Fraction(2)] * 4
c = CoordinateComplex.from_rex(h)
p = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))
assert p.verify().commutation_residuals == (Fraction(0),)
section = ExactSheaf(h, stalk_dims=(2, 1), mediator_dims=(1,)*int(h.nV))
section.assign(0, [Fraction(1, 3), Fraction(1, 6)])
section.assign(1, [Fraction(1, 2)])
section.restrict(0, [[1, 1]])
assert section.check_section().compatible
assert section.check_section().incidence_count == 7
section.assign(1, [Fraction(2, 3)])
assert section.check_section().obstructions[0].residual == (Fraction(-1, 6),)
print("  exact/native channels, dual pairing, metric adjoint, weighted Hodge/Dirac/Green, brackets, accession and chain map work")
print("  exact rectangular sheaf maps and compact incidence compatibility work")

# Both initial conditions must reach the numerical field runtime. A cycle's
# harmonic edge velocity drifts linearly even from zero initial position.
position, velocity = r.field_wave_evolve(np.zeros(4), np.ones(4), np.array([0., 2.]))
assert np.allclose(position[1], 2.) and np.allclose(velocity, 1.)
# Explicit dense oracle must not mistake a negative mode for a zero frequency.
_field = sys.modules["rexgraph.core._field"]
values, vectors, frequencies = _field.field_eigendecomposition(np.array([[-1.]]))
position, velocity = _field.wave_evolve(np.ones(1), values, vectors, frequencies, 1.)
assert np.allclose(position, np.cosh(1.)) and np.allclose(velocity, np.sinh(1.))
print("  field initial-velocity drift and signed spectral reference work")
print("  ok")
