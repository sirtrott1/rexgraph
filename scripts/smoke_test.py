#!/usr/bin/env python3
"""What a built wheel has to do before it is worth shipping.

Run by cibuildwheel against each wheel, from a temporary directory so the source tree
is not on sys.path. Kept as a file rather than a -c one-liner because it grew: the
one-liner version checked betti and nothing else, passed on four manylinux wheels whose
_laplacians could not import, and shipped them.

The rule it now follows: touch every compiled module the package actually needs, and
import them individually so a failure names the module rather than the symptom.
"""

import sys

import numpy as np

import rexgraph
from rexgraph.graph import RexGraph

print(f"  python   {sys.version.split()[0]}")
print(f"  rexgraph {rexgraph.__version__} from {rexgraph.__file__}")

# every kernel, by name. A missing BLAS takes out _laplacians and _linalg while leaving
# _boundary fine, so "the package imports" is not the same as "the package works".
missing = []
for name in ("_boundary", "_channels", "_character", "_cycles", "_dirac", "_faces",
             "_harmonic", "_hodge", "_laplacians", "_linalg", "_overlap",
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
print(f"  all {16} probed kernels import")

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
print("  ok")
