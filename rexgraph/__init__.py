"""
RexGraph: relational complex analysis with Cython-accelerated internals.

Classes:
    RexGraph     - Relational complex with lazily computed structural properties.
    TemporalRex  - Temporal sequence of rex snapshots.

Subpackages:
    core  - Cython extensions (boundary maps, Laplacians, spectral, RCF operators).
    io    - Serialization (Zarr, HDF5, Arrow, Parquet, SQL).
"""

import sys as _sys

# Result/enumeration types. The module lives at ``rexgraph.rextypes`` rather
# than ``rexgraph.types`` because a module literally named ``types`` inside the
# package shadows the standard-library ``types`` module whenever this directory
# lands on sys.path[0] (e.g. running a script from inside the package, or
# ``python -m ...`` from here), which breaks ``enum``/``dataclasses`` and much
# of the stdlib. We keep ``rexgraph.types`` working as a backwards-compatible
# import alias via sys.modules: this does NOT put a ``types.py`` file back on
# disk, so the shadow is gone.
from . import core, rextypes

_sys.modules.setdefault("rexgraph.types", rextypes)
types = rextypes

try:
    from . import io
except ImportError:
    io = None

from . import compute
from .mesh_health import harmonic_health, mesh_health
from .harmonic_sparse import (harmonic_basis, harmonic_coordinates,
                              harmonic_projection, harmonic_winding,
                              multiplicity_cycles, multiplicity_dimension,
                              multiplicity_groups,
                              multiplicity_homology_dimension,
                              simple_cycle_dimension)
from .hodge_coords import (complex_structure, coordinate_dims, from_hodge_coords,
                           harmonic_coords, harmonic_frame, harmonic_metric,
                           hodge_coords)
from .rings import (cycle_vector, cycle_vectors, minimum_cycle_basis,
                    relevant_cycles, ring_sizes, shortest_cycles)
from .tower import channel_delta, graded_delta

__version__ = "1.1.1"

__all__ = [
    "core",
    "io",
    "compute",
    "mesh_health",
    "harmonic_health",
    "channel_delta",
    "graded_delta",
    # rings: the cycle space of the 1-skeleton, basis-free
    "cycle_vector",
    "cycle_vectors",
    "minimum_cycle_basis",
    "relevant_cycles",
    "ring_sizes",
    "shortest_cycles",
    # the Hodge chart and its coordinates
    "coordinate_dims",
    "complex_structure",
    "from_hodge_coords",
    "harmonic_coords",
    "harmonic_frame",
    "harmonic_metric",
    "hodge_coords",
    # the harmonic sector: basis, projection, and the exact integer reading
    "harmonic_basis",
    "harmonic_coordinates",
    "harmonic_projection",
    "harmonic_winding",
    # multiplicity: the part of the cycle space that is repetition, not shape
    "multiplicity_cycles",
    "multiplicity_dimension",
    "multiplicity_groups",
    "multiplicity_homology_dimension",
    "simple_cycle_dimension",
]
