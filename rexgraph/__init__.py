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
from .cells import (
    Cell,
    CellBoundary,
    CellCoboundary,
    CellSet,
    CompositeBinary,
    GradedCellPattern,
    boundary_of,
    cell,
    cells,
    coboundary_of,
    composite_binary,
    corelations,
    enclosure,
    star,
)
from .cochain import Chain, Cochain, Field, GradedState
from .temporal_signal import (
    RelationKey,
    TemporalSignal,
    TemporalSignalEvent,
    TemporalSignalFlow,
    relation_identity,
    relation_key,
    signal_flow,
    temporal_signal,
)
from .green import GreenOperator, vertex_green
from .metric_field import MetricCurvature, relation_metric_curvature
from .harmonic_sparse import (
    harmonic_basis,
    harmonic_coordinates,
    harmonic_projection,
    harmonic_winding,
    multiplicity_cycles,
    multiplicity_dimension,
    multiplicity_groups,
    multiplicity_homology_dimension,
    simple_cycle_dimension,
)
from .hodge_coords import (
    complex_structure,
    coordinate_dims,
    from_hodge_coords,
    harmonic_coords,
    harmonic_frame,
    harmonic_metric,
    hodge_coords,
)
from .linear_operator import (
    RexOperator,
    boundary_operator,
    coboundary_operator,
    down_laplacian,
    hodge_operator,
    up_laplacian,
)
from .mesh_health import harmonic_health, mesh_health
from .rings import (
    cycle_vector,
    cycle_vectors,
    minimum_cycle_basis,
    relevant_cycles,
    ring_sizes,
    shortest_cycles,
)
from .tower import channel_delta, graded_delta
# Import this only after the foundational public types above.  graph imports
# ``rexgraph.core`` during construction, so placing the re-export here keeps
# package-root import acyclic while making the primary public class available
# from the documented package surface.
from .graph import RexGraph, TemporalRex

__version__ = "1.1.4"

__all__ = [
    "core",
    "io",
    "compute",
    "RexGraph",
    "TemporalRex",
    "Cell",
    "CellSet",
    "GradedCellPattern",
    "CompositeBinary",
    "CellBoundary",
    "CellCoboundary",
    "cell",
    "cells",
    "composite_binary",
    "boundary_of",
    "coboundary_of",
    "corelations",
    "star",
    "enclosure",
    "Chain",
    "Cochain",
    "Field",
    "GradedState",
    "RelationKey",
    "TemporalSignal",
    "TemporalSignalEvent",
    "TemporalSignalFlow",
    "relation_identity",
    "relation_key",
    "temporal_signal",
    "signal_flow",
    "RexOperator",
    "GreenOperator",
    "boundary_operator",
    "coboundary_operator",
    "down_laplacian",
    "up_laplacian",
    "hodge_operator",
    "vertex_green",
    "MetricCurvature",
    "relation_metric_curvature",
    "mesh_health",
    "harmonic_health",
    "build_mtor_demo",
    "write_demo_artifacts",
    "CellPaintingPlate",
    "DEFAULT_JUMP_SECTIONS",
    "JumpCellPaintingStudy",
    "load_jump_plate",
    "build_jump_cell_painting_temporal",
    "analyze_jump_delta",
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


def __getattr__(name):
    """Load the optional biomedical demonstration API only when a caller asks for it.

    Keeping this lazy leaves the module untouched until a caller explicitly requests
    its builder or artifact writer. The functions remain part of the documented
    package surface without adding an RCDB or Agent dependency to normal core imports.
    """
    if name in {"build_mtor_demo", "write_demo_artifacts"}:
        from . import biomedical_demo
        return getattr(biomedical_demo, name)
    if name in {
        "CellPaintingPlate", "DEFAULT_JUMP_SECTIONS", "JumpCellPaintingStudy", "load_jump_plate",
        "build_jump_cell_painting_temporal", "analyze_jump_delta",
    }:
        from . import jump_cell_painting
        return getattr(jump_cell_painting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
