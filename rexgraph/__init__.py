"""
RexGraph: relational complex analysis with Cython accelerated internals.

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
# package shadows the standard library ``types`` module whenever this directory
# lands on sys.path[0] (e.g. running a script from inside the package, or
# ``python -m ...`` from here), which breaks ``enum``/``dataclasses`` and much
# of the stdlib. We keep ``rexgraph.types`` working as a backwards compatible
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
from .harmonic_modes import effective_modes, grade_traces, harmonic_log
from .resolvent_ranking import resolvent_rank
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
# ``rexgraph.core`` during construction, so placing the re export here keeps
# package root import acyclic while making the primary public class available
# from the documented package surface.
from .graph import RexGraph, TemporalRex

__version__ = "1.2.1"

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
    # rings: the cycle space of the 1 skeleton, basis free
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
    # the harmonic log: exact effective mode counts of the Hodge sectors
    "effective_modes",
    "grade_traces",
    "harmonic_log",
    # the resolvent rank: PageRank's fixed point as a Green response at any grade
    "resolvent_rank",
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
    "ModelState", "ModelOutput", "ModelInput", "ModelBatch", "ModelTimeline",
    "model_input", "native_model", "infer_model", "train_model", "transport_model",
    "certify_native_response",
]


def __getattr__(name):
    """Load the optional biomedical demonstration API only when a caller asks for it.

    Keeping this lazy leaves the module untouched until a caller explicitly requests
    its builder or artifact writer. The functions remain part of the documented
    package surface without adding an RCDB or Agent dependency to normal core imports.
    """
    fields = {'NativeFieldCalculus': 'native_field', 'FieldAction': 'native_field', 'NativeAction': 'native_field', 'bridge_action': 'native_field', 'FieldSource': 'tensor_field', 'TensorField': 'tensor_field', 'TensorChannels': 'tensor_field', 'apply_tensor': 'tensor_field', 'CoordinatePairing': 'tensor_moment', 'RealizedPairing': 'tensor_moment', 'MomentSpan': 'tensor_moment', 'TensorMomentKernel': 'tensor_moment', 'TensorMoments': 'tensor_moment', 'TensorEvolution': 'temporal_field', 'ResolvedEvolution': 'temporal_field', 'NativeFieldEvolution': 'temporal_field', 'SectorTransport': 'temporal_field', 'MomentChange': 'temporal_field', 'moment_change': 'temporal_field', 'AttachmentField': 'attachment_field', 'AttachmentObservation': 'attachment_field', 'common_attachment_observations': 'attachment_field', 'ReconstructionFamily': 'reconstruction'}
    fields.update({"ModelInput": "model_state", "model_input": "model_state", "ModelState": "model_state", "ModelOutput": "model_state", "ModelBatch": "model_state",
                   "ModelTimeline": "model_state", "native_model": "model_runtime", "infer_model": "model_runtime",
                   "certify_native_response": "model_runtime", "train_model": "model_runtime", "transport_model": "model_runtime"})
    fields.update({"MolecularView": "molecular_field", "molecular_changes": "molecular_field",
                   "conformation_field": "molecular_field", "conformation_direction": "molecular_field",
                   "sampled_field": "process_field", "trajectory_comparison": "process_field",
                   "sample_rates": "process_field", "diagonal_axes": "process_field",
                   "factor_contrast": "process_field", "factorial_contrast": "process_field",
                   "response_direction": "process_field"})
    if name in fields:
        from importlib import import_module
        return getattr(import_module("." + fields[name], __name__), name)
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

__all__ += ['NativeFieldCalculus', 'FieldAction', 'NativeAction', 'bridge_action', 'FieldSource', 'TensorField', 'TensorChannels', 'apply_tensor', 'CoordinatePairing', 'RealizedPairing', 'MomentSpan', 'TensorMomentKernel', 'TensorMoments', 'TensorEvolution', 'ResolvedEvolution', 'NativeFieldEvolution', 'SectorTransport', 'MomentChange', 'moment_change', 'AttachmentField', 'AttachmentObservation', 'common_attachment_observations', 'ReconstructionFamily']

from .section_calculus import (
    SectionSystem, SectionRecipe, SectionFamily, SectionImage,
    InconsistentSectionError, UnderdeterminedSectionError,
)

__all__ += ["SectionSystem", "SectionRecipe", "SectionFamily", "SectionImage",
            "InconsistentSectionError", "UnderdeterminedSectionError"]

__all__ += ["ModelState", "ModelOutput", "ModelBatch", "ModelTimeline", "native_model",
            "infer_model", "train_model", "transport_model"]

__all__ += ["MolecularView", "molecular_changes", "conformation_field", "conformation_direction",
            "sampled_field", "trajectory_comparison", "sample_rates", "diagonal_axes",
            "factor_contrast", "factorial_contrast", "response_direction"]
