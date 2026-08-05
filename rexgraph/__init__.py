"""
RexGraph: relational complex analysis with Cython-accelerated internals.

Classes:
    RexGraph     - Relational complex with lazily computed structural properties.
    TemporalRex  - Temporal sequence of rex snapshots.

Subpackages:
    core  - Cython extensions (boundary maps, Laplacians, spectral, RCF operators).
    io    - Serialization (Zarr, HDF5, Arrow, Parquet, SQL).
    viz   - Visualization dashboard.
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

try:
    from . import viz
except ImportError:
    viz = None

from . import compute
from .mesh_health import harmonic_health, mesh_health

__version__ = "1.0.6"

__all__ = [
    "core",
    "io",
    "viz",
    "compute",
    "mesh_health",
    "harmonic_health",
]
