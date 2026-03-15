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

from . import core

try:
    from . import io
except ImportError:
    io = None

try:
    from . import viz
except ImportError:
    viz = None

__version__ = "0.2.0"

__all__ = [
    "core",
    "io",
    "viz",
]
