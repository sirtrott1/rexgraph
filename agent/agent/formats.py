"""Bundle suffixes for agent-owned paths.

The canonical container is RCBD (``.rcbd``).  Legacy ``.rex`` bundles stay
readable, and they need no reader of their own: the core identifies a directory
bundle by the ``MANIFEST.json`` magic and never by its suffix, so the only thing
a legacy bundle needs from the agent is that these suffix checks keep accepting
it.  That is why this module states the read set and the write default
separately -- they are different questions, and only the write default moved.

The export *format key* (``"rex"``, in the HTTP and frontend surface) is a
different thing again and deliberately unchanged: it names the export contract,
not a filename.
"""
from __future__ import annotations

#: What a newly written bundle is called.
BUNDLE_SUFFIX = ".rcbd"

#: Bundles still on disk from before the rename.  Read, never written.
LEGACY_BUNDLE_SUFFIXES = (".rex",)

#: Every suffix a bundle may arrive with, canonical first.
BUNDLE_SUFFIXES = (BUNDLE_SUFFIX, *LEGACY_BUNDLE_SUFFIXES)

__all__ = [
    "BUNDLE_SUFFIX",
    "BUNDLE_SUFFIXES",
    "LEGACY_BUNDLE_SUFFIXES",
    "is_bundle_suffix",
]


def is_bundle_suffix(suffix: str) -> bool:
    """Whether `suffix` names a relational complex bundle, canonical or legacy."""
    return str(suffix).lower() in BUNDLE_SUFFIXES
