"""Compatibility exports for container independent identity primitives.

New code should import these from :mod:`rexgraph.identity`; the IO namespace remains so
existing callers and stored format tests keep their public surface.
"""
from rexgraph.identity import (
    MANIFEST_FORMAT_VERSION as FORMAT_VERSION,
    canonical_json,
    digest_parts,
    manifest_digest,
)

__all__ = ["FORMAT_VERSION", "canonical_json", "digest_parts", "manifest_digest"]
