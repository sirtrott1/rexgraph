"""Default file backed implementation of :mod:`rexgraph.artifacts`."""
from __future__ import annotations

from rexgraph.artifacts import ArtifactPayload


class IOArtifactBackend:
    """Route neutral artifact requests to the current RexGraph IO implementation."""

    def save_vectors(self, matrix, labels, path, **kwargs):
        from .safetensors_bridge import fingerprints_to_safetensors
        return fingerprints_to_safetensors(matrix, labels, path, **kwargs)

    def save_rex_artifact(self, rex, path, *, tensors, metadata, format):
        if format != "safetensors":
            raise ValueError(f"unsupported model artifact format {format!r}")
        from .safetensors_bridge import rex_to_safetensors
        return rex_to_safetensors(
            rex,
            path,
            extra_tensors=dict(tensors),
            extra_meta=dict(metadata),
        )

    def load_rex_artifact(self, path, *, format):
        if format != "safetensors":
            raise ValueError(f"unsupported model artifact format {format!r}")
        from .safetensors_bridge import load_extra, load_safetensors
        full = load_safetensors(path)
        return ArtifactPayload(
            full["object"],
            dict(full["tensors"]),
            dict(load_extra(path)),
        )
