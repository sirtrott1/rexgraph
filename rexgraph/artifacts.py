"""Persistence ports used by mathematical/model layers.

The core and model packages depend on this interface, never on a concrete HDF5/Zarr/
safetensors module.  ``rexgraph.io`` installs the default file backed adapter when it is
available.  Applications may replace that adapter without changing model code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ArtifactPayload:
    """A reconstructed native object plus caller owned tensors and metadata."""

    object: Any
    tensors: Mapping[str, Any]
    metadata: Mapping[str, Any]


@runtime_checkable
class ArtifactBackend(Protocol):
    """Minimal persistence surface required by live model/runtime code."""

    def save_vectors(self, matrix, labels, path, **kwargs): ...

    def save_rex_artifact(
        self, rex, path, *, tensors: Mapping[str, Any], metadata: Mapping[str, Any], format: str
    ): ...

    def load_rex_artifact(self, path, *, format: str) -> ArtifactPayload: ...


_backend: ArtifactBackend | None = None


def register_artifact_backend(backend: ArtifactBackend, *, replace: bool = False) -> None:
    """Install the process artifact adapter.

    Registration is explicit so state/math layers do not import a concrete IO package.
    Registering again the same backend type is harmless; replacing it with another adapter
    requires ``replace=True``.
    """
    global _backend
    if not isinstance(backend, ArtifactBackend):
        raise TypeError("artifact backend does not implement the required port")
    if _backend is not None:
        if type(_backend) is type(backend):
            _backend = backend
            return
        if not replace:
            raise RuntimeError("an artifact backend is already registered")
    _backend = backend


def artifact_backend() -> ArtifactBackend:
    if _backend is None:
        raise RuntimeError(
            "no artifact backend is registered; import rexgraph.io or install an application adapter"
        )
    return _backend


def save_vectors(matrix, labels, path, **kwargs):
    return artifact_backend().save_vectors(matrix, labels, path, **kwargs)


def save_rex_artifact(rex, path, *, tensors=None, metadata=None, format="safetensors"):
    return artifact_backend().save_rex_artifact(
        rex,
        path,
        tensors={} if tensors is None else tensors,
        metadata={} if metadata is None else metadata,
        format=str(format),
    )


def load_rex_artifact(path, *, format="safetensors") -> ArtifactPayload:
    payload = artifact_backend().load_rex_artifact(path, format=str(format))
    if not isinstance(payload, ArtifactPayload):
        raise TypeError("artifact backend returned an invalid payload")
    return payload


__all__ = [
    "ArtifactBackend",
    "ArtifactPayload",
    "artifact_backend",
    "register_artifact_backend",
    "save_vectors",
    "save_rex_artifact",
    "load_rex_artifact",
]
