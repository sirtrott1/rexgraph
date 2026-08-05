"""
agent.interfaces: the embed contract for the engine.

The engine is sovereign: it reads only what it is pointed at, persists only
structure, holds no credentials it isn't given, and **emits nothing** unless the
host wires it up. This module defines the seams a host plugs into and ships
inert defaults so that, out of the box, the engine is silent and self-contained.

Seams:
  * Logger / Metrics: observability. Default: no-op (no telemetry, ever).
  * Identity        - who/what is acting. Default: a local single-tenant identity.
  * Connector       - read a relational complex from a source. Default: none
                      (the host registers the sources it wants).
  * SecretStore     - connection secrets (see agent.secrets). Default: none held.
  * Store           - the RCDB backend (see agent.rcdb.RCStore).

Nothing here reaches the network, writes a file, or records a metric on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# observability seam (silent by default)

@runtime_checkable
class Logger(Protocol):
    def log(self, level: str, message: str, **fields: Any) -> None: ...


@runtime_checkable
class Metrics(Protocol):
    def incr(self, name: str, value: float = 1.0, **tags: Any) -> None: ...
    def observe(self, name: str, value: float, **tags: Any) -> None: ...


class NullLogger:
    """Emits nothing. The default: the engine never logs unless the host asks."""
    def log(self, level: str, message: str, **fields: Any) -> None:
        return None


class NullMetrics:
    """Records nothing. No telemetry leaves the engine by default."""
    def incr(self, name: str, value: float = 1.0, **tags: Any) -> None:
        return None

    def observe(self, name: str, value: float, **tags: Any) -> None:
        return None


# identity seam (host-injected, not owned)

@runtime_checkable
class Identity(Protocol):
    """The engine *accepts* an identity from the host; it does not run an IdP.
    The host's SSO/OIDC/OPA does the authenticating; the engine respects the
    answer (workspace + role) for scoping."""
    @property
    def workspace(self) -> str: ...
    @property
    def role(self) -> str: ...


class LocalIdentity:
    """Default single-tenant identity for embedded/library use."""
    def __init__(self, workspace: str = "default", role: str = "admin"):
        self._workspace = workspace
        self._role = role

    @property
    def workspace(self) -> str:
        return self._workspace

    @property
    def role(self) -> str:
        return self._role


# connector seam (source -> complex)

@dataclass(frozen=True)
class Capabilities:
    """What a connector can supply from a source. Topology is always present
    (a connector that can't produce B₁ isn't a connector); weights, modality,
    and faces are optional enrichments. The engine reads this *before* calling
    ``read`` so it knows which analyses are available (e.g. strain needs
    weights; curvature needs faces).

    ``schemes`` lists the source-URI schemes this connector claims (e.g.
    ``("postgresql", "sqlite")``), used by the registry to route ``open_connector``.
    """
    topology: bool = True          # can emit B₁ (required; always True)
    weights: bool = False          # can emit per-edge weights (enables strain)
    modality: bool = False         # can emit per-edge FK modality (enables lint)
    faces: bool = False            # can emit a B₂ face selection (enables curvature)
    schemes: tuple[str, ...] = ()  # source-URI schemes this connector handles

    def summary(self) -> str:
        have = [n for n in ("topology", "weights", "modality", "faces")
                if getattr(self, n)]
        return "+".join(have)


@runtime_checkable
class Connector(Protocol):
    """Read a relational complex from a source: a live DB, a dump, a stream,
    an in-memory graph, an ontology. **This is THE seam a customer or the
    services team implements** to teach the engine a new system.

    The contract is deliberately tiny, stable, and read-only:

        read(source) -> (rex, meta)

    where

      * ``rex``  - the topology: either a built ``RexGraph`` or the
        ``(sources, targets)`` edge arrays an adapter builds one from
        (optionally with a ``B₂`` face selection). Signed incidence B₁ is the
        sole required datum; vertices are derived from it.
      * ``meta`` - a plain ``dict`` describing the complex. Keys:

          ==================  ========  ==================================
          key                 required  meaning
          ==================  ========  ==================================
          ``vertex_labels``   yes       ``list[str]``, ``len == nV`` - the
                                        one privacy surface (fed to
                                        ``apply_label_privacy``).
          ``edges``           yes       ``list[(src_label, dst_label)]``,
                                        ``len == nE``.
          ``source``          yes       ``str`` provenance tag.
          ``weights``         no        ``list[float]``, ``len == nE`` -
                                        enables data-forced strain.
          ``modality``        no        ``list[dict]`` per edge (nullable /
                                        identifying / on_delete …) - lint.
          ``faces``           no        a ``B₂`` selection, which enables the
                                        face-bound curvatures.
          ==================  ========  ==================================

    Invariants every connector MUST preserve (asserted by the validation
    harness): **read-only** (issues no writes to the source), **structure-only**
    (returns topology + labels, never cell/row values), and **∂²=0** whenever
    ``faces`` are supplied. Customer/proprietary connectors live *outside* the
    core, depending only on this seam, never editing the engine.
    """
    def read(self, source: Any) -> tuple[Any, dict[str, Any]]: ...

    def capabilities(self) -> Capabilities:
        """What this connector can supply (topology always; optionally weights,
        modality, faces) and which URI schemes it handles."""
        ...


# engine-wide configuration (all inert by default)

class _Config:
    def __init__(self):
        import os
        self.logger: Logger = NullLogger()
        self.metrics: Metrics = NullMetrics()
        self.identity: Identity = LocalIdentity()
        # env lets a deployment enable label privacy without code changes
        self.label_privacy: str = os.environ.get("REXGRAPH_LABEL_PRIVACY", "none")
        self.label_salt: str = os.environ.get("REXGRAPH_LABEL_SALT", "")


_CONFIG = _Config()


def tokenize_labels(labels, salt: str = "") -> list:
    """Deterministically tokenize labels (table/column names). Same name -> same
    token, so cross-complex coherence (which aligns by shared labels) still
    works, while the actual names are hidden: the one privacy leak surface a
    structure-only engine has. Irreversible (SHA-256); the host keeps its own
    map if it needs one."""
    import hashlib
    out = []
    for lbl in labels:
        h = hashlib.sha256((salt + "\x00" + str(lbl)).encode()).hexdigest()[:12]
        out.append("t_" + h)
    return out


def apply_label_privacy(meta: dict) -> dict:
    """If label privacy is enabled, tokenize vertex labels in a meta dict
    (returns a shallow copy; leaves meta untouched when disabled)."""
    if _CONFIG.label_privacy != "hash" or not meta:
        return meta
    labels = meta.get("vertex_labels")
    if not labels:
        return meta
    m = dict(meta)
    m["vertex_labels"] = tokenize_labels(labels, _CONFIG.label_salt)
    m["_label_privacy"] = "hash"
    return m


def configure(logger: Logger | None = None,
              metrics: Metrics | None = None,
              identity: Identity | None = None,
              label_privacy: str | None = None,
              label_salt: str | None = None) -> None:
    """Host wires in its own observability/identity/privacy. Anything left None
    keeps the inert default. This is the only way telemetry is ever enabled or
    label privacy is turned on."""
    if logger is not None:
        _CONFIG.logger = logger
    if metrics is not None:
        _CONFIG.metrics = metrics
    if identity is not None:
        _CONFIG.identity = identity
    if label_privacy is not None:
        _CONFIG.label_privacy = label_privacy
    if label_salt is not None:
        _CONFIG.label_salt = label_salt


def get_logger() -> Logger:
    return _CONFIG.logger


def get_metrics() -> Metrics:
    return _CONFIG.metrics


def get_identity() -> Identity:
    return _CONFIG.identity


def reset() -> None:
    """Restore all seams to their inert defaults (used by tests/embedders)."""
    global _CONFIG
    _CONFIG = _Config()
