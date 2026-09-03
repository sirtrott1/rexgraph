"""Resolve sources and classify them before any operator runs.

The existing binder hands the executor a live object and a policy. That is enough to run
an operator and not enough to type one: nothing says what KIND of source it is, so a
signature cannot check that ``RCDB_GET`` received a store rather than a catalog, and
``STATE_HASH`` cannot be stopped from being handed the catalog it is filed beside.

This module answers that question once, at binding time, from the object's own surface
rather than from its class name. A store is anything that can answer ``get`` and
``history``; a catalog is anything that can ``list`` and ``hash``. Classifying by surface
keeps RCQL from importing rcdb or the catalog module to do an isinstance check, which is
the layering rule the distributions exist to hold.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capabilities import BoundSource, SourcePolicy
from .signatures import OperatorSignature, lookup
from .types import SourceRef, TemporalRef, ValueKind


class SourceKindError(TypeError):
    """The bound source is not the kind of thing this operator reads."""


class UnreachableOperator(TypeError):
    """The operator is registered but cannot execute against any available source."""


@dataclass(frozen=True)
class SourceSchema:
    """What a bound source can answer, decided from its surface.

    ``kind`` is the ValueKind an operator signature declares it needs. ``capabilities`` is
    what the policy grants, so an operator's ``requires`` can be checked before it runs
    rather than raising PermissionError from inside an adapter.
    """

    kind: ValueKind
    capabilities: frozenset[str]
    surface: frozenset[str]

    def can(self, name: str) -> bool:
        return name in self.surface


def classify(value: object) -> ValueKind:
    """Name the kind of a live source object without importing the layer that defines it.

    Order matters: a store answers ``list`` too, so the store test has to come first or a
    store would classify as a catalog.
    """
    if hasattr(value, "get") and hasattr(value, "history"):
        return ValueKind.REX  # an RCDB store, addressed as the complex source it serves
    if hasattr(value, "hash_all") and hasattr(value, "list"):
        return ValueKind.CATALOG_ENTRY_SET
    if hasattr(value, "reconstruct_at") and hasattr(value, "T"):
        return ValueKind.TEMPORAL_REX
    if hasattr(value, "betti") and hasattr(value, "nV"):
        return ValueKind.REX
    return ValueKind.UNKNOWN


@dataclass(frozen=True)
class Binding:
    """One resolved source: the live value, its policy, its schema and its state.

    ``ref`` and ``temporal`` travel into every result type inferred from this binding, so
    a value can always say which source and which state it came from.
    """

    name: str
    source: BoundSource
    schema: SourceSchema
    ref: SourceRef
    temporal: TemporalRef | None = None

    @property
    def value(self) -> object:
        return self.source.value


def bind(name: str, value: object, policy: SourcePolicy, *,
         temporal: TemporalRef | None = None) -> Binding:
    """Resolve one named source into a Binding, classifying it from its surface."""
    bound = BoundSource(value, policy)
    kind = classify(value)
    surface = frozenset(
        attribute for attribute in
        ("get", "history", "stats", "list", "search", "hash", "hash_all", "info",
         "tensors", "search_tensors", "commit_history", "verify_commits",
         "security_status", "state_digest")
        if hasattr(value, attribute)
    )
    schema = SourceSchema(kind=kind, capabilities=frozenset(policy.permissions),
                          surface=surface)
    return Binding(
        name=name, source=bound, schema=schema,
        ref=SourceRef(name=name, policy_digest=policy.digest),
        temporal=temporal,
    )


def resolve(binding: Binding, operator: str, args: tuple[object, ...] = ()) -> OperatorSignature:
    """Check a call against its signature and the binding, before anything executes.

    Refusals are ordered so the most fundamental answer comes first: an operator that
    cannot run at all, then one pointed at the wrong kind of source, then a capability the
    policy withholds, then arity and argument kinds. Reporting a missing capability for a
    call that could never have worked would be misleading.
    """
    signature = lookup(operator)

    if signature.unreachable:
        raise UnreachableOperator(f"{signature.name} is not executable: {signature.unreachable}")

    source_kinds = (
        signature.source_kind
        if isinstance(signature.source_kind, tuple)
        else (signature.source_kind,)
    )
    if ValueKind.UNKNOWN not in source_kinds and binding.schema.kind not in source_kinds:
        wanted = " or ".join(kind.value for kind in source_kinds)
        raise SourceKindError(
            f"{signature.name} reads a {wanted} source, but "
            f"{binding.name!r} is a {binding.schema.kind.value} source"
        )

    missing = sorted(signature.requires - binding.schema.capabilities)
    if missing and "*" not in binding.schema.capabilities:
        raise PermissionError(
            f"{signature.name} requires {missing} which source {binding.name!r} does not grant"
        )

    signature.check_inputs(args, source=binding.ref)
    return signature
