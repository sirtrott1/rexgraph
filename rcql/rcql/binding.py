"""Resolve sources and classify them before any operator runs.

The existing binder hands the executor a live object and a policy. That is enough to run
an operator and not enough to type one: nothing says what KIND of source it is, so a
signature cannot check that ``RCDB_GET`` received a store rather than a catalog, and
``STATE_HASH`` cannot be stopped from being handed the catalog it is filed beside.

This module answers that question once, at binding time, from the object's own surface
rather than from its class name. A store is anything that can answer ``get`` and
``history``; a catalog is anything that can ``list`` and ``hash``. Classifying by surface
keeps RCQL from importing rcdb or the catalog module to do an isinstance check, which is
the layering rule the distributions exist to hold. TurnField is a nominal Core
source because its snapshot and preview methods carry a specific conversation
contract; arbitrary objects with those common method names do not qualify.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from inspect import getattr_static

from .capabilities import BoundSource, SourcePolicy
from .signatures import OperatorSignature, lookup
from .types import SourceRef, TemporalRef, ValueKind


class SourceKindError(TypeError):
    """The bound source is not the kind of thing this operator reads."""


class UnreachableOperator(TypeError):
    """The operator is registered but cannot execute against any available source."""


def _has_method(value, name):
    member = getattr_static(value, name, None)
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    return callable(member)


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
    def has(name):
        return getattr_static(value, name, None) is not None

    if any((cls.__module__, cls.__name__) == ("rexgraph.flow.turn_field", "TurnField")
           for cls in type(value).__mro__):
        return ValueKind.TURN_FIELD
    if has("get") and has("history"):
        return ValueKind.RCDB_STORE
    if has("hash_all") and has("list"):
        return ValueKind.CATALOG_ENTRY_SET
    if has("reconstruct_at") and has("T"):
        return ValueKind.TEMPORAL_REX
    if has("betti") and has("nV"):
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
         temporal: TemporalRef | None = None,
         source_ref: SourceRef | None = None) -> Binding:
    """Resolve one named source into a Binding, classifying it from its surface."""
    bound = BoundSource(value, policy)
    kind = classify(value)
    surface = frozenset(
        attribute for attribute in
        ("get", "history", "stats", "list", "search", "query", "hash", "hash_all", "info",
         "tensors", "search_tensors", "commit_history", "verify_commits",
         "security_status", "state_digest", "read_record", "commit_mutation", "corpus_snapshot")
        if _has_method(value, attribute)
    )
    schema = SourceSchema(kind=kind, capabilities=frozenset(policy.permissions),
                          surface=surface)
    ref = source_ref or SourceRef(name=name, policy_digest=policy.digest)
    if ref.policy_digest != policy.digest:
        ref = replace(ref, policy_digest=policy.digest)
    # Native state identity is canonical input serialization, not an analytic read.
    # Never reconstruct a whole temporal history or inspect a denied source here.
    if (ref.state_digest is None and policy.permits("read")
            and any((cls.__module__, cls.__name__) == ("rexgraph.graph", "RexGraph")
                    for cls in type(value).__mro__)):
        from rexgraph.io.catalog import object_digest
        ref = replace(ref, state_digest=object_digest(value))
    return Binding(
        name=name, source=bound, schema=schema,
        ref=ref,
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
    missing_methods = signature.source_methods - binding.schema.surface
    if missing_methods:
        raise SourceKindError(
            f"{signature.name} requires source methods {sorted(missing_methods)}; "
            f"{binding.name!r} does not implement the RCDB data contract"
        )
    return signature
