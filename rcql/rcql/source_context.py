"""Pinned native sources and explicit historical evidence closure."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from collections.abc import Mapping
from types import MappingProxyType
import math

from .capabilities import BoundSource, SourcePolicy
from .types import SourceRef, TemporalRef


def field_references(value, seen=None):
    """Read declared native references without inferring dependencies from labels."""
    from rexgraph.tensor_field import FieldSource, _native_sources
    seen = set() if seen is None else seen
    if id(value) in seen:
        return ()
    seen.add(id(value))
    if isinstance(value, FieldSource):
        value.check()
        return (value,)
    if isinstance(value, BoundSource):
        ref = value.ref
        own = () if ref is None or ref.state_digest is None else (
            FieldSource(value.value, ref.record_id, ref.record_version, ref.state_digest),)
        return (*own, *field_references(value.value, seen))
    if isinstance(value, SourceRef):
        own = () if value.state_digest is None else (FieldSource(None, value.record_id,
                                                                 value.record_version, value.state_digest),)
        return (*own, *field_references(value.contributors, seen))
    if isinstance(value, Mapping):
        return tuple(ref for item in value.values() for ref in field_references(item, seen))
    if isinstance(value, (tuple, list)):
        return tuple(ref for item in value for ref in field_references(item, seen))
    if type(value).__module__ == "rexgraph.graph" and type(value).__name__ == "RexGraph":
        metadata = getattr(value, "_cell_metadata", None)
        evidence = getattr(value, "_agent_meta", {}).get("rcql_evidence", ())
        declared = []
        for row in evidence:
            if not isinstance(row, dict) or set(row) != {"record_id", "version", "state_digest"}:
                raise ValueError("invalid declared native evidence reference")
            declared.append(FieldSource(None, row["record_id"], row["version"], row["state_digest"]))
        return (*declared, *field_references(metadata, seen))
    if type(value).__module__.startswith(("rexgraph.", "rcql.")):
        refs = []
        if is_dataclass(value):
            for item in fields(value):
                child = getattr(value, item.name)
                if item.name == "source" and not isinstance(child, (FieldSource, SourceRef, BoundSource)):
                    continue
                refs.extend(field_references(child, seen))
        elif hasattr(value, "source"):
            child = value.source
            if isinstance(child, FieldSource):
                refs.extend(field_references(child, seen))
        refs.extend(FieldSource(source) for source in _native_sources(value)
                    if not any(ref.source is source for ref in refs))
        return tuple(refs)
    # Object arrays are part of the existing cell metadata carrier.
    if type(value).__module__.startswith("numpy") and getattr(getattr(value, "dtype", None), "kind", None) == "O":
        return tuple(ref for item in value.flat for ref in field_references(item, seen))
    return ()


def _key(ref):
    return ref.state_digest, ref.record_id, ref.version


def _clock(value):
    if type(value) is int or isinstance(value, Fraction):
        return Fraction(value)
    if type(value) is float and math.isfinite(value):
        return Fraction.from_float(value)
    raise TypeError("cutoff requires a finite recorded clock value")


@dataclass(frozen=True)
class SourceSelection:
    name: str
    record_id: str
    version: int | None = None
    as_of: int | float | Fraction | None = None
    valid_at: int | float | Fraction | None = None

    def __post_init__(self):
        if not self.name or not self.record_id or not isinstance(self.name, str) or not isinstance(self.record_id, str):
            raise ValueError("source selections require names and record identities")
        if self.version is not None and (type(self.version) is not int or self.version < 1):
            raise ValueError("source version must be positive")
        if self.version is not None and (self.as_of is not None or self.valid_at is not None):
            raise ValueError("an exact version and a time selector are distinct requests")


class SnapshotContext:
    """A pinned selection and its declared dependency closure from one store handle."""

    def __init__(self, sources, entries, cutoff=None):
        self.sources = MappingProxyType(dict(sources))
        self.entries = MappingProxyType(dict(entries))
        self.cutoff = cutoff

    @classmethod
    def select(cls, store, selections, *, cutoff=None):
        from rexgraph.tensor_field import FieldSource
        selections = tuple(selections)
        if not selections or any(not isinstance(s, SourceSelection) for s in selections):
            raise TypeError("select a nonempty family of declared source versions")
        if len({s.name for s in selections}) != len(selections):
            raise ValueError("source selection names must be unique")
        if isinstance(store, BoundSource):
            store.require("read"); store.require("identity")
            policy, raw = store.policy, store.value
        else:
            policy, raw = SourcePolicy.allow("*"), store
        lock = getattr(raw, "_transaction_lock", None)
        if lock is None or not callable(getattr(raw, "read_record", None)):
            raise TypeError("snapshot selection requires the native RCDB handle contract")
        limit = None if cutoff is None else _clock(cutoff)
        entries, sources = {}, {}

        def retain(snapshot):
            ref = FieldSource(snapshot.value, snapshot.record.id, snapshot.record.version, snapshot.state_digest)
            key = _key(ref)
            if key in entries:
                return entries[key][0]
            if limit is not None and _clock(float(snapshot.record.tx_from)) > limit:
                raise ValueError("selected evidence was recorded after the query cutoff")
            bound_ref = SourceRef("snapshot/"+ref.record_id+"@"+str(ref.version),
                                  state_digest=ref.state_digest, record_id=ref.record_id,
                                  record_version=ref.version, policy_digest=policy.digest)
            selected = BoundSource(snapshot.value, policy, ref=bound_ref, temporal=TemporalRef(version=ref.version))
            entries[key] = (selected, float(snapshot.record.tx_from))
            for parent in field_references(snapshot.value):
                if _key(parent) == key:
                    continue
                if parent.record_id is None or parent.version is None:
                    raise ValueError("evidence closure requires recorded identities for every dependency")
                dependency = raw.read_record(parent.record_id, version=parent.version)
                if dependency is None or dependency.state_digest != parent.state_digest:
                    raise ValueError("recorded evidence dependency is missing or changed")
                retain(dependency)
            return selected

        with lock:
            for selection in selections:
                as_of = selection.as_of
                if limit is not None and selection.version is None:
                    as_of = limit if as_of is None else min(_clock(as_of), limit)
                snap = raw.read_record(selection.record_id, version=selection.version,
                                       as_of=as_of, valid_at=selection.valid_at)
                if snap is None:
                    raise KeyError("no state satisfies the source selection")
                sources[selection.name] = retain(snap)
        return cls(sources, entries, limit)

    @property
    def digest(self):
        from .program_codec import digest
        return digest(("rcql.snapshot-context.v1", self.cutoff,
                       tuple(sorted((k, v[1], v[0].policy.digest) for k, v in self.entries.items())),
                       tuple(sorted((name, value.ref.record_id, value.ref.record_version, value.ref.state_digest)
                                    for name, value in self.sources.items()))))

    def as_record(self):
        return {"schema": "rcql.evidence-context", "version": 1, "digest": self.digest,
                "cutoff": None if self.cutoff is None else str(self.cutoff),
                "sources": tuple({"name": name, "record_id": bound.ref.record_id,
                                   "version": bound.ref.record_version, "state_digest": bound.ref.state_digest}
                                  for name, bound in self.sources.items()),
                "closure": tuple({"record_id": key[1], "version": key[2], "state_digest": key[0],
                                   "recorded_at": str(_clock(value[1]))}
                                  for key, value in self.entries.items())}

    def check(self):
        from rexgraph.tensor_field import FieldSource
        for (digest, record_id, version), (bound, recorded_at) in self.entries.items():
            FieldSource(bound.value, record_id, version, digest).check()
            if self.cutoff is not None and _clock(recorded_at) > self.cutoff:
                raise ValueError("evidence cutoff was violated")

    def validate_reference(self, reference):
        key = _key(reference)
        if key not in self.entries:
            # Unversioned references can name an action on an already pinned object.
            matches = [k for k, (b, _) in self.entries.items()
                       if reference.source is b.value and reference.state_digest == k[0]
                       and reference.record_id is None and reference.version is None]
            if len(matches) != 1:
                raise ValueError("query uses a dependency outside the pinned evidence closure")
            key = matches[0]
        bound = self.entries[key][0]
        reference.check()
        if reference.source is not None and reference.source is not bound.value:
            from rexgraph.tensor_field import FieldSource
            FieldSource(reference.source, key[1], key[2], key[0]).check()
        return bound

    def validate_binding(self, binding):
        from rexgraph.tensor_field import FieldSource
        reference = FieldSource(binding.value, binding.ref.record_id,
                                binding.ref.record_version, binding.ref.state_digest)
        permitted = self.validate_reference(reference)
        if SourcePolicy.intersection(binding.source.policy, permitted.policy).digest != binding.source.policy.digest:
            raise PermissionError("query policy exceeds the pinned source policy")

    def validate_values(self, values):
        self.check()
        for reference in field_references(values):
            self.validate_reference(reference)

    def bindings(self):
        from .binding import bind
        return tuple(bind(bound.ref.name, bound.value, bound.policy, source_ref=bound.ref)
                     for bound, _ in self.entries.values())
