"""Exact tensor fields with retained axes and explicit source identities."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import prod
from numbers import Integral
import numpy as np

from rexgraph.coordinate_map import _identity, _is_action
from rexgraph.graded_metric import _fraction
from rexgraph.type_accession import CoordinateSpace

__all__ = ["FieldSource", "TensorField", "TensorChannels", "apply_tensor"]


@dataclass(frozen=True, eq=False)
class FieldSource:
    """A native state reference, with optional live binding for validation."""
    source: object = field(default=None, repr=False)
    record_id: str | None = None
    version: int | None = None
    state_digest: str | None = None

    def __post_init__(self):
        if self.record_id is not None and (not isinstance(self.record_id, str) or not self.record_id):
            raise ValueError("record identity must be a nonempty string")
        if self.version is not None and (isinstance(self.version, bool) or
                not isinstance(self.version, Integral) or self.version < 0):
            raise ValueError("record version must be a nonnegative integer")
        digest = self.state_digest
        if self.source is not None:
            from rexgraph.io.rex_state import to_state
            from rexgraph.io.catalog import state_object_digest
            current = state_object_digest(to_state(self.source))
            if digest is not None and digest != current:
                raise ValueError("source does not match the declared native state identity")
            digest = current
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("a detached source requires its native state digest")
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError("source digest is not hexadecimal") from exc
        object.__setattr__(self, "state_digest", digest)
        if self.version is not None:
            object.__setattr__(self, "version", int(self.version))

    @property
    def coefficient_digest(self):
        return _identity(("field_source_v1", self.record_id, self.version, self.state_digest))

    def check(self):
        if self.source is None:
            return
        from rexgraph.io.rex_state import to_state
        from rexgraph.io.catalog import state_object_digest
        if state_object_digest(to_state(self.source)) != self.state_digest:
            raise ValueError("native source changed; select a fresh state")

    def bind(self, source):
        return FieldSource(source, self.record_id, self.version, self.state_digest)

    def matches(self, other):
        return isinstance(other, FieldSource) and self.coefficient_digest == other.coefficient_digest

    def as_record(self):
        return {"record_id": self.record_id, "version": self.version, "state_digest": self.state_digest}


@dataclass(frozen=True, eq=False)
class TensorField:
    """Rational coefficients on one action axis and explicitly retained other axes."""
    space: CoordinateSpace
    values: object
    axes: tuple[CoordinateSpace, ...] = ()
    source: FieldSource | None = None
    grade: int | None = None
    variance: str = "coordinate"
    provenance: tuple = ()
    dependencies: tuple[FieldSource, ...] = ()

    def __post_init__(self):
        if not isinstance(self.space, CoordinateSpace):
            raise TypeError("tensor field requires a named action axis")
        axes = tuple(self.axes)
        if any(not isinstance(a, CoordinateSpace) for a in axes):
            raise TypeError("each retained axis requires a CoordinateSpace")
        if len({a.name for a in (self.space, *axes)}) != len(axes)+1:
            raise ValueError("tensor axis names must be distinct")
        if self.source is not None and not isinstance(self.source, FieldSource):
            raise TypeError("tensor source must be a FieldSource")
        if self.grade is not None and (isinstance(self.grade, bool) or
                not isinstance(self.grade, Integral) or self.grade < 0):
            raise ValueError("tensor grade must be a nonnegative integer")
        if self.variance not in {"chain", "cochain", "coordinate"}:
            raise ValueError("unknown tensor variance")
        shape = (len(self.space.keys), *(len(a.keys) for a in axes))
        values = np.asarray(self.values, dtype=object)
        if values.shape != shape:
            raise ValueError("tensor values must match every declared axis")
        values = np.asarray([_fraction(v) for v in values.flat], dtype=object).reshape(shape)
        values.flags.writeable = False
        provenance = tuple(self.provenance)
        if any(not isinstance(v, str) for v in provenance):
            raise TypeError("tensor provenance requires digest strings")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(v, FieldSource) for v in dependencies):
            raise TypeError("tensor dependencies must be native FieldSources")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "provenance", provenance)
        if self.grade is not None:
            object.__setattr__(self, "grade", int(self.grade))

    @property
    def coefficient_digest(self):
        return _identity(("tensor_field_v1", (self.space.name, self.space.keys),
                          tuple((a.name, a.keys) for a in self.axes),
                          tuple((hex(v.numerator), hex(v.denominator)) for v in self.values.flat),
                          None if self.source is None else self.source.coefficient_digest,
                          self.grade, self.variance, self.provenance,
                          tuple(s.coefficient_digest for s in self.dependencies)))

    def check_state(self):
        if self.source is not None:
            self.source.check()
        for dependency in self.dependencies:
            dependency.check()

    def with_values(self, values, *, provenance=()):
        return replace(self, values=values, provenance=(*self.provenance, *provenance))

    def bind(self, source):
        if self.source is None:
            raise ValueError("unscoped field has no state identity to bind")
        return replace(self, source=self.source.bind(source))

    def bind_dependencies(self, sources):
        dependencies = tuple(s.bind(sources[s.state_digest]) for s in self.dependencies)
        source = self.source
        if source is not None:
            source = source.bind(sources[source.state_digest])
        return replace(self, source=source, dependencies=dependencies)

    def rate(self, interval):
        """Divide a field change by an explicit nonzero rational interval."""
        interval = _fraction(interval)
        if not interval:
            raise ValueError("a finite rate requires a nonzero interval")
        return self.with_values(self.values / interval,
                                provenance=(_identity(("finite_rate_v1", str(interval))),))

    def select(self, axis, key):
        names = [a.name for a in self.axes]
        if axis not in names:
            raise KeyError(axis)
        i = names.index(axis)
        j = self.axes[i].keys.index(key)
        return replace(self, values=np.take(self.values, j, axis=i+1), axes=self.axes[:i]+self.axes[i+1:])

    def contract_axis(self, axis, weights):
        names = [a.name for a in self.axes]
        if axis not in names:
            raise KeyError(axis)
        i = names.index(axis)
        weights = tuple(_fraction(w) for w in weights)
        if len(weights) != len(self.axes[i].keys):
            raise ValueError("contraction requires one explicit weight per coordinate")
        shape = self.values.shape[:i+1]+self.values.shape[i+2:]
        values = np.full(shape, Fraction(0), dtype=object)
        for j, weight in enumerate(weights):
            values += weight*np.take(self.values, j, axis=i+1)
        return replace(self, values=values, axes=self.axes[:i]+self.axes[i+1:])

    def as_record(self):
        return {"space": {"name": self.space.name, "keys": self.space.keys},
                "axes": tuple({"name": a.name, "keys": a.keys} for a in self.axes),
                "values": self.values.tolist(), "shape": self.values.shape,
                "source": None if self.source is None else self.source.as_record(),
                "grade": self.grade, "variance": self.variance,
                "provenance": self.provenance, "dependencies": tuple(s.as_record() for s in self.dependencies),
                "coefficient_domain": "Q"}


def _native_sources(action):
    from rexgraph.native_field import FieldAction, NativeAction, ActionSum, BilinearFormAction
    from rexgraph.coordinate_map import CoordinateWord, CoordinateDifference
    from rexgraph.attachment_field import AttachmentAction
    if isinstance(action, AttachmentAction):
        ref = action.observation.field.source
        return () if ref is None or ref.source is None else (ref.source,)
    if isinstance(action, BilinearFormAction):
        return _native_sources(action.action)
    if isinstance(action, FieldAction):
        return () if action.owner.source is None else (action.owner.source,)
    if isinstance(action, NativeAction):
        return (action.complex.source,)
    if isinstance(action, CoordinateWord):
        return tuple(s for a in action.factors for s in _native_sources(a))
    if isinstance(action, CoordinateDifference):
        return (*_native_sources(action.left), *_native_sources(action.right))
    if isinstance(action, ActionSum):
        return tuple(s for _, a in action.terms for s in _native_sources(a))
    return ()


def apply_tensor(action, tensor, *, source=None):
    """Apply on the named action axis without contracting the retained field axes."""
    if not _is_action(action) or not isinstance(tensor, TensorField):
        raise TypeError("tensor application requires an exact action and TensorField")
    if tensor.space != action.domain:
        raise ValueError("tensor action domain differs from the named field axis")
    tensor.check_state()
    sources = _native_sources(action)
    references = (*tensor.dependencies, *((tensor.source,) if tensor.source is not None else ()))
    if any(not any(s is ref.source for ref in references) for s in sources):
        raise ValueError("native action and tensor require the declared live source bindings")
    expected_variance = getattr(action, "domain_variance", None)
    if expected_variance is not None and tensor.variance != expected_variance:
        raise ValueError("tensor variance differs from the declared action domain")
    domain_grade = getattr(action, "domain_grade", None)
    if domain_grade is not None and tensor.grade is not None and domain_grade != tensor.grade:
        raise ValueError("native action and tensor grades disagree")
    shape = (len(tensor.space.keys), prod(len(a.keys) for a in tensor.axes))
    values = action.apply(tensor.values.reshape(shape)).reshape(
        (len(action.codomain.keys), *tensor.values.shape[1:]))
    result_source = tensor.source if source is None else source
    if result_source is not None:
        result_source.check()
    return TensorField(action.codomain, values, tensor.axes, result_source,
                       getattr(action, "codomain_grade", tensor.grade if action.domain == action.codomain else None),
                       getattr(action, "codomain_variance", tensor.variance),
                       (*tensor.provenance, action.coefficient_digest), tensor.dependencies)


@dataclass(frozen=True)
class TensorChannels:
    """Named fields with explicit support, field axes and endpoint identities."""
    names: tuple[str, ...]
    fields: tuple[TensorField, ...]
    declaration_digest: str | None = None
    endpoint_sources: tuple = ()

    def __post_init__(self):
        names, fields = tuple(self.names), tuple(self.fields)
        if (not fields or len(names) != len(fields) or len(set(names)) != len(names)
                or any(not isinstance(n, str) or not n for n in names)
                or any(not isinstance(f, TensorField) for f in fields)):
            raise ValueError("channels require unique names and tensor fields")
        sources = tuple(self.endpoint_sources)
        if any(not isinstance(s, FieldSource) for s in sources):
            raise TypeError("channel endpoints must be FieldSources")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "endpoint_sources", sources)

    def field(self, name):
        return self.fields[self.names.index(name)]

    def total(self, weights=None):
        first = self.fields[0]
        for f in self.fields:
            f.check_state()
            if (f.space, f.axes, f.grade, f.variance) != (first.space, first.axes, first.grade, first.variance):
                raise ValueError("channel summation requires identical declared target axes")
            if (f.source is None) != (first.source is None) or (
                    f.source is not None and not f.source.matches(first.source)):
                raise ValueError("channel summation requires one target source state")
        weights = (Fraction(1),)*len(self.fields) if weights is None else tuple(_fraction(v) for v in weights)
        if len(weights) != len(self.fields):
            raise ValueError("one explicit weight is required per channel")
        values = np.full(first.values.shape, Fraction(0), dtype=object)
        for weight, f in zip(weights, self.fields, strict=True):
            values += weight*f.values
        dependencies = {s.coefficient_digest: s for f in self.fields for s in f.dependencies}
        dependencies.update({s.coefficient_digest: s for s in self.endpoint_sources})
        return replace(first.with_values(values, provenance=(() if self.declaration_digest is None else (self.declaration_digest,))),
                       dependencies=tuple(dependencies.values()))

    def as_record(self):
        return {"names": self.names, "fields": tuple(f.as_record() for f in self.fields),
                "declaration_digest": self.declaration_digest,
                "endpoint_sources": tuple(s.as_record() for s in self.endpoint_sources),
                "coefficient_domain": "Q"}
