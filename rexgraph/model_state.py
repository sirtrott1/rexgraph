"""Durable local model declarations and retained output coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from fractions import Fraction
from numbers import Integral
import hashlib
import json
import math
import numpy as np

from rexgraph.tensor_field import FieldSource, TensorField
from rexgraph.type_accession import CoordinateSpace

__all__ = ["ModelInput", "model_input", "ModelState", "ModelOutput", "ModelBatch", "ModelTimeline", "freeze_tree", "thaw_tree"]


def freeze_tree(value):
    """Copy supported data without retaining mutable caller storage."""
    if value is None or isinstance(value, (str, bytes, bool, Fraction)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("model metadata must be finite")
        return value
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True, order="C")
        if result.dtype.hasobject:
            from rexgraph.io.rex_state import _encode_exact
            _encode_exact(result)
        elif result.dtype.kind not in "biuf" or not result.dtype.isnative or result.dtype.itemsize not in {1, 2, 4, 8}:
            raise TypeError("unsupported model tensor dtype")
        elif result.dtype.kind == "f" and not np.isfinite(result).all():
            raise ValueError("model tensor must be finite")
        result.flags.writeable = False
        return result
    if isinstance(value, ModelInput):
        value.check_state()
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, (str, int)) or isinstance(k, bool) for k in value):
            raise TypeError("model mapping keys must be strings or integers")
        return MappingProxyType({k: freeze_tree(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze_tree(v) for v in value)
    raise TypeError(f"unsupported model value {type(value).__name__}")


def thaw_tree(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {k: thaw_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(thaw_tree(v) for v in value)
    return value


def _source(ref):
    if not isinstance(ref, FieldSource):
        raise TypeError("model requires an explicit native source reference")
    ref.check()
    return FieldSource(None, ref.record_id, ref.version, ref.state_digest)


def _sha(value, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("expected a SHA256 identity")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid SHA256 identity") from exc


def _digest(value):
    from rexgraph.io.model_state import model_payload
    from rexgraph.io.rex_state import state_digest, encode_tensors
    tensors = model_payload(value)
    codecs = encode_tensors(tensors)
    tensors["codec"] = np.frombuffer(json.dumps(codecs, sort_keys=True).encode(), dtype=np.uint8)
    return state_digest(tensors)


@dataclass(frozen=True, eq=False)
class ModelState:
    """Source bound parameters, observations and optional continuation state."""
    adapter: str
    adapter_version: str
    source: FieldSource
    space: CoordinateSpace
    output_axes: tuple[CoordinateSpace, ...]
    configuration: object
    payload: object
    arithmetic: str
    step: int = 0
    parent: str | None = None
    dependencies: tuple[FieldSource, ...] = ()
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.adapter, str) or not self.adapter or not isinstance(self.adapter_version, str) or not self.adapter_version:
            raise ValueError("model adapter and version must be declared")
        if not isinstance(self.space, CoordinateSpace):
            raise TypeError("model requires named primary coordinates")
        axes = tuple(self.output_axes)
        if any(not isinstance(v, CoordinateSpace) for v in axes):
            raise TypeError("model outputs require named axes")
        if len({v.name for v in (self.space, *axes)}) != len(axes) + 1:
            raise ValueError("model axis names must be distinct")
        if self.arithmetic not in {"rational", "approximate"}:
            raise ValueError("model arithmetic must be rational or approximate")
        if isinstance(self.step, bool) or not isinstance(self.step, Integral) or self.step < 0:
            raise ValueError("model step must be a nonnegative integer")
        _sha(self.parent, optional=True)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "dependencies", tuple(_source(v) for v in self.dependencies))
        object.__setattr__(self, "configuration", freeze_tree(self.configuration))
        object.__setattr__(self, "payload", freeze_tree(self.payload))
        if not isinstance(self.configuration, Mapping) or not isinstance(self.payload, Mapping):
            raise TypeError("model configuration and payload require mappings")
        object.__setattr__(self, "output_axes", axes)
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "_digest", _digest(self))

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        if _digest(self) != self._digest:
            raise ValueError("model state changed after construction")

    def bind(self, source):
        self.check_state()
        return self.source.bind(source)

    def child(self, *, payload, configuration=None, source=None, space=None, step=None, dependencies=None):
        return ModelState(self.adapter, self.adapter_version, self.source if source is None else source,
            self.space if space is None else space, self.output_axes,
            self.configuration if configuration is None else configuration, payload, self.arithmetic,
            self.step if step is None else step, self.coefficient_digest,
            self.dependencies if dependencies is None else dependencies)


@dataclass(frozen=True, eq=False)
class ModelOutput:
    """Numerical or exact values with their model and source identities."""
    values: object
    space: CoordinateSpace
    axes: tuple[CoordinateSpace, ...]
    source: FieldSource
    model_digest: str
    arithmetic: str
    method: str
    dependencies: tuple[FieldSource, ...] = ()
    grade: int | None = None
    variance: str = "coordinate"
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        _sha(self.model_digest)
        if self.grade is not None and (isinstance(self.grade, bool) or not isinstance(self.grade, Integral) or self.grade < 0):
            raise ValueError("output grade must be a nonnegative integer")
        if self.variance not in {"coordinate", "chain", "cochain"}:
            raise ValueError("invalid output variance")
        if not isinstance(self.space, CoordinateSpace) or any(not isinstance(a, CoordinateSpace) for a in self.axes):
            raise TypeError("model output requires named coordinates")
        axes = tuple(self.axes)
        if len({a.name for a in (self.space, *axes)}) != len(axes) + 1:
            raise ValueError("output axis names must be distinct")
        if self.arithmetic not in {"rational", "approximate"}:
            raise ValueError("unknown output arithmetic")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("model output requires its executed method")
        values = np.asarray(self.values, dtype=object if self.arithmetic == "rational" else None)
        if values.shape != (len(self.space.keys), *(len(a.keys) for a in axes)):
            raise ValueError("output values differ from declared axes")
        if self.arithmetic == "rational":
            from rexgraph.graded_metric import _fraction
            values = np.array([_fraction(v) for v in values.flat], dtype=object).reshape(values.shape)
        elif values.dtype.kind != "f":
            raise TypeError("numerical model output requires floating coefficients")
        object.__setattr__(self, "values", freeze_tree(values))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "dependencies", tuple(_source(v) for v in self.dependencies))
        object.__setattr__(self, "_digest", _digest(self))

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        if _digest(self) != self._digest:
            raise ValueError("model output changed")

    def tensor(self):
        """Expose rational output without changing numerical predictions into exact claims."""
        self.check_state()
        if self.arithmetic != "rational":
            raise TypeError("numerical model output is not an exact tensor field")
        return TensorField(self.space, self.values, self.axes, self.source, self.grade, self.variance,
            provenance=(self.model_digest, self.coefficient_digest), dependencies=self.dependencies)

    def recorded_binary(self):
        """Represent the recorded finite binary values as rationals, not exact inference."""
        self.check_state()
        if self.arithmetic == "rational":
            return self.tensor()
        values = np.array([Fraction.from_float(float(v)) for v in self.values.flat], dtype=object).reshape(self.values.shape)
        marker = hashlib.sha256(b"recorded_binary_model_output_not_exact_training").hexdigest()
        return TensorField(self.space, values, self.axes, self.source, self.grade, self.variance,
            provenance=(self.model_digest, self.coefficient_digest, marker), dependencies=self.dependencies)


@dataclass(frozen=True, eq=False)
class ModelInput:
    """Declared numerical conversion with the original field and coordinate identity."""
    original: object
    values: object
    space: CoordinateSpace
    axes: tuple[CoordinateSpace, ...]
    source: FieldSource
    origin_digest: str
    origin_arithmetic: str
    dependencies: tuple[FieldSource, ...] = ()
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        _sha(self.origin_digest)
        if self.origin_arithmetic not in {"rational", "approximate"}:
            raise ValueError("input requires the original arithmetic contract")
        if not isinstance(self.space, CoordinateSpace) or any(not isinstance(a, CoordinateSpace) for a in self.axes):
            raise TypeError("model input requires named axes")
        axes = tuple(self.axes)
        if len({a.name for a in (self.space, *axes)}) != len(axes) + 1:
            raise ValueError("model input axes must have distinct names")
        original = freeze_tree(np.asarray(self.original, dtype=object if self.origin_arithmetic == "rational" else None))
        values = freeze_tree(np.asarray(self.values))
        expected = (len(self.space.keys), *(len(a.keys) for a in axes))
        if original.shape != expected or values.shape != expected or values.dtype.kind != "f":
            raise ValueError("model input conversion must preserve every named axis")
        if self.origin_arithmetic == "rational":
            from rexgraph.graded_metric import _fraction
            original = freeze_tree(np.array([_fraction(v) for v in original.flat], dtype=object).reshape(expected))
        if np.asarray(original, dtype=values.dtype).tobytes() != values.tobytes():
            raise ValueError("numerical input differs from its declared conversion")
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "dependencies", tuple(_source(v) for v in self.dependencies))
        object.__setattr__(self, "_digest", _digest(self))

    @property
    def arithmetic(self):
        return "rounded" if self.origin_arithmetic == "rational" else "approximate"

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        if _digest(self) != self._digest:
            raise ValueError("model input changed")


def model_input(value, dtype="float64"):
    """Convert a retained field for numerical evaluation without discarding its source."""
    if not isinstance(value, (TensorField, ModelOutput)):
        raise TypeError("numerical model conversion requires TensorField or ModelOutput")
    value.check_state()
    if value.source is None:
        raise ValueError("numerical model input requires a source reference")
    dt = np.dtype(dtype)
    if dt not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError("model conversion requires float16, float32 or float64")
    arithmetic = "rational" if isinstance(value, TensorField) else value.arithmetic
    return ModelInput(value.values, np.asarray(value.values, dtype=dt), value.space, value.axes,
        value.source, value.coefficient_digest, arithmetic, value.dependencies)


@dataclass(frozen=True, eq=False)
class ModelBatch:
    """Observed targets and optional inputs on declared source coordinates."""
    source: FieldSource
    space: CoordinateSpace
    targets: object
    observed: object
    inputs: object = None
    dependencies: tuple[FieldSource, ...] = ()
    target_input: ModelInput | None = None
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.space, CoordinateSpace):
            raise TypeError("batch requires named coordinates")
        declared_target = self.targets if isinstance(self.targets, ModelInput) else self.target_input
        target = np.asarray(self.targets.values if isinstance(self.targets, ModelInput) else self.targets)
        if declared_target is not None:
            if not isinstance(declared_target, ModelInput):
                raise TypeError("target conversion must be ModelInput")
            declared_target.check_state()
            if declared_target.space != self.space or not declared_target.source.matches(self.source):
                raise ValueError("target conversion source or coordinates differ")
            if target.dtype != declared_target.values.dtype or target.tobytes() != declared_target.values.tobytes():
                raise ValueError("target values differ from their declared conversion")
        object.__setattr__(self, "target_input", declared_target)
        mask = np.asarray(self.observed)
        if target.ndim < 1 or target.shape[0] != len(self.space.keys):
            raise ValueError("target rows must match source coordinates")
        if mask.dtype != bool or mask.shape != (len(self.space.keys),):
            raise ValueError("observation mask must be boolean and match rows")
        object.__setattr__(self, "targets", freeze_tree(target))
        object.__setattr__(self, "observed", freeze_tree(mask))
        prepared = freeze_tree(self.inputs)
        def validate(v):
            if isinstance(v, ModelInput):
                if v.space != self.space or not v.source.matches(self.source):
                    raise ValueError("batch input source or coordinates differ")
            elif isinstance(v, Mapping):
                for x in v.values(): validate(x)
            elif isinstance(v, tuple):
                for x in v: validate(x)
        validate(prepared)
        object.__setattr__(self, "inputs", prepared)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "dependencies", tuple(_source(v) for v in self.dependencies))
        object.__setattr__(self, "_digest", _digest(self))

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        if _digest(self) != self._digest:
            raise ValueError("model batch changed")


@dataclass(frozen=True, eq=False)
class ModelTimeline:
    """Exact observation times and model references beside a verified temporal index."""
    times: tuple[Fraction, ...]
    model_digests: tuple[str, ...]
    source_digests: tuple[str, ...]
    temporal_header: object
    temporal_tensors: object
    time_axis: str = "model_observation"
    time_unit: str = "step"
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        from rexgraph.graded_metric import _fraction
        from rexgraph.io.temporal_state import TemporalState, verify_temporal_state
        if any(not isinstance(v, str) or not v for v in (self.time_axis, self.time_unit)):
            raise ValueError("model history requires a named time axis and unit")
        times = tuple(_fraction(v) for v in self.times)
        if not times or any(a >= b for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("model history requires strictly increasing exact times")
        if len(times) != len(self.model_digests) or len(times) != len(self.source_digests):
            raise ValueError("history identities and times disagree")
        for digest in (*self.model_digests, *self.source_digests):
            _sha(digest)
        temporal = TemporalState(dict(self.temporal_tensors), dict(self.temporal_header))
        if not verify_temporal_state(temporal) or temporal.header["T"] != len(times):
            raise ValueError("model timeline requires a verified matching TemporalState")
        if temporal.header["times"] != [float(i) for i in range(len(times))]:
            raise ValueError("model temporal index uses ordinal steps, not rounded event times")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "model_digests", tuple(self.model_digests))
        object.__setattr__(self, "source_digests", tuple(self.source_digests))
        object.__setattr__(self, "temporal_header", freeze_tree(self.temporal_header))
        object.__setattr__(self, "temporal_tensors", freeze_tree(self.temporal_tensors))
        object.__setattr__(self, "_digest", _digest(self))

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        if _digest(self) != self._digest:
            raise ValueError("model timeline changed")

    def at(self, time):
        from bisect import bisect_right
        from rexgraph.graded_metric import _fraction
        index = bisect_right(self.times, _fraction(time)) - 1
        if index < 0:
            raise KeyError("no recorded model exists at this time")
        return index

    def temporal_state(self):
        from rexgraph.io.temporal_state import TemporalState
        self.check_state()
        # TemporalState uses JSON lists while immutable model declarations use tuples.
        def lists(v):
            if isinstance(v, Mapping): return {k: lists(x) for k, x in v.items()}
            if isinstance(v, tuple): return [lists(x) for x in v]
            return v
        return TemporalState(thaw_tree(self.temporal_tensors), lists(self.temporal_header))
