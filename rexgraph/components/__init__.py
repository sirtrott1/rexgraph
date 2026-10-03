"""Registered tensor ownership, native codecs and carried state transport."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from types import MappingProxyType
from typing import Callable

import numpy as np

__all__ = ["ComponentCodec", "ComponentDescriptor", "ComponentPayload", "DecodedComponent", "ComponentRegistry", "component_registry", "CellMaps"]


@dataclass(frozen=True)
class ComponentPayload:
    """Logical tensors and semantic header fields emitted by one owner."""
    tensors: Mapping = field(default_factory=dict)
    header: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.tensors, Mapping) or not isinstance(self.header, Mapping):
            raise TypeError("component payload requires tensor and header mappings")
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
        object.__setattr__(self, "header", MappingProxyType(dict(self.header)))


@dataclass(frozen=True)
class DecodedComponent:
    """Constructor arguments and an optional restoration hook for one owner."""
    arguments: Mapping = field(default_factory=dict)
    restore: Callable | None = None

    def __post_init__(self):
        if not isinstance(self.arguments, Mapping) or (self.restore is not None and not callable(self.restore)):
            raise TypeError("invalid decoded component")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@dataclass(frozen=True)
class ComponentDescriptor:
    name: str
    version: int
    tensors: tuple[str, ...]
    virtual_tensors: tuple[str, ...]
    exactness: str
    absence: str

    def as_record(self):
        return {"name": self.name, "version": self.version, "tensors": list(self.tensors),
                "virtual_tensors": list(self.virtual_tensors), "exactness": self.exactness, "absence": self.absence}


@dataclass(frozen=True)
class ComponentCodec:
    """Trusted hooks for one versioned semantic owner.

    Encoders return ComponentPayload; decoders return DecodedComponent with
    constructor arguments or a restoration callback. Header fields and tensor
    claims must match emitted ownership. Registration never executes wire code.
    Capture/remap hooks are additionally required for carried state edits.
    """
    name: str
    version: int
    claim: Callable
    exactness: str = "declared"
    absence: str = "retained"
    capture: Callable | None = None
    remap: Callable | None = None
    encode: Callable | None = None
    decode: Callable | None = None
    header_fields: tuple[str, ...] = ()

    def claims(self, header, names):
        result = frozenset(self.claim(header, names))
        if any(type(name) is not str or not name for name in result) or not result <= names:
            raise ValueError(f"invalid claims from component {self.name!r}")
        return result


class ComponentRegistry:
    def __init__(self):
        self._codecs = {}

    def register(self, codec: ComponentCodec):
        if (not isinstance(codec, ComponentCodec) or not codec.name or type(codec.version) is not int
                or codec.version < 1 or not callable(codec.claim)):
            raise TypeError("invalid component codec")
        if codec.name in self._codecs:
            raise ValueError(f"component {codec.name!r} is already registered")
        if any(hook is not None and not callable(hook) for hook in (codec.capture, codec.remap, codec.encode, codec.decode)):
            raise TypeError("component hooks must be callable")
        if (not isinstance(codec.header_fields, tuple) or any(type(key) is not str for key in codec.header_fields)
                or len(set(codec.header_fields)) != len(codec.header_fields)):
            raise TypeError("component header fields must be distinct names")
        self._codecs[codec.name] = codec

    def _ordered(self):
        return [name for name in self._codecs if name != "relations"] + (["relations"] if "relations" in self._codecs else [])

    def _state_contract(self):
        for name, codec in self._codecs.items():
            if codec.encode is None or codec.decode is None:
                raise ValueError(f"component {name!r} has no native encode/decode contract")

    def encode(self, rex):
        """Run registered encoders once and refuse conflicting or misowned output."""
        from .codecs import EncodeContext
        self._state_contract()
        context = EncodeContext(rex)
        tensors, header, produced = {}, {}, {}
        for name in self._ordered():
            codec = self._codecs[name]
            payload = codec.encode(rex, context)
            if not isinstance(payload, ComponentPayload):
                raise TypeError(f"component {name!r} did not return a ComponentPayload")
            if set(payload.header) - set(codec.header_fields):
                raise ValueError(f"component {name!r} emitted unclaimed header fields")
            if set(tensors) & set(payload.tensors) or set(header) & set(payload.header):
                raise ValueError(f"component {name!r} emitted conflicting state fields")
            if any(type(key) is not str or not key or not isinstance(value, np.ndarray)
                   for key, value in payload.tensors.items()):
                raise TypeError("component tensors require named NumPy arrays")
            produced[name] = frozenset(payload.tensors)
            tensors.update(payload.tensors)
            header.update(payload.header)
        for name, codec in self._codecs.items():
            if codec.claims(header, frozenset(tensors)) != produced[name]:
                raise ValueError(f"component {name!r} output differs from its tensor ownership")
        return tensors, header

    def decode(self, tensors, header):
        """Decode logical payloads, construct once, then restore each owned carrier."""
        from .codecs import DecodeContext
        from rexgraph.graph import RexGraph
        self._state_contract()
        self.describe(header, tensors, {})
        context = DecodeContext(header)
        arguments, restored = {}, []
        names = frozenset(tensors)
        for name in self._ordered():
            codec = self._codecs[name]
            payload = ComponentPayload({key: tensors[key] for key in codec.claims(header, names)},
                                       {key: header[key] for key in codec.header_fields if key in header})
            decoded = codec.decode(payload, context)
            if not isinstance(decoded, DecodedComponent):
                raise TypeError(f"component {name!r} did not return a DecodedComponent")
            if set(arguments) & set(decoded.arguments):
                raise ValueError(f"component {name!r} emitted conflicting constructor arguments")
            arguments.update(decoded.arguments)
            if decoded.restore is not None:
                restored.append(decoded.restore)
        rex = RexGraph(**arguments)
        for restore in restored:
            restore(rex)
        return rex

    def _owners(self, components):
        owners = set(self._codecs) if components is None else set(components)
        if owners - self._codecs.keys():
            raise ValueError("unknown component transport owners")
        return owners

    def capture(self, rex, *, components=None):
        """Capture owned carriers before editing their cell bases; never read caches."""
        result = {}
        owners = self._owners(components)
        for name, codec in self._codecs.items():
            if name not in owners:
                continue
            if codec.capture is None or codec.remap is None:
                raise ValueError(f"component {name!r} has no carried-state transport contract")
            result[name] = codec.capture(rex)
        return result

    def remap(self, captured, rex, maps, *, mode="restrict", components=None):
        """Apply every registered owner through checked old to new cell maps.

        Primary support is built by the native edit/constructor. Its registered
        hook transports the upper tower and extends the maps before other owners.
        Append mode retains the newly staged primary attribution.
        """
        owners = self._owners(components)
        if set(captured) != owners:
            raise ValueError("captured components differ from the registered transport owners")
        if mode not in {"restrict", "append", "structural"}:
            raise ValueError("unknown component transport mode")
        checked = maps if isinstance(maps, CellMaps) else CellMaps(maps)
        if mode != "append":
            for grade, size in enumerate((rex._nV, rex._nE, rex._nF)):
                checked.take(grade, size)
        elif any(not np.array_equal(checked[grade], np.arange(len(checked[grade]))) for grade in (0, 1, 2)):
            raise ValueError("append must preserve every existing cell index")
        rex._cell_metadata = {}
        # Relations reads the newly transported identity/attributes/embedding.
        for name in self._ordered():
            if name not in owners:
                continue
            codec = self._codecs[name]
            codec.remap(captured[name], rex, checked, mode)
        return checked

    def describe(self, header, tensors, tensor_codecs):
        stored = frozenset(tensors)
        logical = stored | frozenset(tensor_codecs)
        owners, result = {}, []
        for name, codec in sorted(self._codecs.items()):
            claims = codec.claims(header, logical)
            for tensor in claims:
                if tensor in owners:
                    raise ValueError(f"duplicate component claim for {tensor!r}: {owners[tensor]!r} and {name!r}")
                owners[tensor] = name
            if claims:
                result.append(ComponentDescriptor(name, codec.version, tuple(sorted(claims & stored)),
                                                  tuple(sorted(claims - stored)), codec.exactness, codec.absence))
        unknown = logical - owners.keys()
        if unknown:
            raise ValueError(f"unclaimed state tensors: {sorted(unknown)!r}")
        return tuple(result)

    def validate(self, descriptors, header, tensors, tensor_codecs):
        if not isinstance(descriptors, list):
            raise ValueError("component descriptors must be a list")
        for record in descriptors:
            if not isinstance(record, dict):
                raise ValueError("invalid component descriptor")
            codec = self._codecs.get(record.get("name"))
            if codec is None or type(record.get("version")) is not int or record["version"] != codec.version:
                raise ValueError("unknown component or component version")
        expected = [d.as_record() for d in self.describe(header, tensors, tensor_codecs)]
        if descriptors != expected:
            raise ValueError("component schema or tensor ownership differs from its registry")


def _fixed(*names):
    return lambda header, available: set(names) & available


def _support(header, names):
    expected = {"boundary_ptr", "boundary_idx", "B2_col_ptr", "B2_row_idx", "B2_vals"}
    for g in range(header.get("n_graded_duals", 0)):
        expected.update(f"gd{g}_{suffix}" for suffix in ("indptr", "indices", "data"))
    return names & expected


def _attributes(header, names):
    expected = set()
    for col in header.get("cell_meta", []):
        prefix = f"cm_{col['dim']}_{col['key']}"
        if col["kind"] != "rex":
            expected.add(prefix + "_idx")
        if col["kind"] == "num":
            expected.add(prefix + "_val")
        if col["kind"] in ("str", "structured", "value"):
            expected.update((prefix + "_valbytes", prefix + "_valoffs"))
    return expected & names


def _attachments(kind, directory):
    def claims(header, names):
        prefixes = [f"{directory}/cm_{col['dim']}_{col['key']}/" for col in header.get("cell_meta", []) if col["kind"] == kind]
        return {name for name in names if any(name.startswith(prefix) for prefix in prefixes)}
    return claims


def _nested(header, names):
    prefixes = [f"nested/{entry['group']}/{entry['j']}/" for entry in header.get("nested", [])]
    return {name for name in names if any(name.startswith(prefix) for prefix in prefixes)}


def _sections(header, names):
    expected = {f"sections/{entry['name']}/{suffix}" for entry in header.get("sectionings", [])
                for suffix in ("indptr", "indices", "parent", "labels", "label_offsets", "spans")}
    return expected & names


from .transport import CellMaps, transport_hooks
from .codecs import codec_hooks

_registry = ComponentRegistry()
for _codec in (
    ComponentCodec("support", 1, _support, "exact", "forbidden"),
    ComponentCodec("declaration", 1, _fixed("column_head", "column_share_num", "column_share_den"), "exact"),
    ComponentCodec("weights", 1, _fixed("w_E", "weight_presence", "signs", "wb_keys", "wb_offsets", "wb_values", "wb_scalar", "wb_record")),
    ComponentCodec("identity", 1, _fixed("relation_ids", "label_bytes", "label_offsets")),
    ComponentCodec("relations", 1, _fixed("relation_record")),
    ComponentCodec("signals", 1, _fixed("signals")),
    ComponentCodec("attributes", 1, _attributes),
    ComponentCodec("field", 1, _attachments("field", "field"), "exact"),
    ComponentCodec("section", 1, _attachments("section", "section"), "exact"),
    ComponentCodec("model", 1, _attachments("model", "model")),
    ComponentCodec("span", 1, _attachments("span", "annotation"), "exact"),
    ComponentCodec("nested", 1, _nested),
    ComponentCodec("sectioning", 1, _sections, "exact"),
):
    _capture, _remap = transport_hooks(_codec.name)
    _encode, _decode, _headers = codec_hooks(_codec.name)
    _registry.register(replace(_codec, capture=_capture, remap=_remap,
                              encode=_encode, decode=_decode, header_fields=_headers))
del _codec


def component_registry():
    """Registered owners for native semantic state; duplicate names are refused."""
    return _registry
