"""Explicit live dataset bindings over Core declarations and sealed state.

Opening a source is a trusted application action. Query text only names an already
registered binding; persisted declarations never open paths or load reader code.
"""
from __future__ import annotations

import hashlib


class DatasetSource:
    """Pinned declared data, detached from input files and caller objects.

    Each query gets a fresh verified graph. Changed inputs require a new binding.
    Reader versions identify declared contracts, not authentication of plugin code.
    """
    __slots__ = ("_declaration", "_reader", "_payload", "_state_digest", "_digest")

    def __init__(self, declaration, source, *, registry=None):
        from rexgraph.io.declaration import DatasetDeclaration
        from rexgraph.io.readers import READERS, ReaderRegistry
        from rexgraph.io.safetensors_bridge import state_to_safetensors_bytes
        from rexgraph.state import to_state
        from rexgraph.object_identity import object_digest
        from rexgraph.value_codec import pack_value
        if not isinstance(declaration, DatasetDeclaration):
            raise TypeError("dataset source requires a Core DatasetDeclaration")
        registry = READERS if registry is None else registry
        if not isinstance(registry, ReaderRegistry):
            raise TypeError("dataset source requires a ReaderRegistry")
        self._declaration = declaration.to_bytes()
        owned = DatasetDeclaration.from_bytes(self._declaration)
        spec = registry.resolve(source, reader=owned.reader)
        # Pin this capability; concurrent registry replacement cannot change it.
        selected = ReaderRegistry(); selected.register(spec)
        self._reader = pack_value(spec.as_record())
        graph = owned.to_rex(source, registry=selected)
        owned.check_state()
        self._state_digest = object_digest(graph)
        self._payload = state_to_safetensors_bytes(to_state(graph))
        self._digest = hashlib.sha256(b"rcql.dataset-source.v1\x00"+pack_value(
            (self._declaration, self._reader, self._state_digest))).hexdigest()

    @property
    def digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        from rexgraph.value_codec import pack_value
        expected = hashlib.sha256(b"rcql.dataset-source.v1\x00"+pack_value(
            (self._declaration, self._reader, self._state_digest))).hexdigest()
        if expected != self._digest:
            raise ValueError("dataset source declaration differs from its pinned identity")

    @property
    def declaration(self):
        from rexgraph.io.declaration import DatasetDeclaration
        self.check_state()
        return DatasetDeclaration.from_bytes(self._declaration)

    def as_record(self):
        from rexgraph.value_codec import unpack_value
        return {"schema": "rcql.dataset-source", "version": 1, "digest": self.digest,
                "declaration_digest": hashlib.sha256(self._declaration).hexdigest(),
                "reader": unpack_value(self._reader), "state_digest": self._state_digest}

    def materialize(self):
        from rexgraph.io.safetensors_bridge import state_from_safetensors_bytes
        from rexgraph.state import from_state
        from rexgraph.object_identity import object_digest
        self.check_state()
        graph = from_state(state_from_safetensors_bytes(self._payload))
        if object_digest(graph) != self._state_digest:
            raise ValueError("dataset source state differs from its declared identity")
        return graph


__all__ = ["DatasetSource"]
