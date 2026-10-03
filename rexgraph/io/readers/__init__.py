"""Bounded file to record readers with explicit schema and number rules."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import os
from threading import RLock
from typing import Callable

from rexgraph.registry import Registry
from rexgraph.io.records import RecordBatch, RecordSchema, SourcePointer
from .contracts import ParameterSchema, ReaderParameter, ReaderProfile

PREFIX_LIMIT = 64 * 1024


@dataclass(frozen=True)
class ReaderSpec:
    name: str
    extensions: tuple[str, ...]
    batch_reader: Callable
    priority: int = 0
    media_types: tuple[str, ...] = ()
    sniffer: Callable | None = None
    parameters: ParameterSchema = ParameterSchema()
    profiler: Callable | None = None
    version: int = 1

    def __post_init__(self):
        if type(self.name) is not str or not self.name or not callable(self.batch_reader) or type(self.priority) is not int:
            raise TypeError("reader requires a name, callable and integer priority")
        if any(type(e) is not str for e in self.extensions):
            raise TypeError("reader extensions must be text")
        extensions = tuple(e.lower() for e in self.extensions)
        if any(not e.startswith(".") or len(e) < 2 or "/" in e or "\\" in e for e in extensions) or len(set(extensions)) != len(extensions):
            raise ValueError("invalid reader extensions")
        if self.sniffer is not None and not callable(self.sniffer):
            raise TypeError("reader sniffer must be callable")
        if self.profiler is not None and not callable(self.profiler):
            raise TypeError("reader profiler must be callable")
        if not isinstance(self.parameters, ParameterSchema):
            raise TypeError("reader parameters require a ParameterSchema")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("reader contract version must be positive")
        media_types = tuple(_media_type(m) for m in self.media_types)
        if len(set(media_types)) != len(media_types):
            raise ValueError("reader media types must be distinct")
        object.__setattr__(self, "extensions", extensions)
        object.__setattr__(self, "media_types", media_types)

    def as_record(self):
        return {"name": self.name, "version": self.version, "extensions": self.extensions,
                "media_types": self.media_types, "priority": self.priority,
                "sniffer": self.sniffer is not None, "profiler": self.profiler is not None,
                "parameters": tuple({"name": f.name, "kind": f.kind, "default": f.default_value(),
                                     "nullable": f.nullable, "minimum": f.minimum, "maximum": f.maximum}
                                    for f in self.parameters.fields)}

    @property
    def digest(self):
        import hashlib
        from rexgraph.value_codec import pack_value
        return hashlib.sha256(b"rexgraph.reader-contract.v1\x00"+pack_value(self.as_record())).hexdigest()


def _media_type(value):
    if type(value) is not str:
        raise TypeError("reader media type must be text")
    media = value.split(";", 1)[0].strip().lower()
    if media.count("/") != 1 or any(not p or any(c.isspace() for c in p) for p in media.split("/")):
        raise ValueError("invalid reader media type")
    return media


class ReaderRegistry(Registry):
    def __init__(self):
        super().__init__("record reader")
        self._lock = RLock()

    def register(self, spec, *, replace=False):
        if not isinstance(spec, ReaderSpec):
            raise TypeError("register a ReaderSpec")
        if type(replace) is not bool:
            raise TypeError("reader replacement must be explicit boolean")
        with self._lock:
            if spec.name in self and not replace:
                raise ValueError(f"reader {spec.name!r} is already registered")
            return super().register(spec.name, spec)

    def unregister(self, name):
        with self._lock:
            return super().unregister(name)

    def clear(self):
        with self._lock:
            super().clear()

    def load_plugins(self, *, group="rexgraph.readers"):
        """Explicitly load trusted installed entry points, publishing as one set.

        Persisted declarations never invoke this method or import providers.
        Entry point loading executes installed application code by caller request.
        """
        from importlib.metadata import entry_points
        if type(group) is not str or not group:
            raise ValueError("reader plugin group must be named")
        specs = tuple(entry.load() for entry in sorted(entry_points(group=group), key=lambda e: e.name))
        if any(not isinstance(s, ReaderSpec) for s in specs):
            raise TypeError("reader entry points must export ReaderSpec values")
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("reader plugins declare duplicate names")
        with self._lock:
            duplicate = sorted(s.name for s in specs if s.name in self)
            if duplicate:
                raise ValueError("reader plugins already registered: " + ", ".join(duplicate))
            for spec in specs:
                super().register(spec.name, spec)
        return tuple(s.name for s in specs)

    def resolve(self, source, *, reader=None, media_type=None, prefix=b""):
        if type(prefix) is not bytes or len(prefix) > PREFIX_LIMIT:
            raise ValueError("reader sniff prefix must be bounded bytes")
        media_type = None if media_type is None else _media_type(media_type)
        if reader is not None:
            if type(reader) is not str or not reader:
                raise TypeError("reader must be named")
            with self._lock:
                return self.require(reader)
        source_name = os.fspath(source) if isinstance(source, (str, os.PathLike)) else getattr(source, "name", "")
        source_name = source_name.lower() if isinstance(source_name, str) else ""
        with self._lock:
            specs = tuple(self._values.values())
        matches = []
        for spec in specs:
            declared = any(source_name.endswith(e) for e in spec.extensions) or media_type is not None and media_type in spec.media_types
            sniffed = False
            if spec.sniffer is not None:
                sniffed = spec.sniffer(prefix)
                if type(sniffed) is not bool:
                    raise TypeError(f"reader {spec.name!r} sniffer must return a boolean")
            if declared or sniffed:
                matches.append(spec)
        if not matches:
            raise ValueError("no record reader matches; name a reader explicitly")
        priority = max(s.priority for s in matches)
        winners = [s for s in matches if s.priority == priority]
        if len(winners) != 1:
            raise ValueError("ambiguous record readers: " + ", ".join(sorted(s.name for s in winners)))
        return winners[0]


READERS = ReaderRegistry()


def source_uri(source):
    if isinstance(source, (str, os.PathLike)):
        return Path(source).expanduser().resolve().as_uri()
    return str(getattr(source, "name", "memory:records"))


@contextmanager
def source_stream(source, *, binary=False):
    if isinstance(source, (str, os.PathLike)):
        with Path(source).expanduser().open("rb" if binary else "r", **({} if binary else {"encoding": "utf-8", "newline": ""})) as stream:
            yield stream
    else:
        yield source


def batches(rows, schema, source, batch_size):
    pending = []
    uri = source_uri(source)
    for i, row in enumerate(rows):
        pointer = SourcePointer(uri, record=i)
        pending.append(schema.convert(row, pointer))
        if len(pending) == batch_size:
            yield RecordBatch.from_records(schema, pending)
            pending = []
    if pending:
        yield RecordBatch.from_records(schema, pending)


def read_batches(source, *, schema: RecordSchema, reader=None, registry=None, batch_size=4096,
                 max_record_bytes=8*1024*1024, media_type=None, prefix=b"", **options):
    """Read fixed size batches without constructing a complex or inferring numbers."""
    if not isinstance(schema, RecordSchema):
        raise TypeError("record reading requires a declared RecordSchema")
    if type(batch_size) is not int or not 0 < batch_size <= 1_000_000 or type(max_record_bytes) is not int or max_record_bytes <= 0:
        raise ValueError("reader batch and record limits must be positive and bounded")
    registry = READERS if registry is None else registry
    if not isinstance(registry, ReaderRegistry):
        raise TypeError("record reading requires a ReaderRegistry")
    selected = registry.resolve(source, reader=reader, media_type=media_type, prefix=prefix)
    options = selected.parameters.bind(options)
    stream = iter(selected.batch_reader(source, schema=schema, batch_size=batch_size,
                                        max_record_bytes=max_record_bytes, **options))
    try:
        for batch in stream:
            if not isinstance(batch, RecordBatch):
                raise TypeError(f"reader {selected.name!r} must return RecordBatch values")
            if batch.schema != schema or len(batch) > batch_size:
                raise ValueError(f"reader {selected.name!r} batch violates its declared schema or size")
            yield batch
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def profile(source, *, reader=None, registry=None, media_type=None, prefix=b"",
            max_records=1000, max_record_bytes=8*1024*1024, **options):
    """Observe a bounded source explicitly; no declaration is inferred or applied."""
    if type(max_records) is not int or not 0 < max_records <= 1_000_000 or type(max_record_bytes) is not int or max_record_bytes <= 0:
        raise ValueError("reader profile limits must be positive and bounded")
    registry = READERS if registry is None else registry
    if not isinstance(registry, ReaderRegistry):
        raise TypeError("source profiling requires a ReaderRegistry")
    selected = registry.resolve(source, reader=reader, media_type=media_type, prefix=prefix)
    if selected.profiler is None:
        raise ValueError(f"reader {selected.name!r} has no declared profile capability")
    result = selected.profiler(source, max_records=max_records, max_record_bytes=max_record_bytes,
                               **selected.parameters.bind(options))
    if not isinstance(result, ReaderProfile) or result.reader != selected.name or result.record_limit != max_records:
        raise ValueError("reader profiler violated its declared profile contract")
    return result


from .tabular import csv_batches, csv_profile, tsv_profile, json_batches, jsonl_batches, arrow_batches, parquet_batches, sqlite_batches, xlsx_batches
for _spec in (
    ReaderSpec("csv", (".csv",), csv_batches, media_types=("text/csv",),
               parameters=ParameterSchema((ReaderParameter("delimiter", "character", None, nullable=True),)), profiler=csv_profile),
    ReaderSpec("tsv", (".tsv",), csv_batches, media_types=("text/tab-separated-values",),
               parameters=ParameterSchema((ReaderParameter("delimiter", "character", "\t"),)), profiler=tsv_profile),
    ReaderSpec("json", (".json",), json_batches, media_types=("application/json",),
               parameters=ParameterSchema((ReaderParameter("max_document_bytes", "integer", 64*1024*1024, minimum=1),))),
    ReaderSpec("jsonl", (".jsonl", ".ndjson"), jsonl_batches, media_types=("application/x-ndjson",)),
    ReaderSpec("arrow", (".arrow", ".feather"), arrow_batches),
    ReaderSpec("parquet", (".parquet", ".pq"), parquet_batches,
               parameters=ParameterSchema((ReaderParameter("columns", "text_sequence", None, nullable=True),))),
    ReaderSpec("sqlite", (".db", ".sqlite", ".sqlite3"), sqlite_batches,
               parameters=ParameterSchema((ReaderParameter("query", "text"), ReaderParameter("parameters", default=())))),
    ReaderSpec("xlsx", (".xlsx",), xlsx_batches,
               parameters=ParameterSchema((ReaderParameter("sheet", "text", None, nullable=True), ReaderParameter("data_only", "boolean", True)))),
):
    READERS.register(_spec)
del _spec

__all__ = ["ReaderSpec", "ReaderRegistry", "ReaderParameter", "ParameterSchema", "ReaderProfile", "READERS", "read_batches", "profile"]
