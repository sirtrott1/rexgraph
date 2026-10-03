"""Streaming tabular readers; interpretation is supplied by RecordSchema."""
from __future__ import annotations

import csv
from contextlib import closing, contextmanager
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from . import batches, source_stream


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _json_decoder():
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON value {value}")
    return json.JSONDecoder(parse_float=Decimal, parse_constant=nonfinite, object_pairs_hook=_object)


def _json(raw):
    return _json_decoder().decode(raw)


@contextmanager
def _csv_records(source, *, max_record_bytes, delimiter=None):
    delimiter = delimiter or ("\t" if Path(str(source)).suffix.lower() == ".tsv" else ",")
    if type(delimiter) is not str or len(delimiter) != 1:
        raise ValueError("CSV delimiter must be a single declared character")
    with source_stream(source) as stream:
        consumed = 0
        def lines():
            nonlocal consumed
            while True:
                line = stream.readline(max_record_bytes+1)
                if not line:
                    break
                consumed += len(line.encode("utf-8"))
                if consumed > max_record_bytes:
                    raise ValueError("CSV record exceeds the declared byte limit")
                yield line
        reader = csv.reader(lines(), delimiter=delimiter, strict=True)
        header = next(reader, None)
        if header is None:
            header = []
        if any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError("CSV header must have distinct named fields")
        consumed = 0
        def rows():
            nonlocal consumed
            for values in reader:
                if len(values) != len(header):
                    raise ValueError("CSV row does not match its header")
                consumed = 0
                yield dict(zip(header, values, strict=True))
        yield tuple(header), rows(), delimiter


def csv_batches(source, *, schema, batch_size, max_record_bytes, delimiter=None):
    with _csv_records(source, max_record_bytes=max_record_bytes, delimiter=delimiter) as (_header, rows, _delimiter):
        yield from batches(rows, schema, source, batch_size)


def csv_profile(source, *, max_records, max_record_bytes, delimiter=None):
    from .contracts import ReaderProfile
    from itertools import islice
    with _csv_records(source, max_record_bytes=max_record_bytes, delimiter=delimiter) as (header, rows, delimiter):
        sampled = sum(1 for _ in islice(rows, max_records))
        return ReaderProfile("csv", header, sampled, max_records, {"delimiter": delimiter})


def tsv_profile(source, **options):
    from dataclasses import replace
    return replace(csv_profile(source, **options), reader="tsv")


def jsonl_batches(source, *, schema, batch_size, max_record_bytes):
    with source_stream(source) as stream:
        def rows():
            while True:
                raw = stream.readline(max_record_bytes+1)
                if not raw:
                    break
                if len(raw.encode("utf-8")) > max_record_bytes:
                    raise ValueError("JSONL record exceeds the declared byte limit")
                if not raw.strip():
                    raise ValueError("JSONL records cannot be blank")
                yield _json(raw)
        yield from batches(rows(), schema, source, batch_size)


def json_batches(source, *, schema, batch_size, max_record_bytes, max_document_bytes=64*1024*1024):
    if type(max_document_bytes) is not int or max_document_bytes <= 0:
        raise ValueError("JSON document limit must be positive")
    with source_stream(source) as stream:
        raw = stream.read(max_document_bytes+1)
    if len(raw.encode("utf-8")) > max_document_bytes:
        raise ValueError("JSON document exceeds the declared byte limit; use JSONL for streaming")
    def whitespace(index):
        while index < len(raw) and raw[index] in " \t\r\n":
            index += 1
        return index
    position = whitespace(0)
    if position >= len(raw) or raw[position] != "[":
        raise TypeError("JSON record source must contain a record array")
    decoder = _json_decoder()
    def rows():
        index = whitespace(position+1)
        if index < len(raw) and raw[index] == "]":
            index += 1
        else:
            while True:
                value, end = decoder.raw_decode(raw, index)
                if len(raw[index:end].encode("utf-8")) > max_record_bytes:
                    raise ValueError("JSON record exceeds the declared byte limit")
                yield value
                index = whitespace(end)
                if index < len(raw) and raw[index] == "]":
                    index += 1
                    break
                if index >= len(raw) or raw[index] != ",":
                    raise ValueError("JSON record array requires a comma or closing bracket")
                index = whitespace(index+1)
        if whitespace(index) != len(raw):
            raise ValueError("JSON document contains trailing data")
    yield from batches(rows(), schema, source, batch_size)


def _columnar_rows(batch, max_record_bytes):
    # Materialize one bounded Arrow batch, never the entire source table.
    if batch.nbytes > max_record_bytes*max(1, batch.num_rows):
        raise ValueError("columnar batch exceeds the declared record byte limit")
    if any(batch.slice(i, 1).nbytes > max_record_bytes for i in range(batch.num_rows)):
        raise ValueError("columnar record exceeds the declared byte limit")
    from ..columnar import column_declarations, decoded_columns
    declaration = column_declarations(batch.schema)
    if declaration:
        columns = decoded_columns(batch, declaration)
        for i in range(batch.num_rows):
            yield {name: value[i] for name, value in columns.items()}
    else:
        yield from batch.to_pylist()


def parquet_batches(source, *, schema, batch_size, max_record_bytes, columns=None):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(source)
    try:
        from ..columnar import column_declarations, projected_columns
        projection = projected_columns(file.schema_arrow, column_declarations(file.schema_arrow), columns)
        def rows():
            for batch in file.iter_batches(batch_size=batch_size, columns=projection):
                yield from _columnar_rows(batch, max_record_bytes)
        yield from batches(rows(), schema, source, batch_size)
    finally:
        file.close()


def arrow_batches(source, *, schema, batch_size, max_record_bytes):
    import pyarrow as pa
    with source_stream(source, binary=True) as stream:
        try:
            reader = pa.ipc.open_file(stream)
        except pa.ArrowInvalid:
            stream.seek(0)
            reader = pa.ipc.open_stream(stream)
        if hasattr(reader, "get_batch"):
            batch_iter = (reader.get_batch(i) for i in range(reader.num_record_batches))
        else:
            batch_iter = iter(reader)
        def rows():
            for batch in batch_iter:
                for start in range(0, batch.num_rows, batch_size):
                    yield from _columnar_rows(batch.slice(start, batch_size), max_record_bytes)
        yield from batches(rows(), schema, source, batch_size)


def sqlite_batches(source, *, schema, batch_size, max_record_bytes, query, parameters=()):
    # A dedicated read only connection keeps a declared query from mutating the
    # source, including queries whose WITH clause conceals a write.
    uri = Path(source).expanduser().resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        # mode=ro applies to the primary file. ATTACH can otherwise create a
        # second writable database before a non result query is refused.
        connection.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY
                                  if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH)
                                  else sqlite3.SQLITE_OK)
        cursor = connection.execute(query, parameters)
        if cursor.description is None:
            raise ValueError("SQL reader requires a result-producing query")
        names = tuple(column[0] for column in cursor.description)
        if len(set(names)) != len(names):
            raise ValueError("SQL result requires distinct named fields")
        def rows():
            while True:
                values = cursor.fetchmany(batch_size)
                if not values:
                    break
                for row in values:
                    if sum(len(v) if isinstance(v, bytes) else len(v.encode("utf-8")) if isinstance(v, str) else 8 for v in row) > max_record_bytes:
                        raise ValueError("SQL record exceeds the declared byte limit")
                    yield dict(zip(names, row, strict=True))
        yield from batches(rows(), schema, source, batch_size)


def xlsx_batches(source, *, schema, batch_size, max_record_bytes, sheet=None, data_only=True):
    import openpyxl
    book = openpyxl.load_workbook(source, read_only=True, data_only=data_only)
    try:
        worksheet = book.active if sheet is None else book[sheet]
        rows = worksheet.iter_rows(values_only=True)
        header = next(rows, ())
        if any(type(n) is not str or not n for n in header) or len(set(header)) != len(header):
            raise ValueError("XLSX header must have distinct named fields")
        def records():
            for row in rows:
                if sum(len(v.encode("utf-8")) if isinstance(v, str) else 8 for v in row) > max_record_bytes:
                    raise ValueError("XLSX record exceeds the declared byte limit")
                yield dict(zip(header, row, strict=True))
        yield from batches(records(), schema, source, batch_size)
    finally:
        book.close()
