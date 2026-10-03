"""Native record state operations over SQLStore's owned transaction and indexes.

SQLStateJournal persists the common grammar; StoreState owns the semantics.
This adapter uses Core's state/payload and metadata codecs, never a SQL specific
graph serializer or a second public store interface.
"""
from __future__ import annotations

import hashlib
import json

from .sql_journal import SQLStateJournal, _blob
from .envelope import _RECORD_HEADER_LIMIT, _RECORD_PAYLOAD_LIMIT

_BLOB_LIMIT = _RECORD_PAYLOAD_LIMIT+_RECORD_HEADER_LIMIT+1024*1024


def _projection_values(record):
    """The exact SQL projection written and checked from native metadata."""
    from .core import _sig_index_values
    from .envelope import dumps_mapping
    return dict(id=record.id, version=record.version,
        signature=dumps_mapping(record.signature), meta=dumps_mapping(record.meta), created=record.created,
        tx_from=record.tx_from, tx_to=record.tx_to, valid_from=record.valid_from, valid_to=record.valid_to,
        record_envelope=json.dumps(record.envelope.as_record()),
        **_sig_index_values(record.signature, is_complex=record.envelope.codec == "rexgraph.safetensors"))


class SQLRecordEngine:
    def __init__(self, store, connection, header):
        self.store = store
        self._projection_certificate = None
        self._projection_delta = None
        self.journal = SQLStateJournal.open(connection, store.table.name, header=header)
        self.header = self.journal.header
        self.state = self.journal.load_state(connection)
        self.check_projection(connection)

    def discard(self):
        self.state = None
        self._projection_certificate = None
        self._projection_delta = None

    def refresh(self, connection):
        if self.state is None:
            self.state = self.journal.load_state(connection)
        else:
            for frame in self.journal.read_after(connection, self.state.cursor):
                self.state.apply(frame)
        self.check_projection(connection)

    def check_projection(self, connection):
        """Reuse checked projection metadata on a qualified SQLite connection.

        SQLite's data_version catches other connections; total_changes catches
        writes through the same DBAPI connection, including rolled back writes.
        Main/temp schema versions and the checked journal cursor also belong to
        the certificate. A native write may nominate an exact set of changed
        addresses, but cannot certify them: those rows and labels are audited
        before reuse. Other changes/drivers retain the full audit. Failed
        transactions discard both certificates and provisional changes.
        """
        revision = self._projection_revision(connection)
        prior = self._projection_certificate
        if (revision is not None and prior is not None and prior[0] is revision[0]
                and prior[1:] == (revision[1], self.state.cursor)):
            return
        delta = self._projection_delta
        incremental = (revision is not None and prior is not None and delta is not None
                       and delta[0] is revision[0]
                       and delta[1:3] == (revision[1], self.state.cursor))
        self._projection_certificate = None
        self._projection_delta = None
        if incremental:
            self._audit_projection_delta(connection, delta[3])
        else:
            self._audit_projection(connection)
        after = self._projection_revision(connection)
        if (revision is not None and after is not None and revision[0] is after[0]
                and revision[1] == after[1]):
            self._projection_certificate = (after[0], after[1], self.state.cursor)

    @staticmethod
    def _projection_revision(connection):
        from contextlib import closing
        import sqlite3
        if connection.dialect.name != "sqlite":
            return None
        raw = connection.connection.driver_connection
        # Unqualified custom drivers use the existing full audit. Revision
        # counters are connection local and never transferred to another handle.
        if type(raw) is not sqlite3.Connection or not raw.in_transaction:
            return None
        with closing(raw.cursor()) as cursor:
            data = cursor.execute("PRAGMA main.data_version").fetchone()[0]
            schema = cursor.execute("PRAGMA main.schema_version").fetchone()[0]
            temp_schema = cursor.execute("PRAGMA temp.schema_version").fetchone()[0]
        return raw, (data, schema, temp_schema, raw.total_changes)

    def _prepare_projection_delta(self, connection):
        """Nominate changes only from an already checked, trigger free revision.

        total_changes counts every changed row, including trigger/FK side effects
        and rolled back writes. The expected budget comes from the engine grammar,
        never a driver reported row count. A changed counter or any SQL trigger
        prevents nomination. Provisional nominations can accumulate in one owned
        write scope, but are not a certificate until their exact projection audit.
        """
        revision = self._projection_revision(connection)
        prior = self._projection_delta or self._projection_certificate
        if (revision is None or self._projection_certificate is None or prior is None
                or prior[0] is not revision[0]
                or prior[1:3] != (revision[1], self.state.cursor)):
            self._projection_delta = None
            return None
        with connection.exec_driver_sql(
                "SELECT 1 FROM main.sqlite_schema WHERE type = 'trigger' "
                "UNION ALL SELECT 1 FROM temp.sqlite_schema WHERE type = 'trigger' LIMIT 1") as rows:
            if rows.first() is not None:
                self._projection_delta = None
                return None
        addresses = set() if self._projection_delta is None else self._projection_delta[3]
        return revision, self.state.cursor, addresses

    def _remember_projection_delta(self, connection, candidate, addresses, expected_changes):
        """Record an exact native DML budget; inspection still happens on read."""
        self._projection_delta = None
        if candidate is None:
            return
        after = self._projection_revision(connection)
        if after is None:
            return
        before, cursor, prior_addresses = candidate
        if (before[0] is after[0] and before[1][:-1] == after[1][:-1]
                and after[1][-1] == before[1][-1]+expected_changes
                and self.state.cursor.sequence == cursor.sequence+1):
            # The provider lock owns this set; accumulate large write scopes
            # without copying the complete nomination at every publication.
            prior_addresses.update(addresses)
            self._projection_delta = (after[0], after[1], self.state.cursor,
                                      prior_addresses)

    def _audit_projection(self, connection):
        """Refuse stale metadata/indexes before an indexed query can lose a record.

        Only metadata is inspected; payload graphs and payload BLOBs stay closed.
        Full audit remains mandatory on reopen and after changes without an exact
        native nomination. Payload BLOBs are checked separately when selected.
        """
        expected = {(r.id, r.version): r for r in self.state.records(include_history=True)}
        self._audit_projection_rows(connection, expected)

    def _audit_projection_delta(self, connection, addresses):
        """Audit changed metadata and vocabulary in bounded SQL parameter batches."""
        addresses = sorted(addresses)
        for start in range(0, len(addresses), 256):
            page = addresses[start:start+256]
            expected = {address: self.state.record(*address) for address in page}
            if any(record is None for record in expected.values()):
                raise ValueError("SQL projection nomination has no published record")
            self._audit_projection_rows(connection, expected, addresses=page)

    def _audit_projection_rows(self, connection, expected, *, addresses=None):
        from sqlalchemy import select, tuple_
        from .core import _record_labels
        table = self.store.table
        columns = [*self.store._record_cols(), *(table.c[key] for key in self.store._INDEX_COLS)]
        records = select(*columns)
        vocabulary = select(self.store.labels_table)
        if addresses is not None:
            records = records.where(tuple_(table.c.id, table.c.version).in_(addresses))
            lt = self.store.labels_table
            vocabulary = vocabulary.where(tuple_(lt.c.id, lt.c.version).in_(addresses))
        seen = set()
        labels = set()
        with connection.execute(records) as rows:
            for row in rows:
                address = (row.id, row.version)
                canonical = expected.get(address)
                if canonical is None or address in seen:
                    raise ValueError("SQL metadata projection has an unpublished record address")
                values = _projection_values(canonical)
                if any(row._mapping[name] != value for name, value in values.items()):
                    raise ValueError("SQL metadata projection differs from its checked record history")
                labels.update((row.id, row.version, term) for term in _record_labels(canonical.signature, canonical.meta))
                seen.add(address)
        if seen != set(expected):
            raise ValueError("SQL metadata projection is missing a published record")
        with connection.execute(vocabulary) as rows:
            actual_labels = {tuple(row) for row in rows}
        if actual_labels != labels:
            raise ValueError("SQL label index differs from its checked record vocabulary")

    def selected(self, record_id, as_of=None, valid_at=None):
        return self.state.selected(record_id, as_of=as_of, valid_at=valid_at)

    def put(self, connection, record_id, rex, signature, metadata, valid_from, valid_to, tx_time):
        from sqlalchemy import insert, update
        from .core import ComplexRecord, _now, _record_labels
        now = _now() if tx_time is None else float(tx_time)
        record = ComplexRecord(record_id, signature, created=now, meta=metadata,
                               version=self.state.next_version(record_id), tx_from=now,
                               valid_from=now if valid_from is None else valid_from, valid_to=valid_to)
        record, blob = self.store._record_payload(record, rex)
        artifact = self.store._load_commit_bytes(record.id, record.version)
        frame = self.state.prepare_put(record, blob_digest=hashlib.sha256(blob).hexdigest(),
            commit_digest=None if artifact is None else hashlib.sha256(artifact).hexdigest())
        # Validate the proposal before altering any transaction interval/index.
        self.state.validate(frame)
        candidate = self._prepare_projection_delta(connection)
        table = self.store.table
        connection.execute(update(table).where(table.c.id == record.id, table.c.tx_to.is_(None)).values(tx_to=now))
        connection.execute(insert(table).values(blob=blob, **_projection_values(record)))
        labels = sorted(_record_labels(record.signature, record.meta))
        if labels:
            connection.execute(insert(self.store.labels_table),
                [{"id": record.id, "version": record.version, "label": term} for term in labels])
        self.state = self.journal.publish(connection, frame, state=self.state)
        addresses = [(record.id, record.version)]
        if frame.mutation.previous_version:
            addresses.append((record.id, frame.mutation.previous_version))
        self._remember_projection_delta(connection, candidate, addresses,
            int(bool(frame.mutation.previous_version))+1+len(labels)+2)
        return record

    def read(self, connection, record, *, verify=True):
        from sqlalchemy import select
        table = self.store.table
        row = connection.execute(select(table.c.blob).where(table.c.id == record.id,
                                                           table.c.version == record.version)).first()
        if row is None or row.blob is None:
            raise ValueError("published SQL record payload is missing")
        blob = _blob(row.blob, _BLOB_LIMIT)
        change = self.state.change_for(record.id, record.version)
        if change is None or hashlib.sha256(blob).hexdigest() != change.blob_digest:
            raise ValueError("SQL record payload differs from its published content digest")
        if change.commit_digest is not None:
            self.store._load_commit_bytes(record.id, record.version)
        return self.store._read_record_payload(record, blob, verify=verify)

    def delete(self, connection, record_id, *, tx_time=None, expected_version=None):
        from sqlalchemy import update
        from .core import VersionConflictError, _now, _read_selector
        _read_selector(record_id, None, tx_time, None)
        if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
            raise ValueError("expected version requires a nonnegative native integer")
        self.store._claim_delete(record_id)
        current = self.state.current(record_id)
        actual = 0 if current is None else current.version
        if expected_version is not None and actual != expected_version:
            raise VersionConflictError("deletion expected version differs from current SQL record")
        frame = self.state.prepare_delete(record_id, _now() if tx_time is None else tx_time)
        if frame is None:
            return False
        candidate = self._prepare_projection_delta(connection)
        table = self.store.table
        connection.execute(update(table).where(table.c.id == record_id, table.c.tx_to.is_(None)).values(tx_to=frame.mutation.tx_time))
        self.state = self.journal.publish(connection, frame, state=self.state)
        self._remember_projection_delta(connection, candidate, [(record_id, current.version)], 1+2)
        self.store._emit("rcdb.delete", record_id, frame.mutation.version, {})
        return True

    def commit_bytes(self, record_id, version, raw):
        """Verify a claimed artifact, including where ordinary writes are allowed."""
        change = self.state.change_for(record_id, version)
        if raw is None:
            if change is not None and change.commit_digest is not None:
                raise ValueError("published SQL mutation artifact is missing")
            return None
        raw = _blob(raw, _BLOB_LIMIT)
        if change is not None and hashlib.sha256(raw).hexdigest() != change.commit_digest:
            raise ValueError("SQL mutation artifact differs from its published digest")
        if change is None and not self.store._sql_writing:
            raise ValueError("SQL mutation artifact has no published engine transition")
        return raw
