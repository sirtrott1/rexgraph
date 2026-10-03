"""SQL persistence for the shared native header, checked frames and logical head.

The provider owns the database transaction and writer arbitration. This adapter
never opens a second connection, commits, or invents a checked legacy history.
"""
from __future__ import annotations

from .engine import ChangeCursor, StoreState
from .header import HEADER_LIMIT, StoreHeader
from .journal import FRAME_LIMIT, GENESIS_DIGEST, JournalFrame


def _blob(value, limit):
    if isinstance(value, memoryview):
        size = value.nbytes
    elif isinstance(value, (bytes, bytearray)):
        size = len(value)
    else:
        size = None
    if size is None or size > limit:
        raise ValueError("SQL journal payload requires bounded binary bytes")
    return bytes(value)


def _transaction(connection):
    if not connection.in_transaction():
        raise RuntimeError("SQL state journal requires an owned transaction")
    if connection.dialect.name == "sqlite" and not connection.connection.driver_connection.in_transaction:
        raise RuntimeError("SQL state journal requires an explicit SQLite transaction")


def sql_state_header(connection, prefix):
    """Discover checked ownership before SQLStore can initialize an identity.

    This only reads a bounded declaration; it neither creates tables nor assumes
    that headerless legacy metadata describes an engine publication history.
    """
    from sqlalchemy import inspect, select
    from .store_identity import StoreIdentity, sql_identity_payload, sql_identity_table
    declaration = SQLStateJournal(prefix, StoreHeader(StoreIdentity("0"*32, "sql")))
    identity_table = sql_identity_table(str(prefix))
    anchor = None
    if inspect(connection).has_table(identity_table.name):
        claim = connection.execute(select(identity_table.c.value).where(
            identity_table.c.key == "engine_header")).scalar_one_or_none()
        if claim is not None:
            anchor = StoreHeader.from_bytes(_blob(claim, HEADER_LIMIT))
            owner = connection.execute(select(identity_table.c.value).where(
                identity_table.c.key == "identity")).scalar_one_or_none()
            if owner is None:
                raise ValueError("SQL store identity is missing from native durable state; explicit recovery required")
            if StoreIdentity.from_bytes(sql_identity_payload(owner), backend="sql") != anchor.identity:
                raise ValueError("SQL native header and identity ownership claims name different stores")
    tables = (declaration.header_table, declaration.frames_table, declaration.head_table)
    present = [inspect(connection).has_table(table.name) for table in tables]
    if any(present) and not all(present):
        raise ValueError("SQL state journal schema is incomplete; explicit recovery required")
    if not any(present):
        if anchor is not None:
            raise ValueError("claimed SQL state journal schema is missing; explicit recovery required")
        return None
    rows = connection.execute(select(declaration.header_table).limit(2)).fetchall()
    if len(rows) != 1 or rows[0].key != "header":
        raise ValueError("SQL store header is missing or has extra declarations; explicit recovery required")
    header = StoreHeader.from_bytes(_blob(rows[0].payload, HEADER_LIMIT))
    if header.identity.backend != "sql":
        raise ValueError("SQL store header names another provider")
    if anchor is None:
        raise ValueError("SQL native header ownership anchor is missing; explicit recovery required")
    if anchor != header:
        raise ValueError("SQL native header differs from its durable ownership anchor")
    return header


class SQLStateJournal:
    """One checked engine history inside the caller's SQL transaction."""

    def __init__(self, prefix, header):
        from sqlalchemy import BigInteger, Column, LargeBinary, MetaData, String, Table
        if not isinstance(prefix, str) or not prefix or "\x00" in prefix:
            raise ValueError("SQL journal prefix requires a nonempty literal identifier")
        prefix = str(prefix)  # SQLAlchemy Table.name is a quoted_name str subclass.
        if not isinstance(header, StoreHeader) or header.identity.backend != "sql":
            raise ValueError("SQL journal requires a declared SQL store header")
        self._header = header
        self._prefix = prefix
        self.metadata = MetaData()
        self.header_table = Table(prefix+"_engine_header", self.metadata,
                                  Column("key", String(32), primary_key=True),
                                  Column("payload", LargeBinary, nullable=False))
        self.frames_table = Table(prefix+"_engine_frames", self.metadata,
                                  Column("sequence", BigInteger, primary_key=True),
                                  Column("payload", LargeBinary, nullable=False))
        self.head_table = Table(prefix+"_engine_head", self.metadata,
                                Column("key", String(32), primary_key=True),
                                Column("sequence", BigInteger, nullable=False),
                                Column("digest", String(64), nullable=False),
                                Column("header_digest", String(64), nullable=False))

    @property
    def header(self):
        return self._header

    @classmethod
    def open(cls, connection, prefix, *, header=None):
        """Initialize an empty journal or open its unchanged durable declaration.

        Partial schemas, missing metadata and a changed requested header require
        explicit recovery/migration. Creation belongs to the caller's transaction.
        """
        _transaction(connection)
        if header is not None and not isinstance(header, StoreHeader):
            raise TypeError("SQL journal header must be a declared StoreHeader")
        stored = sql_state_header(connection, prefix)
        if stored is None:
            if header is None:
                raise ValueError("new SQL state journal requires an explicit store header")
            declaration = cls(prefix, header)
            from sqlalchemy import select
            from .store_identity import StoreIdentity, sql_identity_payload, sql_identity_table
            identity_table = sql_identity_table(str(prefix))
            with connection.begin_nested():
                identity_table.create(connection, checkfirst=True)
                payload = connection.execute(select(identity_table.c.value).where(
                    identity_table.c.key == "identity")).scalar_one_or_none()
                if payload is None:
                    connection.execute(identity_table.insert().values(key="identity", value=header.identity.to_bytes()))
                elif StoreIdentity.from_bytes(sql_identity_payload(payload), backend="sql") != header.identity:
                    raise ValueError("SQL journal header differs from the existing store identity")
                connection.execute(identity_table.insert().values(key="engine_header", value=header.to_bytes()))
                declaration.metadata.create_all(connection)
                connection.execute(declaration.header_table.insert().values(key="header", payload=header.to_bytes()))
                connection.execute(declaration.head_table.insert().values(key="head", sequence=0,
                    digest=GENESIS_DIGEST, header_digest=header.digest))
            return declaration
        if header is not None and stored != header:
            raise ValueError("SQL store header differs; explicit migration required")
        declaration = cls(prefix, stored)
        declaration._head(connection)
        return declaration

    def _head(self, connection):
        from sqlalchemy import select
        _transaction(connection)
        if sql_state_header(connection, self._prefix) != self.header:
            raise ValueError("SQL store header differs from its opened declaration")
        rows = connection.execute(select(self.head_table).limit(2)).fetchall()
        if len(rows) != 1 or rows[0].key != "head":
            raise ValueError("SQL journal head is missing or has extra declarations")
        row = rows[0]
        cursor = ChangeCursor(self.header.identity.id, row.header_digest, row.sequence, row.digest)
        if cursor.header_digest != self.header.digest:
            raise ValueError("SQL journal head differs from its store header")
        return cursor

    def load_state(self, connection, *, after=None):
        """Strictly replay the durable history; an optional prior cursor must survive."""
        from sqlalchemy import select
        head = self._head(connection)
        state = StoreState(self.header)
        with connection.execute(select(self.frames_table).order_by(self.frames_table.c.sequence)) as rows:
            for row in rows:
                if type(row.sequence) is not int or row.sequence != state.cursor.sequence+1:
                    raise ValueError("SQL journal sequence address differs from its checked chain")
                frame = JournalFrame.from_bytes(_blob(row.payload, FRAME_LIMIT+64))
                if frame.sequence != row.sequence:
                    raise ValueError("SQL journal frame differs from its sequence address")
                state.apply(frame)
        if state.cursor != head:
            raise ValueError("SQL journal replay differs from its published head")
        if after is not None:
            state.check_cursor(after)
        return state

    def read_after(self, connection, after):
        """Check a retained anchor and read only its contiguous published suffix.

        The provider has already replayed the prefix into StoreState. Each
        returned frame still needs StoreState.apply to check record transitions.
        Complete reopen uses load_state, which checks the entire durable history.
        """
        from sqlalchemy import select
        head = self._head(connection)
        if not isinstance(after, ChangeCursor):
            raise TypeError("SQL journal tail requires a declared logical cursor")
        if (after.store_id != self.header.identity.id or after.header_digest != self.header.digest
                or after.sequence > head.sequence):
            raise ValueError("SQL journal cursor differs from its retained history")
        if after.sequence:
            row = connection.execute(select(self.frames_table).where(
                self.frames_table.c.sequence == after.sequence)).first()
            if row is None:
                raise ValueError("SQL journal cursor anchor is missing")
            anchor = JournalFrame.from_bytes(_blob(row.payload, FRAME_LIMIT+64))
            if (anchor.store_id, anchor.sequence, anchor.digest) != (after.store_id, after.sequence, after.digest):
                raise ValueError("SQL journal cursor anchor differs from its checked frame")
        frames = []
        cursor = after
        with connection.execute(select(self.frames_table).where(
                self.frames_table.c.sequence > after.sequence).order_by(self.frames_table.c.sequence)) as rows:
            for row in rows:
                if type(row.sequence) is not int or row.sequence != cursor.sequence+1:
                    raise ValueError("SQL journal suffix is missing a sequence address")
                frame = JournalFrame.from_bytes(_blob(row.payload, FRAME_LIMIT+64))
                frame.check_successor(store_id=cursor.store_id, sequence=cursor.sequence+1, previous=cursor.digest)
                if frame.mutation is None or frame.mutation.header_digest != self.header.digest or frame.extra is not None:
                    raise ValueError("SQL journal suffix differs from its declared state engine")
                cursor = ChangeCursor(frame.store_id, self.header.digest, frame.sequence, frame.digest)
                frames.append(frame)
        if cursor != head:
            raise ValueError("SQL journal suffix differs from its published head")
        return tuple(frames)

    def publish(self, connection, frame, *, state=None):
        """Append one shared engine proposal atomically within the owned transaction.

        A savepoint ensures a caught refusal cannot leave a frame without a head.
        The returned state is provisional until the provider commits the outer
        transaction. It must be discarded if that outer transaction rolls back.
        An already checked, provider owned StoreState can skip full replay after
        its retained cursor is checked against the current database head.
        """
        from sqlalchemy import update
        from sqlalchemy.exc import IntegrityError
        from .core import VersionConflictError
        if not isinstance(frame, JournalFrame):
            raise TypeError("SQL publication requires a declared checked journal frame")
        if state is None:
            state = self.load_state(connection)
        else:
            if not isinstance(state, StoreState) or state.header != self.header:
                raise ValueError("SQL publication state differs from its declared store header")
            if self.read_after(connection, state.cursor):
                raise VersionConflictError("SQL publication state precedes the published head")
        old = state.cursor
        if (frame.store_id, frame.sequence, frame.previous) != (old.store_id, old.sequence+1, old.digest):
            raise VersionConflictError("SQL journal proposal differs from the published head")
        state.validate(frame)
        raw = frame.to_bytes()
        digest = frame.digest
        try:
            with connection.begin_nested():
                connection.execute(self.frames_table.insert().values(sequence=frame.sequence, payload=raw))
                result = connection.execute(update(self.head_table).where(
                    self.head_table.c.key == "head", self.head_table.c.sequence == old.sequence,
                    self.head_table.c.digest == old.digest, self.head_table.c.header_digest == old.header_digest,
                ).values(sequence=frame.sequence, digest=digest))
                if result.rowcount != 1:
                    raise VersionConflictError("SQL journal head changed during publication")
        except IntegrityError as exc:
            # A provider constraint error is not automatically a competing writer.
            # The savepoint restored the connection: inspect the winning head
            # before translating only an actual sequence/address conflict.
            current = self.load_state(connection)
            if current.cursor.sequence >= frame.sequence and current.cursor != old:
                raise VersionConflictError("SQL journal sequence was occupied during publication") from exc
            raise
        state.apply(frame)
        return state


__all__ = ["SQLStateJournal", "sql_state_header"]
