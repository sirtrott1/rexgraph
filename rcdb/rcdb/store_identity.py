"""Durable opaque store identities, independent of paths and credentials."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
from threading import RLock
import uuid

from rexgraph.value_codec import pack_value, unpack_value
from .envelope import _hex_identity

_MAGIC = b"RGSI1"
_LIMIT = 4096
_MEMORY_FS_LOCK = RLock()


@dataclass(frozen=True)
class StoreIdentity:
    id: str
    backend: str

    def __post_init__(self):
        _hex_identity(self.id, 32, "store identity")
        if type(self.backend) is not str or not self.backend:
            raise ValueError("store backend must be declared")

    @classmethod
    def create(cls, backend):
        return cls(uuid.uuid4().hex, backend)

    def to_bytes(self):
        body = pack_value({"object_type": "RCDBStoreIdentity", "version": 1,
                           "id": self.id, "backend": self.backend})
        if len(body) > _LIMIT-37:
            raise ValueError("store identity exceeds its byte limit")
        return _MAGIC+hashlib.sha256(body).digest()+body

    @classmethod
    def from_bytes(cls, payload, *, backend):
        if (type(payload) is not bytes or not payload.startswith(_MAGIC)
                or not 37 < len(payload) <= _LIMIT):
            raise ValueError("invalid store identity frame")
        body = payload[37:]
        if not hmac.compare_digest(hashlib.sha256(body).digest(), payload[5:37]):
            raise ValueError("store identity digest mismatch")
        record = unpack_value(body)
        if (type(record) is not dict or set(record) != {"object_type", "version", "id", "backend"}
                or record["object_type"] != "RCDBStoreIdentity" or type(record["version"]) is not int
                or record["version"] != 1 or record["backend"] != backend):
            raise ValueError("store identity backend or format differs")
        return cls(record["id"], record["backend"])


def bound_identity(records):
    """Find durable ownership claims without inventing a replacement identity."""
    identity = None
    for record in records:
        envelope = record.envelope
        if envelope is not None:
            if identity is not None and identity != envelope.store_id:
                raise ValueError("record ownership claims name different stores")
            identity = envelope.store_id
    return identity


def _check_existing(value, existing_id):
    if existing_id is not None and value.id != existing_id:
        raise ValueError("store identity differs from its durable ownership claim")
    return value


def local_identity(path, *, backend, existing_id=None, read_only=False):
    """Publish once under the shared local lock; subsequent opens only read."""
    from rexgraph.io.publication import staged_publication
    path = Path(path)
    if path.is_symlink():
        raise ValueError("store identity cannot be a symbolic link")
    if path.exists():
        if path.stat().st_size > _LIMIT:
            raise ValueError("store identity exceeds its byte limit")
        return _check_existing(StoreIdentity.from_bytes(path.read_bytes(), backend=backend), existing_id)
    if read_only:
        if existing_id is not None:
            raise ValueError("store identity is missing from bound durable state; explicit recovery required")
        from .legacy import source_fingerprint
        # Unowned historical layouts never had a durable UUID. Identify the
        # exact source bytes for migration without claiming a persistent owner
        # or creating a file during a read only open.
        digest = source_fingerprint(path.parent)
        return StoreIdentity(hashlib.sha256(b"rexgraph-legacy-source-identity\x00" +
                             backend.encode("utf-8") + bytes.fromhex(digest)).hexdigest()[:32], backend)
    with staged_publication(path, update=True) as staged:
        if staged.exists():
            if staged.stat().st_size > _LIMIT:
                raise ValueError("store identity exceeds its byte limit")
            value = StoreIdentity.from_bytes(staged.read_bytes(), backend=backend)
        else:
            if existing_id is not None:
                raise ValueError("store identity is missing from bound durable state; explicit recovery required")
            value = StoreIdentity.create(backend)
            staged.write_bytes(value.to_bytes())
        _check_existing(value, existing_id)
    return value


def sql_identity_table(table_name):
    """The durable identity row also arbitrates this store's SQL transactions."""
    from sqlalchemy import Column, LargeBinary, MetaData, String, Table
    return Table(table_name+"_store_identity", MetaData(),
                 Column("key", String(32), primary_key=True), Column("value", LargeBinary, nullable=False))


def sql_identity_payload(value):
    """Normalize DBAPI buffers only after their declared type and size are bounded."""
    if isinstance(value, memoryview):
        size = value.nbytes
    elif isinstance(value, (bytes, bytearray)):
        size = len(value)
    else:
        size = None
    if size is None or size > _LIMIT:
        raise ValueError("SQL store identity requires bounded binary bytes")
    return bytes(value)


def sql_identity(engine, table_name, *, existing_id=None):
    """Use a unique identity row, with transactional first writer arbitration."""
    from sqlalchemy import insert, select
    from sqlalchemy.exc import IntegrityError
    table = sql_identity_table(table_name)
    if engine.dialect.name == "sqlite":
        from sqlalchemy.schema import CreateTable
        with engine.begin() as connection:
            connection.execute(CreateTable(table, if_not_exists=True))
    else:
        table.create(engine, checkfirst=True)
    with engine.connect() as connection:
        payload = connection.execute(select(table.c.value).where(table.c.key == "identity")).scalar_one_or_none()
    if payload is not None:
        return _check_existing(StoreIdentity.from_bytes(sql_identity_payload(payload), backend="sql"), existing_id)
    if existing_id is not None:
        raise ValueError("store identity is missing from bound durable state; explicit recovery required")
    value = StoreIdentity.create("sql")
    try:
        with engine.begin() as connection:
            connection.execute(insert(table).values(key="identity", value=value.to_bytes()))
        return value
    except IntegrityError:
        # A competing first writer may have won the unique row. Other schema or
        # storage errors remain errors; this branch must read a valid winner.
        with engine.connect() as connection:
            payload = connection.execute(select(table.c.value).where(table.c.key == "identity")).scalar_one()
        return StoreIdentity.from_bytes(sql_identity_payload(payload), backend="sql")


def memory_filesystem_identity(fs, path, *, existing_id=None):
    """Arbitrate fsspec's process local memory filesystem across store handles."""
    with _MEMORY_FS_LOCK:
        if fs.exists(path):
            with fs.open(path, "rb") as stream:
                return _check_existing(StoreIdentity.from_bytes(stream.read(_LIMIT+1), backend="object"), existing_id)
        if existing_id is not None:
            raise ValueError("store identity is missing from bound durable state; explicit recovery required")
        value = StoreIdentity.create("object")
        with fs.open(path, "wb") as stream:
            stream.write(value.to_bytes())
        return value
