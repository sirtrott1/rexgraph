"""Explicit collection of unpublished staging; retained history is never pruned.

Plans are portable checked declarations, not delete capabilities. Application
rechecks ownership, the current cursor and every candidate under the provider's
writer gate. Object maintenance advances the conditional head epoch before any
deletion, fencing writers that staged bytes against an older head.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import time

from rexgraph.value_codec import pack_value

from .checkpoint import _decode, _encode
from .engine import ChangeCursor
from .envelope import _hex_identity

PLAN_LIMIT = 4*1024*1024
INVENTORY_LIMIT = 100000
_NAMES = {"blobs", "commits", "frames", "replay"}


@dataclass(frozen=True)
class RetentionPolicy:
    grace_seconds: float = 86400.
    max_objects: int = 256
    max_bytes: int = 256*1024*1024

    def __post_init__(self):
        if (type(self.grace_seconds) not in (int, float) or not math.isfinite(self.grace_seconds)
                or not 0 <= self.grace_seconds <= 2**53):
            raise ValueError("retention grace requires a finite nonnegative number")
        if type(self.max_objects) is not int or not 0 < self.max_objects <= 4096:
            raise ValueError("retention requires an object limit from 1 to 4096")
        if type(self.max_bytes) is not int or not 0 < self.max_bytes <= 2**40:
            raise ValueError("retention requires a bounded positive byte limit")

    def as_record(self):
        return {"grace_seconds": self.grace_seconds, "max_objects": self.max_objects, "max_bytes": self.max_bytes}


@dataclass(frozen=True)
class OrphanObject:
    namespace: str
    address: str | tuple[str, int]
    digest: str
    size: int
    modified: float | None

    def __post_init__(self):
        if type(self.namespace) is not str or self.namespace not in _NAMES:
            raise ValueError("unknown staging namespace")
        if type(self.address) is str:
            _hex_identity(self.address, 64, "staging address")
        elif (type(self.address) is not tuple or len(self.address) != 2
                or type(self.address[0]) is not str or not self.address[0]
                or type(self.address[1]) is not int or not 0 < self.address[1] < 2**63):
            raise ValueError("invalid staging record address")
        _hex_identity(self.digest, 64, "staging content digest")
        if type(self.size) is not int or not 0 <= self.size <= 2**40:
            raise ValueError("invalid staging size")
        if self.modified is not None and (type(self.modified) not in (int, float) or not math.isfinite(self.modified)):
            raise ValueError("invalid staging modification time")

    def as_record(self):
        return dict(namespace=self.namespace, address=self.address, digest=self.digest,
                    size=self.size, modified=self.modified)


@dataclass(frozen=True)
class RetentionPlan:
    cursor: ChangeCursor
    epoch: int
    policy: RetentionPolicy
    planned_at: float
    candidates: tuple[OrphanObject, ...]

    def __post_init__(self):
        if not isinstance(self.cursor, ChangeCursor) or not isinstance(self.policy, RetentionPolicy):
            raise TypeError("retention requires a checked cursor and policy")
        if type(self.epoch) is not int or not 0 <= self.epoch < 2**63:
            raise ValueError("invalid retention maintenance epoch")
        if type(self.planned_at) not in (int, float) or not math.isfinite(self.planned_at):
            raise ValueError("invalid retention planning time")
        if (type(self.candidates) is not tuple or len(self.candidates) > self.policy.max_objects
                or any(not isinstance(item, OrphanObject) for item in self.candidates)
                or len({(item.namespace, item.address) for item in self.candidates}) != len(self.candidates)
                or sum(item.size for item in self.candidates) > self.policy.max_bytes):
            raise ValueError("retention candidates exceed their declared bounds")

    def to_bytes(self):
        return _encode(b"RGRT1", pack_value(dict(retention_version=1, cursor=self.cursor.as_record(),
            epoch=self.epoch, policy=self.policy.as_record(), planned_at=self.planned_at,
            candidates=tuple(item.as_record() for item in self.candidates))), PLAN_LIMIT)

    @classmethod
    def from_bytes(cls, raw):
        value = _decode(raw, b"RGRT1", PLAN_LIMIT)
        if (type(value) is not dict or set(value) != {"retention_version", "cursor", "epoch", "policy", "planned_at", "candidates"}
                or type(value["retention_version"]) is not int or value["retention_version"] != 1
                or type(value["policy"]) is not dict or set(value["policy"]) != {"grace_seconds", "max_objects", "max_bytes"}
                or type(value["candidates"]) is not tuple or len(value["candidates"]) > 4096):
            raise ValueError("unknown retention plan declaration")
        entries = []
        for item in value["candidates"]:
            if type(item) is not dict or set(item) != {"namespace", "address", "digest", "size", "modified"}:
                raise ValueError("unknown retention candidate declaration")
            entries.append(OrphanObject(**item))
        result = cls(ChangeCursor.from_record(value["cursor"]), value["epoch"], RetentionPolicy(**value["policy"]),
                     value["planned_at"], tuple(entries))
        if result.to_bytes() != raw:
            raise ValueError("noncanonical retention plan")
        return result


@contextmanager
def _scope(store, *, write):
    from .core import MemoryStore, SQLStore
    from .localstore import LocalStore
    from .object_native import NativeObjectStore
    from .object_publication import LocalObjectPublication, MemoryObjectPublication
    if type(store) is MemoryStore:
        with store.read_transaction(): yield "memory"
    elif type(store) is LocalStore:
        with store._scope(write=write): yield "local"
    elif type(store) is NativeObjectStore:
        if type(store._provider) not in (LocalObjectPublication, MemoryObjectPublication):
            raise NotImplementedError("object retention requires a qualified exclusive maintenance capability")
        # Planning must inspect a stable object inventory as well as the head.
        with store._transaction_lock, store._provider.transaction(), store._scope(write=write): yield "object"
    elif type(store) is SQLStore and store.engine.dialect.name == "sqlite" and store._native is not None:
        with store._sql_transaction(write=write): yield "sql"
    else:
        raise NotImplementedError("retention is qualified for native memory, local, SQLite and included object providers")


def _state(store, kind):
    return store._native.state if kind == "sql" else store._state


def _protected(store, kind):
    result = set()
    state = _state(store, kind)
    if kind in ("memory", "sql"):
        # Also preserve injected compatibility metadata: it is not orphan staging.
        for rows in state._rows.values():
            for row in rows:
                result.update((("blobs", (row.id, row.version)), ("commits", (row.id, row.version))))
    else:
        for frame in state.changes(limit=2**63-1):
            result.add(("frames", frame.digest))
            change = frame.mutation
            if change.blob_digest is not None: result.add(("blobs", change.blob_digest))
            if change.commit_digest is not None:
                address = store._commit_path(frame.record_id, change.version).name if kind == "local" else change.commit_digest
                result.add(("commits", address))
        checkpoint = store._read_checkpoint() if kind == "local" else store._head_checkpoint
        if checkpoint is not None:
            checkpoint = checkpoint.checkpoint if kind == "local" else checkpoint
            # Old published segment references remain in newer descriptors.
            result.update(("replay", reference.digest) for reference in checkpoint.segments)
    return result


def _inventory(store, kind):
    """Metadata first; candidate reads obey the requested total byte budget."""
    result = []
    def add(namespace, address, size, modified):
        if len(result) >= INVENTORY_LIMIT: raise ValueError("retention inventory exceeds its object limit")
        result.append((namespace, address, size, modified))
    if kind == "memory":
        for namespace, values in (("blobs", store._blobs), ("commits", store._commit_blobs)):
            for address, raw in values.items(): add(namespace, address, len(raw), None)
    elif kind == "sql":
        from sqlalchemy import func, select
        table = store.commits_table
        with store._sql_connection.execute(select(table.c.id, table.c.version, func.length(table.c.artifact))) as rows:
            for rid, version, size in rows: add("commits", (rid, version), size, None)
    else:
        from .object_publication import MemoryObjectPublication
        if kind == "object" and type(store._provider) is MemoryObjectPublication:
            prefix = store._provider.root.rstrip("/")+"/"
            for path, stream in store.fs.store.items():
                if not path.startswith(prefix): continue
                key = path[len(prefix):].split("/")
                if len(key) == 2 and key[0] in _NAMES and len(key[1]) == 64 and all(c in "0123456789abcdef" for c in key[1]):
                    add(key[0], key[1], stream.size, stream.modified.timestamp())
        else:
            root = store._root if kind == "local" else Path(store._provider.root)
            for namespace in sorted(_NAMES-({"frames"} if kind == "local" else set())):
                directory = root/namespace
                if directory.is_symlink(): raise ValueError("retention storage cannot use symbolic links")
                if not directory.exists(): continue
                if not directory.is_dir(): raise ValueError("retention storage requires directories")
                for path in directory.iterdir():
                    if path.is_symlink() or not path.is_file(): raise ValueError("retention storage requires regular files")
                    if len(path.name) == 64 and all(c in "0123456789abcdef" for c in path.name):
                        stat = path.stat(); add(namespace, path.name, stat.st_size, stat.st_mtime)
    return sorted(result, key=lambda item: pack_value(item[:2]))


def _read(store, kind, namespace, address, limit):
    if kind == "memory": return {"blobs": store._blobs, "commits": store._commit_blobs}[namespace][address]
    if kind == "sql":
        from sqlalchemy import select
        table = store.commits_table
        return bytes(store._sql_connection.execute(select(table.c.artifact).where(
            table.c.id == address[0], table.c.version == address[1])).scalar_one())
    if kind == "object":
        value = store._read(namespace+"/"+address, limit)
        if value is None: raise ValueError("retention candidate is missing")
        return value.data
    from .localstore import _read_bytes
    return _read_bytes(store._root/namespace/address, limit)


def _eligible(item, protected, policy, now):
    namespace, address, size, modified = item
    return ((namespace, address) not in protected and
            (policy.grace_seconds == 0 or modified is not None and modified <= now-policy.grace_seconds))


def plan_retention(store, policy=None):
    policy = RetentionPolicy() if policy is None else policy
    if not isinstance(policy, RetentionPolicy): raise TypeError("retention requires a declared policy")
    with _scope(store, write=False) as kind:
        now, used, candidates = time.time(), 0, []
        protected = _protected(store, kind)
        for item in _inventory(store, kind):
            if not _eligible(item, protected, policy, now) or item[2] > policy.max_bytes-used: continue
            if len(candidates) >= policy.max_objects: break
            raw = _read(store, kind, item[0], item[1], item[2])
            if len(raw) != item[2]: raise ValueError("staging bytes changed during retention planning")
            candidates.append(OrphanObject(item[0], item[1], hashlib.sha256(raw).hexdigest(), item[2], item[3]))
            used += len(raw)
        result = RetentionPlan(_state(store, kind).cursor, getattr(store, "_head_epoch", 0), policy, now, tuple(candidates))
        result.to_bytes()
        return result


def apply_retention(store, plan):
    if not isinstance(plan, RetentionPlan): raise TypeError("retention application requires a declared plan")
    plan.to_bytes()
    with _scope(store, write=True) as kind:
        if _state(store, kind).cursor != plan.cursor or getattr(store, "_head_epoch", 0) != plan.epoch:
            raise ValueError("retention plan is stale or belongs to another store")
        protected = _protected(store, kind)
        inventory = {(item[0], item[1]): item for item in _inventory(store, kind)}
        # Verify ALL candidates before deleting anything. A forged plan cannot
        # nominate published data or change the policy's age/size preconditions.
        for candidate in plan.candidates:
            item = inventory.get((candidate.namespace, candidate.address))
            if (item is None or item[2:] != (candidate.size, candidate.modified)
                    or not _eligible(item, protected, plan.policy, min(plan.planned_at, time.time()))
                    or hashlib.sha256(_read(store, kind, candidate.namespace, candidate.address, candidate.size)).hexdigest() != candidate.digest):
                raise ValueError("retention candidate changed or is protected; no deletion performed")
        if kind == "object" and plan.candidates:
            from .core import PublicationUncertainError, VersionConflictError
            from .object_native import ObjectHead
            head = ObjectHead(store.header, store._state.cursor, store._head_checkpoint, store._head_epoch+1)
            raw = head.to_bytes()
            try:
                value = store._acknowledged(store._provider.compare_and_swap("store.head", store._head_token, raw), raw)
            except VersionConflictError: raise
            except BaseException as exc:
                store._publication_uncertain = True
                if isinstance(exc, Exception):
                    raise PublicationUncertainError("retention fence publication is uncertain; reopen and verify") from exc
                raise
            store._head_token, store._head_epoch = value.token, head.epoch
        directories = set()
        for candidate in plan.candidates:
            namespace, address = candidate.namespace, candidate.address
            if kind == "memory": del {"blobs": store._blobs, "commits": store._commit_blobs}[namespace][address]
            elif kind == "sql":
                from sqlalchemy import delete
                table = store.commits_table
                store._sql_connection.execute(delete(table).where(table.c.id == address[0], table.c.version == address[1]))
            elif kind == "object" and store.fs.protocol == "memory":
                store.fs.rm(store._provider.path(namespace+"/"+address))
            else:
                root = store._root if kind == "local" else Path(store._provider.root)
                path = root/namespace/address
                path.unlink(); directories.add(path.parent)
        for directory in sorted(directories):
            fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try: os.fsync(fd)
            finally: os.close(fd)
        return {"deleted_objects": len(plan.candidates), "deleted_bytes": sum(item.size for item in plan.candidates),
                "history_retained": True, "cursor": plan.cursor, "epoch": getattr(store, "_head_epoch", 0)}
