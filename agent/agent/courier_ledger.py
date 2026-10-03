"""Bounded sender receipts with shared local arbitration and durable publication.

This ledger attests acknowledged shipments. It is not a receiver state cache or
an exactly once protocol. A lost remote response still needs reconciliation.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import stat
from threading import Lock, RLock, local
import time

from rexgraph.io.publication import publication_lock, staged_publication
from rexgraph.value_codec import pack_value, unpack_value

try:
    import fcntl
except ImportError:
    fcntl = None

LEDGER_LIMIT = 16*1024*1024
_FIELDS = {"peer", "record_id", "remote_id", "structure", "at"}
_GUARD, _GATES, _FDS, _LOCAL = Lock(), {}, set(), local()


def _child():
    global _GUARD, _GATES, _FDS, _LOCAL
    for fd in _FDS:
        os.close(fd)
    _GUARD, _GATES, _FDS, _LOCAL = Lock(), {}, set(), local()


def _before_fork():
    _GUARD.acquire()


def _parent():
    _GUARD.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_parent=_parent, after_in_child=_child)


@contextmanager
def _file_gate(path, *, create):
    # The stable sidecar must survive atomic ledger replacement. Core's parent
    # directory gate remains responsible for the short publication itself; never
    # hold that broad gate across a network call to a co hosted receiver.
    if path.is_symlink():
        raise ValueError("courier ledger lock must not be a symbolic link")
    fd, key, gate, owner = None, None, None, os.getpid()
    held = getattr(_LOCAL, "held", None)
    if held is None:
        held = _LOCAL.held = {}
    try:
        with _GUARD:
            fd = os.open(path, (os.O_RDWR | os.O_CREAT if create else os.O_RDONLY) | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0), 0o600)
            _FDS.add(fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("courier ledger lock requires a regular file")
            key = (info.st_dev, info.st_ino)
            canonical = os.path.normcase(os.path.realpath(path))
            if key not in held and any(other > canonical for other in held.values()):
                raise RuntimeError("distinct courier ledger locks require canonical path order")
            gate = _GATES.setdefault(key, [RLock(), 0])
            gate[1] += 1
        with gate[0]:
            if key not in held:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                held[key] = canonical
                try:
                    yield
                finally:
                    held.pop(key)
            else:
                yield
    finally:
        if owner == os.getpid():
            with _GUARD:
                if fd is not None:
                    os.close(fd)
                    _FDS.discard(fd)
                if gate is not None:
                    gate[1] -= 1
                    if not gate[1]:
                        del _GATES[key]


class LedgerUncertainError(RuntimeError):
    """Ledger publication may have succeeded; explicitly reload before reuse."""


def _text(value):
    if type(value) is not str or not value or len(value.encode("utf-8")) > 1024*1024:
        raise ValueError("ledger addresses require bounded nonempty literal text")
    return value


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate courier ledger field")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("courier ledger requires finite JSON numbers")


def _float(value):
    number = float(value)
    if not isfinite(number):
        _constant(value)
    return number


def _entry(value):
    if (type(value) is not dict or not _FIELDS <= value.keys()
            or set(value)-(_FIELDS|{"receipt"}) or type(value["structure"]) is not dict
            or type(value["at"]) not in (int, float) or not isfinite(value["at"])):
        raise ValueError("invalid courier ledger entry")
    for key in ("peer", "record_id", "remote_id"):
        _text(value[key])
    structure = value["structure"]
    for key in ("state_digest", "record_digest"):
        if key in structure and (type(structure[key]) is not str or len(structure[key]) != 64
                                 or any(c not in "0123456789abcdef" for c in structure[key])):
            raise ValueError("ledger state identities require canonical SHA256 digests")
    if "record_digest" in structure or "receipt" in value:
        from rcdb import CopyReceipt
        from rcdb.transfer import RECEIPT_LIMIT
        raw = value.get("receipt")
        if (type(raw) is not str or len(raw) > RECEIPT_LIMIT*2
                or set(structure) != {"state_digest", "record_digest"}):
            raise ValueError("record ledger identity requires its checked copy receipt")
        try:
            receipt = CopyReceipt.from_bytes(bytes.fromhex(raw))
            if receipt.to_bytes().hex() != raw:
                raise ValueError("noncanonical copy receipt text")
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid courier ledger copy receipt") from exc
        if (receipt.source_record_id != value["record_id"]
                or receipt.destination_record_id != value["remote_id"]
                or receipt.source_digest != structure["state_digest"]
                or receipt.destination_digest != structure["state_digest"]):
            raise ValueError("ledger copy receipt differs from its address or state")
    return value


def _canonical(entries):
    raw = json.dumps(entries, ensure_ascii=False, allow_nan=False,
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > LEDGER_LIMIT:
        raise ValueError("courier ledger exceeds its byte limit")
    return raw


def _archive_entries(raw):
    if type(raw) is not bytes or len(raw) > LEDGER_LIMIT:
        raise ValueError("invalid courier retention archive bytes")
    data = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    if type(data) is not dict or len(data) > 4096:
        raise ValueError("courier retention archive exceeds its entry limit")
    for key, value in data.items():
        _entry(value)
        if key != Ledger._key(value["peer"], value["record_id"]):
            raise ValueError("courier retention archive differs from its literal addresses")
    if _canonical(data) != raw:
        raise ValueError("courier retention archive is not canonical")
    return data


@dataclass(frozen=True)
class LedgerRetentionPlan:
    owner: str
    base_digest: str
    before: float
    peer: str | None
    archive_bytes: bytes

    def __post_init__(self):
        for digest in (self.owner, self.base_digest):
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("courier retention requires canonical identities")
        if type(self.before) not in (int, float) or not isfinite(self.before):
            raise ValueError("courier retention cutoff requires a finite number")
        if self.peer is not None: _text(self.peer)
        entries = _archive_entries(self.archive_bytes)
        if any(e["at"] >= self.before or self.peer is not None and e["peer"] != self.peer for e in entries.values()):
            raise ValueError("courier retention archive differs from its cutoff or peer")

    def to_bytes(self):
        body = pack_value(dict(ledger_retention_version=1, owner=self.owner, base_digest=self.base_digest,
                              before=self.before, peer=self.peer, archive_bytes=self.archive_bytes))
        if len(body) > LEDGER_LIMIT+4096: raise ValueError("courier retention plan exceeds its byte limit")
        return b"RGLR1"+hashlib.sha256(b"RGLR1"+body).digest()+body

    @classmethod
    def from_bytes(cls, raw):
        if (type(raw) is not bytes or not 37 < len(raw) <= LEDGER_LIMIT+4096+37
                or raw[:5] != b"RGLR1" or raw[5:37] != hashlib.sha256(raw[:5]+raw[37:]).digest()):
            raise ValueError("invalid courier retention plan frame")
        value = unpack_value(raw[37:])
        if (type(value) is not dict or set(value) != {"ledger_retention_version", "owner", "base_digest", "before", "peer", "archive_bytes"}
                or type(value["ledger_retention_version"]) is not int or value.pop("ledger_retention_version") != 1):
            raise ValueError("unknown courier retention plan declaration")
        result = cls(**value)
        if result.to_bytes() != raw: raise ValueError("noncanonical courier retention plan")
        return result


class Ledger:
    """Owned sender entries, refreshed under one stable ledger sidecar gate.

    Independent handles and POSIX processes sharing a file serialize a complete
    shipment decision and write. A memory ledger serializes its own threads.
    File fallback without flock supplies thread arbitration only. A shared ledger
    serializes network shipments; separate ledgers can proceed independently.
    Missing previously observed files and malformed data require operator recovery.
    """
    def __init__(self, path: str | None = None):
        self.path = Path(path).expanduser().absolute() if path else None
        self._gate_path = (self.path.with_name(".courier-"+hashlib.sha256(
            os.fsencode(self.path.name)).hexdigest()+".lock") if self.path else None)
        self._entries, self._depth = {}, 0
        self._lock, self._pid = RLock(), os.getpid()
        self._seen_file, self._uncertain = False, False
        self._maintenance_owner = hashlib.sha256(b"rexgraph-courier-retention\x00"+
            (os.fsencode(os.path.realpath(self.path)) if self.path else os.urandom(32))).hexdigest()
        self.load()

    @staticmethod
    def _key(peer, record_id):
        return hashlib.sha256(b"rexgraph-courier-ledger\x00"+pack_value((_text(peer), _text(record_id)))).hexdigest()

    def _refresh(self):
        if self.path is None:
            return
        if self.path.is_symlink() or self.path.exists() and not self.path.is_file():
            raise ValueError("courier ledger requires a regular file")
        try:
            with self.path.open("rb") as stream:
                raw = stream.read(LEDGER_LIMIT+1)
        except FileNotFoundError as exc:
            if self._seen_file:
                raise ValueError("previously published courier ledger is missing; explicit recovery required") from exc
            self._entries = {}
            return
        if len(raw) > LEDGER_LIMIT:
            raise ValueError("courier ledger exceeds its byte limit")
        try:
            data = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
            if type(data) is not dict:
                raise ValueError("courier ledger requires an entry mapping")
            entries = {}
            for value in data.values():
                _entry(value)
                # Historical separator keys upgrade only from their literal fields.
                key = self._key(value["peer"], value["record_id"])
                if key in entries:
                    raise ValueError("ambiguous courier ledger address")
                entries[key] = value
        except (TypeError, KeyError, RecursionError, OverflowError) as exc:
            raise ValueError("invalid courier ledger declaration") from exc
        self._entries, self._seen_file = entries, True

    @contextmanager
    def _scope(self, *, write=False, reload=False):
        if os.getpid() != self._pid:
            raise RuntimeError("inherited courier ledger cannot be reused after fork; open a fresh handle")
        with self._lock:
            if self._uncertain and not reload:
                raise LedgerUncertainError("courier ledger publication is uncertain; explicitly reload and verify")
            if self._depth:
                self._refresh()
                yield self
                return
            if self.path is not None and write:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            gate = (_file_gate(self._gate_path, create=write) if self.path is not None
                    and (write or self._gate_path.exists() or self._gate_path.is_symlink()) else nullcontext())
            with gate:
                self._refresh()
                self._depth = 1
                try:
                    yield self
                finally:
                    self._depth = 0

    @contextmanager
    def delivery_scope(self, peer, record_id):
        """Serialize check, network acknowledgement and note for this sender ledger."""
        self._key(peer, record_id)
        with self._scope(write=True):
            yield self

    def _publish(self, entries):
        # Refuse unsupported or nonfinite data before modifying visible state.
        raw = _canonical(entries)
        if self.path is not None:
            prepared = False
            try:
                with staged_publication(self.path) as staged:
                    staged.write_bytes(raw)
                    prepared = True
            except BaseException as exc:
                if prepared:
                    self._uncertain = True
                    if isinstance(exc, Exception):
                        raise LedgerUncertainError("courier ledger publication outcome is uncertain; reload and verify") from exc
                raise
            self._seen_file = True
        self._entries = deepcopy(entries)

    def note(self, peer, record_id, remote_id, structure, *, receipt=None):
        key = self._key(peer, record_id)
        value = dict(peer=peer, record_id=record_id, remote_id=remote_id,
                     structure=deepcopy(structure), at=time.time())
        if receipt is not None:
            from rcdb import CopyReceipt
            if not isinstance(receipt, CopyReceipt):
                raise ValueError("ledger receipt requires a declared CopyReceipt")
            value["receipt"] = receipt.to_bytes().hex()
        _entry(value)
        with self._scope(write=True):
            entries = dict(self._entries)
            entries[key] = value
            self._publish(entries)

    def entry(self, peer, record_id):
        key = self._key(peer, record_id)
        with self._scope():
            return deepcopy(self._entries.get(key))

    def remote_id(self, peer, record_id):
        value = self.entry(peer, record_id)
        return None if value is None else value["remote_id"]

    def structure(self, peer, record_id):
        value = self.entry(peer, record_id)
        return None if value is None else value["structure"]

    def forget(self, peer, record_id):
        key = self._key(peer, record_id)
        with self._scope(write=True):
            if key not in self._entries:
                return False
            entries = dict(self._entries)
            entries.pop(key)
            self._publish(entries)
            return True

    def entries(self, peer=None):
        if peer is not None:
            _text(peer)
        with self._scope():
            return deepcopy([e for e in self._entries.values() if peer is None or e["peer"] == peer])

    def to_dict(self):
        with self._scope():
            return deepcopy(self._entries)

    def load(self):
        """Explicitly reload and verify, also resolving an unknown local write outcome."""
        with self._scope(reload=True):
            self._uncertain = False

    def save(self):
        """Republish the current refreshed image, preserving other handles' entries."""
        with self._scope(write=True):
            self._publish(self._entries)

    def plan_retention(self, *, before, peer=None, max_entries=256):
        """Select inactive acknowledgements for explicit archive, without removal.

        The ledger already keeps one latest entry per peer/record. Retiring an
        entry forgets the live shipment decision, so the checked plan carries its
        complete receipt for operator reconciliation and must be archived first.
        """
        if type(before) not in (int, float) or not isfinite(before):
            raise ValueError("courier retention cutoff requires a finite number")
        if peer is not None: _text(peer)
        if type(max_entries) is not int or not 0 < max_entries <= 4096:
            raise ValueError("courier retention requires an entry limit from 1 to 4096")
        with self._scope():
            selected = sorted(((key, value) for key, value in self._entries.items()
                if value["at"] < before and (peer is None or value["peer"] == peer)),
                key=lambda item: (item[1]["at"], item[0]))[:max_entries]
            return LedgerRetentionPlan(self._maintenance_owner, hashlib.sha256(_canonical(self._entries)).hexdigest(),
                                       before, peer, _canonical(dict(selected)))

    def apply_retention(self, plan, *, archive_path):
        """Durably archive the checked plan before removing any selected entries."""
        if not isinstance(plan, LedgerRetentionPlan): raise TypeError("courier retention requires a declared plan")
        archive = Path(archive_path).expanduser().absolute()
        if archive.is_symlink() or archive.exists() and not archive.is_file():
            raise ValueError("courier retention archive requires a regular file")
        if self.path is not None and os.path.realpath(archive) in {
                os.path.realpath(self.path), os.path.realpath(self._gate_path)}:
            raise ValueError("courier retention archive cannot replace the ledger or its gate")
        with self._scope(write=True):
            if (plan.owner != self._maintenance_owner or plan.base_digest != hashlib.sha256(_canonical(self._entries)).hexdigest()):
                raise ValueError("courier retention plan is stale or belongs to another ledger")
            retired = _archive_entries(plan.archive_bytes)
            if any(self._entries.get(key) != value for key, value in retired.items()):
                raise ValueError("courier retention candidate differs from the current acknowledgement")
            raw = plan.to_bytes()
            archive.parent.mkdir(parents=True, exist_ok=True)
            with publication_lock(archive.parent):
                if archive.is_symlink() or archive.exists() and not archive.is_file():
                    raise ValueError("courier retention archive requires a regular file")
                if archive.exists():
                    with archive.open("rb") as stream: existing = stream.read(len(raw)+1)
                    if existing != raw: raise ValueError("courier retention archive already contains different bytes")
                else:
                    with staged_publication(archive) as staged: staged.write_bytes(raw)
            entries = {key: value for key, value in self._entries.items() if key not in retired}
            self._publish(entries)
            return {"retired_entries": len(retired), "remaining_entries": len(entries), "archive": str(archive)}
