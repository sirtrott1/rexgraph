"""
agent.courier_remote: couriers that cross machines, over the rexgraph native protocol.

A local courier moves records between two RCStores it can both open. Across machines it
cannot: the far side is a running server, reachable only through `/rex/v1`, where a complex
crosses as the layered binary it is already stored as and is checked twice on arrival (the
header digest for the bytes, the chain condition for whether those bytes are a complex).
`agent.client.RexClient` already speaks that surface, including the HMAC both directions
when a deployment sets `REXGRAPH_FRAME_KEY`, so a peer here is that client plus the one
thing the protocol deliberately does not provide.

WHAT THE PROTOCOL DOES NOT PROVIDE, AND WHY IT MATTERS HERE. `/rex/v1/store` mints the
record id itself, randomly, and there is no list or query over stored records. That is a
deliberate choice on the server: an id that can be guessed from the one before it makes the
ownership check the only thing standing between a caller and every record in the store. The
consequence for a courier is precise. The sender is the only party who can answer "have I
already shipped this", because it is the only party that knows both its own record id and
the id the server minted back. So the ledger lives on the sending side. It is not a cache of
the peer's state and must not be read as one: it records what this courier sent and what it
was told in reply.

The ceilings are read rather than discovered by failing. `/rex/v1/hello` reports max_cells,
so a record too large for the peer is reported as `oversize` before anything crosses the
wire, using the same comparison the server's own `check_size` makes against a frame header.

Failures are per-record, never per-trip. A refused shipment, an unreachable peer, an
oversized record: each lands as one shipment with a reason, and the rest of the trip
continues, which is the same fail-soft contract the local courier gives an unreadable blob.

    peer = Peer("gpu-box", RexClient("https://gpu-box:8000", api_key=tok))
    courier.attach_peer(peer)
    courier.deliver("alpha", "gpu-box")
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .courier import structure_of

logger = logging.getLogger(__name__)

# Header fields the server measures against its cell ceiling. Checking the same three
# locally means a record too big for the peer never leaves this machine.
SIZE_FIELDS = ("nV", "nE", "nF")


@dataclass
class Shipment:
    """One record's fate on one crossing.

    `shipped` when the peer took it and named it, `held` when the ledger already records
    this structure at this peer, `oversize` when it is past the peer's declared ceiling,
    `refused` when the peer answered with an error, and `unreadable` when the local store
    could not produce the complex."""
    record_id: str
    reason: str
    remote_id: str | None = None
    detail: str = ""

    @property
    def shipped(self) -> bool:
        return self.reason == "shipped"

    def public(self) -> dict:
        return {"record_id": self.record_id, "reason": self.reason,
                "shipped": self.shipped, "remote_id": self.remote_id,
                "detail": self.detail}


class Ledger:
    """What this courier has shipped to which peer, and the structure it sent.

    Keyed by (peer, record id) and holding the id the peer minted back, so a second trip
    can skip a record without asking the peer a question the protocol has no route for.
    `path` makes it durable across processes, which is what a courier that runs from cron
    on two machines needs; without one it lives for the process."""

    def __init__(self, path: str | None = None):
        self.path = Path(path).expanduser() if path else None
        self._entries: dict[str, dict] = {}
        if self.path is not None and self.path.exists():
            self.load()

    @staticmethod
    def _key(peer: str, record_id: str) -> str:
        return f"{peer}\x1f{record_id}"

    def note(self, peer: str, record_id: str, remote_id: str, structure: dict) -> None:
        self._entries[self._key(peer, record_id)] = {
            "peer": peer, "record_id": record_id, "remote_id": remote_id,
            "structure": structure, "at": time.time()}
        self.save()

    def entry(self, peer: str, record_id: str) -> dict | None:
        return self._entries.get(self._key(peer, record_id))

    def remote_id(self, peer: str, record_id: str) -> str | None:
        e = self.entry(peer, record_id)
        return e["remote_id"] if e else None

    def structure(self, peer: str, record_id: str) -> dict | None:
        e = self.entry(peer, record_id)
        return e["structure"] if e else None

    def forget(self, peer: str, record_id: str) -> bool:
        """Drop one entry, so the next trip ships it again. This is the repair when a peer
        has lost a record: the ledger records what was sent, and cannot notice a deletion
        on the far side."""
        gone = self._entries.pop(self._key(peer, record_id), None) is not None
        if gone:
            self.save()
        return gone

    def entries(self, peer: str | None = None) -> list[dict]:
        return [e for e in self._entries.values() if peer is None or e["peer"] == peer]

    def to_dict(self) -> dict:
        return dict(self._entries)

    def load(self) -> None:
        try:
            self._entries = json.loads(self.path.read_text() or "{}")
        except (OSError, ValueError):
            logger.debug("ledger at %s is unreadable, starting empty", self.path,
                         exc_info=True)
            self._entries = {}

    def save(self) -> None:
        """Write through, atomically. A ledger torn by a crash mid-write would claim a
        record was shipped when it was not, or the reverse."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._entries))
            tmp.replace(self.path)
        except OSError:
            logger.debug("could not write ledger at %s", self.path, exc_info=True)


@dataclass
class Peer:
    """A remote rexgraph server as a courier destination.

    `confirm` fetches each shipment back and compares fingerprints. It costs a second
    round trip per record and is off by default, because the protocol already verifies the
    digest and the chain condition on arrival; turn it on when the receipt matters more
    than the traffic."""
    name: str
    client: object
    ledger: Ledger = field(default_factory=Ledger)
    confirm: bool = False
    _hello: dict | None = field(default=None, repr=False)

    #### what the peer says it will accept
    def hello(self, *, refresh: bool = False) -> dict:
        if self._hello is None or refresh:
            self._hello = self.client.rex_hello()
        return self._hello

    def limits(self) -> dict:
        try:
            return dict(self.hello().get("limits") or {})
        except Exception:
            logger.debug("peer %s did not answer hello", self.name, exc_info=True)
            return {}

    def oversize(self, signature: dict) -> str:
        """The field that puts this record past the peer's ceiling, or an empty string.
        Mirrors the server's own check_size, which measures the frame header rather than
        the built complex, so the answer here is the answer there."""
        cap = self.limits().get("max_cells")
        if not cap:
            return ""
        for f in SIZE_FIELDS:
            n = (signature or {}).get(f)
            if n is not None and int(n) > int(cap):
                return f"{f}={int(n)} is over the peer's {int(cap)}-cell limit"
        return ""

    #### the crossing
    def ship(self, record, rex, *, source: str, courier: str) -> Shipment:
        from rexgraph.protocol import fingerprint
        sig = dict(record.signature or {})
        over = self.oversize(sig)
        if over:
            return Shipment(record.id, "oversize", detail=over)

        structure = structure_of(sig)
        if self.ledger.structure(self.name, record.id) == structure:
            return Shipment(record.id, "held",
                            remote_id=self.ledger.remote_id(self.name, record.id))

        meta = {"record_id": record.id, "source_hive": source, "courier": courier,
                "source_version": record.version, "shipped_at": time.time(),
                "tags": list(sig.get("tags") or []),
                "kind": (record.meta or {}).get("kind", "")}
        try:
            out = self.client.rex_store(rex, **meta)
        except Exception as e:
            return Shipment(record.id, "refused", detail=_reason(e))
        remote_id = (out or {}).get("record_id")
        if not remote_id:
            return Shipment(record.id, "refused", detail="peer returned no record id")

        if self.confirm:
            try:
                back = self.client.rex_fetch(remote_id)
            except Exception as e:
                return Shipment(record.id, "refused", remote_id=remote_id,
                                detail=f"stored but unconfirmable: {_reason(e)}")
            if fingerprint(back) != fingerprint(rex):
                return Shipment(record.id, "refused", remote_id=remote_id,
                                detail="the peer returned a different complex")

        self.ledger.note(self.name, record.id, remote_id, structure)
        return Shipment(record.id, "shipped", remote_id=remote_id)

    def retrieve(self, record_id: str):
        """A record this courier shipped, back from the peer. Addressed by the LOCAL id,
        since the ledger is what translates it into the id the peer minted."""
        remote_id = self.ledger.remote_id(self.name, record_id)
        if remote_id is None:
            raise ValueError(f"nothing shipped to {self.name!r} under {record_id!r}")
        return self.client.rex_fetch(remote_id)

    def status(self) -> dict:
        return {"peer": self.name, "shipped": len(self.ledger.entries(self.name)),
                "limits": self.limits(), "confirm": self.confirm}


def _reason(exc: Exception) -> str:
    """A failure in one line, with the peer's status code when there was one. The server
    sanitizes its own 5xx bodies, so what reaches here is already what a client may see."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        body = ""
        try:
            body = json.dumps(resp.json())[:200]
        except Exception:
            body = (resp.text or "")[:200]
        return f"{resp.status_code}: {body}" if body else str(resp.status_code)
    return f"{type(exc).__name__}: {exc}"
