"""Transfer selected RCDB records through agent.client.RexClient.

Peers advertising record_transfer_version=1 exchange RecordPackets and checked
CopyReceipts through copy_record. Older peers use the graph only protocol.
Configured frame keys authenticate both directions.

The sender ledger records acknowledgements by literal peer and source address.
It does not track remote deletion or provide exactly once publication. Peer cell
limits are checked before sending; each failed record produces its own Shipment.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from .courier_ledger import Ledger, LedgerUncertainError as LedgerUncertainError

logger = logging.getLogger(__name__)

# Header fields the server measures against its cell ceiling. Checking the same three
# locally means a record too big for the peer never leaves this machine.
SIZE_FIELDS = ("nV", "nE", "nF")


@dataclass
class Shipment:
    """One record's fate on one crossing.

    `shipped` when the peer took it and named it, `held` when the ledger already records
    this native state at this peer, `oversize` when it is past the peer's declared ceiling,
    `refused` when the peer answered with an error, and `unreadable` when the local store
    could not produce the complex."""
    record_id: str
    reason: str
    remote_id: str | None = None
    detail: str = ""
    receipt: dict | None = None

    @property
    def shipped(self) -> bool:
        return self.reason == "shipped"

    def public(self) -> dict:
        result = {"record_id": self.record_id, "reason": self.reason,
                "shipped": self.shipped, "remote_id": self.remote_id,
                "detail": self.detail}
        if self.receipt is not None:
            result["receipt"] = self.receipt
        return result


@dataclass
class Peer:
    """A remote rexgraph server as a courier destination.

    `confirm` fetches each shipment back and compares native state digests. It costs a second
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
    def ship(self, record, rex, *, source: str, courier: str, source_store_id=None, snapshot=None) -> Shipment:
        try:
            with self.ledger.delivery_scope(self.name, record.id):
                return self._ship_locked(record, rex, source=source, courier=courier,
                                         source_store_id=source_store_id, snapshot=snapshot)
        except Exception as exc:
            return Shipment(record.id, "refused", detail="sender ledger unavailable: "+_reason(exc))

    def _ship_locked(self, record, rex, *, source, courier, source_store_id, snapshot):
        try:
            modern = self.hello().get("record_transfer_version") == 1
        except Exception as e:
            return Shipment(record.id, "refused", detail=_reason(e))
        if modern:
            return self._ship_record(record, rex, source=source, courier=courier,
                                     source_store_id=source_store_id, snapshot=snapshot)
        from rexgraph.object_identity import object_digest
        sig = dict(record.signature or {})
        over = self.oversize(sig)
        if over:
            return Shipment(record.id, "oversize", detail=over)

        try:
            identity = {"state_digest": object_digest(rex)}
        except Exception as e:
            return Shipment(record.id, "unreadable", detail=_reason(e))
        # Keep the ledger field/API for compatibility; old shape only entries do
        # not match and are upgraded after one successful shipment.
        if self.ledger.structure(self.name, record.id) == identity:
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
                received_digest = object_digest(back)
            except Exception as e:
                return Shipment(record.id, "refused", remote_id=remote_id,
                                detail=f"stored but unconfirmable: {_reason(e)}")
            if received_digest != identity["state_digest"]:
                return Shipment(record.id, "refused", remote_id=remote_id,
                                detail="the peer returned a different complex")

        try:
            self.ledger.note(self.name, record.id, remote_id, identity)
        except Exception as exc:
            return Shipment(record.id, "refused", remote_id=remote_id,
                            detail="stored but sender ledger was not acknowledged: "+_reason(exc))
        return Shipment(record.id, "shipped", remote_id=remote_id)

    def _ship_record(self, record, value, *, source, courier, source_store_id, snapshot):
        from rcdb import RecordPacket, RecordSnapshot
        try:
            if snapshot is None:
                if record.envelope is None:
                    raise ValueError("modern courier requires a selected source snapshot")
                snapshot = RecordSnapshot(record, value, record.envelope.object_digest)
                source_store_id = source_store_id or record.envelope.store_id
            packet = RecordPacket.from_snapshot(snapshot, source_store_id=source_store_id)
            counts = packet.cell_counts()
            over = self.oversize(counts)
            limits = self.limits()
            if limits.get("max_cells") and counts.get("T", 0) > int(limits["max_cells"]):
                over = "temporal history is over the peer's cell limit"
            if limits.get("max_record_bytes") and len(packet.to_bytes()) > int(limits["max_record_bytes"]):
                over = "record packet is over the peer's byte limit"
            if over:
                return Shipment(record.id, "oversize", detail=over)
        except Exception as e:
            return Shipment(record.id, "unreadable", detail=_reason(e))
        identity = {"state_digest": packet.state_digest, "record_digest": packet.selection_digest}
        if self.ledger.structure(self.name, record.id) == identity:
            return Shipment(record.id, "held", remote_id=self.ledger.remote_id(self.name, record.id))
        remote_id = None
        receipt = None
        try:
            receipt = self.client.rex_store_record(packet, source=source, courier=courier)
            self._check_receipt(packet, receipt)
            remote_id = receipt.destination_record_id
            if self.confirm:
                self._confirm_record(receipt)
        except Exception as e:
            return Shipment(record.id, "refused", remote_id=remote_id, detail=_reason(e),
                            receipt=receipt.as_record() if remote_id is not None else None)
        try:
            self.ledger.note(self.name, record.id, remote_id, identity, receipt=receipt)
        except Exception as exc:
            return Shipment(record.id, "refused", remote_id=remote_id, receipt=receipt.as_record(),
                            detail="stored but sender ledger was not acknowledged: "+_reason(exc))
        return Shipment(record.id, "shipped", remote_id=remote_id, receipt=receipt.as_record())

    @staticmethod
    def _check_receipt(packet, receipt):
        from rcdb import CopyReceipt
        if not isinstance(receipt, CopyReceipt):
            raise ValueError("peer returned no checked copy receipt")
        if ((receipt.source_store_id, receipt.source_record_id, receipt.source_version, receipt.source_digest)
                != (packet.source_store_id, packet.record.id, packet.record.version, packet.state_digest)
                or receipt.destination_digest != packet.state_digest):
            raise ValueError("peer receipt differs from the selected source")

    def _confirm_record(self, receipt):
        from rcdb import RecordPacket
        packet = self.client.rex_fetch_record(receipt.destination_record_id, version=receipt.destination_version)
        if not isinstance(packet, RecordPacket) or (
                packet.source_store_id, packet.record.id, packet.record.version, packet.state_digest) != (
                receipt.destination_store_id, receipt.destination_record_id,
                receipt.destination_version, receipt.destination_digest):
            raise ValueError("peer returned a different record")
        return packet.snapshot()

    def reconcile(self, packet, receipt):
        """Verify a known destination receipt and record it without another POST.

        Explicit reconciliation reloads an uncertain local ledger, then fetches
        the exact immutable destination address/version through RexClient. An
        unknown remote address after a lost response cannot be recovered here.
        """
        from rcdb import RecordPacket
        if not isinstance(packet, RecordPacket):
            raise TypeError("courier reconciliation requires a RecordPacket")
        self._check_receipt(packet, receipt)
        self.ledger.load()
        with self.ledger.delivery_scope(self.name, packet.record.id):
            self._confirm_record(receipt)
            identity = {"state_digest": packet.state_digest, "record_digest": packet.selection_digest}
            self.ledger.note(self.name, packet.record.id, receipt.destination_record_id, identity, receipt=receipt)
        return Shipment(packet.record.id, "held", remote_id=receipt.destination_record_id,
                        detail="verified destination receipt reconciled", receipt=receipt.as_record())

    def retrieve(self, record_id: str):
        """A record this courier shipped, back from the peer. Addressed by the LOCAL id,
        since the ledger is what translates it into the id the peer minted."""
        entry = self.ledger.entry(self.name, record_id)
        if entry is None:
            raise ValueError(f"nothing shipped to {self.name!r} under {record_id!r}")
        remote_id = entry["remote_id"]
        if "record_digest" in (entry.get("structure") or {}):
            from rcdb import CopyReceipt
            receipt = CopyReceipt.from_bytes(bytes.fromhex(entry["receipt"]))
            return self._confirm_record(receipt).value
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
            try:
                body = (resp.text or "")[:200]
            except Exception:
                body = ""
        return f"{resp.status_code}: {body}" if body else str(resp.status_code)
    return f"{type(exc).__name__}: {exc}"
