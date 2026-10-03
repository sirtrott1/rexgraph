"""
agent.courier: a worker that carries stored complexes between hives.

A hive's learning is not its chat log, it is what it has catalogued: its worker type
structure, its schema lineage, the complexes its work produced. All of that already lives in
an RCStore with native payloads and structural signatures, so distributing learning is a
transfer between stores, and the thing that performs it is an ordinary member. A courier
declares capability `transform`, so it is routed to, invoked, typed and monitored exactly
like any other worker rather than sitting beside the hive as machinery.

Two properties of RCDB keep this small. Version snapshots carry framework state digests,
so "does the destination already know this" compares full payload identity, not cell counts
or Betti numbers. A repeat trip reads and hashes the selected payloads but writes nothing.
A store versions a lineage natively, so a delivery that DOES carry something appends a version
at the destination instead of overwriting, leaving the receiving hive a history of what
arrived and when.

A trip is recorded through `HiveNetwork.relay`, which is the network's own `add_message`.
Courier traffic is therefore an edge at the network grade, and `HiveNetwork.monitor()` reads
routes that carry the most as load bearing without being told a courier exists.

    courier = Courier("mule", network=net)
    courier.attach_store("alpha", alpha_store)
    courier.attach_store("beta", beta_store)
    courier.join("alpha")                          # now a bee: hive.invoke("mule", {...})
    courier.deliver("alpha", "beta", carry=CarrySpec(tags=["hive-schema"]))

What a courier will carry is a `CarrySpec`, not a hardcoded rule. An empty spec carries
everything the source store holds, up to its limit; tags and ids narrow it. Selection runs as
a store query where it can, so a spec that names tags does not pull every record to
filter in Python.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock

logger = logging.getLogger(__name__)

# The fields of a structural signature that say what SHAPE a complex is, named rather than
# subtracted. A denylist was tried first and was wrong: the rest of the signature carries
# provenance (tags, source) AND analytics whose presence depends on how the complex reached
# the store, so a memory store that keeps the object and a file store that serialises it
# disagree on labels_sample, n_labels and n_voids for the very same complex. Comparing
# everything but provenance therefore reported a record as changed the moment it crossed a
# backend boundary, which is every real store, and a courier silently re carried on every
# trip. These six survive any round trip because they are read off the boundary data itself.
STRUCTURE_FIELDS = ("object_type", "nV", "nE", "nF", "betti", "chain_valid")

DEFAULT_LIMIT = 100
DEFAULT_COPY_ATTEMPTS = 4
_NOT_SELECTED = object()


def _name(value):
    if type(value) is not str or not value or len(value.encode("utf-8")) > 1024*1024:
        raise ValueError("courier names require bounded nonempty literal text")
    return value


def structure_of(signature: dict) -> dict:
    """The part of a signature that says what shape a complex is.

    A survey summary, not a complete state identity: different boundaries and weights
    can share these invariants. Delivery uses the framework payload digest instead.
    Absent fields are left out rather than defaulted."""
    sig = signature or {}
    return {k: sig[k] for k in STRUCTURE_FIELDS if k in sig}


@dataclass
class CarrySpec:
    """What a courier is willing to carry on one trip.

    `tags` matches any of the given tags, `ids` names records directly, and `limit` caps the
    trip. All three empty means everything the source holds, up to the limit, which is the
    useful default for a first exchange between two hives that have never met."""
    tags: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    limit: int = DEFAULT_LIMIT

    def __post_init__(self):
        if type(self.limit) is not int or not 0 <= self.limit < 2**63:
            raise ValueError("carry limit requires a nonnegative bounded native integer")
        for name in ("tags", "ids"):
            values = getattr(self, name)
            if type(values) not in (list, tuple):
                raise ValueError("carry tags and ids require sequences of literal text")
            setattr(self, name, [_name(value) for value in values])

    def snapshot(self):
        # Preserve a subclass's selection extension while owning the standard
        # configuration. Extra subclass state remains its author's contract.
        from copy import copy
        validated = CarrySpec(tags=self.tags, ids=self.ids, limit=self.limit)
        result = copy(self)
        result.tags, result.ids, result.limit = validated.tags, validated.ids, validated.limit
        return result

    def select(self, store) -> list:
        """The records this spec picks out of a store. Named ids are fetched directly;
        tags go to the store's own query so the filter runs where the index is."""
        if not self.limit:
            return []
        if self.ids:
            got = []
            for record_id in self.ids:
                record = store.get_record(record_id)
                if record is not None:
                    got.append(record)
                    if len(got) == self.limit:
                        break
            return got
        if self.tags:
            return store.query(limit=self.limit, tags_any=list(self.tags))
        return store.list(limit=self.limit)

    @classmethod
    def from_dict(cls, d: dict | None) -> CarrySpec:
        if d is not None and type(d) is not dict:
            raise ValueError("carry selection requires a mapping")
        d = {} if d is None else d
        return cls(tags=[] if d.get("tags") is None else d["tags"],
                   ids=[] if d.get("ids") is None else d["ids"],
                   limit=d.get("limit", DEFAULT_LIMIT))


@dataclass
class Delivery:
    """One record's fate on one trip.

    `reason` is `carried` when the destination gained a version, `held` when it already had
    this native state under this id, and `unreadable` when the store could not produce the
    payload. `conflict` reports an intervening destination change, `refused` a policy
    or publication refusal, and `uncertain` an unknown publication outcome. These
    outcomes are reported per record so one failure does not strand the trip."""
    record_id: str
    reason: str
    version: int | None = None
    parent_version: int | None = None
    detail: str = ""

    @property
    def carried(self) -> bool:
        return self.reason == "carried"

    def public(self) -> dict:
        result = {"record_id": self.record_id, "reason": self.reason, "carried": self.carried,
                  "version": self.version, "parent_version": self.parent_version}
        if self.detail:
            result["detail"] = self.detail
        return result


class Courier:
    """A transform worker that moves catalogued complexes between hives' stores.

    The courier holds the routes (which store belongs to which hive) because a store is not a
    property of a `Hive`: the same hive can be catalogued into a throwaway store for a probe
    and a persistent one for real work, and which of those an exchange should use is a
    decision about the exchange."""

    def __init__(self, name: str = "courier", *, network=None, carry: CarrySpec | None = None,
                 network_factory=None, copy_attempts: int = DEFAULT_COPY_ATTEMPTS):
        self.name = _name(name)
        if network_factory is not None and (network is not None or not callable(network_factory)):
            raise ValueError("provide a network or a callable network_factory")
        if type(copy_attempts) is not int or not 1 <= copy_attempts <= 32:
            raise ValueError("copy_attempts requires a native integer from 1 to 32")
        if carry is not None and not isinstance(carry, CarrySpec):
            raise ValueError("carry requires a CarrySpec")
        self._lock, self._pid = RLock(), os.getpid()
        self._network, self._network_factory = network, network_factory
        self._copy_attempts = copy_attempts
        self.carry = (carry or CarrySpec()).snapshot()
        self._stores: dict[str, object] = {}
        self._peers: dict[str, object] = {}
        self._trips = 0
        self._carried = 0

    def _guard(self):
        if os.getpid() != self._pid:
            raise RuntimeError("inherited courier cannot be reused after fork; open a fresh courier and stores")
        return self._lock

    @property
    def network(self):
        with self._guard():
            network, factory = self._network, self._network_factory
        return factory() if factory is not None else network

    @network.setter
    def network(self, value):
        with self._guard():
            self._network, self._network_factory = value, None

    #### routes: which store a hive is catalogued into
    def attach_store(self, hive: str, store) -> None:
        """Register a hive's store. `store` is an open RCStore or an RCDB uri."""
        from .rcdb import open_store
        hive = _name(hive)
        with self._guard():
            if hive in self._peers:
                raise ValueError("courier destination is already registered as a peer")
        opened = open_store(store) if isinstance(store, str) else store
        try:
            with self._guard():
                if hive in self._peers:
                    raise ValueError("courier destination is already registered as a peer")
                self._stores[hive] = opened
        except BaseException:
            if isinstance(store, str):
                try:
                    opened.close()
                except Exception:
                    logger.debug("could not close refused courier store", exc_info=True)
            raise
        _record(self.name, "route", {"hive": hive})

    def store_of(self, hive: str):
        with self._guard():
            st = self._stores.get(_name(hive))
        if st is None:
            raise ValueError(f"courier {self.name!r} has no store for hive {hive!r}")
        return st

    def hives(self) -> list[str]:
        with self._guard():
            return sorted(self._stores)

    def attach_peer(self, peer) -> None:
        """Register a remote server as a destination. `peer` is an `courier_remote.Peer`,
        which is a `RexClient` plus the ledger of what has already crossed to it."""
        name = _name(peer.name)
        with self._guard():
            if name in self._stores:
                raise ValueError("courier destination is already registered as a store")
            self._peers[name] = peer
        _record(self.name, "route", {"peer": peer.name})

    def peers(self) -> list[str]:
        with self._guard():
            return sorted(self._peers)

    def peer_of(self, peer: str):
        with self._guard():
            result = self._peers.get(_name(peer))
        if result is None:
            raise ValueError(f"courier {self.name!r} has no registered peer {peer!r}")
        return result

    def destinations(self) -> list[str]:
        """Every place a trip can go, local and remote. A peer is a destination like any
        other: which machine a hive is on does not change how it is addressed."""
        with self._guard():
            return sorted({*self._stores, *self._peers})

    #### what is available to carry, without carrying it
    def survey(self, hive: str, *, carry: CarrySpec | None = None) -> list[dict]:
        """What a trip out of this hive would consider, and the shape of each record. This is
        the read only half of `deliver`, so a caller can decide whether a trip is worth it."""
        with self._guard():
            m = (carry or self.carry).snapshot()
            store = self.store_of(hive)
        return [{"record_id": r.id, "version": r.version, "tags": r.signature.get("tags") or [],
                 "kind": (r.meta or {}).get("kind", ""), "structure": structure_of(r.signature)}
                for r in m.select(store)]

    #### the trip
    def deliver(self, source: str, dest: str, *, carry: CarrySpec | None = None) -> dict:
        """Carry everything the spec selects from source to dest, skipping what dest
        already holds. Returns the trip: per record deliveries and the counts.

        Delivering to a store that is already the source's is not an error and is not a
        special case: every record compares equal to itself and the whole trip reads `held`.

        A dest that names a peer crosses machines instead, over `/rex/v1`. The counters
        are the same either way; what differs is that a crossing reports `shipments` with
        the peer's own record ids rather than `deliveries` with versions."""
        src, routes, spec, network = self._bindings(source, [dest], carry)
        return self._deliver_trips(src, source, routes, spec, network)[0]

    def broadcast(self, source: str, dests: list[str] | None = None, *,
                  carry: CarrySpec | None = None) -> dict:
        """One trip per destination, defaulting to every other hive the courier routes for.

        This is a fan out of `deliver` and not a cheaper path: each destination is compared
        against separately, because two destinations do not hold the same thing and a record
        one already has is a record the other may still need."""
        src, routes, spec, network = self._bindings(source, dests, carry, broadcast=True)
        targets = [route[0] for route in routes]
        trips = self._deliver_trips(src, source, routes, spec, network)
        return {"courier": self.name, "source": source, "dests": targets,
                "carried": sum(t["carried"] for t in trips), "trips": trips}

    def _bindings(self, source, dests, carry, *, broadcast=False):
        """Own one route/configuration selection; release the registry before I/O."""
        source = _name(source)
        if dests is not None and type(dests) not in (list, tuple):
            raise ValueError("courier destinations require a sequence of literal names")
        if carry is not None and not isinstance(carry, CarrySpec):
            raise ValueError("carry requires a CarrySpec")
        with self._guard():
            src = self.store_of(source)
            targets = self.destinations() if dests is None else [_name(dest) for dest in dests]
            routes = []
            for dest in targets:
                if broadcast and dest == source:
                    continue
                dst, peer = self._stores.get(dest), self._peers.get(dest)
                if dst is None and peer is None:
                    raise ValueError(f"no destination {dest!r}; register it first")
                routes.append((dest, dst, peer))
            spec = (carry or self.carry).snapshot()
            network, factory = self._network, self._network_factory
        return src, routes, spec, factory() if factory is not None else network

    def _deliver_trips(self, src, source, routes, spec, network):
        """Select source versions once; fan out one decoded record at a time."""
        from .courier_remote import Shipment
        if not routes:
            return []
        with src.read_transaction():
            records, owner = spec.select(src), src.store_id
        outcomes = [[] for _ in routes]
        for rec in records:
            try:
                snapshot = src.read_record(rec.id, version=rec.version)
            except Exception:
                logger.debug("courier %s could not read %s", self.name, rec.id, exc_info=True)
                snapshot = None
            for values, (_, dst, peer) in zip(outcomes, routes, strict=True):
                if peer is not None:
                    result = (Shipment(rec.id, "unreadable") if snapshot is None else
                              peer.ship(snapshot.record, snapshot.value, source=source, courier=self.name,
                                        source_store_id=owner, snapshot=snapshot))
                else:
                    result = self._one(src, dst, rec, source, snapshot=snapshot, source_owner=owner)
                values.append(result)
        trips = []
        for (dest, _, peer), values in zip(routes, outcomes, strict=True):
            remote = peer is not None
            carried = sum(value.shipped if remote else value.carried for value in values)
            trip = {"courier": self.name, "source": source, "dest": dest,
                    "considered": len(values), "carried": carried}
            reasons = ("held", "oversize", "refused", "unreadable") if remote else (
                "held", "unreadable", "refused", "conflict", "uncertain")
            for reason in reasons:
                trip[reason] = sum(value.reason == reason for value in values)
            trip["shipments" if remote else "deliveries"] = [value.public() for value in values]
            if remote:
                trip["remote"] = True
            with self._guard():
                self._trips += 1
                self._carried += carried
            self._relay(source, dest, trip, network=network)
            _record_trip(self.name, "ship" if remote else "deliver", trip, source, dest)
            trips.append(trip)
        return trips

    def _one(self, src, dst, rec, source: str, *, snapshot=_NOT_SELECTED, source_owner=None) -> Delivery:
        """One record's trip. The destination is compared on payload identity, so a record that
        arrived on an earlier trip and was re tagged there still reads as held.

        The write goes through `rcdb.copy_record`, the one place a record crosses between
        stores, so a delivery keeps the valid time the record was true for rather than
        being stamped with the time it was carried at."""
        from .rcdb import copy_record, PublicationUncertainError, VersionConflictError
        try:
            if snapshot is _NOT_SELECTED:
                snapshot = src.read_record(rec.id, version=rec.version)
        except Exception:
            logger.debug("courier %s could not read %s", self.name, rec.id, exc_info=True)
            snapshot = None
        if snapshot is None:
            return Delivery(rec.id, "unreadable")
        try:
            rec = snapshot.record
            owner = src.store_id if source_owner is None else source_owner
            from hashlib import sha256
            from rexgraph.value_codec import pack_value
            fields = rec.to_dict()
            fields.pop("envelope", None)
            selected = sha256(b"rexgraph-local-courier\x00"+pack_value(
                (owner, fields, snapshot.state_digest))).hexdigest()
            # The same logical store already holds the selected immutable version,
            # even if a newer version became current. Respect scoped visibility.
            if owner == dst.store_id:
                existing = dst.read_record(rec.id, version=rec.version)
                if self._same_payload(existing, snapshot):
                    return Delivery(rec.id, "held", version=existing.record.version)
            cursor, policy, have, original = self._destination(dst, rec.id)
            if self._held(have, snapshot, selected):
                return Delivery(rec.id, "held", version=have.record.version)
            meta = dict(rec.meta or {})
            meta["courier"] = {"by": self.name, "from": source, "at": time.time(),
                               "source_version": rec.version, "source_store_id": owner,
                               "selection_digest": selected}
            tags = sorted({*(rec.signature.get("tags") or []), "courier", f"from:{source}"})
            if cursor is not None and dst._stored_meta(meta).get("courier") != meta["courier"]:
                return Delivery(rec.id, "refused", detail="destination policy discards courier selection identity")
            for attempt in range(self._copy_attempts):
                try:
                    out = copy_record(src, dst, rec, meta=meta, tags=tags, expected_digest=snapshot.state_digest,
                                      expected_cursor=cursor, expected_policy_digest=policy)
                    break
                except VersionConflictError:
                    # A conditional refusal is known to have published nothing.
                    # Rebase only if this literal address and its retained highwater
                    # did not change. Identical concurrent arrivals count as held.
                    next_cursor, next_policy, next_have, view = self._destination(dst, rec.id)
                    if self._held(next_have, snapshot, selected):
                        return Delivery(rec.id, "held", version=next_have.record.version)
                    if view != original or attempt+1 == self._copy_attempts or cursor is None:
                        return Delivery(rec.id, "conflict", detail="destination changed before conditional copy")
                    if next_policy != policy:
                        return Delivery(rec.id, "refused", detail="destination policy changed before conditional copy")
                    cursor, have = next_cursor, next_have
            if out is None:
                return Delivery(rec.id, "unreadable")
            parent = (None if have is None else have.record.version) if cursor is not None else (
                out.version-1 if out.version > 1 else None)
            return Delivery(rec.id, "carried", version=out.version, parent_version=parent)
        except PublicationUncertainError:
            logger.debug("courier %s has uncertain destination %s", self.name, rec.id, exc_info=True)
            return Delivery(rec.id, "uncertain", detail="publication outcome is uncertain; reopen and verify")
        except Exception:
            logger.debug("courier %s could not publish %s", self.name, rec.id, exc_info=True)
            return Delivery(rec.id, "refused", detail="could not verify or publish the selected record")

    @staticmethod
    def _same_payload(have, snapshot):
        def codec(record):
            envelope = record.envelope
            return ("rexgraph.safetensors", 1, record.object_type) if envelope is None else (
                envelope.codec, envelope.codec_version, record.object_type)
        return (have is not None and have.record.id == snapshot.record.id
                and have.state_digest == snapshot.state_digest
                and codec(have.record) == codec(snapshot.record))

    @classmethod
    def _held(cls, have, snapshot, selected):
        if not cls._same_payload(have, snapshot):
            return False
        previous = have.record.meta.get("courier", {})
        return isinstance(previous, dict) and previous.get("selection_digest") == selected

    @staticmethod
    def _destination(dst, record_id):
        from rcdb.header import StoreHeader
        with dst.read_transaction():
            native = isinstance(getattr(dst, "header", None), StoreHeader)
            cursor = dst.change_cursor if native else None
            policy = dst.transfer_policy_digest() if native else None
            have = dst.read_record(record_id)
            if have is not None and have.record.id != record_id:
                have = None
            view = ((dst.next_version(record_id), None if have is None else have.record.version)
                    if native else None)
            return cursor, policy, have, view

    def _ship(self, source: str, dest: str, carry: CarrySpec) -> dict:
        """One trip across machines. Reading the complex is local and can fail locally, so
        that stays here; everything past the wire is the peer's to report."""
        return self.deliver(source, dest, carry=carry)

    def reconcile(self, source, dest, record_id, receipt):
        """Select registered source/peer together, then verify a known receipt."""
        from rcdb import CopyReceipt, record_packet
        if not isinstance(receipt, CopyReceipt):
            raise ValueError("courier reconciliation requires a CopyReceipt")
        _name(record_id)
        _name(receipt.destination_record_id)
        with self._guard():
            src, peer = self.store_of(source), self.peer_of(dest)
        packet = record_packet(src, record_id, version=receipt.source_version)
        return peer.reconcile(packet, receipt)

    def _relay(self, source: str, dest: str, trip: dict, *, network=_NOT_SELECTED) -> None:
        """Record the trip as inter hive traffic. A trip that carried nothing is still a trip,
        so the edge is recorded either way and the network complex sees the route."""
        network = self.network if network is _NOT_SELECTED else network
        if network is None:
            return
        try:
            network.relay(source, dest,
                               f"courier {self.name} carried {trip['carried']} of "
                               f"{trip['considered']}", courier=self.name,
                               carried=trip["carried"])
        except Exception:
            logger.debug("courier %s could not relay %s to %s", self.name, source, dest,
                         exc_info=True)

    #### membership: a courier is a worker, not machinery beside the hive
    def join(self, hive: str, *, specialties=None):
        """Register as a transform worker on a hive, so the courier is invoked, routed and
        typed like any member. Needs a network to resolve the hive by name."""
        network = self.network
        if network is None:
            raise ValueError("join needs a network to resolve the hive by name")
        h = network.get(hive)
        if h is None:
            raise ValueError(f"no hive {hive!r} in the network")
        return h.add_worker(self.name, self.handler, capability="transform",
                            specialties=list(specialties or
                                             ["courier", "exchange", "learning", "rcdb"]),
                            worker_type="courier:rcdb")

    def handler(self, data, **kw):
        """The transform capability. `{"source": a, "dest": b}` is one trip; omitting `dest`
        broadcasts. `tags`, `ids` and `limit` build the spec for this call only, so one
        registered courier serves both a narrow exchange and a full one."""
        d = dict(data or {})
        d.update(kw)
        source = d.get("source") or d.get("from")
        if not source:
            raise ValueError("a delivery needs 'source'")
        m = CarrySpec.from_dict(d) if any(k in d for k in ("tags", "ids", "limit")) else None
        dest = d.get("dest") or d.get("to")
        if dest:
            return self.deliver(source, dest, carry=m)
        return self.broadcast(source, d.get("dests"), carry=m)

    def status(self) -> dict:
        with self._guard():
            spec = self.carry.snapshot()
            return {"name": self.name, "hives": self.hives(), "peers": self.peers(),
                    "trips": self._trips, "carried": self._carried,
                    "copy_attempts": self._copy_attempts,
                    "carry": {"tags": spec.tags, "ids": spec.ids, "limit": spec.limit}}


def _record(name: str, action: str, detail: dict, *, on: str = "", flow: str = "") -> None:
    """Journal one courier action. Recording must never break the trip it describes."""
    try:
        from . import activity
        activity.record("worker:" + name, action, scope="worker", detail=detail,
                        on=on, flow=flow)
    except Exception:
        logger.debug("activity record failed for %s/%s", name, action, exc_info=True)


def _record_trip(name: str, action: str, trip: dict, source: str, dest: str,
                 trip_id: str = "") -> None:
    """A trip as the two oriented acts it is: read the source, write the destination.

    One event per end rather than one per record. The journal already shows what a log of
    40k single record events looks like: 39.5k objects of degree one, which adds vertices
    and no cycles, so the topology it carries is the same one the two ends carry and it
    costs 20,000 times the volume to say it."""
    keep = {k: trip[k] for k in ("considered", "carried", "held") if k in trip}
    tid = trip_id or uuid.uuid4().hex[:12]
    # the SAME trip id on both ends, so a reader pairs them by identity rather than by
    # adjacency in a journal many processes are appending to at once
    _record(name, action, dict(keep, end="source", trip=tid), on="hive:" + source, flow="read")
    _record(name, action, dict(keep, end="dest", trip=tid), on="hive:" + dest, flow="write")


# process wide courier

_COURIER: Courier | None = None
_COURIER_LOCK = RLock()


def get_courier() -> Courier:
    """The courier this process routes through, wired to the process wide hive network so
    its trips land as edges of the same complex the hives are cells of."""
    global _COURIER
    with _COURIER_LOCK:
        if _COURIER is None:
            from .hive_network import get_network
            _COURIER = Courier("courier", network_factory=get_network)
        return _COURIER


def reset_courier() -> None:
    global _COURIER
    with _COURIER_LOCK:
        _COURIER = None


def _after_fork():
    global _COURIER, _COURIER_LOCK
    _COURIER, _COURIER_LOCK = None, RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _spec_from_args(a) -> CarrySpec:
    return CarrySpec(tags=[t for t in a.tags.split(",") if t],
                     ids=[i for i in a.ids.split(",") if i], limit=a.limit)


def main(argv=None):
    """CLI: `python -m agent.courier deliver <source-uri> <dest-uri>`.

    Stores are named by RCDB uri rather than by hive, because a command that ends when it
    returns has no hive to belong to. The library keeps the hive addressed form, where a
    trip is also an edge in the network complex."""
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="rexgraph-courier", description=(
        "Carry catalogued complexes between stores, on this machine or across the wire."))
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _carry(p):
        p.add_argument("--tags", default="", help="comma-separated tags, or omit for all")
        p.add_argument("--ids", default="", help="comma-separated record ids")
        p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)

    sv = sub.add_parser("survey",
                        help="what a trip out of a store would consider, carrying nothing")
    sv.add_argument("store", help="RCDB uri")
    _carry(sv)

    dl = sub.add_parser("deliver", help="carry between two stores this machine can open")
    dl.add_argument("source", help="RCDB uri"); dl.add_argument("dest", help="RCDB uri")
    _carry(dl)

    sh = sub.add_parser("ship", help="carry to a remote rexgraph server over /rex/v1")
    sh.add_argument("source", help="RCDB uri")
    sh.add_argument("url", help="base url of the peer, e.g. https://gpu-box:8000")
    sh.add_argument("--token-ref", default="",
                    help="env var or secret-store name holding the bearer token, never the token")
    sh.add_argument("--ledger", default="",
                    help="file to keep the shipped-ledger in, so repeat trips stay idempotent")
    sh.add_argument("--confirm", action="store_true",
                    help="fetch each shipment back and compare native state digests")
    _carry(sh)

    a = ap.parse_args(argv)
    c = Courier("cli", carry=_spec_from_args(a))

    if a.cmd == "survey":
        c.attach_store("source", a.store)
        print(json.dumps(c.survey("source"), indent=2, default=str))
        return 0

    c.attach_store("source", a.source)
    if a.cmd == "deliver":
        c.attach_store("dest", a.dest)
        out = c.deliver("source", "dest")
    else:
        from .client import RexClient
        from .courier_remote import Ledger, Peer
        from .secrets import resolve_ref
        key = resolve_ref(a.token_ref) if a.token_ref else ""
        peer = Peer(a.url, RexClient(a.url, api_key=key or None),
                    ledger=Ledger(a.ledger or None), confirm=a.confirm)
        c.attach_peer(peer)
        out = c.deliver("source", a.url)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
