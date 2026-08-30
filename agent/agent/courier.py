"""
agent.courier: a worker that carries stored complexes between hives.

A hive's learning is not its chat log, it is what it has catalogued: its worker-type
structure, its schema lineage, the complexes its work produced. All of that already lives in
an RCStore keyed by structural signature, so distributing learning between hives is a
transfer between stores, and the thing that performs it is an ordinary member. A courier
declares capability `transform`, so it is routed to, invoked, typed and monitored exactly
like any other worker rather than sitting beside the hive as machinery.

Two properties of the RCDB are what keep this small. A record carries its structural
signature, so "does the destination already know this" is a comparison of signatures and
never of bytes, and a repeat trip costs one lookup per record and writes nothing. And a
store versions a lineage natively, so a delivery that DOES carry something appends a version
at the destination instead of overwriting, leaving the receiving hive a history of what
arrived and when.

A trip is recorded through `HiveNetwork.relay`, which is the network's own `add_message`.
Courier traffic is therefore an edge at the network grade, and `HiveNetwork.monitor()` reads
routes that carry the most as load-bearing without being told a courier exists.

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
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# The fields of a structural signature that say what SHAPE a complex is, named rather than
# subtracted. A denylist was tried first and was wrong: the rest of the signature carries
# provenance (tags, source) AND analytics whose presence depends on how the complex reached
# the store, so a memory store that keeps the object and a file store that serialises it
# disagree on labels_sample, n_labels and n_voids for the very same complex. Comparing
# everything-but-provenance therefore reported a record as changed the moment it crossed a
# backend boundary, which is every real store, and a courier silently re-carried on every
# trip. These six survive any round trip because they are read off the boundary data itself.
STRUCTURE_FIELDS = ("object_type", "nV", "nE", "nF", "betti", "chain_valid")

DEFAULT_LIMIT = 100


def structure_of(signature: dict) -> dict:
    """The part of a signature that says what shape a complex is.

    The identity a delivery compares on, so re-tagging a record on arrival does not make
    it look new, and neither does carrying it from one backend into another. Absent fields
    are left out rather than defaulted, so two signatures that recorded different amounts
    still compare on what they both actually state."""
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

    def select(self, store) -> list:
        """The records this spec picks out of a store. Named ids are fetched directly;
        tags go to the store's own query so the filter runs where the index is."""
        if self.ids:
            got = [store.get_record(i) for i in self.ids]
            return [r for r in got if r is not None][:self.limit]
        if self.tags:
            return store.query(limit=self.limit, tags_any=list(self.tags))
        return store.list(limit=self.limit)

    @classmethod
    def from_dict(cls, d: dict | None) -> CarrySpec:
        d = d or {}
        return cls(tags=list(d.get("tags") or []), ids=list(d.get("ids") or []),
                   limit=int(d.get("limit") or DEFAULT_LIMIT))


@dataclass
class Delivery:
    """One record's fate on one trip.

    `reason` is `carried` when the destination gained a version, `held` when it already had
    this structure under this id, and `unreadable` when the source could not produce the
    complex. An unreadable record is reported rather than raised: one bad blob must not
    strand the rest of the trip."""
    record_id: str
    reason: str
    version: int | None = None
    parent_version: int | None = None

    @property
    def carried(self) -> bool:
        return self.reason == "carried"

    def public(self) -> dict:
        return {"record_id": self.record_id, "reason": self.reason, "carried": self.carried,
                "version": self.version, "parent_version": self.parent_version}


class Courier:
    """A transform worker that moves catalogued complexes between hives' stores.

    The courier holds the routes (which store belongs to which hive) because a store is not a
    property of a `Hive`: the same hive can be catalogued into a throwaway store for a probe
    and a persistent one for real work, and which of those an exchange should use is a
    decision about the exchange."""

    def __init__(self, name: str = "courier", *, network=None, carry: CarrySpec | None = None):
        self.name = name
        self.network = network
        self.carry = carry or CarrySpec()
        self._stores: dict[str, object] = {}
        self._peers: dict[str, object] = {}
        self._trips = 0
        self._carried = 0

    #### routes: which store a hive is catalogued into
    def attach_store(self, hive: str, store) -> None:
        """Register a hive's store. `store` is an open RCStore or an RCDB uri."""
        from .rcdb import open_store
        self._stores[hive] = open_store(store) if isinstance(store, str) else store
        _record(self.name, "route", {"hive": hive})

    def store_of(self, hive: str):
        st = self._stores.get(hive)
        if st is None:
            raise ValueError(f"courier {self.name!r} has no store for hive {hive!r}")
        return st

    def hives(self) -> list[str]:
        return sorted(self._stores)

    def attach_peer(self, peer) -> None:
        """Register a remote server as a destination. `peer` is an `courier_remote.Peer`,
        which is a `RexClient` plus the ledger of what has already crossed to it."""
        self._peers[peer.name] = peer
        _record(self.name, "route", {"peer": peer.name})

    def peers(self) -> list[str]:
        return sorted(self._peers)

    def destinations(self) -> list[str]:
        """Every place a trip can go, local and remote. A peer is a destination like any
        other: which machine a hive is on does not change how it is addressed."""
        return sorted({*self._stores, *self._peers})

    #### what is available to carry, without carrying it
    def survey(self, hive: str, *, carry: CarrySpec | None = None) -> list[dict]:
        """What a trip out of this hive would consider, and the shape of each record. This is
        the read-only half of `deliver`, so a caller can decide whether a trip is worth it."""
        m = carry or self.carry
        return [{"record_id": r.id, "version": r.version, "tags": r.signature.get("tags") or [],
                 "kind": (r.meta or {}).get("kind", ""), "structure": structure_of(r.signature)}
                for r in m.select(self.store_of(hive))]

    #### the trip
    def deliver(self, source: str, dest: str, *, carry: CarrySpec | None = None) -> dict:
        """Carry everything the spec selects from source to dest, skipping what dest
        already holds. Returns the trip: per-record deliveries and the counts.

        Delivering to a store that is already the source's is not an error and is not a
        special case: every record compares equal to itself and the whole trip reads `held`.

        A dest that names a peer crosses machines instead, over `/rex/v1`. The counters
        are the same either way; what differs is that a crossing reports `shipments` with
        the peer's own record ids rather than `deliveries` with versions."""
        m = carry or self.carry
        if dest in self._peers:
            return self._ship(source, dest, m)
        src, dst = self.store_of(source), self.store_of(dest)
        deliveries = [self._one(src, dst, rec, source) for rec in m.select(src)]

        carried = [d for d in deliveries if d.carried]
        self._trips += 1
        self._carried += len(carried)
        trip = {"courier": self.name, "source": source, "dest": dest,
                "considered": len(deliveries), "carried": len(carried),
                "held": sum(1 for d in deliveries if d.reason == "held"),
                "unreadable": sum(1 for d in deliveries if d.reason == "unreadable"),
                "deliveries": [d.public() for d in deliveries]}
        self._relay(source, dest, trip)
        _record_trip(self.name, "deliver", trip, source, dest)
        return trip

    def broadcast(self, source: str, dests: list[str] | None = None, *,
                  carry: CarrySpec | None = None) -> dict:
        """One trip per destination, defaulting to every other hive the courier routes for.

        This is a fan-out of `deliver` and not a cheaper path: each destination is compared
        against separately, because two destinations do not hold the same thing and a record
        one already has is a record the other may still need."""
        targets = [d for d in (dests if dests is not None else self.destinations())
                   if d != source]
        trips = [self.deliver(source, d, carry=carry) for d in targets]
        return {"courier": self.name, "source": source, "dests": targets,
                "carried": sum(t["carried"] for t in trips), "trips": trips}

    def _one(self, src, dst, rec, source: str) -> Delivery:
        """One record's trip. The destination is compared on structure alone, so a record that
        arrived on an earlier trip and was re-tagged there still reads as held.

        The write goes through `rcdb.copy_record`, the one place a record crosses between
        stores, so a delivery keeps the valid time the record was true for rather than
        being stamped with the time it was carried at."""
        try:
            have = dst.get_record(rec.id)
        except Exception:
            have = None
        if have is not None and structure_of(have.signature) == structure_of(rec.signature):
            return Delivery(rec.id, "held", version=have.version)
        from .rcdb import copy_record
        meta = dict(rec.meta or {})
        meta["courier"] = {"by": self.name, "from": source, "at": time.time(),
                           "source_version": rec.version}
        tags = sorted({*(rec.signature.get("tags") or []), "courier", f"from:{source}"})
        try:
            out = copy_record(src, dst, rec, meta=meta, tags=tags)
        except Exception:
            logger.debug("courier %s could not read %s", self.name, rec.id, exc_info=True)
            out = None
        if out is None:
            return Delivery(rec.id, "unreadable")
        return Delivery(rec.id, "carried", version=out.version,
                        parent_version=out.version - 1 if out.version > 1 else None)

    def _ship(self, source: str, dest: str, carry: CarrySpec) -> dict:
        """One trip across machines. Reading the complex is local and can fail locally, so
        that stays here; everything past the wire is the peer's to report."""
        from .courier_remote import Shipment
        peer, src = self._peers[dest], self.store_of(source)
        shipments = []
        for rec in carry.select(src):
            try:
                rex = src.get(rec.id)
            except Exception:
                logger.debug("courier %s could not read %s", self.name, rec.id, exc_info=True)
                rex = None
            shipments.append(Shipment(rec.id, "unreadable") if rex is None else
                             peer.ship(rec, rex, source=source, courier=self.name))

        shipped = [x for x in shipments if x.shipped]
        self._trips += 1
        self._carried += len(shipped)
        trip = {"courier": self.name, "source": source, "dest": dest, "remote": True,
                "considered": len(shipments), "carried": len(shipped)}
        for reason in ("held", "oversize", "refused", "unreadable"):
            trip[reason] = sum(1 for x in shipments if x.reason == reason)
        trip["shipments"] = [x.public() for x in shipments]
        self._relay(source, dest, trip)
        _record_trip(self.name, "ship", trip, source, dest)
        return trip

    def _relay(self, source: str, dest: str, trip: dict) -> None:
        """Record the trip as inter-hive traffic. A trip that carried nothing is still a trip,
        so the edge is recorded either way and the network complex sees the route."""
        if self.network is None:
            return
        try:
            self.network.relay(source, dest,
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
        if self.network is None:
            raise ValueError("join needs a network to resolve the hive by name")
        h = self.network.get(hive)
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
        m = CarrySpec.from_dict(d) if (d.get("tags") or d.get("ids") or d.get("limit")) \
            else self.carry
        dest = d.get("dest") or d.get("to")
        if dest:
            return self.deliver(source, dest, carry=m)
        return self.broadcast(source, d.get("dests"), carry=m)

    def status(self) -> dict:
        return {"name": self.name, "hives": self.hives(), "peers": self.peers(),
                "trips": self._trips,
                "carried": self._carried,
                "carry": {"tags": self.carry.tags, "ids": self.carry.ids,
                          "limit": self.carry.limit}}


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
    40k single-record events looks like: 39.5k objects of degree one, which adds vertices
    and no cycles, so the topology it carries is the same one the two ends carry and it
    costs 20,000 times the volume to say it."""
    keep = {k: trip[k] for k in ("considered", "carried", "held") if k in trip}
    tid = trip_id or uuid.uuid4().hex[:12]
    # the SAME trip id on both ends, so a reader pairs them by identity rather than by
    # adjacency in a journal many processes are appending to at once
    _record(name, action, dict(keep, end="source", trip=tid), on="hive:" + source, flow="read")
    _record(name, action, dict(keep, end="dest", trip=tid), on="hive:" + dest, flow="write")


# process-wide courier

_COURIER: Courier | None = None


def get_courier() -> Courier:
    """The courier this process routes through, wired to the process-wide hive network so
    its trips land as edges of the same complex the hives are cells of."""
    global _COURIER
    if _COURIER is None:
        from .hive_network import get_network
        _COURIER = Courier("courier", network=get_network())
    return _COURIER


def reset_courier() -> None:
    global _COURIER
    _COURIER = None


def _spec_from_args(a) -> CarrySpec:
    return CarrySpec(tags=[t for t in a.tags.split(",") if t],
                     ids=[i for i in a.ids.split(",") if i], limit=a.limit)


def main(argv=None):
    """CLI: `python -m agent.courier deliver <source-uri> <dest-uri>`.

    Stores are named by RCDB uri rather than by hive, because a command that ends when it
    returns has no hive to belong to. The library keeps the hive-addressed form, where a
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
                    help="fetch each shipment back and compare fingerprints")
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
