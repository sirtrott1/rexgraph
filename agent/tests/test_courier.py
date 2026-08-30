"""agent.courier: carrying catalogued complexes between hives, and the edges the trips leave."""
import numpy as np
import pytest
from agent.courier import CarrySpec, Courier, structure_of

from agent import activity, rcdb
from agent import hive as hivemod
from rexgraph.graph import RexGraph


@pytest.fixture(autouse=True)
def _isolated_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", str(tmp_path / "activity.jsonl"))
    activity.reset()
    hivemod.reset_network()
    yield
    activity.get_log().close()


def _rex(n):
    """A cycle on n vertices, so two sizes are two different structures."""
    v = np.arange(n)
    return RexGraph.from_graph(v, np.roll(v, -1))


@pytest.fixture
def pair():
    """Two stores, the source holding a schema and an interaction complex."""
    src, dst = rcdb.open_store("memory://"), rcdb.open_store("memory://")
    src.put("alpha-schema", _rex(3), meta={"kind": "hive-schema"}, tags=["hive-schema"])
    src.put("alpha-work", _rex(4), meta={"kind": "interaction"}, tags=["interaction"])
    return src, dst


def _courier(src, dst, **kw):
    c = Courier("mule", **kw)
    c.attach_store("alpha", src)
    c.attach_store("beta", dst)
    return c


def test_a_first_exchange_carries_everything_the_source_holds(pair):
    src, dst = pair
    trip = _courier(src, dst).deliver("alpha", "beta")

    assert trip["considered"] == 2 and trip["carried"] == 2 and trip["held"] == 0
    assert {r.id for r in dst.list()} == {"alpha-schema", "alpha-work"}
    assert dst.get("alpha-work").nV == 4, "the complex arrives, not just its signature"


def test_a_repeat_trip_carries_nothing(pair):
    src, dst = pair
    c = _courier(src, dst)
    c.deliver("alpha", "beta")
    again = c.deliver("alpha", "beta")

    assert again["carried"] == 0 and again["held"] == 2
    assert dst.get_record("alpha-schema").version == 1, "a held record gains no version"


def test_a_changed_record_is_carried_as_a_new_version(pair):
    src, dst = pair
    c = _courier(src, dst)
    c.deliver("alpha", "beta")
    src.put("alpha-schema", _rex(5), meta={"kind": "hive-schema"}, tags=["hive-schema"])
    trip = c.deliver("alpha", "beta")

    assert trip["carried"] == 1 and trip["held"] == 1
    rec = dst.get_record("alpha-schema")
    assert rec.version == 2 and dst.get("alpha-schema").nV == 5
    assert [r.signature["nV"] for r in dst.history("alpha-schema")] == [3, 5]


def test_the_carry_spec_narrows_by_tag(pair):
    src, dst = pair
    trip = _courier(src, dst).deliver("alpha", "beta", carry=CarrySpec(tags=["hive-schema"]))

    assert trip["considered"] == 1 and trip["carried"] == 1
    assert [r.id for r in dst.list()] == ["alpha-schema"]


def test_the_carry_spec_names_ids_directly(pair):
    src, dst = pair
    m = CarrySpec(ids=["alpha-work", "does-not-exist"])
    trip = _courier(src, dst).deliver("alpha", "beta", carry=m)

    assert trip["considered"] == 1, "a named id that is absent is skipped, not an error"
    assert [r.id for r in dst.list()] == ["alpha-work"]


def test_provenance_rides_with_the_record(pair):
    src, dst = pair
    _courier(src, dst).deliver("alpha", "beta")

    rec = dst.get_record("alpha-schema")
    assert rec.meta["kind"] == "hive-schema", "the source's own meta survives the trip"
    assert rec.meta["courier"]["from"] == "alpha" and rec.meta["courier"]["by"] == "mule"
    assert "from:alpha" in rec.signature["tags"]
    assert dst.query(tags_any=["from:alpha"]), "provenance is queryable at the destination"


def test_retagging_at_the_destination_does_not_make_it_look_new(pair):
    src, dst = pair
    c = _courier(src, dst)
    c.deliver("alpha", "beta")
    # the destination re-files what it received under its own tag
    dst.put("alpha-work", dst.get("alpha-work"), meta=dst.get_record("alpha-work").meta,
            tags=["reviewed"])
    trip = c.deliver("alpha", "beta")

    assert trip["carried"] == 0, "identity is the structure, and tags are not part of it"
    assert structure_of({"nV": 3, "tags": ["x"], "source": "y"}) == {"nV": 3}


def test_a_trip_is_an_edge_in_the_network_complex(pair):
    src, dst = pair
    net = hivemod.get_network()
    net.hive("alpha"), net.hive("beta")
    c = _courier(src, dst, network=net)
    c.deliver("alpha", "beta")

    edges = net.monitor()["edges"]
    assert {"from": "alpha", "to": "beta", "weight": 1} in edges
    # a second trip carries nothing, and is still a trip, so the route gains weight
    c.deliver("alpha", "beta")
    assert {"from": "alpha", "to": "beta", "weight": 2} in net.monitor()["edges"]
    assert {a["agent"] for a in net.monitor()["agents"]} >= {"alpha", "beta"}


def test_broadcast_reaches_every_other_hive(pair):
    src, dst = pair
    third = rcdb.open_store("memory://")
    c = _courier(src, dst)
    c.attach_store("gamma", third)
    out = c.broadcast("alpha")

    assert out["dests"] == ["beta", "gamma"] and out["carried"] == 4
    assert len(third.list()) == 2 and len(dst.list()) == 2


def test_the_courier_is_an_ordinary_member(pair):
    src, dst = pair
    net = hivemod.get_network()
    net.hive("alpha"), net.hive("beta")
    c = _courier(src, dst, network=net)
    bee = c.join("alpha")

    assert bee.capability == "transform" and bee.worker_type == "courier:rcdb"
    assert net.get("alpha").providers("transform") == ["mule"]
    # invoked through the hive, and routable across the network by capability
    trip = net.get("alpha").invoke("mule", {"source": "alpha", "dest": "beta"})
    assert trip["carried"] == 2
    out = net.dispatch_capability("transform", {"source": "alpha", "dest": "beta"})
    assert out["hive"] == "alpha" and out["worker"] == "mule"
    labels = c.network.get("alpha").type_complex()[1]["vertex_labels"]
    assert "courier:rcdb" in labels, "the courier joins the worker-type ontology"


def test_an_unreadable_record_does_not_strand_the_trip(pair):
    src, dst = pair

    class Blocked:
        """A source whose second blob will not deserialize."""
        def __init__(self, real):
            self._real = real

        def __getattr__(self, k):
            return getattr(self._real, k)

        def get_version(self, id, version):
            if id == "alpha-schema":
                raise ValueError("blob is corrupt")
            return self._real.get_version(id, version)

    c = Courier("mule")
    c.attach_store("alpha", Blocked(src))
    c.attach_store("beta", dst)
    trip = c.deliver("alpha", "beta")

    assert trip["unreadable"] == 1 and trip["carried"] == 1
    assert [r.id for r in dst.list()] == ["alpha-work"], "the good record still arrived"


def test_delivering_into_the_source_store_holds_everything(pair):
    src, _ = pair
    c = Courier("mule")
    c.attach_store("alpha", src)
    c.attach_store("mirror", src)
    trip = c.deliver("alpha", "mirror")

    assert trip["carried"] == 0 and trip["held"] == 2


def test_a_delivery_keeps_the_records_valid_time(pair):
    """A record is true for a stretch of time that is not the time it was carried at.
    `rcdb.copy_record` is what preserves it, and is the one place a record crosses
    between stores so that `migrate` and a courier cannot drift on this."""
    src, dst = pair
    src.put("dated", _rex(3), meta={"kind": "reading"}, tags=["reading"],
            valid_from=1000.0, valid_to=2000.0)
    _courier(src, dst).deliver("alpha", "beta")

    got = dst.get_record("dated")
    assert (got.valid_from, got.valid_to) == (1000.0, 2000.0)


def test_dedup_survives_a_backend_boundary(tmp_path):
    """The case memory stores cannot show. A file store serialises a complex where a
    memory store keeps the object, so the two record different optional analytics
    (labels_sample, n_labels, n_voids) for the SAME complex. Comparing the whole
    signature minus provenance therefore called every record changed the moment it
    crossed backends, and a courier re-carried everything on every trip."""
    src = rcdb.open_store(f"file://{tmp_path}/a")
    dst = rcdb.open_store(f"file://{tmp_path}/b")
    src.put("schema", _rex(3), meta={"kind": "hive-schema"}, tags=["hive-schema"])

    c = Courier("mule")
    c.attach_store("alpha", src)
    c.attach_store("beta", dst)
    assert c.deliver("alpha", "beta")["carried"] == 1
    assert c.deliver("alpha", "beta")["held"] == 1, "a second trip re-carried"


def test_a_changed_record_still_reads_as_changed_across_backends(tmp_path):
    """The half worth checking as hard as the false positive: a comparison narrow enough
    to stop re-carrying could also stop noticing real change."""
    src = rcdb.open_store(f"file://{tmp_path}/a")
    dst = rcdb.open_store(f"file://{tmp_path}/b")
    src.put("work", _rex(3), meta={"kind": "x"}, tags=["x"])
    c = Courier("mule")
    c.attach_store("alpha", src)
    c.attach_store("beta", dst)
    c.deliver("alpha", "beta")

    src.put("work", _rex(6), meta={"kind": "x"}, tags=["x"])
    assert c.deliver("alpha", "beta")["carried"] == 1
    assert dst.get("work").nV == 6
