"""The hive network (agent.hive_network): hives as cells one grade up - inter-hive routing,
cross-hive capability dispatch, and the network-grade RCFE field."""
import pytest

from agent import agent_complex
from agent import hive as H
from agent import hive_network as HN


@pytest.fixture(autouse=True)
def clean():
    H.reset_hive(); agent_complex.reset_live(); agent_complex.reset_drift(); HN.reset_network()
    yield
    H.reset_hive(); agent_complex.reset_live(); agent_complex.reset_drift(); HN.reset_network()


def _net():
    h1 = H.Hive()
    h1.add_worker("bio-clf", lambda d, **k: {"pred": 1}, capability="predict", specialties=["protein"])
    h2 = H.Hive()
    h2.add_worker("stat", lambda d, **k: {"score": 0.5}, capability="score", specialties=["stats"])
    net = HN.get_network()
    net.add_hive("bio", h1, specialties=["protein", "biology"])
    net.add_hive("chem", h2, specialties=["stats", "chemistry"])
    return net


def test_network_routes_and_cross_hive_capability_dispatch():
    net = _net()
    assert net.route("protein binding")[0]["hive"] == "bio"           # routed by inter-hive specialty
    assert net.dispatch_capability("predict", [1, 2, 3])["hive"] == "bio"   # predict lives in bio
    assert net.dispatch_capability("score", [1, 2, 3])["hive"] == "chem"    # score lives in chem
    with pytest.raises(ValueError):
        net.dispatch_capability("embed", [1])                        # no hive provides it


def test_network_grade_field_and_drift():
    net = _net()
    net.add_hive("mat", H.Hive(), specialties=["materials"])
    for a, b in [("bio", "chem"), ("chem", "mat"), ("mat", "bio")]:   # a coordinating triad of hives
        net.relay(a, b, "coord"); net.relay(b, a, "ack")
    m = net.monitor(track=True)
    assert set(m["hives"]) == {"bio", "chem", "mat"}
    assert m["strain"] is not None and m["strain"] > 0               # coordination -> a network field
    assert "drift" in m and "drifting" in m["drift"]


def test_network_persist_stores_ambient_and_subcomplexes():
    """persist catalogues the inter-hive complex and each member hive as a subcomplex in one store."""
    from agent.rcdb import open_store
    store = open_store("memory://")
    net = _net()
    net.relay("bio", "chem", "coord"); net.relay("chem", "bio", "ack")
    net.persist(store, name="net")
    ids = {r.id for r in store.list()}
    assert "net" in ids                                  # the ambient network complex
    assert "net:bio" in ids and "net:chem" in ids        # member hives as subcomplexes
