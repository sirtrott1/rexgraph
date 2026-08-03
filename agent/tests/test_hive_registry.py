"""agent.hive_network registry: named hives, isolation, status, and network-scope logging."""
from agent import activity, agent_complex
from agent import hive as hivemod


def test_named_hives_registered_and_isolated():
    hivemod.reset_network(); activity.reset()
    net = hivemod.get_network()
    a = net.create("alpha")
    b = net.hive("beta")                                      # get-or-create
    a.attach("q1", "http://x", role="queen", model="m")
    b.attach("q2", "http://x", role="queen", model="m")

    assert set(net.names()) >= {"alpha", "beta"}
    assert net.get("alpha").get("q1") is not None
    assert net.get("alpha").get("q2") is None                # rosters are separate
    st = net.status()
    assert st["n_hives"] >= 2 and st["n_bees"] >= 2
    # creation is logged at network/hive scope
    assert any(e["entity"] == "hive:alpha" and e["action"] == "create"
               for e in activity.get_log().events())
    assert net.remove("alpha") and net.get("alpha") is None


def test_named_hive_coordination_is_isolated():
    hivemod.reset_network(); agent_complex.reset_live()
    net = hivemod.get_network()
    alpha = net.hive("alpha")
    for x, y in [("planner", "coder"), ("coder", "reviewer"), ("reviewer", "planner")]:
        alpha.relay(x, y, "waiting")                          # a circular wait inside alpha only
    assert alpha.monitor()["deadlock_cycles"] == 1
    assert net.hive("default").monitor().get("deadlock_cycles", 0) == 0   # default hive unaffected


def test_default_hive_is_backward_compatible():
    hivemod.reset_network(); agent_complex.reset_live()
    h1 = hivemod.get_hive()
    h1.attach("w", "http://x", role="worker", model="m")
    assert hivemod.get_hive() is h1 and hivemod.get_hive().get("w") is not None   # same object
    assert hivemod.get_hive().name == "default"
