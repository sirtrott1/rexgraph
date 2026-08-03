"""agent.dashboard: the hive-network snapshot + terminal render."""
from agent.dashboard import hive_dashboard, render

from agent import agent_complex
from agent import hive as hivemod


def test_dashboard_snapshot_and_render():
    hivemod.reset_hive(); agent_complex.reset_live()
    h = hivemod.get_hive()
    h.attach("lead", "http://x", role="queen", model="m", specialties=["coordinate"])
    h.attach("w1", "http://x", role="worker", model="m", specialties=["work"])
    h.relay("lead", "w1", "do it"); h.relay("w1", "lead", "done")   # some information flow
    d = hive_dashboard(h)
    assert d["overview"]["bees"] == 2 and d["overview"]["queen"] == "lead"
    assert "coordination" in d and "information_flow" in d
    assert d["information_flow"]["edges"]                            # who -> whom is captured
    out = render(d)
    assert isinstance(out, str) and "HIVE NETWORK" in out
