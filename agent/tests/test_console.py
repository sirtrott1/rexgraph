"""agent.console: scoped commands, chat routing, and governed consequential verbs."""
from agent.console import CommandConsole
from agent.reactive_hive import ReactiveHive

from agent import agent_complex, rcdb
from agent import hive as hivemod


def _hive():
    hivemod.reset_hive(); agent_complex.reset_live()
    h = hivemod.get_hive()
    h.attach("lead", "http://x", role="queen", model="m", specialties=["coordinate"])
    return h


def test_status_help_and_unknown():
    con = CommandConsole(_hive())
    assert con.command("status")["ok"] is True
    assert "kill" in con.command("help")["commands"]
    assert con.command("frobnicate")["ok"] is False              # unknown verb


def test_require_grows_the_team():
    h = _hive()
    con = CommandConsole(h, reactive=ReactiveHive(h, store=rcdb.MemoryStore()))
    r = con.command("require review test", scope="hive")
    assert r["ok"] and h.get("reviewer") is not None and h.get("tester") is not None


def test_kill_is_governed():
    h = _hive()
    h.add_worker("rogue", lambda d, **k: d, capability="analyze")
    con = CommandConsole(h)
    proposed = con.command("kill rogue")                          # no confirm -> proposal only
    assert proposed["ok"] is False and proposed.get("governed") and h.get("rogue") is not None
    applied = con.command("kill rogue", confirm=True)             # confirmed -> applied
    assert applied["ok"] and h.get("rogue") is None


def test_chat_routes_to_a_worker(monkeypatch):
    h = _hive()
    h.attach("payments", "http://x", role="worker", model="m-pay", specialties=["payment"])
    monkeypatch.setattr(hivemod, "_chat", lambda url, model, prompt, **k: "retries with backoff")
    con = CommandConsole(h)
    r = con.command("chat how do you handle a 503", scope="worker:payments")
    assert r["ok"] and r["from"] == "payments" and "backoff" in r["reply"]


def test_set_overwrites_specialties():
    h = _hive()
    con = CommandConsole(h)
    r = con.command("set lead planning routing", scope="hive")
    assert r["ok"] and h.get("lead").specialties == ["planning", "routing"]
