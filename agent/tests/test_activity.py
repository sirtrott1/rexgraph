"""agent.activity: the activity log + model-usage registry, and its wiring into the hive/foundry."""
import json
import time

from agent import activity, hive as hivemod, agent_complex


def _wait(cond, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_journal_tailer_folds_peer_events(tmp_path):
    # a peer process (different src) appends to the shared journal; the tailer folds it into this log
    # AND pushes it live to subscribers - this is exactly what a CLI action -> running server looks like.
    jp = tmp_path / "activity.jsonl"
    activity.reset()
    log = activity.get_log()
    got = []
    log.subscribe(lambda ev: got.append(ev))
    log.enable_journal(str(jp), warm=False, tail=True)
    peer = {"ts": time.time(), "entity": "hive:peer", "scope": "hive", "action": "create",
            "detail": {"via": "cli"}, "src": "peerabcd"}
    with open(jp, "a") as f:
        f.write(json.dumps(peer) + "\n")
    assert _wait(lambda: any(e["entity"] == "hive:peer" for e in log.events()))
    assert any(e["entity"] == "hive:peer" for e in got)       # pushed live, not just stored
    log.close()


def test_journal_own_writes_not_doubled(tmp_path):
    # the server records AND tails the same file; its own line must not be re-folded (own-src skip)
    jp = tmp_path / "activity.jsonl"
    activity.reset()
    log = activity.get_log()
    log.enable_journal(str(jp), warm=False, tail=True)
    log.record("worker:coder", "deploy")
    time.sleep(0.25)                                          # give the tailer time to (wrongly) re-add
    assert sum(1 for e in log.events() if e["entity"] == "worker:coder") == 1
    log.close()


def test_journal_warm_load_restores_history(tmp_path):
    # a prior session's journal on disk -> a fresh log warm-loads its history (persistence across restarts)
    jp = tmp_path / "activity.jsonl"
    with open(jp, "w") as f:
        f.write(json.dumps({"ts": 1.0, "entity": "hive:old", "scope": "hive", "action": "create",
                            "detail": {}, "src": "prev1"}) + "\n")
        f.write(json.dumps({"ts": 2.0, "entity": "model:qwen", "scope": "model", "action": "use.open",
                            "detail": {"purpose": "x", "by": "y", "handle": 1}, "src": "prev1"}) + "\n")
    activity.reset()
    log = activity.get_log()
    log.enable_journal(str(jp), warm=True, tail=False)
    ents = {e["entity"] for e in log.events()}
    assert "hive:old" in ents and "model:qwen" in ents        # history restored
    u = log.usage()["qwen"]
    assert u["total_uses"] == 1 and u["concurrent"] == 0      # a dead process's open use is not "active"
    log.close()


def test_log_records_and_filters_by_scope_and_prefix():
    activity.reset()
    log = activity.get_log()
    log.record("hive:alpha", "compose", detail={"n": 3})
    log.record("worker:coder", "dispatch", detail={"q": "x"})
    log.record("worker:reviewer", "deploy")
    assert len(log.events()) == 3                              # newest-first
    assert log.events()[0]["entity"] == "worker:reviewer"
    assert len(log.events(scope="worker")) == 2
    assert len(log.events(action="deploy")) == 1
    # entity prefix: a hive covers its workers
    log.record("hive:alpha:worker:coder", "note")             # (illustrative nested id)
    assert log.events(entity="hive:alpha")                    # prefix match


def test_model_usage_concurrency_and_runtime():
    activity.reset()
    log = activity.get_log()
    h1 = log.open_use("qwen-7b", "collaborate", by="worker:coder")
    h2 = log.open_use("qwen-7b", "consensus", by="worker:reviewer")   # same model, concurrent
    u = log.usage()["qwen-7b"]
    assert u["concurrent"] == 2 and u["total_uses"] == 2
    assert u["instantiated"] is not None and u["runtime_s"] >= 0
    log.close_use(h1)
    assert log.usage()["qwen-7b"]["concurrent"] == 1          # one still active
    log.close_use(h2)
    assert log.usage()["qwen-7b"]["concurrent"] == 0 and log.usage()["qwen-7b"]["total_uses"] == 2


def test_pubsub_subscribe_notify_unsubscribe():
    activity.reset()
    log = activity.get_log()
    got = []
    def cb(ev):
        got.append(ev)
    log.subscribe(cb)
    log.record("hive:alpha", "create")
    assert len(got) == 1 and got[0]["entity"] == "hive:alpha" and got[0]["action"] == "create"
    log.unsubscribe(cb)
    log.record("hive:beta", "create")
    assert len(got) == 1                                      # unsubscribed -> no more pushes


def test_events_route_registered():
    # the SSE endpoint is registered (streaming it under TestClient hangs on the infinite generator,
    # so this asserts registration via the OpenAPI schema; the stream is exercised live)
    from agent.server.app import app
    assert "/api/v1/agents/events" in app.openapi().get("paths", {})


def test_hive_actions_and_asks_are_logged(monkeypatch):
    activity.reset(); hivemod.reset_hive(); agent_complex.reset_live()
    h = hivemod.get_hive()
    h.attach("coder", "http://x", role="worker", model="qwen-7b", specialties=["code"])
    h.add_worker("net", lambda d, **k: d, capability="predict", worker_type="model:mlp")
    monkeypatch.setattr(hivemod, "_chat", lambda url, model, prompt, **k: "ok")
    h.ask("coder", "hi")                                       # opens+closes a use on qwen-7b
    h.invoke("net", None)                                      # opens+closes a use on net

    actions = {e["action"] for e in activity.get_log().events()}
    assert {"attach", "deploy", "use.open", "use.close"} <= actions
    usage = activity.get_log().usage()
    assert usage["qwen-7b"]["total_uses"] >= 1 and usage["qwen-7b"]["concurrent"] == 0
