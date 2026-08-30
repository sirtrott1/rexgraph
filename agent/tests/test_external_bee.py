"""agent.external_bee: a bee answered by a process that polls, and its wiring into the hive."""
import threading
import time

import httpx
import pytest

from agent import activity, external_bee
from agent import hive as hivemod


@pytest.fixture(autouse=True)
def _isolated_journal(tmp_path, monkeypatch):
    # keep every recorded event in the test's own journal rather than the operator's
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", str(tmp_path / "activity.jsonl"))
    activity.reset()
    hivemod.reset_network()
    yield
    activity.get_log().close()


@pytest.fixture
def broker():
    srv = external_bee.serve(port=0, name="claude", model="claude-probe", reply_timeout=3.0)
    yield srv
    srv.stop()


def _responder(srv, reply, *, wait=3.0, token=""):
    """Claim one request and answer it, the way an external agent drives the endpoint."""
    seen = {}

    def run():
        h = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(timeout=wait + 2) as c:
            r = c.get(f"{srv.url}/agent/next?wait={wait}", headers=h)
            if r.status_code != 200:
                return
            seen.update(r.json())
            body = dict(reply, id=seen["id"])
            c.post(f"{srv.url}/agent/reply", json=body, headers=h)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, seen


def test_a_polling_responder_answers_the_hives_ask(broker):
    hive = hivemod.get_hive()
    hive.attach("claude", broker.url, role="worker", specialties=["topology"])
    t, _ = _responder(broker, {"content": "the kernel is the constant vector"})

    assert hive.ask("claude", "what is in ker L0") == "the kernel is the constant vector"
    t.join(5)
    assert broker.broker.status()["served"] == 1


def test_messages_reach_the_responder_verbatim(broker):
    hive = hivemod.get_hive()
    hive.attach("claude", broker.url)
    t, seen = _responder(broker, {"content": "ok"})
    hive.ask("claude", "count the channels", system="answer in one word")
    t.join(5)

    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user"], "the tool loop depends on history arriving unrewritten"
    assert seen["messages"][0]["content"] == "answer in one word"
    assert seen["params"]["max_tokens"] == 512 and "temperature" in seen["params"]


def test_an_unanswered_request_times_out_and_the_hive_degrades():
    srv = external_bee.serve(port=0, name="idle", reply_timeout=0.4)
    try:
        hive = hivemod.get_hive()
        hive.attach("idle", srv.url)
        t0 = time.monotonic()
        # nobody polls, so the broker gives up first and the hive reads the 504 as unreachable
        assert hive.ask("idle", "anyone home") is None
        assert time.monotonic() - t0 < 5.0
        assert srv.broker.status()["timed_out"] == 1
    finally:
        srv.stop()


def test_tool_calls_survive_the_round_trip(broker):
    hive = hivemod.get_hive()
    hive.attach("claude", broker.url)
    call = {"id": "call_1", "type": "function",
            "function": {"name": "rexgraph_homology", "arguments": "{}"}}
    t, _ = _responder(broker, {"content": None, "tool_calls": [call]})

    res = hive.ask_full("claude", "read the betti numbers")
    t.join(5)
    assert res is not None and res.wants_tools
    assert res.finish_reason == "tool_calls"
    assert res.tool_calls[0]["function"]["name"] == "rexgraph_homology"


def test_the_token_reference_guards_both_faces(monkeypatch):
    monkeypatch.setenv("BEE_TOKEN", "s3cret")
    srv = external_bee.serve(port=0, name="guarded", token_ref="BEE_TOKEN", reply_timeout=1.0)
    try:
        with httpx.Client(timeout=5) as c:
            assert c.get(f"{srv.url}/agent/next?wait=0").status_code == 401
            assert c.post(f"{srv.url}/v1/chat/completions",
                          json={"messages": [{"role": "user", "content": "x"}]}).status_code == 401
            # /health stays open: a liveness probe that needs the credential cannot report on it
            assert c.get(f"{srv.url}/health").status_code == 200
        t, _ = _responder(srv, {"content": "admitted"}, wait=2.0, token="s3cret")
        hive = hivemod.get_hive()
        hive.attach("guarded", srv.url, api_key_ref="BEE_TOKEN")
        assert hive.ask("guarded", "knock") == "admitted"
        t.join(5)
    finally:
        srv.stop()


def test_models_route_lets_attach_live_discover_the_broker(broker):
    with httpx.Client(timeout=5) as c:
        data = c.get(f"{broker.url}/v1/models").json()
    assert [m["id"] for m in data["data"]] == ["claude-probe"]


def test_a_late_answer_is_dropped_not_delivered():
    srv = external_bee.serve(port=0, name="slow", reply_timeout=0.3)
    try:
        p = srv.broker.submit([{"role": "user", "content": "x"}])
        assert srv.broker.wait(p) is None
        with httpx.Client(timeout=5) as c:
            r = c.post(f"{srv.url}/agent/reply", json={"id": p.id, "content": "too late"})
        assert r.status_code == 409 and r.json()["ok"] is False
    finally:
        srv.stop()


def test_the_exchange_lands_in_the_activity_journal(broker):
    hive = hivemod.get_hive()
    hive.attach("claude", broker.url)
    t, _ = _responder(broker, {"content": "recorded"})
    hive.ask("claude", "log this")
    t.join(5)

    actions = [e["action"] for e in activity.get_log().events(entity="worker:claude")]
    assert {"request", "claim", "reply"} <= set(actions)


def test_a_command_can_be_the_bee(broker):
    """The generalisation the broker exists for: a responder that can poll but not listen,
    which is what a CLI harness is. `cat` stands in for the command so the test does not
    need whatever tool the operator actually wires up."""
    stop = threading.Event()
    t = threading.Thread(target=external_bee.respond_with, daemon=True,
                         kwargs={"url": broker.url, "command": "cat", "stop": stop,
                                 "wait": 2.0})
    t.start()
    try:
        hive = hivemod.get_hive()
        hive.attach("shell", broker.url)
        assert hive.ask("shell", "the prompt comes back") == "the prompt comes back"
    finally:
        stop.set()


def test_a_broken_command_answers_instead_of_hanging(broker):
    """A command that cannot run reports so. The hive's own timeout would fire eventually
    and report a bee that is up and idle, which is a different fault from a broken one."""
    stop = threading.Event()
    threading.Thread(target=external_bee.respond_with, daemon=True,
                     kwargs={"url": broker.url, "command": "no-such-command-here",
                             "stop": stop, "wait": 2.0}).start()
    try:
        reply = hivemod.get_hive().attach("broken", broker.url) and \
            hivemod.get_hive().ask("broken", "anything")
        assert reply and "could not run" in reply
    finally:
        stop.set()


def test_the_transcript_keeps_the_roles_it_was_sent():
    """A system turn arriving indistinguishable from the user's is a different request."""
    text = external_bee.render_prompt([{"role": "system", "content": "be terse"},
                                       {"role": "user", "content": "why"}])
    assert text.splitlines()[0] == "[system] be terse"
    assert text.strip().endswith("why")
