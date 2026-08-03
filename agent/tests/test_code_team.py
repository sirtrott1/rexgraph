"""agent.code_team: concurrent generate+evaluate, self-heal via debugger, lead unifies a build."""
import threading
import time

from agent.code_team import CodeTeam

from agent import agent_complex
from agent import hive as hivemod


def _hive():
    hivemod.reset_hive(); agent_complex.reset_live()
    h = hivemod.get_hive()
    h.attach("coder", "http://x", role="queen", model="m", specialties=["code", "implement"])
    return h


def test_build_unifies_passing_pieces_and_grows_evaluators():
    team = CodeTeam(hive=_hive())
    tasks = [
        {"name": "add", "code": "def add(a, b):\n    return a + b",
         "tests": [("add(2, 3)", "5"), ("add(-1, 1)", "0")]},
        {"name": "mul", "code": "def mul(a, b):\n    return a * b",
         "tests": [("mul(2, 3)", "6")]},
    ]
    r = team.build(tasks)
    assert r["committed"] is True                        # the unified build passes every test
    assert r["integration"]["passed"] == 3
    assert all(p["verdict"]["ok"] for p in r["pieces"])
    # the team grew the review + test roles up front
    assert team.hive.get("reviewer") is not None and team.hive.get("tester") is not None


def test_failing_piece_self_heals_with_a_debugger():
    def gen(task, feedback):
        return "def sub(a, b):\n    return a - b" if feedback else "def sub(a, b):\n    return a + b"

    team = CodeTeam(hive=_hive())
    tasks = [{"name": "sub", "generate": gen, "tests": [("sub(5, 2)", "3")]}]
    r = team.build(tasks)
    assert r["committed"] is True                        # the retry fixed it
    assert r["pieces"][0]["retried"] is True
    assert team.hive.get("debugger") is not None         # reactive layer deployed a debugger


def test_generation_and_evaluation_overlap():
    intervals, lock = [], threading.Lock()

    def gen(task, feedback):
        s = time.perf_counter(); time.sleep(0.05); e = time.perf_counter()
        with lock:
            intervals.append((s, e))
        return task["code"]

    team = CodeTeam(hive=_hive(), max_workers=3)
    tasks = [{"name": f"t{i}", "generate": gen, "code": "def f():\n    return 1",
              "tests": [("f()", "1")]} for i in range(3)]
    team.build(tasks)
    intervals.sort()
    # concurrency, proven without a timing threshold: two generation windows overlap in time
    assert any(intervals[i][1] > intervals[i + 1][0] for i in range(len(intervals) - 1))


def test_partial_build_commits_only_passing_pieces():
    team = CodeTeam(hive=_hive())
    tasks = [
        {"name": "ok", "code": "def ok():\n    return 42", "tests": [("ok()", "42")]},
        {"name": "bad", "generate": lambda t, fb: "def bad():\n    return 0", "tests": [("bad()", "1")]},
    ]
    r = team.build(tasks)
    assert r["committed"] is False                       # 'bad' never passes even after retry
    verdicts = {p["name"]: p["verdict"]["ok"] for p in r["pieces"]}
    assert verdicts == {"ok": True, "bad": False}
    assert "def ok" in r["build"] and "def bad" not in r["build"]   # only passing pieces committed
