"""agent.code_team: a team that generates and evaluates code concurrently, then unifies a build.

Code agents take on tasks in parallel: each piece is generated and immediately evaluated (its tests
run), so while one piece is being written another is being tested - evaluation overlaps generation
across the team. A lead then unifies the passing pieces into a single module and runs the whole
test set against it (the integration "commit"). The team rides the reactive layer (agent.
reactive_hive): it grows review/test roles up front, and when a piece fails it deploys a debugger
and retries with the failure as feedback - a self-healing build, every structural change versioned
in the hive's self-schema.

Generation is pluggable: a task may carry a `generate(task, feedback)` callable (deterministic), a
static `code` string, or fall back to the hive's chat path. Evaluation is real - the code and its
tests run in a subprocess.
"""
from __future__ import annotations

import contextlib
import os
import re
import subprocess
import tempfile
import textwrap
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .reactive_hive import ReactiveHive

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def _extract_code(reply: str) -> str:
    if not reply:
        return ""
    blocks = _CODE_RE.findall(reply)
    return (blocks[-1] if blocks else reply).strip()


def _run_tests(code: str, tests: list) -> dict[str, Any]:
    """Run candidate code + its hidden tests in a subprocess. tests: list of (call_expr, expected)."""
    if not tests:
        return {"passed": 0, "total": 0, "ok": False, "error": "no tests"}
    checks = "\n".join(f"_t(lambda: {call}, {exp})" for call, exp in tests)
    harness = textwrap.dedent("""
        _p = [0]; _n = [0]; _err = []
        def _t(fn, expected):
            _n[0] += 1
            try:
                if fn() == expected: _p[0] += 1
                else: _err.append('mismatch')
            except Exception as e: _err.append(repr(e))
    """) + "\n" + (code or "") + "\n" + checks + "\nprint(_p[0], _n[0], '|', '; '.join(_err[:3]))"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(harness); path = f.name
        out = subprocess.run(["python", path], capture_output=True, text=True, timeout=10)
        head, _, err = out.stdout.strip().partition("|")
        p, n = (head.split() + ["0", "0"])[:2]
        passed, total = int(p), int(n)
        return {"passed": passed, "total": total, "ok": total > 0 and passed == total,
                "error": (err.strip() or out.stderr.strip()[:200])}
    except Exception as e:
        return {"passed": 0, "total": len(tests), "ok": False, "error": repr(e)}
    finally:
        if path:
            with contextlib.suppress(Exception): os.unlink(path)


class CodeTeam:
    """A concurrent, self-healing code team on top of ReactiveHive."""

    def __init__(self, hive=None, reactive: ReactiveHive | None = None, *, store=None,
                 generate: Callable | None = None, max_workers: int = 4):
        if reactive is None:
            if hive is None:
                from . import hive as hivemod
                hive = hivemod.get_hive()
            reactive = ReactiveHive(hive, store=store)
        self.reactive = reactive
        self.hive = reactive.hive
        self.generate = generate
        self.max_workers = max_workers

    # -- generation + evaluation ----------------------------------------------

    def _generate(self, task: dict, feedback: str | None = None) -> str:
        gen = task.get("generate") or self.generate
        if callable(gen):
            return _extract_code(gen(task, feedback))
        if "code" in task and feedback is None:
            return task["code"]
        # hive chat fallback
        spec = task.get("spec", task.get("name", ""))
        if feedback:
            spec += f"\n\nYour previous attempt failed: {feedback}\nReturn the corrected function."
        try:
            reply = self.hive.dispatch(spec).get("reply")
        except Exception:
            reply = None
        return _extract_code(reply or task.get("code", ""))

    def _gen_eval(self, task: dict, feedback: str | None = None) -> dict:
        """One piece: generate, then evaluate. Run concurrently across pieces so evaluation of one
        overlaps generation of another."""
        code = self._generate(task, feedback=feedback)
        verdict = _run_tests(code, task.get("tests", []))
        return {"name": task["name"], "task": task, "code": code, "verdict": verdict,
                "retried": feedback is not None}

    # -- the build ------------------------------------------------------------

    def build(self, tasks: list[dict]) -> dict[str, Any]:
        """Generate + evaluate all pieces concurrently, self-heal failures, then unify a build."""
        reactions: list[dict] = []
        reactions += self.reactive.require("review", "test")     # grow the evaluators up front

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            pieces = list(ex.map(self._gen_eval, tasks))

        # self-heal: a failing piece deploys a debugger and retries with the error as feedback
        failed = [p for p in pieces if not p["verdict"]["ok"]]
        if failed:
            reactions += self.reactive.require("debug")
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                fixes = list(ex.map(lambda p: self._gen_eval(p["task"], feedback=p["verdict"]["error"]),
                                    failed))
            by_name = {p["name"]: p for p in pieces}
            for fx in fixes:
                if fx["verdict"]["passed"] >= by_name[fx["name"]]["verdict"]["passed"]:
                    by_name[fx["name"]] = fx
            pieces = [by_name[t["name"]] for t in tasks]
        wall = time.perf_counter() - t0

        # the lead unifies the passing pieces into one module and runs the whole test set
        passing = [p for p in pieces if p["verdict"]["ok"]]
        build_code = "\n\n".join(p["code"] for p in passing)
        all_tests = [t for task in tasks for t in task.get("tests", [])]
        integration = _run_tests(build_code, all_tests)

        return {
            "committed": integration["ok"],
            "integration": integration,
            "build": build_code,
            "pieces": [{"name": p["name"], "verdict": p["verdict"], "retried": p["retried"]}
                       for p in pieces],
            "reactions": reactions,
            "team": [b.name for b in self.hive.bees()],
            "wall_seconds": round(wall, 3),
        }
