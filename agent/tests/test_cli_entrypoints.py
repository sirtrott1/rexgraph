"""The console scripts, which nothing exercised.

Fifteen entry points ship in pyproject and 5,452 lines of the agent were never
executed by any test. `cli/auth.py` alone is 493 of them, and it is the surface that
decides who can do what.

These are smoke tests against the real argument parsers: an entry point resolves, it
answers --help without raising, and it rejects nonsense rather than crashing. That is
the floor. Anything that needs a server, a model or SLURM is asserted to fail
cleanly rather than to succeed.
"""
from __future__ import annotations

import contextlib
import importlib
import io

import pytest

#: name -> module:function, read from pyproject's console_scripts
ENTRY_POINTS = {
    "rexgraph-ui": ("agent.server.app", "main"),
    "rexgraph-ocr": ("agent.cli.ocr", "ocr_main"),
    "rexgraph-test": ("agent.cli.test_all", "main"),
    "rexgraph-run": ("agent.cli.run_pipeline", "main"),
    "rexgraph-deploy": ("agent.cli.deploy", "main"),
    "rexgraph-auth": ("agent.cli.auth", "main"),
    "rexgraph-serve": ("agent.cli.serve", "main"),
    "rexgraph-setup": ("agent.cli.setup", "main"),
    "rexgraph-config": ("agent.cli.config", "main"),
    "rexgraph-connect": ("agent.cli.connect", "main"),
    "rexgraph-local": ("agent.local_runtime", "main"),
    "rexgraph-hive": ("agent.hive", "main"),
    "rexgraph-models": ("agent.models.cli", "main"),
    "rexgraph-ops": ("agent.lifecycle", "main"),
    "rcf-server": ("agent.server.app", "main"),
}


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_the_entry_point_resolves(name):
    """Every console_scripts target imports and is callable. A broken one is a
    package that installs and then fails at the shell."""
    mod_name, fn_name = ENTRY_POINTS[name]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    assert fn is not None, f"{mod_name}:{fn_name} does not exist"
    assert callable(fn), f"{mod_name}:{fn_name} is not callable"


def _run(fn, argv):
    """Call a main() with argv, capturing output. Returns (exit_code, text)."""
    import sys
    buf = io.StringIO()
    old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rc = fn()
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = old
    return rc, buf.getvalue()



#: entry points whose main() parses argv; excludes the servers, which bind a port
PARSERS = [n for n in ENTRY_POINTS
           if n not in ("rexgraph-ui", "rcf-server", "rexgraph-serve", "rexgraph-test")]


@pytest.mark.parametrize("name", PARSERS)
def test_help_does_not_raise(name):
    """--help is the one invocation every CLI must survive. argparse exits 0."""
    mod_name, fn_name = ENTRY_POINTS[name]
    fn = getattr(importlib.import_module(mod_name), fn_name)
    rc, out = _run(fn, [name, "--help"])
    assert rc in (0, None), f"{name} --help exited {rc}: {out[:300]}"
    assert out.strip(), f"{name} --help printed nothing"


@pytest.mark.parametrize("name", PARSERS)
def test_an_unknown_subcommand_is_rejected_cleanly(name):
    """A typo gets a usage error, not a traceback."""
    mod_name, fn_name = ENTRY_POINTS[name]
    fn = getattr(importlib.import_module(mod_name), fn_name)
    try:
        rc, out = _run(fn, [name, "definitely-not-a-real-subcommand"])
    except Exception as e:                       # noqa: BLE001 - that is the finding
        pytest.fail(f"{name} raised {type(e).__name__} on a bad subcommand: {e}")
    assert rc != 0 or "usage" in out.lower() or "unknown" in out.lower(), (
        f"{name} accepted a nonsense subcommand (rc={rc})")


#### rexgraph-auth: the surface that decides who can do what


@pytest.fixture
def auth_home(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    import agent.server.auth as A
    if hasattr(A, "_MANAGER"):
        monkeypatch.setattr(A, "_MANAGER", None, raising=False)
    return tmp_path


def test_auth_status_runs_on_a_fresh_config(auth_home):
    from agent.cli.auth import main
    rc, out = _run(main, ["rexgraph-auth", "status"])
    assert rc in (0, None), f"status exited {rc}: {out[:300]}"
    assert out.strip(), "status printed nothing"


def test_auth_create_without_a_server_fails_with_guidance(auth_home):
    """`create` is an HTTP client of a running server, not a local config editor.
    With nothing to talk to it has to say so and say what to do, not traceback."""
    from agent.cli.auth import main
    rc, out = _run(main, ["rexgraph-auth", "create", "--name", "probe",
                          "--role", "admin", "--save"])
    assert rc != 0, "create appeared to succeed with no server running"
    low = out.lower()
    assert "auth" in low or "401" in out or "connect" in low, (
        f"the failure does not explain itself: {out[:300]}")
    assert "--admin-token" in out or "login" in low, (
        f"the failure does not say what to do next: {out[:300]}")
    assert "Traceback" not in out, f"create raised instead of reporting: {out[:300]}"


def test_auth_rejects_an_unknown_role_before_reaching_the_network(auth_home):
    """A bad --role is the caller's mistake and argparse should catch it locally."""
    from agent.cli.auth import main
    rc, out = _run(main, ["rexgraph-auth", "create", "--name", "x",
                          "--role", "wizard", "--save"])
    assert rc != 0, f"an unknown role was accepted (rc={rc})"
    assert "Traceback" not in out, out[:300]


#### rexgraph-config and rexgraph-connect: read-only verbs


def test_config_show_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    from agent.cli.config import main
    rc, out = _run(main, ["rexgraph-config", "show"])
    assert rc in (0, None), f"config show exited {rc}: {out[:300]}"


def test_connect_list_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    from agent.cli.connect import main
    rc, out = _run(main, ["rexgraph-connect", "list"])
    assert rc in (0, None), f"connect list exited {rc}: {out[:300]}"
