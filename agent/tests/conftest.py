"""Suite-wide test isolation - never read or mutate real user state.

``agent.server.auth`` resolves ``REXGRAPH_CONFIG_DIR`` at IMPORT time (module-level
``_CONFIG_DIR``), so a test that triggers a token/recovery/TLS write without first
overriding it would land in the real ``~/.config/rexgraph/`` - leaking or clobbering
a developer's actual tokens. This conftest runs before any test module imports the
agent package, so pointing the config dir at a throwaway temp dir here guarantees
the whole suite stays hermetic. Individual tests may still ``monkeypatch`` it to
their own ``tmp_path``; that just overrides this (already-safe) default per-test.
"""
import atexit
import contextlib
import os
import pathlib
import shutil
import sys
import tempfile

import pytest


# rexgraph-rcdb is a sibling distribution. Preferring an installed one and falling back
# to the copy beside this checkout means the suite exercises the real package without
# anything being installed into the caller's environment, which is the same rule the rcql
# and system suites follow. The bare import is dropped first because the repository root
# shadows both as namespace packages and an empty one would otherwise pass for the real
# thing.
def _ensure_sibling(name: str) -> None:
    try:
        module = __import__(name)
        if getattr(module, "__file__", None):
            return
    except ImportError:
        pass
    sys.modules.pop(name, None)
    root = pathlib.Path(__file__).resolve().parents[2] / name
    if not root.is_dir():
        return
    sys.path.insert(0, str(root))
    # And for CHILD processes: several tests reopen a store from a fresh interpreter,
    # which is the only way to prove persistence rather than one process's memory. A
    # sys.path insert does not reach them, so the environment has to carry it.
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if str(root) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(root), *parts])


_ensure_sibling("rcdb")

# Put the agent project root on the path before anything imports from it.
#
# Two things resolve wrongly without this when pytest is invoked from the REPOSITORY
# root rather than from agent/. The directory `agent/` has no __init__.py, so from the
# repository root `import agent` finds it as a namespace package and binds to the outer
# directory instead of the real package at agent/agent/, which has no __version__ and
# made three version tests fail for a reason that had nothing to do with versions. And
# five test modules import their fixtures with `from tests.test_... import`, which needs
# agent/ on the path to resolve at all, so they were not collected and the suite reported
# a pass without them.
#
# Both worked when the suite was run from agent/, so the failures looked like a broken
# checkout rather than a rootdir difference. Fixing it here means the same command works
# from either directory.
_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

# If `agent` already bound to the namespace directory, drop it so the next import picks
# up the real package. A namespace package has no __file__, which is what distinguishes
# it; a correctly imported agent is left alone.
_bound = sys.modules.get("agent")
if _bound is not None and getattr(_bound, "__file__", None) is None:
    for _name in [n for n in sys.modules if n == "agent" or n.startswith("agent.")]:
        del sys.modules[_name]

# Set at module import time (before agent.* is imported by any test module) so the
# import-time config-dir binding picks up the temp location. Force it - the whole
# point is that a stray REXGRAPH_CONFIG_DIR pointing at real config can't leak in.
_TEST_HOME = tempfile.mkdtemp(prefix="rexgraph_test_config_")
os.environ["REXGRAPH_CONFIG_DIR"] = _TEST_HOME

# The suite drives many endpoints from a single client IP; the global rate limiter
# would otherwise 429 later tests. Disable it here (the limiter itself is covered
# by its own dedicated test that configures a low limit on an isolated app).
os.environ.setdefault("RCF_RATE_LIMIT", "0")

atexit.register(lambda: shutil.rmtree(_TEST_HOME, ignore_errors=True))


@pytest.fixture(autouse=True)
def _isolate_auth_state():
    """Reset the auth singleton and its persisted config after every test.

    The auth manager is a process-wide singleton that persists to auth.json in
    the (shared, session-scoped) config dir. Without this, a test that enables
    auth or mints a token would leak that state into unrelated later tests. The
    reset runs on teardown, so the next test loads a fresh (auth-off) manager.
    """
    yield
    from agent.server import auth
    auth.reset_auth_manager()
    with contextlib.suppress(OSError):
        (auth._CONFIG_DIR / "auth.json").unlink()
