"""Suite-wide test isolation - never read or mutate real user state.

``agent.server.auth`` resolves ``REXGRAPH_CONFIG_DIR`` at IMPORT time (module-level
``_CONFIG_DIR``), so a test that triggers a token/recovery/TLS write without first
overriding it would land in the real ``~/.config/rexgraph/`` - leaking or clobbering
a developer's actual tokens. This conftest runs before any test module imports the
agent package, so pointing the config dir at a throwaway temp dir here guarantees
the whole suite stays hermetic. Individual tests may still ``monkeypatch`` it to
their own ``tmp_path``; that just overrides this (already-safe) default per-test.
"""
import os
import atexit
import shutil
import tempfile

import pytest

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
    try:
        (auth._CONFIG_DIR / "auth.json").unlink()
    except OSError:
        pass
