"""Turning auth OFF is the unsafe direction, and it used to be the easy one.

`disable_auth()` persisted unconditionally, so any in-process caller wrote
`enabled: false` into the host's own `~/.config/rexgraph/auth.json`. Six test fixtures
did exactly that, which is how a test suite turned auth off on a live install and left it
off. Enabling needs no ceremony: the worst an accidental enable does is ask for a token
the caller already has. Disabling needs two, and they are separate on purpose:

    persist=False   flip the flag for this process only, never touch disk. What a test
                    wants, because it needs the server object open rather than the host
                    reconfigured.
    confirm=True    required before a disable is WRITTEN to a config that has tokens,
                    because that is someone's live install. Missing it raises rather
                    than writing, so an accident is loud instead of silent.

The network path is stricter and unchanged: host-local, admin token, disable passphrase.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """A manager over a config directory of its own, with one token, auth on."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    import agent.server.auth as auth

    importlib.reload(auth)
    mgr = auth.get_auth_manager()
    mgr.enable_auth()
    mgr.bootstrap_admin()
    return mgr, tmp_path


def _written(path):
    return json.loads((Path(path) / "auth.json").read_text())["enabled"]


#### the guard


def test_a_naive_disable_is_refused_when_tokens_exist(manager):
    mgr, _path = manager
    with pytest.raises(PermissionError, match="refusing to write auth off"):
        mgr.disable_auth()


def test_the_refusal_leaves_auth_on(manager):
    """A guard that half-applied would be worse than none."""
    mgr, path = manager
    with pytest.raises(PermissionError):
        mgr.disable_auth()
    assert mgr._auth_enabled is True
    assert _written(path) is True


def test_the_refusal_names_what_is_at_stake(manager):
    mgr, _path = manager
    with pytest.raises(PermissionError) as caught:
        mgr.disable_auth()
    message = str(caught.value)
    assert "token(s)" in message
    assert "confirm=True" in message and "persist=False" in message


#### the two ways through


def test_persist_false_flips_the_process_and_not_the_host(manager):
    mgr, path = manager
    mgr.disable_auth(persist=False)
    assert mgr._auth_enabled is False
    assert _written(path) is True, "an in-process toggle reached the host's config"


def test_confirm_true_writes(manager):
    mgr, path = manager
    mgr.disable_auth(confirm=True)
    assert _written(path) is False


def test_a_fresh_config_needs_no_confirmation(manager, tmp_path, monkeypatch):
    """Nothing is at stake before there are tokens, and a first-run install should not
    have to argue with the library."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "fresh"))
    import agent.server.auth as auth

    importlib.reload(auth)
    fresh = auth.get_auth_manager()
    fresh.disable_auth()
    assert fresh._auth_enabled is False


#### enabling stays easy


def test_enabling_needs_nothing(manager):
    mgr, path = manager
    mgr.disable_auth(confirm=True)
    mgr.enable_auth()
    assert _written(path) is True


def test_enable_can_also_be_process_only(manager):
    mgr, path = manager
    mgr.disable_auth(confirm=True)
    mgr.enable_auth(persist=False)
    assert mgr._auth_enabled is True
    assert _written(path) is False


#### nothing in the tree disables auth the old way


def test_no_call_site_disables_without_saying_which_it_means():
    """The recurring theme, pinned. Every caller has to say persist=False or confirm=True,
    so a new fixture cannot quietly reconfigure the host again."""
    import re

    root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in list((root / "agent").rglob("*.py")) + list((root / "rexgraph").rglob("*.py")):
        if path.name == "auth.py" or path.name == Path(__file__).name:
            continue
        for line in path.read_text().splitlines():
            if re.search(r"\.disable_auth\(\s*\)", line):
                offenders.append(f"{path.relative_to(root)}: {line.strip()}")
    assert not offenders, "these disable auth without saying how:\n" + "\n".join(offenders)
