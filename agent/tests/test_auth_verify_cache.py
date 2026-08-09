"""Authenticating a request should not cost more than answering it.

bcrypt is deliberately slow, and token verification scanned EVERY stored hash on EVERY
request. At cost 12 with five tokens on file, and the matching one last, a request that
returned 0.1 KB took 0.86s and the app took about three seconds a page. The work was the
authentication, not the data, and it grew with the number of tokens.

Verify once and remember the answer for the process. The key is a SHA-256 of the presented
token rather than the token itself, so the cache never holds the secret. A MISS is cached
too, or an unauthenticated caller could still make the server do five bcrypts per request
by repeating one bad token. The whole cache is cleared whenever the token set moves, so a
revoked token stops working on the next request rather than at restart.
"""
from __future__ import annotations

import importlib
import time

import pytest


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    import agent.server.auth as auth

    importlib.reload(auth)
    mgr = auth.get_auth_manager()
    mgr.enable_auth()
    tokens = [mgr.create_token(f"u{i}", ["default"]) for i in range(4)]
    return mgr, tokens


#### it still authenticates


def test_a_good_token_verifies(manager):
    mgr, tokens = manager
    assert mgr.verify(tokens[-1]) is not None


def test_a_bad_token_does_not(manager):
    mgr, _tokens = manager
    assert mgr.verify("not-a-token") is None


def test_the_cached_answer_is_the_same_entry(manager):
    mgr, tokens = manager
    first = mgr.verify(tokens[0])
    assert mgr.verify(tokens[0]) is first


#### and it stops paying for it


def test_the_second_verify_is_orders_of_magnitude_faster(manager):
    mgr, tokens = manager
    start = time.perf_counter()
    mgr.verify(tokens[-1])
    cold = time.perf_counter() - start
    start = time.perf_counter()
    mgr.verify(tokens[-1])
    warm = time.perf_counter() - start
    assert warm < cold / 100, f"cold {cold:.4f}s, warm {warm:.6f}s"


def test_a_repeated_bad_token_is_not_re_scanned(manager):
    """Otherwise an unauthenticated caller sets the cost, which is the wrong way round."""
    mgr, _tokens = manager
    mgr.verify("wrong")
    start = time.perf_counter()
    mgr.verify("wrong")
    assert time.perf_counter() - start < 0.01


#### and it cannot go stale or leak


def test_minting_a_token_clears_the_cache(manager):
    mgr, tokens = manager
    mgr.verify(tokens[0])
    assert mgr._verify_cache
    mgr.create_token("later", ["default"])
    assert mgr._verify_cache == {}


def test_the_cache_never_holds_the_token(manager):
    """Keys are sha256 hexdigests, so the secret is not sitting in a dict."""
    mgr, tokens = manager
    mgr.verify(tokens[0])
    assert all(len(key) == 64 for key in mgr._verify_cache)
    assert not any(tokens[0] in str(key) for key in mgr._verify_cache)


def test_the_cache_is_bounded(manager):
    """A flood of bad tokens must not grow it without limit."""
    mgr, _tokens = manager
    mgr._verify_cache = {f"k{i}": False for i in range(4096)}
    mgr.verify("one-more-bad-token")
    assert len(mgr._verify_cache) == 4096
