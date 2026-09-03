"""The key context a container is sealed with, and who may name a key.

rexgraph.io never sees key material: it calls seal and open on an opaque object and
records only the key identifier. That makes resolving an identifier to a key an
authorization question, and this is where it is answered.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cryptography")


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", f"file://{tmp_path}/connections.json")
    from agent.server import auth, scope
    scope.reset_secret_store(); auth.reset_auth_manager()
    auth.get_auth_manager().enable_auth()
    yield scope
    scope.reset_secret_store(); auth.reset_auth_manager()


def _in_workspace(scope, name):
    """A keyring as it would be built inside a request for `name`."""
    from agent.kms import WorkspaceKeyring
    token = scope.set_workspace(name)
    try:
        store = scope.secret_store()
        store.put("grade0", f"secret-of-{name}", "key")
        return WorkspaceKeyring().load("grade0"), token
    finally:
        pass


def test_a_sealed_envelope_opens_again(scoped):
    ring, token = _in_workspace(scoped, "alpha")
    try:
        sealed = ring.seal("grade0", b"boundary bytes", b"aad")
        assert sealed != b"boundary bytes"
        assert ring.open(sealed, b"aad") == b"boundary bytes"
    finally:
        scoped.reset_workspace(token)


def test_a_different_aad_does_not_open(scoped):
    ring, token = _in_workspace(scoped, "alpha")
    try:
        sealed = ring.seal("grade0", b"boundary bytes", b"grade0/chunk0")
        with pytest.raises(PermissionError):
            ring.open(sealed, b"grade0/chunk1")
    finally:
        scoped.reset_workspace(token)


def test_tampered_ciphertext_does_not_open(scoped):
    ring, token = _in_workspace(scoped, "alpha")
    try:
        sealed = bytearray(ring.seal("grade0", b"boundary bytes", b"aad"))
        sealed[-1] ^= 0x01
        with pytest.raises(PermissionError):
            ring.open(bytes(sealed), b"aad")
    finally:
        scoped.reset_workspace(token)


def test_the_same_identifier_in_two_workspaces_is_two_keys(scoped):
    """The identifier resolves through the scoped store, so naming grade0 in beta
    cannot reach the key alpha stored under that name."""
    alpha, t1 = _in_workspace(scoped, "alpha")
    sealed = alpha.seal("grade0", b"alpha's boundary", b"aad")
    scoped.reset_workspace(t1)
    beta, t2 = _in_workspace(scoped, "beta")
    try:
        with pytest.raises(PermissionError):
            beta.open(sealed, b"aad")
    finally:
        scoped.reset_workspace(t2)


def test_an_unknown_identifier_is_refused_rather_than_guessed(scoped):
    from agent.kms import WorkspaceKeyring
    token = scoped.set_workspace("alpha")
    try:
        with pytest.raises(PermissionError):
            WorkspaceKeyring().seal("never-stored", b"x", b"aad")
    finally:
        scoped.reset_workspace(token)


def test_an_identifier_from_a_request_is_denied_unless_the_operator_listed_it(
        scoped, monkeypatch):
    from agent.kms import WorkspaceKeyring
    monkeypatch.setenv("GRADE0_KEY", "operator-secret")
    monkeypatch.delenv("REXGRAPH_REQUEST_KEY_REFS", raising=False)
    with pytest.raises(PermissionError):
        WorkspaceKeyring(caller_named=True).seal("GRADE0_KEY", b"x", b"aad")
    monkeypatch.setenv("REXGRAPH_REQUEST_KEY_REFS", "GRADE0_KEY")
    ring = WorkspaceKeyring(caller_named=True)
    assert ring.open(ring.seal("GRADE0_KEY", b"x", b"aad"), b"aad") == b"x"


def test_it_seals_a_real_rex_bundle_end_to_end(scoped, tmp_path):
    """The point of the contract: core writes a container with this and never sees a key."""
    import numpy as np

    from rexgraph.graph import RexGraph
    from rexgraph.io import ContainerEncryptionConfig, load_rex, save_rex

    ring, token = _in_workspace(scoped, "alpha")
    try:
        ring.configuration = ContainerEncryptionConfig(footer_key="grade0", tensor_keys={})
        rex = RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])
        out = str(tmp_path / "sealed.rex")
        save_rex(out, rex, encryption_properties=ring)

        with pytest.raises(PermissionError):
            load_rex(out)                            # no keys, no complex

        back = load_rex(out, decryption_properties=ring)
        assert (back.nV, back.nE) == (rex.nV, rex.nE)
        b1 = np.asarray(back.B1.todense() if hasattr(back.B1, "todense") else back.B1)
        assert np.allclose(b1, np.asarray(
            rex.B1.todense() if hasattr(rex.B1, "todense") else rex.B1))
        # the boundary survives as a boundary: every column still sums to zero
        assert np.allclose(b1.sum(axis=0), 0.0), b1.sum(axis=0)
    finally:
        scoped.reset_workspace(token)
