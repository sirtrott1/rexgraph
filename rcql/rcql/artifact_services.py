"""Explicit Core providers for in memory artifact operations.

Providers are executor configuration, never query parameters or serialized
plan values. Their methods run only at execution, not during EXPLAIN.
"""
from types import MappingProxyType


class ArtifactServices:
    __slots__ = ("keys", "identity_keys", "signers", "verifiers")

    def __init__(self, *, keys=None, identity_keys=None, signers=None, verifiers=None):
        self.keys = keys
        self.identity_keys = identity_keys
        self.signers = MappingProxyType(dict(signers or {}))
        self.verifiers = MappingProxyType(dict(verifiers or {}))
        for providers in (self.signers, self.verifiers):
            if any(not isinstance(name, str) or not name for name in providers):
                raise ValueError("artifact provider identities must be nonempty strings")

    def __repr__(self):
        return "<ArtifactServices>"

    def provider(self, role, identity=None):
        if role in {"keys", "identity_keys"}:
            value = getattr(self, role)
            method = "key"
        else:
            value = getattr(self, role).get(identity)
            method = "sign" if role == "signers" else "verify"
            if value is not None and value.signer_id != identity:
                raise ValueError("configured artifact provider identity differs from its registration")
        if value is None or not callable(getattr(value, method, None)):
            raise ValueError(f"no configured artifact provider for {role}")
        return value
