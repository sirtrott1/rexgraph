"""Query local adapter observations; no global last method or value serialization."""
from contextlib import contextmanager
from contextvars import ContextVar

_BINDING = ContextVar("rcql_binding", default=None)


def current_binding():
    return _BINDING.get()


_OBSERVATIONS = ContextVar("rcql_method_observations", default=None)
_POLICY_DIGEST = ContextVar("rcql_policy_digest", default="")
_POLICY = ContextVar("rcql_policy", default=None)
_ARTIFACT_SERVICES = ContextVar("rcql_artifact_services", default=None)


def current_policy():
    from .capabilities import SourcePolicy
    return _POLICY.get() or SourcePolicy.allow("*")


def current_artifact_services(*, required=True):
    services = _ARTIFACT_SERVICES.get()
    if services is None and required:
        raise ValueError("artifact operation requires configured Executor artifacts")
    return services


def current_policy_digest():
    """Read the executor's bound policy identity, never a caller supplied grant."""
    return _POLICY_DIGEST.get()


def record_method(method, **details):
    target = _OBSERVATIONS.get()
    if target is not None:
        target.append({"method": method, **details})


@contextmanager
def capture_methods(*, policy_digest="", policy=None, artifact_services=None, binding=None):
    binding_token = _BINDING.set(binding)
    observations = []
    token = _OBSERVATIONS.set(observations)
    policy_token = _POLICY_DIGEST.set(policy_digest)
    source_token = _POLICY.set(policy)
    services_token = _ARTIFACT_SERVICES.set(artifact_services)
    try:
        yield observations
    finally:
        _BINDING.reset(binding_token)
        _OBSERVATIONS.reset(token)
        _POLICY_DIGEST.reset(policy_token)
        _POLICY.reset(source_token)
        _ARTIFACT_SERVICES.reset(services_token)


_EVIDENCE = ContextVar("rcql_evidence", default=None)


def current_evidence():
    return _EVIDENCE.get()


@contextmanager
def evidence_scope(evidence):
    token = _EVIDENCE.set(evidence)
    try:
        yield
    finally:
        _EVIDENCE.reset(token)
