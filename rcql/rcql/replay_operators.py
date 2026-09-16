"""Local replay adapters using Core endpoint and lineage verification."""
from .execution_trace import current_artifact_services, record_method
from .replay_contracts import validate_options


def _verifiers():
    services = current_artifact_services(required=False)
    if services is None:
        return {}
    return {identity: services.provider("verifiers", identity) for identity in services.verifiers}


def apply_delta(source, delta, parent_digest=None, policy=None):
    from rexgraph.io.mutation import apply_mutation, mutation_from_bytes
    policy = validate_options(parent_digest, policy)
    package = mutation_from_bytes(delta)
    result = apply_mutation(package, previous=source, parent_digest=parent_digest,
                            policy=policy, verifiers=_verifiers())
    record_method("core-canonical-mutation-replay", commits=1, terminal_commit=package.link.digest)
    return result


def replicate(source, package, parent_digest=None, policy=None):
    from rexgraph.io.replication import apply_replication
    policy = validate_options(parent_digest, policy)
    # The caller has already materialized the checkpoint as the bound source.
    # Core checks its actual state digest against the manifest before replay.
    applied = apply_replication(package, checkpoint_loader=lambda _: source,
                                policy=policy, verifiers=_verifiers(), checkpoint_commit=parent_digest)
    record_method("core-canonical-replication", commits=len(applied.packages),
                  terminal_commit=applied.packages[-1].link.digest if applied.packages else parent_digest)
    return applied.result


def install(register):
    register("APPLY_DELTA")(apply_delta)
    register("REPLICATE")(replicate)
