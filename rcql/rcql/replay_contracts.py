"""Verified local state replay through canonical Core mutation packages."""
from collections.abc import Mapping

from .types import Domain, Exactness, PredicateResult, RCType, ValueKind

ARGUMENTS = {
    "APPLY_DELTA": (("delta", "parent_digest", "policy"), (None, None)),
    "REPLICATE": (("package", "parent_digest", "policy"), (None, None)),
}


def mutation_policy(value):
    from rexgraph.io.mutation import MutationPolicy
    if value is None:
        return MutationPolicy()
    keys = {"require_transition_signature", "require_lineage_signature", "allowed_signers"}
    if not isinstance(value, Mapping) or set(value) - keys:
        raise TypeError("mutation policy requires only declared signature policy fields")
    flags = [value.get(key, False) for key in ("require_transition_signature", "require_lineage_signature")]
    if any(type(flag) is not bool for flag in flags):
        raise TypeError("mutation signature requirements must be booleans")
    signers = value.get("allowed_signers", ())
    if (not isinstance(signers, (list, tuple)) or
            any(not isinstance(signer, str) or not signer for signer in signers) or
            len(set(signers)) != len(signers)):
        raise TypeError("allowed_signers requires distinct nonempty text identities")
    return MutationPolicy(*flags, tuple(signers))


def validate_options(parent_digest, policy):
    if parent_digest is not None and (not isinstance(parent_digest, str) or not parent_digest):
        raise ValueError("parent_digest must be nonempty text or None for a root link")
    return mutation_policy(policy)


def refine(typed, children, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native RexGraph checkpoint")
    args = typed.args
    validate_options(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None)
    return [PredicateResult("canonical_mutation_replay", "deferred",
        "Core verifies actual endpoints, explicit parent links and signature policy at execution; "
        "the returned Rex owns its complete state and is not published to a store")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    for name, (parameters, _) in ARGUMENTS.items():
        register(OperatorSignature(name=name, source_kind=ValueKind.REX,
            inputs=(TypePattern(parameters[0], kind=ValueKind.ARTIFACT_BYTES, literal=bytes),
                    TypePattern("parent_digest", literal=(str, type(None)), optional=True),
                    TypePattern("policy", optional=True)),
            result=RCType("Rex", kind=ValueKind.REX, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL),
            implementation_key="rexgraph.io." + ("mutation.apply_mutation" if name == "APPLY_DELTA"
                                                  else "replication.apply_replication"),
            requires=frozenset({"read", "identity", "history", "security"}), memoizable=False,
            preconditions=("canonical mutation bytes, not a C1 TemporalSignal",
                           "None parent means root link; no implicit lineage or policy downgrade",
                           "signature providers run only at execution; rebind the returned Rex before querying its basis")))
