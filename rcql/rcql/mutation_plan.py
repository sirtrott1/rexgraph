"""Typed mutation plans: native input DAG followed by one governed publication.

Keys and signers belong to the bound store's execution capabilities, never to
syntax. EXPLAIN plans inputs but neither prepares/signs nor publishes a package.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

from .ast import MutationQuery, Query
from .binding import SourceKindError
from .native_plan import plain
from .planning import QueryPlan, plan_query
from .types import RCType, ValueKind

_FIELDS = ("record_id", "resulting", "actor", "valid_from", "valid_to", "expected_version", "expected_hash")


def validate_arguments(values, *, static=False, file=False):
    """Check the write contract both before execution and at the publication boundary."""
    from rexgraph.graph import RexGraph, TemporalRex
    record_id, resulting, actor, valid_from, valid_to, expected, expected_hash = values
    if not isinstance(record_id, str) or not record_id:
        raise TypeError("mutation record id must be a nonempty string")
    if not isinstance(actor, str):
        raise TypeError("mutation actor must be a string")
    kinds = (ValueKind.REX, ValueKind.TEMPORAL_REX) if file else (ValueKind.REX,)
    if not isinstance(resulting, (RexGraph, TemporalRex) if file else RexGraph) and not (
        static and isinstance(resulting, RCType) and resulting.kind in kinds
    ):
        raise TypeError("mutation state must be a native RexGraph, not a history or untyped object")
    for name, value in (("valid_from", valid_from), ("valid_to", valid_to)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not isfinite(float(value))):
            raise ValueError(f"mutation {name} must be a finite real time")
    if valid_from is not None and valid_to is not None and float(valid_to) <= float(valid_from):
        raise ValueError("valid_to must follow valid_from (half-open interval)")
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, Integral) or expected < 0):
        raise ValueError("expected_version must be a nonnegative integer or None")
    if file and any(v is not None for v in (expected, valid_from, valid_to)):
        raise ValueError("standalone files have no RCDB version or validity interval; use expected_hash")
    if expected_hash is not None:
        if not file:
            raise ValueError("expected_hash is for standalone files; RCDB uses expected_version")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(
            c not in "0123456789abcdef" for c in expected_hash
        ):
            raise ValueError("expected_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class MutationPlan:
    """A fully typed input DAG and one non reusable commit terminal."""

    inputs: QueryPlan
    query: MutationQuery

    def explain(self):
        dag = self.inputs.dag().explain()
        terminal = f"n{len(dag['nodes'])}"
        commit = {
            "id": terminal, "kind": "commit", "operator": "COMMIT", "source": "s0",
            "inputs": list(dag["outputs"]), "fields": list(_FIELDS), "reusable": False,
            "requires": ["identity", "mutate"], "source_methods": ["commit_mutation"],
            "effects": ["mutate", "publish"],
            "result": {"kind": "Record", "exactness": "structural"},
            "physical": {"status": "selected", "method": "rcdb-temporal-mutation-commit",
                         "adapter": "rcdb.commit_mutation"},
            "obligations": {
                "inputs": "checked",
                "expected_version": "checked under the store handle's publication lock at execution",
                "delta_replay_and_policy": "checked by RCDB before publication",
                "signing": "bound-store capability; no signing during EXPLAIN",
            },
        }
        if self.inputs.binding.schema.kind is ValueKind.CATALOG_ENTRY_SET:
            commit["requires"] = ["identity", "mutate", "file_write"]
            commit["physical"] = {"status": "selected", "method": "recoverable-native-file-replacement",
                                  "adapter": "catalog.commit_mutation"}
            commit["obligations"] = {"inputs": "checked", "state_identity": "verified after staged core roundtrip",
                                      "publication": "parent-lock, expected_hash recheck, retained recovery copy",
                                      "lineage": "standalone file replacement is not an RCDB version"}
        dag["nodes"].append(commit)
        dag["outputs"] = [terminal]
        return plain(dag)


def plan_mutation(binding, query, *, parameters=None):
    """Reject the whole mutation before executing its first input adapter."""
    if not isinstance(query, MutationQuery):
        raise TypeError("plan_mutation expects a MutationQuery")
    file = binding.schema.kind is ValueKind.CATALOG_ENTRY_SET
    if binding.schema.kind not in (ValueKind.RCDB_STORE, ValueKind.CATALOG_ENTRY_SET):
        raise SourceKindError("RCQL mutation expects an RCDB store or a file catalog")
    for permission in ("mutate", "identity"):
        binding.source.require(permission)
    if file:
        binding.source.require("file_write")
    if not binding.schema.can("commit_mutation"):
        raise SourceKindError("mutation requires the RCDB commit_mutation contract")
    values = tuple(getattr(query, field) for field in _FIELDS)
    inputs = plan_query(binding, Query(query.source, values, bindings=query.bindings,
                                      source_alias=query.source_alias), parameters=parameters)
    validate_arguments(tuple(expr.result for expr in inputs.returns), static=True, file=file)
    return MutationPlan(inputs, query)
