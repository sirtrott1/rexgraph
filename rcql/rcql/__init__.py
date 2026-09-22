"""Relational Complex Query Language."""
#: Kept here rather than read back from installed metadata, so a source checkout reports
#: what it is. pyproject.toml has to match; a test enforces it.
__version__ = "1.2.2"

from .ast import (
    Alias,
    Comparison,
    MatchBinding,
    StructuralEdit,
    Call,
    LetBinding,
    ListExpr,
    Literal,
    Member,
    MutationQuery,
    Parameter,
    Query,
    Reference,
)
from .binding import (
    Binding,
    SourceKindError,
    SourceSchema,
    UnreachableOperator,
    bind,
    classify,
)
from .builder import (
    alias,
    at,
    at_time,
    call,
    let,
    member,
    mutation,
    param,
    phrase,
    query,
    rcdb,
    rcdb_as_of,
    rcdb_get,
    rcdb_valid_at,
    rcdb_version,
    ref,
    source,
    source_call,
)
from .capabilities import BoundSource, SourcePolicy
from .inference import TypedCall, infer
from .inventory import operator_inventory
from .mutation_plan import MutationPlan, plan_mutation
from .native_plan import NativeNode, NativePlan
from .parser import parse
from .planning import PlannedExpression, QueryPlan, plan_query
from .signatures import OperatorSignature, TypePattern, catalogued, lookup
from .types import (
    AccessionDescriptor,
    BasisRef,
    CoordinateDescriptor,
    CrossMetricDescriptor,
    Domain,
    Effect,
    Exactness,
    FamilyMetricDescriptor,
    MetricDescriptor,
    OperatorDescriptor,
    PredicateResult,
    RCType,
    RealizationDescriptor,
    ShapeRef,
    SourceRef,
    TemporalRef,
    ValueKind,
    Variance,
)

# The type, signature, binding and inference surfaces are eager because none of them
# touches numpy or the operator registry: deciding what a call WOULD produce must not
# require the machinery that would produce it. The executor stays lazy for that reason.
__all__ = [
    "ArtifactServices",
    "Comparison", "MatchBinding", "StructuralEdit",
    "Alias", "ListExpr", "Member", "alias", "member", "source_call",
    "MutationPlan", "plan_mutation",
    "rcdb",
    "LetBinding", "Reference", "let", "ref",
    "AccessionDescriptor",
    "CoordinateDescriptor", "CrossMetricDescriptor",
    "FamilyMetricDescriptor", "RealizationDescriptor",
    "BasisRef", "Binding", "BoundSource", "Call", "Domain", "Effect", "Exactness",
    "Executor", "Literal", "MetricDescriptor", "MutationQuery", "NativeNode", "NativePlan", "OperatorDescriptor", "OperatorSignature", "Parameter", "PredicateResult", "Query",
    "PlannedExpression", "QueryPlan", "RCType", "Result", "ShapeRef", "SourceKindError", "SourcePolicy", "SourceRef",
    "SourceSchema", "TemporalRef", "TypePattern", "TypedCall", "UndeclaredRestrictionError", "UnreachableOperator",
    "ValueKind", "Variance", "at", "at_time", "bind", "call", "catalogued", "classify", "infer",
    "lookup", "mutation", "operator_inventory", "param", "parse", "phrase", "plan_query", "query", "rcdb_as_of", "rcdb_get",
    "rcdb_valid_at", "rcdb_version", "source", "PhraseCorrespondence", "PhraseGlueResult",
    "PhraseGluingObstruction", "PhraseMapError", "PhraseSectionCheck", "PhraseSheaf", "PhraseStalk",
]


def __getattr__(name):
    """Load the executor on first use.

    Importing it eagerly would pull in the operator registry, and through it the whole
    numeric stack, for a caller that only wanted to parse or build a query.
    """
    if name == "ArtifactServices":
        from .artifact_services import ArtifactServices
        return ArtifactServices
    if name in ("Executor", "Result"):
        from .executor import Executor, Result
        return {"Executor": Executor, "Result": Result}[name]
    if name in {
        "PhraseCorrespondence", "PhraseGlueResult", "PhraseGluingObstruction", "PhraseMapError", "PhraseSheaf", "PhraseStalk",
        "UndeclaredRestrictionError",
        "PhraseSectionCheck",
    }:
        from .phrase import (
            PhraseCorrespondence,
            PhraseGlueResult,
            PhraseGluingObstruction,
            PhraseMapError,
            PhraseSectionCheck,
            PhraseSheaf,
            PhraseStalk,
            UndeclaredRestrictionError,
        )
        return {
            "PhraseCorrespondence": PhraseCorrespondence,
            "PhraseGlueResult": PhraseGlueResult,
            "PhraseGluingObstruction": PhraseGluingObstruction,
            "PhraseMapError": PhraseMapError,
            "PhraseSectionCheck": PhraseSectionCheck,
            "PhraseSheaf": PhraseSheaf,
            "PhraseStalk": PhraseStalk,
            "UndeclaredRestrictionError": UndeclaredRestrictionError,
        }[name]
    if name in _PROGRAM_EXPORTS:
        from importlib import import_module
        module, symbol = _PROGRAM_EXPORTS[name]
        return getattr(import_module("."+module, __name__), symbol)
    raise AttributeError(name)

_PROGRAM_EXPORTS = {
    "Program": ("program", "Program"), "ProgramStep": ("program", "ProgramStep"),
    "ProgramInput": ("program", "ProgramInput"), "OutputRef": ("program", "OutputRef"),
    "ProgramResult": ("program", "ProgramResult"), "PlanTopology": ("plan_topology", "PlanTopology"),
    "PlanScheduler": ("scheduling", "PlanScheduler"), "QueryCancelledError": ("scheduling", "QueryCancelledError"),
    "SnapshotContext": ("source_context", "SnapshotContext"), "SourceSelection": ("source_context", "SourceSelection"),
    "QueryCache": ("query_cache", "QueryCache"), "ReadoutEquivalence": ("readout_equivalence", "ReadoutEquivalence"),
}
__all__ += list(_PROGRAM_EXPORTS)

_RELATION_EXPORTS = {
    "ProgramTransformation": ("program_transformation", "ProgramTransformation"),
    "ProgramAssembly": ("program_family", "ProgramAssembly"),
    "ProgramFamily": ("program_family", "ProgramFamily"),
    "ProgramEvolution": ("program_evolution", "ProgramEvolution"),
    "NameRelation": ("name_relation", "NameRelation"),
    "RecursiveDefinition": ("recursive_program", "RecursiveDefinition"),
    "RecursiveProgram": ("recursive_program", "RecursiveProgram"),
    "RecursionResult": ("recursive_program", "RecursionResult"),
    "recur": ("recursive_program", "recur"),
    "RecursionLimits": ("relation_runtime", "RecursionLimits"),
    "RecursionLimitError": ("relation_runtime", "RecursionLimitError"),
    "RecursiveCycleError": ("relation_runtime", "RecursiveCycleError"),
    "RelationTopology": ("relation_topology", "RelationTopology"),
    "OperationRelationCell": ("relation_topology", "OperationRelationCell"),
}
_PROGRAM_EXPORTS.update(_RELATION_EXPORTS)
__all__ += list(_RELATION_EXPORTS)
