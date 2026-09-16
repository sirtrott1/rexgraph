"""Relational Complex Query Language."""
#: Kept here rather than read back from installed metadata, so a source checkout reports
#: what it is. pyproject.toml has to match; a test enforces it.
__version__ = "1.1.6"

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
    raise AttributeError(name)
