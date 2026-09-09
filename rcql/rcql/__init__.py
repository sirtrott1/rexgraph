"""Relational Complex Query Language."""
#: Kept here rather than read back from installed metadata, so a source checkout reports
#: what it is. pyproject.toml has to match; a test enforces it.
__version__ = "1.1.5"

from .ast import Call, Literal, MutationQuery, Parameter, Query
from .binding import (
    Binding,
    SourceKindError,
    SourceSchema,
    UnreachableOperator,
    bind,
    classify,
)
from .builder import (
    at,
    at_time,
    call,
    mutation,
    param,
    phrase,
    query,
    rcdb_as_of,
    rcdb_get,
    rcdb_valid_at,
    rcdb_version,
    source,
)
from .capabilities import BoundSource, SourcePolicy
from .inference import TypedCall, infer
from .parser import parse
from .planning import PlannedExpression, QueryPlan, plan_query
from .signatures import OperatorSignature, TypePattern, catalogued, lookup
from .types import (
    BasisRef,
    Domain,
    Effect,
    Exactness,
    RCType,
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
    "BasisRef", "Binding", "BoundSource", "Call", "Domain", "Effect", "Exactness",
    "Executor", "Literal", "MutationQuery", "OperatorSignature", "Parameter", "Query",
    "PlannedExpression", "QueryPlan", "RCType", "Result", "ShapeRef", "SourceKindError", "SourcePolicy", "SourceRef",
    "SourceSchema", "TemporalRef", "TypePattern", "TypedCall", "UndeclaredRestrictionError", "UnreachableOperator",
    "ValueKind", "Variance", "at", "at_time", "bind", "call", "catalogued", "classify", "infer",
    "lookup", "mutation", "param", "parse", "phrase", "plan_query", "query", "rcdb_as_of", "rcdb_get",
    "rcdb_valid_at", "rcdb_version", "source", "PhraseCorrespondence", "PhraseGlueResult",
    "PhraseGluingObstruction", "PhraseMapError", "PhraseSheaf", "PhraseStalk",
]


def __getattr__(name):
    """Load the executor on first use.

    Importing it eagerly would pull in the operator registry, and through it the whole
    numeric stack, for a caller that only wanted to parse or build a query.
    """
    if name in ("Executor", "Result"):
        from .executor import Executor, Result
        return {"Executor": Executor, "Result": Result}[name]
    if name in {
        "PhraseCorrespondence", "PhraseGlueResult", "PhraseGluingObstruction", "PhraseMapError", "PhraseSheaf", "PhraseStalk",
        "UndeclaredRestrictionError",
    }:
        from .phrase import (
            PhraseCorrespondence,
            PhraseGlueResult,
            PhraseGluingObstruction,
            PhraseMapError,
            PhraseSheaf,
            PhraseStalk,
            UndeclaredRestrictionError,
        )
        return {
            "PhraseCorrespondence": PhraseCorrespondence,
            "PhraseGlueResult": PhraseGlueResult,
            "PhraseGluingObstruction": PhraseGluingObstruction,
            "PhraseMapError": PhraseMapError,
            "PhraseSheaf": PhraseSheaf,
            "PhraseStalk": PhraseStalk,
            "UndeclaredRestrictionError": UndeclaredRestrictionError,
        }[name]
    raise AttributeError(name)
