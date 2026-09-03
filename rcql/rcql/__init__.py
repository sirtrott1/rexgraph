"""Relational Complex Query Language."""
#: Kept here rather than read back from installed metadata, so a source checkout reports
#: what it is. pyproject.toml has to match; a test enforces it.
__version__ = "1.1.4"

from .ast import Call, Literal, MutationQuery, Parameter, Query
from .binding import (
    Binding,
    SourceKindError,
    SourceSchema,
    UnreachableOperator,
    bind,
    classify,
)
from .builder import at, at_time, call, mutation, param, query, source
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
    "SourceSchema", "TemporalRef", "TypePattern", "TypedCall", "UnreachableOperator",
    "ValueKind", "Variance", "at", "at_time", "bind", "call", "catalogued", "classify", "infer",
    "lookup", "mutation", "param", "parse", "plan_query", "query", "source",
]


def __getattr__(name):
    """Load the executor on first use.

    Importing it eagerly would pull in the operator registry, and through it the whole
    numeric stack, for a caller that only wanted to parse or build a query.
    """
    if name in ("Executor", "Result"):
        from .executor import Executor, Result
        return {"Executor": Executor, "Result": Result}[name]
    raise AttributeError(name)
