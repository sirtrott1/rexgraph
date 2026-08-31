"""Relational Complex Query Language."""
#: Kept here rather than read back from installed metadata, so a source checkout reports
#: what it is. pyproject.toml has to match; a test enforces it.
__version__ = "1.1.3"

from .ast import Call, Literal, MutationQuery, Parameter, Query
from .builder import call, mutation, param, query, source
from .capabilities import BoundSource, SourcePolicy
from .parser import parse

__all__ = [
    "BoundSource", "Call", "Executor", "Literal", "MutationQuery", "Parameter",
    "Query", "Result", "SourcePolicy", "call", "mutation", "param", "parse",
    "query", "source",
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
