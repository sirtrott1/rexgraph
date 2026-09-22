"""Static operator signatures: what an operation is, before it is a Python callable.

The binder resolves a name here, checks arity, source, capabilities and input kinds,
then inference produces a result type before an expression adapter runs. Execution
uses these same capability and query local memoization declarations; there is no
second permission table or cache allow list to drift from EXPLAIN.

This catalogue describes current storage, catalog, metadata and native mathematics
adapters. Historical planning names live separately in ``inventory.py``. A declared
implementation key identifies an adapter contract, not a selected physical kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction

from .types import Domain, Effect, Exactness, RCType, ShapeRef, TemporalRef, ValueKind, Variance

# Capability names. These are separate on purpose: reading a projected summary is not the
# same right as resolving an identity, and neither implies mutating.
# Historical spelling pairs (files/file_read and security/admin) are retained as
# independent requirements: this reconciles the previous planner/executor union,
# without silently widening policies or inventing permission aliases.
IDENTITY = "identity"
HISTORY = "history"
MUTATE = "mutate"
FILES = "files"
SEARCH = "search"
SECURITY = "security"


@dataclass(frozen=True)
class TypePattern:
    """What one argument must be for a call to type check.

    ``literal`` accepts a plain Python value of the given type, which is how a record id or
    a limit arrives. ``kind`` constrains a typed RCQL value. A pattern with neither
    accepts anything, which is only correct where the operator genuinely does not care.
    """

    name: str
    kind: ValueKind | tuple[ValueKind, ...] | None = None
    literal: type | tuple[type, ...] | None = None
    grade: int | tuple[int, ...] | None = None
    variance: Variance | tuple[Variance, ...] | None = None
    domain: Domain | tuple[Domain, ...] | None = None
    exactness: Exactness | tuple[Exactness, ...] | None = None
    source_bound: bool = False
    basis_bound: bool = False
    optional: bool = False

    def accepts(self, value: object, *, source=None) -> bool:
        if isinstance(value, RCType):
            if self.literal is not None and self.kind is None:
                return False
            if self.kind is not None:
                kinds = self.kind if isinstance(self.kind, tuple) else (self.kind,)
                if value.kind not in kinds:
                    return False
            if self.grade is not None:
                grades = self.grade if isinstance(self.grade, tuple) else (self.grade,)
                if value.grade not in grades:
                    return False
            if self.variance is not None:
                variances = self.variance if isinstance(self.variance, tuple) else (self.variance,)
                if value.variance not in variances:
                    return False
            if self.domain is not None:
                domains = self.domain if isinstance(self.domain, tuple) else (self.domain,)
                if value.domain not in domains:
                    return False
            if self.exactness is not None:
                contracts = self.exactness if isinstance(self.exactness, tuple) else (self.exactness,)
                if value.exactness not in contracts:
                    return False
            if self.source_bound and (source is None or value.source != source):
                return False
            if self.basis_bound:
                if value.basis is None:
                    return False
                if source is not None and value.basis.source_id != source.name:
                    return False
                if value.grade is not None and value.basis.grade != value.grade:
                    return False
            return True
        if self.literal is None:
            return self.kind is None
        literals = self.literal if isinstance(self.literal, tuple) else (self.literal,)
        allows_bool = bool in literals
        return isinstance(value, literals) and (allows_bool or not isinstance(value, bool))

    def describe(self) -> str:
        if self.kind is not None:
            kinds = self.kind if isinstance(self.kind, tuple) else (self.kind,)
            body = "|".join(k.value for k in kinds)
        elif self.literal is not None:
            lits = self.literal if isinstance(self.literal, tuple) else (self.literal,)
            body = "|".join(t.__name__ for t in lits)
        else:
            body = "any"
        details = []
        if self.grade is not None:
            details.append(f"grade={self.grade}")
        if self.variance is not None:
            details.append(f"variance={self.variance}")
        if self.domain is not None:
            details.append(f"domain={self.domain}")
        if self.exactness is not None:
            details.append(f"exactness={self.exactness}")
        if self.source_bound:
            details.append("source-bound")
        if self.basis_bound:
            details.append("basis-bound")
        if details:
            body += " (" + ", ".join(details) + ")"
        return f"{self.name}: {body}{' = optional' if self.optional else ''}"


@dataclass(frozen=True)
class OperatorSignature:
    """One operation, declared as data.

    ``result`` is either a fixed type or a rule over the argument types, because some
    results depend on what was passed. ``unreachable`` marks an operation that is
    registered but cannot execute against any available source; the binder refuses it
    rather than letting it fail inside the adapter.
    """

    name: str
    source_kind: ValueKind | tuple[ValueKind, ...]
    inputs: tuple[TypePattern, ...]
    result: RCType | Callable[[tuple[object, ...]], RCType]
    implementation_key: str
    requires: frozenset[str] = field(default_factory=lambda: frozenset({"read"}))
    effects: frozenset[Effect] = field(default_factory=lambda: frozenset({Effect.READ}))
    preconditions: tuple[str, ...] = ()
    unreachable: str = ""
    # An explicit opt in for query local reuse, not a cross query cache promise.
    # External/cache-observable reads remain false even when their effect is READ.
    memoizable: bool = False
    source_methods: frozenset[str] = frozenset()

    @property
    def arity(self) -> tuple[int, int]:
        """Minimum and maximum argument counts, excluding the bound source."""
        required = sum(1 for pattern in self.inputs if not pattern.optional)
        return required, len(self.inputs)

    def check_arity(self, count: int) -> None:
        low, high = self.arity
        if not low <= count <= high:
            want = f"{low}" if low == high else f"{low} to {high}"
            raise TypeError(f"{self.name} takes {want} arguments, got {count}")

    def check_inputs(self, args: tuple[object, ...], *, source=None) -> None:
        self.check_arity(len(args))
        for pattern, value in zip(self.inputs, args, strict=False):
            if not pattern.accepts(value, source=source):
                raise TypeError(
                    f"{self.name} argument {pattern.describe()} rejected {value!r}"
                )

    def result_type(self, args: tuple[object, ...] = ()) -> RCType:
        return self.result(args) if callable(self.result) else self.result


_CATALOGUE: dict[str, OperatorSignature] = {}


def register(signature: OperatorSignature) -> OperatorSignature:
    from .names import canonical_name, insert_unique
    signature = replace(signature, name=canonical_name(signature.name))
    return insert_unique(_CATALOGUE, signature.name, signature)


def lookup(name: str) -> OperatorSignature:
    from .names import canonical_name
    name = canonical_name(name)
    try:
        return _CATALOGUE[name]
    except KeyError as exc:
        from .names import RETIRED
        if name in RETIRED:
            raise KeyError(f"RCQL operator {name!r} was renamed; use {RETIRED[name]}") from exc
        raise KeyError(f"no RCQL signature for operator {name!r}") from exc


def catalogued() -> frozenset[str]:
    return frozenset(_CATALOGUE)


def _t(name: str, kind: ValueKind, **kw) -> RCType:
    return RCType(name, kind=kind, **kw)


_STR = TypePattern("name", literal=str)
_LIMIT = TypePattern("limit", literal=int, optional=True)
_OFFSET = TypePattern("offset", literal=int, optional=True)

def _aggregate_result(name, args):
    from .aggregates import count_type, result_type
    return count_type(args[0]) if name == "COUNT" else result_type(args[0], mean=name == "MEAN")


for _aggregate in ("COUNT", "SUM", "MEAN"):
    register(OperatorSignature(
        name=_aggregate, source_kind=ValueKind.UNKNOWN, inputs=(TypePattern("values", source_bound=True),),
        result=lambda args, name=_aggregate: _aggregate_result(name, args),
        implementation_key=f"rcql.aggregate.{_aggregate.lower()}", memoizable=True,
        preconditions=("finite explicit sequence; scalar SUM and MEAN retain exact integers and rationals",),
    ))

register(OperatorSignature(
    name="SHOW_OPERATORS", source_kind=ValueKind.UNKNOWN, inputs=(_LIMIT, _OFFSET),
    result=_t("OperatorSignatureSet", ValueKind.OPERATOR_SIGNATURE_SET,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="rcql.operator_inventory",
    preconditions=(
        "limit is in [0, 1000] and offset is nonnegative",
        "static global inventory; source eligibility and authorization are checked on actual calls",
        "implementation keys identify adapters, not a selected physical kernel or pushdown plan",
    ),
))

# file catalog
#
# The catalog indexes loadable kinds only, so an entry name that exists on disk is not
# necessarily an entry. That is a precondition rather than a runtime KeyError, and is
# recorded here so Phase 2 can check it during binding.

_ENTRY_EXISTS = ("name is an indexed catalog entry; the catalog holds loadable kinds only",)

register(OperatorSignature(
    name="FILES", source_kind=ValueKind.CATALOG_ENTRY_SET, inputs=(_LIMIT, _OFFSET),
    result=_t("CatalogEntrySet", ValueKind.CATALOG_ENTRY_SET,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.list", requires=frozenset({FILES, "file_read"}),
))
register(OperatorSignature(
    name="SEARCH", source_kind=ValueKind.CATALOG_ENTRY_SET,
    inputs=(TypePattern("text", literal=str), _LIMIT),
    result=_t("CatalogEntrySet", ValueKind.CATALOG_ENTRY_SET,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.search", requires=frozenset({FILES, SEARCH}),
))
register(OperatorSignature(
    name="FILE_INFO", source_kind=ValueKind.CATALOG_ENTRY_SET, inputs=(_STR,),
    result=_t("CatalogEntry", ValueKind.CATALOG_ENTRY,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.info", requires=frozenset({FILES, "file_read"}),
    preconditions=_ENTRY_EXISTS + (
        "the sha256 field is populated only once a hash has been computed; a caller that "
        "needs the digest must request FILE_HASH rather than read it opportunistically",
    ),
))
register(OperatorSignature(
    name="FILE_HASH", source_kind=ValueKind.CATALOG_ENTRY_SET, inputs=(_STR,),
    result=_t("Digest", ValueKind.DIGEST, domain=Domain.BYTES,
              exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.hash", requires=frozenset({FILES, "file_read"}),
    effects=frozenset({Effect.READ, Effect.FILESYSTEM}),
    preconditions=_ENTRY_EXISTS,
))
register(OperatorSignature(
    name="HASH_FILES", source_kind=ValueKind.CATALOG_ENTRY_SET, inputs=(),
    result=_t("Integer", ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
              exactness=Exactness.INTEGER),
    implementation_key="catalog.hash_all", requires=frozenset({FILES, "file_read"}),
    effects=frozenset({Effect.READ, Effect.FILESYSTEM}),
))
register(OperatorSignature(
    name="TENSORS", source_kind=ValueKind.CATALOG_ENTRY_SET, inputs=(_STR, _LIMIT),
    result=_t("TensorManifest", ValueKind.TENSOR_MANIFEST,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.tensors", requires=frozenset({FILES, "file_read"}),
    preconditions=_ENTRY_EXISTS,
))
register(replace(lookup("TENSORS"), name="TENSOR_MANIFEST"))
register(OperatorSignature(
    name="SEARCH_TENSORS", source_kind=ValueKind.CATALOG_ENTRY_SET,
    inputs=(_STR, TypePattern("text", literal=str), _LIMIT),
    result=_t("TensorManifest", ValueKind.TENSOR_MANIFEST,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.search_tensors", requires=frozenset({FILES, SEARCH}),
    preconditions=_ENTRY_EXISTS,
))

# complex digest
#
# STATE_HASH reads a Rex, not a catalog, despite sitting among the catalog operators in
# the registry. Declaring its source kind is what stops it being called on the catalog it
# is filed beside, which currently fails inside the digest rather than at the boundary.

register(OperatorSignature(
    name="STATE_HASH", source_kind=(ValueKind.REX, ValueKind.TEMPORAL_REX), inputs=(),
    result=_t("Digest", ValueKind.DIGEST, domain=Domain.BYTES,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rex.object_digest",
))

# RCDB readings

register(OperatorSignature(
    name="RCDB_LIST", source_kind=ValueKind.RCDB_STORE, inputs=(_LIMIT, _OFFSET),
    source_methods=frozenset({"list"}),
    result=_t("RecordSet", ValueKind.RECORD_SET, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.list", requires=frozenset({"records"}),
    preconditions=("returns projected summaries; it does not decode a stored complex",),
))
register(OperatorSignature(
    name="RCDB_SEARCH", source_kind=ValueKind.RCDB_STORE,
    source_methods=frozenset({"query", "list"}),
    inputs=(TypePattern("text", literal=str), _LIMIT),
    result=_t("RecordSet", ValueKind.RECORD_SET, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.search", requires=frozenset({SEARCH, "records"}),
))
register(OperatorSignature(
    name="RCDB_GET", source_kind=ValueKind.RCDB_STORE, inputs=(_STR,),
    source_methods=frozenset({"read_record"}),
    result=_t("Rex", ValueKind.REX, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.get", requires=frozenset({IDENTITY}),
    preconditions=("decodes a stored complex, so it resolves an identity",),
))
register(OperatorSignature(
    name="RCDB_HISTORY", source_kind=ValueKind.RCDB_STORE, inputs=(_STR,),
    source_methods=frozenset({"history"}),
    result=_t("History", ValueKind.HISTORY, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.history", requires=frozenset({HISTORY, IDENTITY}),
))
register(OperatorSignature(
    name="RCDB_STATS", source_kind=ValueKind.RCDB_STORE, inputs=(),
    source_methods=frozenset({"stats"}),
    result=_t("StoreStats", ValueKind.STORE_STATS, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.stats",
))
register(OperatorSignature(
    name="RCDB_HASH", source_kind=ValueKind.RCDB_STORE, inputs=(_STR,),
    source_methods=frozenset({"read_record"}),
    result=_t("Digest", ValueKind.DIGEST, domain=Domain.BYTES,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.record_digest", requires=frozenset({IDENTITY}),
))
register(OperatorSignature(
    name="RCDB_COMMITS", source_kind=ValueKind.RCDB_STORE, inputs=(_STR, _LIMIT),
    source_methods=frozenset({"commit_history"}),
    result=_t("CommitLink", ValueKind.COMMIT_LINK, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.commit_history", requires=frozenset({HISTORY, IDENTITY}),
    preconditions=("a plain put contributes no commit link; only a governed transition does",),
))
register(OperatorSignature(
    name="RCDB_VERIFY", source_kind=ValueKind.RCDB_STORE, inputs=(_STR,),
    source_methods=frozenset({"verify_commits"}),
    result=_t("Boolean", ValueKind.BOOLEAN, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.verify_commits", requires=frozenset({HISTORY, IDENTITY}),
))
register(OperatorSignature(
    name="RCDB_SECURITY", source_kind=ValueKind.RCDB_STORE, inputs=(),
    source_methods=frozenset({"security_status"}),
    result=_t("SecurityStatus", ValueKind.SECURITY_STATUS, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.security_status", requires=frozenset({SECURITY, "admin"}),
    preconditions=("bounded configuration only; never key material or backend paths",),
))

# primary cells and fields
#
# These signatures describe the native relational complex carriers.  They deliberately
# name C1 relations and their derived C0 boundaries before any projected graph reading.
# ``TypePattern`` performs only static checks here: runtime still validates that a literal
# grade/index occurs in the particular bound source.

_GRADE = TypePattern("grade", literal=int)
_INDEX = TypePattern("index", literal=int)
_INDICES = TypePattern("indices", literal=(tuple, list), optional=True)
_CELL = TypePattern(
    "cell", kind=ValueKind.CELL, source_bound=True, basis_bound=True,
)
_CELLS = TypePattern(
    "cells", kind=ValueKind.CELL_SET, source_bound=True, basis_bound=True,
)
_CELL_OR_SET = TypePattern(
    "cell", kind=(ValueKind.CELL, ValueKind.CELL_SET),
    source_bound=True, basis_bound=True,
)
_C1_CELL = TypePattern(
    "C1 relation", kind=ValueKind.CELL, grade=1,
    source_bound=True, basis_bound=True,
)
_C1_COMPOSITE = TypePattern(
    "C1 relation or composite binary", kind=(ValueKind.CELL, ValueKind.COMPOSITE_BINARY),
    grade=1, source_bound=True, basis_bound=True,
)
_BOUND_C1_COCHAIN = TypePattern(
    "C1 metric cochain", kind=(ValueKind.COCHAIN, ValueKind.CELL_COBOUNDARY, ValueKind.FIELD),
    grade=1,
    variance=Variance.COCHAIN, source_bound=True, basis_bound=True,
)
_TEMPORAL_DELTA = TypePattern(
    "C1 temporal delta", kind=ValueKind.DELTA, grade=1,
    source_bound=True, basis_bound=True,
)
_CHANNEL = TypePattern("channel", literal=str, optional=True)
_RELATION_KEY = TypePattern("relation support key", literal=(tuple, list))
_REX_OR_TEMPORAL = (ValueKind.REX, ValueKind.TEMPORAL_REX)


def _exact_integer(name: str = "Integer") -> RCType:
    return _t(name, ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
              exactness=Exactness.INTEGER)


def _cell_result(args: tuple[object, ...], *, plural: bool = False) -> RCType:
    grade = int(args[0])
    return _t("CellSet" if plural else "Cell", ValueKind.CELL_SET if plural else ValueKind.CELL,
              grade=grade, variance=Variance.CELL, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL)


def _boundary_result(args: tuple[object, ...]) -> RCType:
    first = args[0]
    if isinstance(first, RCType):
        if len(args) != 1:
            raise TypeError("BOUNDARY accepts either a cell value or grade plus Chain")
        if first.kind is ValueKind.CELL_SET:
            grade = None if first.grade is None else first.grade - 1
            if first.grade == 1:
                return _t("Chain", ValueKind.CHAIN, grade=grade, variance=Variance.CHAIN,
                          domain=Domain.RATIONAL, exactness=Exactness.RATIONAL)
            return _t("Chain", ValueKind.CHAIN, grade=grade, variance=Variance.CHAIN,
                      domain=first.domain, exactness=first.exactness)
        grade = None if first.grade is None or first.grade == 0 else first.grade - 1
        return _t("CellBoundary", ValueKind.CELL_BOUNDARY, grade=grade, variance=Variance.CHAIN,
                  domain=Domain.RATIONAL if first.grade == 1 else Domain.METADATA,
                  exactness=Exactness.RATIONAL if first.grade == 1 else Exactness.STRUCTURAL)

    grade = int(first)
    if grade < 1:
        raise ValueError("BOUNDARY grade must be at least 1")
    if len(args) == 1:
        return _t("BoundaryOperator", ValueKind.OPERATOR, grade=grade,
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    value = args[1]
    assert isinstance(value, RCType)
    if value.grade != grade:
        raise TypeError("BOUNDARY requires a Chain at the requested grade")
    exactness = value.exactness
    domain = value.domain
    if exactness in {Exactness.INTEGER, Exactness.RATIONAL}:
        exactness, domain = Exactness.RATIONAL, Domain.RATIONAL
    return _t("Chain", ValueKind.CHAIN, grade=grade - 1, variance=Variance.CHAIN,
              domain=domain, exactness=exactness)


def _coboundary_result(args: tuple[object, ...]) -> RCType:
    first = args[0]
    if isinstance(first, RCType):
        if len(args) != 1:
            raise TypeError("COBOUNDARY accepts either a cell value or grade plus Cochain")
        if first.kind is ValueKind.CELL_SET:
            grade = None if first.grade is None else first.grade + 1
            if first.grade == 0:
                return _t("Cochain", ValueKind.COCHAIN, grade=grade,
                          variance=Variance.COCHAIN, domain=Domain.RATIONAL,
                          exactness=Exactness.RATIONAL)
            return _t("Cochain", ValueKind.COCHAIN, grade=grade,
                      variance=Variance.COCHAIN, domain=first.domain,
                      exactness=first.exactness)
        grade = None if first.grade is None else first.grade + 1
        return _t("CellCoboundary", ValueKind.CELL_COBOUNDARY, grade=grade,
                  variance=Variance.COCHAIN,
                  domain=Domain.RATIONAL if first.grade == 0 else Domain.METADATA,
                  exactness=Exactness.RATIONAL if first.grade == 0 else Exactness.STRUCTURAL)

    grade = int(first)
    if grade < 0:
        raise ValueError("COBOUNDARY grade must be nonnegative")
    if len(args) == 1:
        return _t("CoboundaryOperator", ValueKind.OPERATOR, grade=grade,
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    value = args[1]
    assert isinstance(value, RCType)
    if value.grade != grade:
        raise TypeError("COBOUNDARY requires a Cochain at the requested grade")
    exactness = value.exactness
    domain = value.domain
    if exactness in {Exactness.INTEGER, Exactness.RATIONAL}:
        exactness, domain = Exactness.RATIONAL, Domain.RATIONAL
    return _t("Cochain", ValueKind.COCHAIN, grade=grade + 1,
              variance=Variance.COCHAIN, domain=domain, exactness=exactness)


def _corelations_result(args: tuple[object, ...]) -> RCType:
    value = args[0]
    assert isinstance(value, RCType)
    return _t("CellSet", ValueKind.CELL_SET,
              grade=None if value.grade is None else value.grade + 1,
              variance=Variance.CELL, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL)


def _indicator_result(args: tuple[object, ...]) -> RCType:
    value = args[0]
    assert isinstance(value, RCType)
    return _t("Cochain", ValueKind.COCHAIN, grade=value.grade, variance=Variance.COCHAIN,
              domain=Domain.INTEGER, exactness=Exactness.INTEGER)


def _channel(args: tuple[object, ...], *, position: int, default: str) -> str:
    return default if len(args) <= position else str(args[position]).lower()


def _temporal_delta_result(args: tuple[object, ...]) -> RCType:
    step = int(args[0])
    return _t("Delta", ValueKind.DELTA, grade=1, variance=Variance.NEUTRAL,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
              temporal=TemporalRef(version=step))


def _signal_source_result(args: tuple[object, ...]) -> RCType:
    channel = _channel(args, position=1, default="structural")
    approximate = channel == "amplitude"
    return _t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.REAL if approximate else Domain.RATIONAL,
              exactness=Exactness.APPROXIMATE if approximate else Exactness.RATIONAL)


def _relation_signal_result(args: tuple[object, ...]) -> RCType:
    channel = _channel(args, position=1, default="amplitude")
    exact_binary = channel in {"existence", "orientation", "signing"}
    return _t("Cochain", ValueKind.COCHAIN, grade=1, variance=Variance.COCHAIN,
              domain=Domain.INTEGER if exact_binary else Domain.REAL,
              exactness=Exactness.INTEGER if exact_binary else Exactness.APPROXIMATE)


def _signal_flow_result(args: tuple[object, ...]) -> RCType:
    channel = _channel(args, position=1, default="structural")
    approximate = channel == "amplitude"
    return _t("TemporalSignalFlow", ValueKind.SIGNAL_FLOW,
              domain=Domain.REAL if approximate else Domain.RATIONAL,
              exactness=Exactness.APPROXIMATE if approximate else Exactness.RATIONAL)


def _metric_curvature_result(args: tuple[object, ...]) -> RCType:
    metric = args[0]
    assert isinstance(metric, RCType)
    exact = metric.exactness in {Exactness.INTEGER, Exactness.RATIONAL} and metric.domain in {
        Domain.INTEGER, Domain.RATIONAL,
    }
    return _t("MetricCurvature", ValueKind.METRIC_CURVATURE,
              domain=Domain.RATIONAL if exact else Domain.REAL,
              exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE)


register(OperatorSignature(
    name="CELL", memoizable=True, source_kind=ValueKind.REX, inputs=(_GRADE, _INDEX),
    result=_cell_result, implementation_key="rex.cell",
    preconditions=("grade and ordered-basis index occur in the bound relational complex",),
))
register(OperatorSignature(
    name="CELLS", memoizable=True, source_kind=ValueKind.REX, inputs=(_GRADE, _INDICES),
    result=lambda args: _cell_result(args, plural=True), implementation_key="rex.cells",
    preconditions=("grade and every selected ordered-basis index occur in the bound relational complex",),
))
register(OperatorSignature(
    name="INDICATOR", memoizable=True, source_kind=ValueKind.REX, inputs=(_CELL_OR_SET,),
    result=_indicator_result, implementation_key="rex.indicator",
    preconditions=(
        "keeps the selected primary cells distinct from their explicit 0/1 cochain",
        "materializing the full coefficient basis is proportional to its grade population",
    ),
))
register(OperatorSignature(
    name="BOUNDARY", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("cell or grade", kind=(ValueKind.CELL, ValueKind.CELL_SET), literal=int,
                        source_bound=True, basis_bound=True),
            TypePattern("chain", kind=ValueKind.CHAIN, variance=Variance.CHAIN,
                        source_bound=True, basis_bound=True, optional=True)),
    result=_boundary_result, implementation_key="rex.boundary",
    preconditions=("a direct C1 boundary retains declared head/share coefficients",),
))
register(OperatorSignature(
    name="COBOUNDARY", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("cell or grade", kind=(ValueKind.CELL, ValueKind.CELL_SET), literal=int,
                        source_bound=True, basis_bound=True),
            TypePattern("cochain", kind=ValueKind.COCHAIN, variance=Variance.COCHAIN,
                        source_bound=True, basis_bound=True, optional=True)),
    result=_coboundary_result, implementation_key="rex.coboundary",
    preconditions=("a direct C0 coboundary retains declared relation-share coefficients",),
))
register(OperatorSignature(
    name="COMPOSITE", memoizable=True, source_kind=ValueKind.REX, inputs=(_C1_CELL,),
    result=_t("CompositeBinary", ValueKind.COMPOSITE_BINARY, grade=1, variance=Variance.CELL,
              domain=Domain.RATIONAL, exactness=Exactness.RATIONAL),
    implementation_key="rex.composite_binary",
    preconditions=("repeated C1 incidence refuses because vertex-basis binary masks would collapse occurrences",
                   "existence, head and share are read from the column, so a declared head or share is returned as declared",),
))
register(OperatorSignature(
    name="EXISTENCE", memoizable=True, source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.INTEGER, exactness=Exactness.INTEGER),
    implementation_key="rex.composite_binary.existence",
))
register(OperatorSignature(
    name="HEAD", memoizable=True, source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.INTEGER, exactness=Exactness.INTEGER),
    implementation_key="rex.composite_binary.head",
))
register(OperatorSignature(
    name="SHARE", memoizable=True, source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.RATIONAL, exactness=Exactness.RATIONAL),
    implementation_key="rex.composite_binary.share",
))
register(OperatorSignature(
    name="SHARE_SUPPORT", memoizable=True, source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.INTEGER, exactness=Exactness.INTEGER),
    implementation_key="rex.composite_binary.share_support",
))
register(OperatorSignature(
    name="ARITY", memoizable=True, source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_exact_integer(), implementation_key="rex.composite_binary.arity",
))
register(OperatorSignature(
    name="CORELATIONS", memoizable=True, source_kind=ValueKind.REX, inputs=(_CELL_OR_SET,),
    result=_corelations_result, implementation_key="rex.corelations",
))
register(OperatorSignature(
    name="STAR", memoizable=True, source_kind=ValueKind.REX, inputs=(_CELL_OR_SET,),
    result=_t("GradedCellPattern", ValueKind.CELL_PATTERN, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL), implementation_key="rex.star",
))
register(OperatorSignature(
    name="ENCLOSURE", memoizable=True, source_kind=ValueKind.REX, inputs=(_CELL_OR_SET,),
    result=_t("GradedCellPattern", ValueKind.CELL_PATTERN, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL), implementation_key="rex.enclosure",
))

# Temporal relations are C1-primary values.  The delta carrier separates its structural
# axes from the optional measured amplitude, so amplitude is conservatively numerical at
# planning time even when an individual unweighted dataset happens to yield exact ones.
register(OperatorSignature(
    name="TEMPORAL_DELTA", memoizable=True, source_kind=ValueKind.TEMPORAL_REX,
    inputs=(TypePattern("step", literal=int),),
    result=_temporal_delta_result, implementation_key="rex.temporal_delta",
    preconditions=(
        "step lies in the source timeline transition range",
        "parallel equal-support C1 relations require stable relation identities and refuse otherwise",
    ),
))
register(OperatorSignature(
    name="SIGNAL_AT", memoizable=True, source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _RELATION_KEY),
    result=_t("TemporalSignalEvent", ValueKind.TEMPORAL_EVENT, grade=1,
              variance=Variance.CELL, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rex.temporal_signal.event",
))
register(OperatorSignature(
    name="SIGNAL_SOURCE", memoizable=True, source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _CHANNEL), result=_signal_source_result,
    implementation_key="rex.temporal_signal.source_field",
    preconditions=("signing is a retained gauge event channel and has an exact zero B1 source",),
))
register(OperatorSignature(
    name="RELATION_SIGNAL", memoizable=True, source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _CHANNEL), result=_relation_signal_result,
    implementation_key="rex.temporal_signal.relation_field",
    preconditions=("the direct C1 field remains separate from its derived C0 boundary source",),
))
register(OperatorSignature(
    name="SIGNAL_FLOW", memoizable=True, source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _CHANNEL), result=_signal_flow_result,
    implementation_key="rex.temporal_signal.flow",
    preconditions=("the local response is B1* followed by B1; it is not a vertex-path search",),
))
register(OperatorSignature(
    name="SIGNAL_HODGE", memoizable=True, source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _CHANNEL),
    result=_t("HodgeSplit", ValueKind.HODGE_SPLIT, grade=1, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE),
    implementation_key="rex.temporal_signal.hodge",
    preconditions=("the present Hodge action is numerical; direct C1 amplitude is not preprojected first",),
))
register(OperatorSignature(
    name="METRIC_CURVATURE", memoizable=True, source_kind=_REX_OR_TEMPORAL, inputs=(_BOUND_C1_COCHAIN,),
    result=_metric_curvature_result, implementation_key="rex.metric_curvature",
    preconditions=(
        "uses declared C1 boundary incidences and exact rational shares",
        "preserves witnesses and repeated incidence; no pairwise projection is applied",
    ),
))

# remaining native readings
#
# These operations already execute in the Rex adapter.  Their declarations make the
# whole phrase planner useful for the Hodge/Green/character layer as well, while keeping
# numerical action distinct from structural addressing and exact rational geometry.

_BOOL = TypePattern("exact", literal=bool, optional=True)
_SCALAR = TypePattern("scalar", literal=(int, float, Fraction), optional=True)
_ANY_VALUE = TypePattern("graded value", optional=True)
_BOUND_OPERATOR_OR_GRADE = TypePattern(
    "bound operator or grade", kind=ValueKind.OPERATOR, literal=int,
    source_bound=True, basis_bound=True,
)
_C1_COCHAIN = TypePattern(
    "C1 cochain", kind=(ValueKind.COCHAIN, ValueKind.CELL_COBOUNDARY, ValueKind.FIELD),
    grade=1, variance=Variance.COCHAIN,
    source_bound=True, basis_bound=True,
)
_C0_COCHAIN = TypePattern(
    "C0 cochain", kind=(ValueKind.COCHAIN, ValueKind.CELL_COBOUNDARY, ValueKind.FIELD),
    grade=0, variance=Variance.COCHAIN,
    source_bound=True, basis_bound=True,
)
_OPTIONAL_C0_COCHAIN = TypePattern(
    "C0 cochain", kind=(ValueKind.COCHAIN, ValueKind.CELL_COBOUNDARY, ValueKind.FIELD),
    grade=0, variance=Variance.COCHAIN,
    source_bound=True, basis_bound=True, optional=True,
)
_BOUND_COCHAIN = TypePattern(
    "source-bound cochain", kind=(ValueKind.COCHAIN, ValueKind.CELL_COBOUNDARY, ValueKind.FIELD),
    variance=Variance.COCHAIN, source_bound=True, basis_bound=True,
)
_ACTION = TypePattern(
    "Green or typed native Rex action", kind=(ValueKind.GREEN_ACTION, ValueKind.OPERATOR),
    source_bound=True, basis_bound=True,
)
_CHAIN_OR_COCHAIN = TypePattern(
    "bound chain or cochain", kind=(
        ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.CELL_BOUNDARY,
        ValueKind.CELL_COBOUNDARY, ValueKind.FIELD,
    ),
    source_bound=True, basis_bound=True,
)
_ACCUMULATABLE = TypePattern(
    "aligned chain or cochain", kind=(ValueKind.CHAIN, ValueKind.COCHAIN),
    source_bound=True, basis_bound=True,
)
_EXACT_SHEAF = TypePattern(
    "exact local-section sheaf", kind=ValueKind.EXACT_SHEAF,
    source_bound=True, basis_bound=True,
)
_METRIC = TypePattern("metric", kind=ValueKind.METRIC, literal=type(None),
                      source_bound=True, basis_bound=True, optional=True)
_CROSS_OR_METRIC = TypePattern("cross block or coherent metric", kind=(ValueKind.METRIC, ValueKind.CROSS_METRIC, ValueKind.FAMILY_METRIC),
                              literal=type(None), source_bound=True, basis_bound=True, optional=True)
_FAMILY_OR_METRIC = TypePattern("ambient or factored family metric", kind=(ValueKind.METRIC, ValueKind.FAMILY_METRIC),
                               literal=type(None), source_bound=True, basis_bound=True, optional=True)


def _coefficient_carrier(value: RCType) -> RCType:
    """View a structural boundary/coboundary wrapper as its declared coefficients.

    ``BOUNDARY(CELL(...))`` retains participant cells and the composite binary witness,
    while QUADRANCE or SPREAD acts on its Chain. This is a typed view of an existing
    carrier, never an implicit projection or newly derived field.
    """
    if value.kind is ValueKind.CELL_BOUNDARY:
        return value.with_(name="BoundaryChain", kind=ValueKind.CHAIN, variance=Variance.CHAIN)
    if value.kind in {ValueKind.CELL_COBOUNDARY, ValueKind.FIELD}:
        return value.with_(name="BoundaryCochain", kind=ValueKind.COCHAIN,
                           variance=Variance.COCHAIN)
    return value


def _rank_result(_args: tuple[object, ...]) -> RCType:
    return _t("Integer", ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
              exactness=Exactness.INTEGER)


def _green_result(args: tuple[object, ...]) -> RCType:
    if not args:
        return _t("GreenAction", ValueKind.GREEN_ACTION, grade=0,
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    return _t("Field", ValueKind.FIELD, grade=0, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE)


def _apply_result(args: tuple[object, ...]) -> RCType:
    action, value = args[:2]
    exact = len(args) > 2 and args[2] is True
    assert isinstance(action, RCType) and isinstance(value, RCType)
    if action.kind is ValueKind.GRADED_OPERATOR:
        from .graded_calculus import graded_application
        return graded_application(args)
    if value.kind is ValueKind.GRADED_CHAIN:
        raise TypeError("GradedChain requires an explicitly graded operator")
    descriptor = action.operator
    if descriptor is not None and descriptor.construction == "channel":
        value = _coefficient_carrier(value)
        if (value.variance is not Variance.COCHAIN or value.grade != descriptor.domain.grade
                or value.basis != descriptor.domain):
            raise TypeError("channel APPLY requires its canonical source-bound C1 Cochain")
        if exact and not descriptor.exact_action:
            raise TypeError("normalized G has no certified rational full action; its exact diagonal remains available")
        allowed = {Domain.INTEGER, Domain.RATIONAL} if exact else {Domain.INTEGER, Domain.RATIONAL, Domain.REAL}
        if value.domain not in allowed:
            raise TypeError("channel exact APPLY requires integer/rational coefficients" if exact else
                            "channel APPLY currently requires real coefficients")
        return _t("Field", ValueKind.FIELD, grade=1, variance=Variance.COCHAIN, basis=descriptor.codomain,
                  domain=Domain.RATIONAL if exact else Domain.REAL,
                  exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE)
    if descriptor is not None and descriptor.construction == "metric-resolvent":
        value = _coefficient_carrier(value)
        if (value.kind is not ValueKind.CHAIN or value.variance is not Variance.CHAIN
                or value.grade != descriptor.domain.grade or value.basis != descriptor.domain):
            raise TypeError("metric Green solve requires a Chain at its domain grade and ordered basis")
        if exact:
            raise TypeError("metric Green solve is numerical; no certified exact solve is available")
        if value.domain not in {Domain.INTEGER, Domain.RATIONAL, Domain.REAL}:
            raise TypeError("metric Green solve currently requires real coefficients")
        return _t("Chain", ValueKind.CHAIN, grade=value.grade, variance=Variance.CHAIN,
                  basis=descriptor.codomain, domain=Domain.REAL, exactness=Exactness.APPROXIMATE)
    if descriptor is not None and descriptor.construction in {"metric-adjoint", "weighted-hodge", "operator-bracket", "cell-chain", "exact-adjugate", "primary-lift", "rational-operator", "participation-walk", "text-overlap"}:
        value = _coefficient_carrier(value)
        if (value.grade != descriptor.domain.grade or value.basis != descriptor.domain
                or value.variance.value != descriptor.action_variance):
            raise TypeError("APPLY typed action requires its domain grade, ordered basis and original variance")
        if exact and not descriptor.exact_action:
            raise TypeError("APPLY typed action has no certified exact action for these coefficients and metrics")
        allowed = {Domain.INTEGER, Domain.RATIONAL} if exact else {
            Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX}
        if value.domain not in allowed:
            raise TypeError("APPLY exact=True requires integer/rational coefficients" if exact
                            else "APPLY typed action requires numeric coefficients")
        domain = Domain.RATIONAL if exact else Domain.COMPLEX if Domain.COMPLEX in {
            value.domain, descriptor.coefficient_domain} else Domain.REAL
        return _t("Chain" if value.variance is Variance.CHAIN else "Cochain", value.kind,
                  grade=descriptor.codomain.grade, variance=value.variance, basis=descriptor.codomain,
                  domain=domain, exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE)
    if exact:
        raise TypeError("APPLY exact=True requires an explicitly certified adjoint or weighted Hodge action")
    if value.variance is not Variance.COCHAIN:
        raise TypeError("APPLY requires a cochain for this action")
    if value.domain not in {Domain.INTEGER, Domain.RATIONAL, Domain.REAL}:
        raise TypeError("APPLY currently requires real coefficients; complex values cannot be discarded")
    if descriptor is None or descriptor.domain.grade != descriptor.codomain.grade:
        raise TypeError(
            "APPLY requires an explicit grade-preserving operator descriptor (e.g. HODGE_OPERATOR); use BOUNDARY or "
            "COBOUNDARY for a graded map"
        )
    if descriptor.domain.grade != value.grade:
        raise TypeError("APPLY requires a cochain at the Green action domain grade")
    if value.basis != descriptor.domain:
        raise TypeError("APPLY requires the operator's domain ordered basis")
    return _t("Field", ValueKind.FIELD, grade=descriptor.codomain.grade, variance=Variance.COCHAIN,
              basis=descriptor.codomain,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE)


def _adjoint_result(args):
    from .validation import adjoint_descriptor
    action = args[0]
    desc = action.operator
    if desc is None or not desc.transpose_available:
        raise TypeError("ADJOINT requires a declared transpose action; no materialization fallback")
    metrics = []
    for metric, basis in zip((*args[1:], None, None), (desc.domain, desc.codomain), strict=False):
        if metric is not None and (metric.metric is None or metric.basis != basis
                or metric.source != action.source or metric.temporal != action.temporal):
            raise TypeError("ADJOINT endpoint metric must match source, grade, basis and time")
        if basis.ordering != "canonical":
            raise TypeError("ADJOINT requires canonical ordered endpoint bases")
        metrics.append(None if metric is None else metric.metric)
    result = adjoint_descriptor(desc, *metrics)
    return _t("RexOperator", ValueKind.OPERATOR, grade=result.domain.grade, basis=result.domain,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
              operator=result, shape=ShapeRef(result.shape))


def _weighted_hodge_result(args, *, sector):
    grade = args[0]
    expected = (grade, grade+1) if sector == "up" else (grade, grade-1, grade+1)
    metrics = args[1:]
    for metric, k in zip(metrics, expected, strict=False):
        if metric is None:
            continue
        if k < 0:
            raise TypeError("grade-zero Hodge has no lower metric input")
        if metric.metric is None or metric.grade != k or metric.basis.ordering != "canonical":
            raise TypeError("Hodge metric requires its declared grade and canonical ordered basis")
    present = [m for m in metrics if m is not None]
    if present and any((m.source, m.temporal) != (present[0].source, present[0].temporal) for m in present):
        raise TypeError("Hodge metrics must share source and temporal state")
    return _t("WeightedHodgeOperator", ValueKind.OPERATOR, grade=grade,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def _metric_check(value, metric, exact):
    if metric is None:
        return None
    if metric.metric is None:
        raise TypeError("contraction requires an explicit metric descriptor")
    if (metric.grade != value.grade or metric.basis != value.basis
            or metric.source != value.source or metric.temporal != value.temporal):
        raise TypeError("contraction metric must match source, grade, basis and time")
    if exact and metric.metric.coefficient_domain not in {Domain.INTEGER, Domain.RATIONAL}:
        raise TypeError("exact contraction requires an integer/rational metric")
    return metric.metric


def _geometry_result(args: tuple[object, ...], *, name: str) -> RCType:
    raw = args[0]
    assert isinstance(raw, RCType)
    value = _coefficient_carrier(raw)
    exact = len(args) > 1 and args[1] is True
    descriptor = _metric_check(value, args[2] if len(args) > 2 else None, exact)
    if exact:
        if value.domain not in {Domain.INTEGER, Domain.RATIONAL}:
            raise TypeError(f"{name} exact=True requires an integer or rational carrier")
        return _t("Rational", ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL,
                  exactness=Exactness.RATIONAL, metric=descriptor)
    return _t("Real", ValueKind.REAL, domain=Domain.REAL,
              exactness=Exactness.APPROXIMATE, metric=descriptor)


def _spread_result(args: tuple[object, ...]) -> RCType:
    raw_left, raw_right = args[:2]
    assert isinstance(raw_left, RCType) and isinstance(raw_right, RCType)
    left, right = _coefficient_carrier(raw_left), _coefficient_carrier(raw_right)
    if not left.same_space(right):
        raise TypeError("SPREAD requires matching grade, variance, basis, source, and temporal state")
    extra = args[2:]
    _geometry_result((right, *extra), name="SPREAD")
    return _geometry_result((left, *extra), name="SPREAD")


def _signed_moment_result(args):
    left, right = map(_coefficient_carrier, args[:2])
    if not left.same_space(right):
        raise TypeError("MOMENT requires the same source, grade, variance, basis and time")
    metric, exact = args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else False
    _geometry_result((right, exact, metric), name="MOMENT")
    result = _geometry_result((left, exact, metric), name="MOMENT")
    if not exact and Domain.COMPLEX in {left.domain, right.domain}:
        result = result.with_(name="Complex", kind=ValueKind.COMPLEX, domain=Domain.COMPLEX)
    return result


def _metric_result(args):
    grade = args[0]
    weights = args[1] if len(args) > 1 else None
    if weights is not None and weights.grade != grade:
        raise TypeError("METRIC diagonal requires the declared grade")
    return _t("Metric", ValueKind.METRIC, grade=grade,
              basis=None if weights is None else weights.basis,
              domain=Domain.RATIONAL if weights is None or weights.domain in {Domain.INTEGER, Domain.RATIONAL} else Domain.REAL,
              exactness=Exactness.STRUCTURAL)


def _integrate_result(args):
    left, right = map(_coefficient_carrier, args[:2])
    if left.kind is not ValueKind.COCHAIN or right.kind is not ValueKind.CHAIN:
        raise TypeError("INTEGRATE requires a Cochain followed by a Chain")
    if (left.variance is not Variance.COCHAIN or right.variance is not Variance.CHAIN
            or left.grade != right.grade or left.basis != right.basis
            or left.source != right.source or left.temporal != right.temporal):
        raise TypeError("INTEGRATE requires dual variance on the same source, grade, basis and time")
    exact = len(args) > 2 and args[2] is True
    allowed = {Domain.INTEGER, Domain.RATIONAL} if exact else {
        Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX}
    if left.domain not in allowed or right.domain not in allowed:
        raise TypeError("INTEGRATE exact=True requires integer/rational coefficients" if exact
                        else "INTEGRATE requires numeric coefficient carriers")
    if left.shape and right.shape and left.shape != right.shape:
        raise ValueError("INTEGRATE requires matching vector/block shapes")
    if exact:
        return _t("Rational", ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL,
                  exactness=Exactness.RATIONAL)
    complex_ = Domain.COMPLEX in {left.domain, right.domain}
    return _t("Complex" if complex_ else "Real", ValueKind.COMPLEX if complex_ else ValueKind.REAL,
              domain=Domain.COMPLEX if complex_ else Domain.REAL, exactness=Exactness.APPROXIMATE)


def _resolvent_result(args):
    from math import isfinite
    operator = args[0]
    desc = operator.operator
    weighted = (desc is not None and desc.construction == "weighted-hodge"
                and dict(desc.parameters).get("sector") in {"down", "up", "sum"})
    if (desc is None or (not weighted and (desc.symmetric is not True or desc.psd is not True))
            or desc.domain.grade != desc.codomain.grade or desc.shape[0] != desc.shape[1]):
        raise TypeError("RESOLVENT requires an explicit symmetric PSD square operator or native metric PSD Hodge sector")
    alpha = args[1] if len(args) > 1 else 1.0
    tol = args[2] if len(args) > 2 else 1e-10
    maxiter = args[3] if len(args) > 3 else 1000
    if not isfinite(float(alpha)) or alpha < 0:
        raise ValueError("RESOLVENT alpha must be finite and nonnegative")
    if not isfinite(float(tol)) or not 0 < float(tol) < 1:
        raise ValueError("RESOLVENT tol must lie in (0, 1)")
    if maxiter <= 0:
        raise ValueError("RESOLVENT maxiter must be positive")
    from .validation import resolvent_descriptor
    result_desc = resolvent_descriptor(desc, alpha, tol, maxiter)
    return _t("GreenAction", ValueKind.GREEN_ACTION, grade=desc.domain.grade,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
              basis=desc.domain, operator=result_desc)


def _access_result(args, *, family=False):
    value, maps = _coefficient_carrier(args[0]), args[1]
    exact = len(args) > 2 and args[2] is True
    if not maps.accessions:
        raise TypeError("ACCESS requires explicit accession descriptors")
    if (value.grade, value.basis, value.source, value.temporal) != (
            maps.grade, maps.basis, maps.source, maps.temporal):
        raise TypeError("ACCESS requires matching source, grade, basis and time")
    if value.domain not in {Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX}:
        raise TypeError("ACCESS requires numerical coefficient carriers")
    if exact and (value.domain not in {Domain.INTEGER, Domain.RATIONAL}
                  or any(a.coefficient_domain not in {Domain.INTEGER, Domain.RATIONAL} for a in maps.accessions)):
        raise TypeError("exact ACCESS requires integer/rational coefficients and accessions")
    shape, members = None, ()
    if value.shape is not None:
        members = tuple((a.shape[0], *value.shape.dims[1:]) for a in maps.accessions)
        rows = members[0][0] if all(s[0] == members[0][0] for s in members) else None
        shape = ShapeRef((len(members), rows, *members[0][1:]) if family else members[0])
    return _t("TypedFamily" if family else "TypeView", ValueKind.TYPED_FAMILY if family else ValueKind.TYPE_VIEW,
              grade=value.grade, variance=value.variance, basis=value.basis, shape=shape,
              domain=Domain.RATIONAL if exact else Domain.COMPLEX if value.domain is Domain.COMPLEX else Domain.REAL,
              exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE, accessions=maps.accessions,
              member_shapes=members if family else ())


def _family_metric_check(value, metric, exact):
    desc = metric.family_metric
    if desc is None or not desc.realizations or desc.psd is not True or desc.metric.positive_definite is not True:
        raise TypeError("family contraction requires an explicit coherent PSD factor descriptor")
    if (metric.source, metric.grade, metric.basis, metric.temporal) != (
            value.source, value.grade, value.basis, value.temporal):
        raise TypeError("family metric must match source, grade, ambient basis and time")
    by_name = {e.accession.name: e for e in desc.realizations}
    if len(by_name) != len(desc.realizations):
        raise TypeError("family metric type names must be unique")
    for a in value.accessions:
        e = by_name.get(a.name)
        if e is None:
            raise TypeError(f"family metric has no realization for type {a.name!r}")
        endpoint = e.accession
        if (a.basis, a.coordinates, a.shape) != (endpoint.basis, endpoint.coordinates, endpoint.shape):
            raise TypeError("family realization requires its declared input coordinate space")
    domains = (desc.metric.coefficient_domain, *(e.coefficient_domain for e in desc.realizations))
    if exact and any(d not in {Domain.INTEGER, Domain.RATIONAL} for d in domains):
        raise TypeError("exact family contraction requires an integer/rational base metric and all realizations")
    if value.domain not in {Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX}:
        raise TypeError("family contraction requires numerical coefficient carriers")
    return desc


def _co_relate_result(args):
    left, right = args[:2]
    if not left.same_space(right):
        raise TypeError("CO_RELATE requires the same ambient source, grade, variance, basis and time")
    metric, exact = args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else False
    if any(len(v.accessions) != 1 for v in (left, right)):
        raise TypeError("CO_RELATE requires one explicit accession per view")
    if any(v.domain not in {Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX} for v in (left, right)):
        raise TypeError("CO_RELATE requires numerical coefficient carriers")
    cross = metric.cross_metric if isinstance(metric, RCType) else None
    if metric is not None and metric.kind is ValueKind.FAMILY_METRIC:
        _family_metric_check(right, metric, exact)
        family = _family_metric_check(left, metric, exact)
        if left.shape and right.shape and left.shape.dims[1:] != right.shape.dims[1:]:
            raise TypeError("CO_RELATE requires matching vector/block shapes")
        _geometry_result((right, exact, None), name="CO_RELATE")
        result = _geometry_result((left, exact, None), name="CO_RELATE").with_(family_metric=family)
    elif metric is not None and metric.kind is ValueKind.CROSS_METRIC:
        if cross is None:
            raise TypeError("CO_RELATE requires an explicit cross-metric descriptor")
        if (metric.source, metric.grade, metric.basis, metric.temporal) != (
                left.source, left.grade, left.basis, left.temporal):
            raise TypeError("cross metric must match source, grade, ambient basis and time")
        for view, endpoint in ((left, cross.left), (right, cross.right)):
            a = view.accessions[0]
            if (a.basis, a.coordinates, a.shape[0]) != (endpoint.basis, endpoint.coordinates, endpoint.shape[0]):
                raise TypeError("cross metric endpoint requires its declared output coordinate space")
        if exact and cross.coefficient_domain not in {Domain.INTEGER, Domain.RATIONAL}:
            raise TypeError("exact contraction requires an integer/rational cross metric")
        if left.shape and right.shape and left.shape.dims[1:] != right.shape.dims[1:]:
            raise TypeError("CO_RELATE requires matching vector/block shapes")
        # Reuse coefficient arithmetic checks, not ambient metric identification.
        _geometry_result((right, exact, None), name="CO_RELATE")
        result = _geometry_result((left, exact, None), name="CO_RELATE").with_(cross_metric=cross)
    else:
        if any(v.accessions[0].coordinates is not None for v in (left, right)):
            raise TypeError("type coordinates require an explicit CrossMetric or FamilyMetric, not an ambient metric")
        _geometry_result((right, exact, metric), name="CO_RELATE")
        result = _geometry_result((left, exact, metric), name="CO_RELATE")
    if not exact and Domain.COMPLEX in {left.domain, right.domain}:
        result = result.with_(name="Complex", kind=ValueKind.COMPLEX, domain=Domain.COMPLEX)
    return result.with_(accessions=left.accessions + right.accessions)


def _moment_tensor_result(args):
    family = args[0]
    if not family.accessions:
        raise TypeError("MOMENT_TENSOR requires a nonempty declared type family")
    metric, exact = args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else False
    if metric is not None and metric.kind is ValueKind.FAMILY_METRIC:
        desc = _family_metric_check(family, metric, exact)
        result = _geometry_result((family, exact, None), name="MOMENT_TENSOR").with_(family_metric=desc)
    else:
        if any(a.coordinates is not None for a in family.accessions):
            raise TypeError("coordinate MOMENT_TENSOR requires a coherent family form; independent cross blocks are not a Gram metric")
        result = _geometry_result((family, exact, metric), name="MOMENT_TENSOR")
    return result.with_(name="TypedMomentTensor", kind=ValueKind.MOMENT_TENSOR,
        grade=family.grade, basis=family.basis, variance=Variance.NEUTRAL,
        domain=Domain.COMPLEX if not exact and family.domain is Domain.COMPLEX else result.domain,
        shape=ShapeRef((len(family.accessions), len(family.accessions))), accessions=family.accessions)


def _accumulate_result(args: tuple[object, ...]) -> RCType:
    left, right = args
    assert isinstance(left, RCType) and isinstance(right, RCType)
    if not left.same_space(right):
        raise TypeError(
            "ACCUMULATE requires matching grade, variance, basis, source, and temporal state"
        )
    exact_inputs = (
        left.domain in {Domain.INTEGER, Domain.RATIONAL}
        and right.domain in {Domain.INTEGER, Domain.RATIONAL}
        and left.exactness in {Exactness.INTEGER, Exactness.RATIONAL}
        and right.exactness in {Exactness.INTEGER, Exactness.RATIONAL}
    )
    if exact_inputs:
        rational = Domain.RATIONAL in {left.domain, right.domain}
        domain = Domain.RATIONAL if rational else Domain.INTEGER
        exactness = Exactness.RATIONAL if rational else Exactness.INTEGER
    else:
        domain = Domain.COMPLEX if Domain.COMPLEX in {left.domain, right.domain} else Domain.REAL
        exactness = Exactness.APPROXIMATE
    return _t(left.name, left.kind, grade=left.grade, variance=left.variance,
              domain=domain, exactness=exactness)


def _closure_result(args: tuple[object, ...]) -> RCType:
    grade = 0 if len(args) < 3 else int(args[2])
    if grade != 0:
        raise NotImplementedError("CLOSURE currently implements only grade-0 cell seeds")
    return _t("SemanticClosure", ValueKind.RECORD,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def _character_result(args: tuple[object, ...]) -> RCType:
    exact = bool(args[0]) if args else False
    return _t("Character", ValueKind.CHARACTER,
              domain=Domain.RATIONAL if exact else Domain.REAL,
              exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE)


def _channel_result(args):
    if args[0].upper() not in {"T", "G", "F", "C"}:
        raise ValueError("CHANNEL name must be T, G, F or C")
    return _t("ChannelOperator", ValueKind.OPERATOR, grade=1,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def _moment_result(args):
    action, order = args[:2]
    desc = action.operator
    local = args[2] if len(args) > 2 else False
    exact = args[3] if len(args) > 3 else False
    if order < 0:
        raise ValueError("SCALE_MOMENT order must be nonnegative")
    if (desc is None or desc.symmetric is not True
            or desc.domain.grade != desc.codomain.grade or desc.shape[0] != desc.shape[1]):
        raise TypeError("SCALE_MOMENT requires an explicit symmetric square operator")
    if exact and order != 0 and (order != 1 or desc.construction != "channel"):
        raise ValueError("exact SCALE_MOMENT supports order 0, or channel order 1 only")
    return _t("Cochain" if local else "Rational" if exact else "Real",
              ValueKind.COCHAIN if local else ValueKind.EXACT_RATIONAL if exact else ValueKind.REAL,
              grade=action.grade if local else None,
              variance=Variance.COCHAIN if local else None,
              basis=desc.codomain if local else None,
              domain=Domain.RATIONAL if exact else Domain.REAL,
              exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE)


register(OperatorSignature(
    name="CHANNEL", memoizable=True, source_kind=ValueKind.REX, inputs=(_STR,),
    result=_channel_result, implementation_key="rex.channel_operator",
    preconditions=("distinct participants; no vertex weighting; G follows source selection",
                   "T/G/F carry relation weights, C does not; F uses raw G; no trace normalization",
                   "exact full action and transpose for T/raw-G/F/C; normalized G supports exact diagonal only",
                   "edge_metric_exact reads declared rational weights or the exact stored binary float"),
))
register(OperatorSignature(
    name="STAR_CHARACTER", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("vertex cell", kind=ValueKind.CELL, grade=0, source_bound=True, basis_bound=True), _BOOL),
    result=lambda args: _character_result(args[1:]).with_(grade=0),
    implementation_key="rex.star_character",
    preconditions=("mean of edge character over one C0 star; isolated vertex uses uniform character",),
))
register(OperatorSignature(
    name="SCALE_MOMENT", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("square operator", kind=ValueKind.OPERATOR, source_bound=True, basis_bound=True), TypePattern("order", literal=int),
            TypePattern("local", literal=bool, optional=True), _BOOL),
    result=_moment_result, implementation_key="rex.scale_moment",
    preconditions=("symmetric real square operator; nonnegative integer order",
                   "exact order 0 or channel order 1; numerical higher sparse powers may fill"),
))
register(OperatorSignature(
    name="CHARACTER_ENERGY", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("square operator", kind=ValueKind.OPERATOR, source_bound=True, basis_bound=True),),
    result=lambda args: _moment_result((args[0], 2, True, False)),
    implementation_key="rex.energy_character",
    preconditions=("explicit symmetric real operator; returns diag(operator squared), not squared character",),
))


def _zero_result(args: tuple[object, ...]) -> RCType:
    grade = int(args[0])
    kind = "cochain" if len(args) == 1 else str(args[1]).lower()
    if kind == "cochain":
        value_kind, variance = ValueKind.COCHAIN, Variance.COCHAIN
    elif kind == "chain":
        value_kind, variance = ValueKind.CHAIN, Variance.CHAIN
    else:
        raise ValueError("ZERO kind must be 'chain' or 'cochain'")
    return _t(kind.title(), value_kind, grade=grade, variance=variance,
              domain=Domain.INTEGER, exactness=Exactness.INTEGER)


def _exact_glue_result(args: tuple[object, ...]) -> RCType:
    section = args[0]
    assert isinstance(section, RCType)
    return _t("ExactGlueResult", ValueKind.EXACT_GLUE, grade=section.grade,
              domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL)


def _grade_result(args):
    if args and isinstance(args[0], RCType) and args[0].kind in {ValueKind.GRADED_CHAIN, ValueKind.GRADED_OPERATOR}:
        raise TypeError("a direct-sum value has no single grade; select GRADE_COMPONENT first")
    return _exact_integer()


register(OperatorSignature(
    name="GRADE", memoizable=True, source_kind=ValueKind.REX, inputs=(_ANY_VALUE,),
    result=_grade_result, implementation_key="rex.grade",
))
register(OperatorSignature(
    name="DESCRIBE", memoizable=True, source_kind=ValueKind.REX, inputs=(),
    result=_t("StructuralDescription", ValueKind.STRUCTURAL_DESCRIPTION,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="rex.describe",
))
register(OperatorSignature(
    name="HODGE_OPERATOR", memoizable=True, source_kind=ValueKind.REX, inputs=(_GRADE, _SCALAR),
    result=lambda args: _t("RexOperator", ValueKind.OPERATOR, grade=int(args[0]),
                           domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="rex.hodge_operator",
))
for _hodge_sector in ("down", "up", "sum", "difference"):
    register(OperatorSignature(
        name="HODGE_" + _hodge_sector.upper(), memoizable=True, source_kind=ValueKind.REX,
        inputs=(_GRADE, _METRIC, _METRIC) + ((_METRIC,) if _hodge_sector in {"sum", "difference"} else ()),
        result=lambda args, sector=_hodge_sector: _weighted_hodge_result(args, sector=sector),
        implementation_key="rex.weighted_hodge." + _hodge_sector,
        preconditions=("canonical Chain coordinates; explicit positive diagonal grade and neighboring metrics, default identity",
                       "metric-self-adjoint sectors; sum/down/up are metric PSD, difference is not granted positivity",
                       "mutual sector annihilation requires the source chain law; no Euclidean symmetry is inferred"),
    ))
register(OperatorSignature(
    name="RANK", memoizable=True, source_kind=ValueKind.REX, inputs=(_BOUND_OPERATOR_OR_GRADE,),
    result=_rank_result, implementation_key="rex.rank",
    preconditions=("exact integer elimination or canonical C1 denominator clearing; unsupported numerical matrices are refused",),
))
register(OperatorSignature(
    name="NULLITY", memoizable=True, source_kind=ValueKind.REX, inputs=(_BOUND_OPERATOR_OR_GRADE,),
    result=_rank_result, implementation_key="rex.nullity",
    preconditions=("exact integer elimination or canonical C1 denominator clearing; unsupported numerical matrices are refused",),
))
register(OperatorSignature(
    name="BETTI", memoizable=True, source_kind=ValueKind.REX, inputs=(_GRADE,),
    result=_t("Integer", ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
              exactness=Exactness.INTEGER), implementation_key="rex.betti",
    preconditions=("full carried rational/integer tower with exact zero consecutive compositions",
                   "denominator clearing is for rank only, never for chain composition"),
))
register(OperatorSignature(
    name="HODGE", memoizable=True, source_kind=_REX_OR_TEMPORAL, inputs=(_C1_COCHAIN,),
    result=_t("HodgeSplit", ValueKind.HODGE_SPLIT, grade=1, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE),
    implementation_key="rex.hodge",
))
register(OperatorSignature(
    name="HARMONIC", memoizable=True, source_kind=_REX_OR_TEMPORAL, inputs=(_C1_COCHAIN,),
    result=_t("Cochain", ValueKind.COCHAIN, grade=1, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE),
    implementation_key="rex.harmonic",
))
register(OperatorSignature(
    name="GREEN", memoizable=True, source_kind=ValueKind.REX, inputs=(_OPTIONAL_C0_COCHAIN,),
    result=_green_result, implementation_key="rex.green",
    preconditions=("the unapplied Green action is structural; applying it is numerical",),
))
register(OperatorSignature(
    name="APPLY", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("native action", kind=(ValueKind.OPERATOR, ValueKind.GREEN_ACTION, ValueKind.GRADED_OPERATOR),
                        source_bound=True),
            TypePattern("typed coefficients", kind=(*_CHAIN_OR_COCHAIN.kind, ValueKind.GRADED_CHAIN),
                        source_bound=True), _BOOL),
    result=_apply_result, implementation_key="rex.operator.apply",
))
for _dirac_name in ("DIRAC", "ANTI_DIRAC"):
    register(OperatorSignature(
        name=_dirac_name, memoizable=True, source_kind=ValueKind.REX,
        inputs=(TypePattern("explicit grade metrics", literal=(tuple, list, type(None)), optional=True),),
        result=_t("GradedOperator", ValueKind.GRADED_OPERATOR, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
        implementation_key="rex.weighted_dirac",
        preconditions=("full carried Chain tower; canonical bases; positive diagonal metrics, identity where omitted",
                       "Dirac is metric self-adjoint; anti-Dirac is metric skew-adjoint; neither is certified PSD",
                       "square, anticommutator and commutator identities require the chain law",
                       "grade/type restrictions and full SPD metrics are not implicit"),
    ))
for _bracket_name in ("COMMUTATOR", "ANTICOMMUTATOR"):
    from .operator_algebra import bracket_result
    register(OperatorSignature(
        name=_bracket_name, memoizable=True, source_kind=ValueKind.REX,
        inputs=(TypePattern("native operator", kind=(ValueKind.OPERATOR, ValueKind.GRADED_OPERATOR), source_bound=True),)*2,
        result=lambda args, anti=_bracket_name == "ANTICOMMUTATOR": bracket_result(args, anti=anti),
        implementation_key="rex.operator_bracket",
        preconditions=("AB-BA or AB+BA; rightmost action first; ordinary rather than Koszul-graded bracket",
                       "matching canonical source/grade/variance or full graded tower; numerical Green handles excluded",
                       "exact action only with both certified factors; no product matrix or eigensolve",
                       "common-metric adjointness only when established; no inferred PSD or chain-law simplification"),
    ))
register(OperatorSignature(
    name="GRADED_CHAIN", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("Chain components", literal=(tuple, list)),),
    result=_t("GradedChain", ValueKind.GRADED_CHAIN, variance=Variance.CHAIN),
    implementation_key="rex.graded_chain",
    preconditions=("unique canonical source-bound Chain grades with matching block shape; omitted grades are zero",
                   "rational seeds stay Q; numerical seeds promote the whole state to real/complex"),
))


def _grade_component_result(args):
    from .graded_calculus import component_result
    return component_result(args)


register(OperatorSignature(
    name="GRADE_COMPONENT", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("graded Chain", kind=ValueKind.GRADED_CHAIN, source_bound=True), _GRADE),
    result=_grade_component_result, implementation_key="rex.graded_chain.component",
    preconditions=("extract one carried Chain grade without flattening or changing its basis",),
))
register(OperatorSignature(
    name="ADJOINT", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("operator with transpose action", kind=ValueKind.OPERATOR,
                        source_bound=True, basis_bound=True), _METRIC, _METRIC),
    result=_adjoint_result, implementation_key="rex.metric_adjoint",
    preconditions=("two positive diagonal metrics name the ORIGINAL operator's domain and codomain; omitted means identity",
                   "variance preserved; metric adjoint is not dual coboundary or a Euclidean PSD certificate"),
))
register(OperatorSignature(
    name="QUADRANCE", memoizable=True, source_kind=_REX_OR_TEMPORAL, inputs=(_CHAIN_OR_COCHAIN, _BOOL, _METRIC),
    result=lambda args: _geometry_result(args, name="QUADRANCE"),
    implementation_key="rex.quadrance",
))
register(OperatorSignature(
    name="SPREAD", memoizable=True, source_kind=ValueKind.REX,
    inputs=(_CHAIN_OR_COCHAIN, _CHAIN_OR_COCHAIN, _BOOL, _METRIC), result=_spread_result,
    implementation_key="rex.spread",
))
register(OperatorSignature(
    name="METRIC", memoizable=True, source_kind=ValueKind.REX,
    inputs=(_GRADE, TypePattern("positive diagonal", kind=ValueKind.COCHAIN, literal=type(None),
        domain=(Domain.INTEGER, Domain.RATIONAL, Domain.REAL), source_bound=True, basis_bound=True, optional=True)),
    result=_metric_result, implementation_key="rex.diagonal_metric",
    preconditions=("identity if omitted; otherwise an explicit positive real diagonal Cochain",
                   "no automatic channel-weight, accession or semidefinite-quotient interpretation"),
))
register(OperatorSignature(
    name="MOMENT", memoizable=True, source_kind=ValueKind.REX,
    inputs=(_CHAIN_OR_COCHAIN, _CHAIN_OR_COCHAIN, _METRIC, _BOOL),
    result=_signed_moment_result, implementation_key="rex.metric_moment",
    preconditions=("aligned coefficient carriers; signed/Hermitian contraction, not squared magnitude",),
))
register(OperatorSignature(
    name="INTEGRATE", memoizable=True, source_kind=ValueKind.REX,
    inputs=(_CHAIN_OR_COCHAIN, _CHAIN_OR_COCHAIN, _BOOL),
    result=_integrate_result, implementation_key="rex.integrate",
    preconditions=("canonical bilinear Cochain/Chain dual evaluation, not a Hermitian metric moment",
                   "same source, grade, ordered basis, temporal state and vector/block shape; Q-only exact mode"),
))
register(OperatorSignature(
    name="RESOLVENT", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("PSD operator", kind=ValueKind.OPERATOR, source_bound=True, basis_bound=True),
            _SCALAR, TypePattern("tolerance", literal=(int, float, Fraction), optional=True),
            TypePattern("maxiter", literal=int, optional=True)),
    result=_resolvent_result, implementation_key="rex.green.resolvent",
    preconditions=("(I + alpha L)^-1, not a pseudoinverse or shifted inverse (L + alpha I)^-1",
                   "real Euclidean PSD or native positive-diagonal metric PSD Hodge action; alpha >= 0; positive tolerance and iteration limit"),
))
register(OperatorSignature(
    name="GREEN_SOLVE", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("Green action", kind=ValueKind.GREEN_ACTION, source_bound=True, basis_bound=True), _CHAIN_OR_COCHAIN),
    result=_apply_result, implementation_key="rex.green.solve",
    preconditions=("same typed action contract as APPLY; current solvers are numerical and real",),
))
_TYPE_VIEW = TypePattern("type view", kind=ValueKind.TYPE_VIEW, source_bound=True, basis_bound=True,
                         domain=(Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX))
_TYPED_FAMILY = TypePattern("typed family", kind=ValueKind.TYPED_FAMILY, source_bound=True, basis_bound=True,
                            domain=(Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX))
for _access_name, _is_family in (("ACCESS", False), ("ACCESS_TYPES", True)):
    register(OperatorSignature(
        name=_access_name, memoizable=True, source_kind=ValueKind.REX,
        inputs=(_CHAIN_OR_COCHAIN, TypePattern("declared accessions",
            kind=ValueKind.ACCESSION_FAMILY if _is_family else ValueKind.TYPE_ACCESSION,
            source_bound=True, basis_bound=True), _BOOL),
        result=lambda args, family=_is_family: _access_result(args, family=family),
        implementation_key="rex.type_accession." + ("AccessionFamily.apply" if _is_family else "TypeAccession.apply"),
        preconditions=("explicit sparse maps from the same ambient basis into declared cell/type coordinates",
                       "overlap and nonprojector maps allowed; chain preservation is not asserted"),
    ))
register(OperatorSignature(
    name="CO_RELATE", memoizable=True, source_kind=ValueKind.REX,
    inputs=(_TYPE_VIEW, _TYPE_VIEW, _CROSS_OR_METRIC, _BOOL), result=_co_relate_result,
    implementation_key="rex.type_accession.co_relate",
    preconditions=("ambient diagonal metric, coherent factored family form, or explicit sparse cross block; no implicit coordinate identification",),
))
register(OperatorSignature(
    name="MOMENT_TENSOR", memoizable=True, source_kind=ValueKind.REX,
    inputs=(_TYPED_FAMILY, _FAMILY_OR_METRIC, _BOOL), result=_moment_tensor_result,
    implementation_key="rex.type_accession.moment_tensor",
    preconditions=("requested type-by-type matrix; block columns are contracted, not additional types",
                   "diagonal quadrance and off-diagonal co relation; explicit ambient metric or coherent family realization form"),
))


def _chain_map_result(args):
    value = args[0]
    desc = value.graded_map
    if desc is None or not desc.domain or len(desc.domain) != len(desc.codomain):
        raise TypeError("CHAIN_MAP requires an explicit full graded-map descriptor")
    if desc.shapes != tuple((len(b.keys), len(a.keys)) for a, b in zip(desc.domain, desc.codomain, strict=True)):
        raise ValueError("graded-map descriptor shapes must match its ordered spaces")
    if len(desc.nnz) != len(desc.shapes) or any(n < 0 for n in desc.nnz):
        raise ValueError("graded-map descriptor requires nonnegative nnz at every grade")
    if any(n > rows*cols for n, (rows, cols) in zip(desc.nnz, desc.shapes, strict=True)):
        raise ValueError("graded-map nnz exceeds its declared shape")
    if desc.construction != "exact-sparse-graded-map":
        raise TypeError("CHAIN_MAP requires the exact sparse declaration contract")
    return value.with_(name="ChainMap", kind=ValueKind.CHAIN_MAP)


register(OperatorSignature(
    name="CHAIN_MAP", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("explicit graded map", kind=(ValueKind.GRADED_MAP, ValueKind.CHAIN_MAP),
                        source_bound=True, domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL),),
    result=_chain_map_result, implementation_key="rex.chain_map.verify",
    preconditions=("full source-bound Q tower with explicit target boundaries and grade maps",
                   "both chain laws and every commuting square must vanish exactly; boundary state must be current"),
))
register(OperatorSignature(
    name="ACCUMULATE", memoizable=True, source_kind=_REX_OR_TEMPORAL,
    inputs=(_ACCUMULATABLE, _ACCUMULATABLE), result=_accumulate_result,
    implementation_key="rex.accumulate",
    preconditions=(
        "combines only source, basis, and temporal-state aligned coefficient carriers",
        "cross-time accumulation requires an explicit transport or alignment action",
    ),
))
register(OperatorSignature(
    name="HODGE_COORDS", memoizable=True, source_kind=_REX_OR_TEMPORAL, inputs=(_C1_COCHAIN,),
    result=_t("HodgeCoordinates", ValueKind.HODGE_COORDINATES, grade=1,
              variance=Variance.COCHAIN, domain=Domain.REAL,
              exactness=Exactness.APPROXIMATE), implementation_key="rex.hodge_coordinates",
))
def _winding_result(args):
    domain = args[0].domain
    exactness = {Domain.INTEGER: Exactness.INTEGER, Domain.RATIONAL: Exactness.RATIONAL}.get(
        domain, Exactness.APPROXIMATE)
    return _t("Winding", ValueKind.WINDING, grade=1, variance=Variance.COCHAIN,
              domain=domain if domain in {Domain.INTEGER, Domain.RATIONAL} else Domain.REAL,
              exactness=exactness)


register(OperatorSignature(
    name="WINDING", memoizable=True, source_kind=_REX_OR_TEMPORAL, inputs=(_C1_COCHAIN,),
    result=_winding_result,
    implementation_key="rex.winding",
))
register(OperatorSignature(
    name="CLOSURE", memoizable=True, source_kind=ValueKind.REX,
    inputs=(TypePattern("C0 seed", literal=int),
            TypePattern("maximum depth", literal=int, optional=True),
            TypePattern("grade", literal=int, optional=True)),
    result=_closure_result, implementation_key="rex.closure",
))
register(OperatorSignature(
    name="SIGNIFICANCE", memoizable=True, source_kind=ValueKind.REX, inputs=(_INDEX,),
    result=_t("Real", ValueKind.REAL, domain=Domain.REAL,
              exactness=Exactness.APPROXIMATE), implementation_key="rex.significance",
))
register(OperatorSignature(
    name="CHARACTER", memoizable=True, source_kind=ValueKind.REX, inputs=(_BOOL,),
    result=_character_result, implementation_key="rex.character",
    preconditions=("exact diagonal character supports raw G, or normalized G with nonnegative relation weights",),
))
register(OperatorSignature(
    name="ZERO", memoizable=True, source_kind=ValueKind.REX,
    inputs=(_GRADE, TypePattern("kind", literal=str, optional=True)),
    result=_zero_result, implementation_key="rex.zero",
))
register(OperatorSignature(
    name="GLUE", source_kind=ValueKind.REX, inputs=(_EXACT_SHEAF,),
    result=_exact_glue_result, implementation_key="rex.exact_sheaf.glue",
    preconditions=(
        "the local-section stalks and incidence restrictions are exact integer/rational values",
        "every restriction is evaluated at every shared mediator; failures retain exact residuals",
        "cross-state section comparison requires explicit incidence restrictions; no chain-map certificate or complex merge is inferred",
    ),
))
register(OperatorSignature(
    name="SECTION_CHECK", source_kind=ValueKind.REX, inputs=(_EXACT_SHEAF,),
    result=lambda args: _t("ExactSectionCheck", ValueKind.EXACT_SECTION_CHECK, grade=args[0].grade,
                          domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL),
    implementation_key="rex.exact_sheaf.check_section",
    preconditions=(
        "integer/rational stalks and rectangular maps have declared incidence dimensions",
        "one transport per incidence and anchor residuals certify all-pair compatibility without a pair graph",
        "cross-state section comparison requires explicit incidence restrictions; no chain-map certificate or complex merge is inferred",
    ),
))

# The logical store manifest includes identities and history, not merely a
# projected current record count. Permissions name everything it can observe.
register(OperatorSignature(
    name="RCDB_STATE_HASH", source_kind=ValueKind.RCDB_STORE, inputs=(),
    source_methods=frozenset({"state_digest"}),
    requires=frozenset({"read", HISTORY, IDENTITY}),
    result=_t("Digest", ValueKind.DIGEST, domain=Domain.BYTES,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.state_digest",
    preconditions=("reads all published native payloads and logical record/commit metadata; not an index hash or lineage verification",),
))

from .value_contracts import install as _install_value_signatures

_install_value_signatures(register)

from .calculus_contracts import install as _install_calculus_signatures

_install_calculus_signatures(register)

from .critical_contracts import install as _install_critical_signatures

_install_critical_signatures(register)

from .certificate_contracts import install as _install_certificate_signatures

_install_certificate_signatures(register)

from .temporal_contracts import install as _install_temporal_signatures

_install_temporal_signatures(register)

from .structure_contracts import install as _install_structure_signatures
from .homology_contracts import install as _install_homology_signatures

_install_homology_signatures(register)

from .harmonic_modes_contracts import install as _install_harmonic_modes_signatures

_install_harmonic_modes_signatures(register)

from .resolvent_rank_contracts import install as _install_resolvent_rank_signatures

_install_resolvent_rank_signatures(register)

_install_structure_signatures(register)

from .partition_contracts import install as _install_partition_signatures
from .symmetry_contracts import install as _install_symmetry_signatures

_install_symmetry_signatures(register)

_install_partition_signatures(register)

from .artifact_contracts import install as _install_artifact_signatures
from .filling_contracts import install as _install_filling_signatures

_install_filling_signatures(register)
from .difference_contracts import install as _install_difference_signatures
from .replay_contracts import install as _install_replay_signatures

_install_replay_signatures(register)

_install_difference_signatures(register)

from .document_contracts import install as _install_document_signatures

_install_document_signatures(register)

from .rational_contracts import install as _install_rational_signatures

_install_rational_signatures(register)

from .markov_contracts import install as _install_markov_signatures

_install_markov_signatures(register)

from .corpus_contracts import install as _install_corpus_signatures

_install_corpus_signatures(register)

from .turn_contracts import install as _install_turn_signatures

_install_turn_signatures(register)

_install_artifact_signatures(register)

from .coordinate_contracts import install as _install_coordinate_signatures
_install_coordinate_signatures(register)

from .tensor_contracts import install as _install_tensor_signatures
_install_tensor_signatures(register)

from .section_contracts import install as _install_section_signatures
_install_section_signatures(register)

from .model_contracts import install as _install_model_signatures
_install_model_signatures(register)

from .molecular_contracts import install as _install_molecular_signatures
_install_molecular_signatures(register)

from .program_contracts import install as _install_program_signatures
_install_program_signatures(register, OperatorSignature, TypePattern)

from .recursion_contracts import install as _install_recursion_signatures
_install_recursion_signatures(register, OperatorSignature, TypePattern)

from .transformation_contracts import install as _install_transformation_signatures
_install_transformation_signatures(register, OperatorSignature, TypePattern)
