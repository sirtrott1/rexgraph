"""Static operator signatures: what an operation is, before it is a Python callable.

The executor currently evaluates an operator and then inspects the result to decide what
it was. That ordering cannot answer anything before the work happens, so EXPLAIN cannot
report a type without paying for the computation, and an impossible request is discovered
by failing rather than by being refused.

A signature moves those answers in front of execution. The binder resolves a name here,
checks arity and input kinds, and attaches the signature; inference applies its rules to
produce a result type; only then does the executor resolve ``implementation_key`` and run
anything.

This module carries the catalogue for the storage, catalog and metadata operators. The
Rex-mathematics signatures have a separate contract and land beside their adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .types import Domain, Effect, Exactness, RCType, TemporalRef, ValueKind, Variance

# Capability names. These are separate on purpose: reading a projected summary is not the
# same right as resolving an identity, and neither implies mutating.
IDENTITY = "identity"
HISTORY = "history"
MUTATE = "mutate"
FILES = "files"
SEARCH = "search"
SECURITY = "security"


@dataclass(frozen=True)
class TypePattern:
    """What one argument must be for a call to type-check.

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
    requires: frozenset[str] = field(default_factory=frozenset)
    effects: frozenset[Effect] = field(default_factory=lambda: frozenset({Effect.READ}))
    preconditions: tuple[str, ...] = ()
    unreachable: str = ""

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
    _CATALOGUE[signature.name] = signature
    return signature


def lookup(name: str) -> OperatorSignature:
    try:
        return _CATALOGUE[name.upper()]
    except KeyError as exc:
        raise KeyError(f"no RCQL signature for operator {name!r}") from exc


def catalogued() -> frozenset[str]:
    return frozenset(_CATALOGUE)


def _t(name: str, kind: ValueKind, **kw) -> RCType:
    return RCType(name, kind=kind, **kw)


_STR = TypePattern("name", literal=str)
_LIMIT = TypePattern("limit", literal=int, optional=True)
_OFFSET = TypePattern("offset", literal=int, optional=True)

# ------------------------------------------------------------------ file catalog
#
# The catalog indexes loadable kinds only, so an entry name that exists on disk is not
# necessarily an entry. That is a precondition rather than a runtime KeyError, and is
# recorded here so Phase 2 can check it during binding.

_ENTRY_EXISTS = ("name is an indexed catalog entry; the catalog holds loadable kinds only",)

register(OperatorSignature(
    name="FILES", source_kind=ValueKind.UNKNOWN, inputs=(_LIMIT, _OFFSET),
    result=_t("CatalogEntrySet", ValueKind.CATALOG_ENTRY_SET,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.list", requires=frozenset({FILES}),
))
register(OperatorSignature(
    name="SEARCH", source_kind=ValueKind.UNKNOWN,
    inputs=(TypePattern("text", literal=str), _LIMIT),
    result=_t("CatalogEntrySet", ValueKind.CATALOG_ENTRY_SET,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.search", requires=frozenset({FILES, SEARCH}),
))
register(OperatorSignature(
    name="FILE_INFO", source_kind=ValueKind.UNKNOWN, inputs=(_STR,),
    result=_t("CatalogEntry", ValueKind.CATALOG_ENTRY,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.info", requires=frozenset({FILES}),
    preconditions=_ENTRY_EXISTS + (
        "the sha256 field is populated only once a hash has been computed; a caller that "
        "needs the digest must request FILE_HASH rather than read it opportunistically",
    ),
))
register(OperatorSignature(
    name="FILE_HASH", source_kind=ValueKind.UNKNOWN, inputs=(_STR,),
    result=_t("Digest", ValueKind.DIGEST, domain=Domain.BYTES,
              exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.hash", requires=frozenset({FILES}),
    preconditions=_ENTRY_EXISTS,
))
register(OperatorSignature(
    name="HASH_FILES", source_kind=ValueKind.UNKNOWN, inputs=(),
    result=_t("Integer", ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
              exactness=Exactness.INTEGER),
    implementation_key="catalog.hash_all", requires=frozenset({FILES}),
    effects=frozenset({Effect.READ, Effect.FILESYSTEM}),
))
register(OperatorSignature(
    name="TENSORS", source_kind=ValueKind.UNKNOWN, inputs=(_STR, _LIMIT),
    result=_t("TensorManifest", ValueKind.TENSOR_MANIFEST,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.tensors", requires=frozenset({FILES}),
    preconditions=_ENTRY_EXISTS,
))
register(OperatorSignature(
    name="SEARCH_TENSORS", source_kind=ValueKind.UNKNOWN,
    inputs=(_STR, TypePattern("text", literal=str), _LIMIT),
    result=_t("TensorManifest", ValueKind.TENSOR_MANIFEST,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="catalog.search_tensors", requires=frozenset({FILES, SEARCH}),
    preconditions=_ENTRY_EXISTS,
))

# ------------------------------------------------------------------ complex digest
#
# STATE_HASH reads a Rex, not a catalog, despite sitting among the catalog operators in
# the registry. Declaring its source kind is what stops it being called on the catalog it
# is filed beside, which currently fails inside the digest rather than at the boundary.

register(OperatorSignature(
    name="STATE_HASH", source_kind=ValueKind.REX, inputs=(),
    result=_t("Digest", ValueKind.DIGEST, domain=Domain.BYTES,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rex.object_digest",
))

# ------------------------------------------------------------------ RCDB readings

register(OperatorSignature(
    name="RCDB_LIST", source_kind=ValueKind.REX, inputs=(_LIMIT, _OFFSET),
    result=_t("RecordSet", ValueKind.RECORD_SET, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.list",
    preconditions=("returns projected summaries; it does not decode a stored complex",),
))
register(OperatorSignature(
    name="RCDB_SEARCH", source_kind=ValueKind.REX,
    inputs=(TypePattern("text", literal=str), _LIMIT),
    result=_t("RecordSet", ValueKind.RECORD_SET, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.search", requires=frozenset({SEARCH}),
))
register(OperatorSignature(
    name="RCDB_GET", source_kind=ValueKind.REX, inputs=(_STR,),
    result=_t("Rex", ValueKind.REX, domain=Domain.METADATA),
    implementation_key="rcdb.get", requires=frozenset({IDENTITY}),
    preconditions=("decodes a stored complex, so it resolves an identity",),
))
register(OperatorSignature(
    name="RCDB_HISTORY", source_kind=ValueKind.REX, inputs=(_STR,),
    result=_t("History", ValueKind.HISTORY, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.history", requires=frozenset({HISTORY}),
))
register(OperatorSignature(
    name="RCDB_STATS", source_kind=ValueKind.REX, inputs=(),
    result=_t("StoreStats", ValueKind.STORE_STATS, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.stats",
))
register(OperatorSignature(
    name="RCDB_HASH", source_kind=ValueKind.REX, inputs=(_STR,),
    result=_t("Digest", ValueKind.DIGEST, domain=Domain.BYTES,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.record_digest", requires=frozenset({IDENTITY}),
))
register(OperatorSignature(
    name="RCDB_COMMITS", source_kind=ValueKind.REX, inputs=(_STR,),
    result=_t("CommitLink", ValueKind.COMMIT_LINK, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.commit_history", requires=frozenset({HISTORY}),
    preconditions=("a plain put contributes no commit link; only a governed transition does",),
))
register(OperatorSignature(
    name="RCDB_VERIFY", source_kind=ValueKind.REX, inputs=(_STR,),
    result=_t("Boolean", ValueKind.BOOLEAN, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.verify_commits", requires=frozenset({HISTORY}),
))
register(OperatorSignature(
    name="RCDB_SECURITY", source_kind=ValueKind.REX, inputs=(),
    result=_t("SecurityStatus", ValueKind.SECURITY_STATUS, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.security_status", requires=frozenset({SECURITY}),
    preconditions=("bounded configuration only; never key material or backend paths",),
))

# ------------------------------------------------------------------ primary cells and fields
#
# These signatures describe the native relational-complex carriers.  They deliberately
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
    if len(args) == 1:
        return _t("BoundaryOperator", ValueKind.OPERATOR, grade=grade,
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    value = args[1]
    assert isinstance(value, RCType)
    exactness = value.exactness
    domain = value.domain
    if grade == 1 and exactness in {Exactness.INTEGER, Exactness.RATIONAL}:
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
    if len(args) == 1:
        return _t("CoboundaryOperator", ValueKind.OPERATOR, grade=grade,
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    value = args[1]
    assert isinstance(value, RCType)
    exactness = value.exactness
    domain = value.domain
    if grade == 0 and exactness in {Exactness.INTEGER, Exactness.RATIONAL}:
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
    name="CELL", source_kind=ValueKind.REX, inputs=(_GRADE, _INDEX),
    result=_cell_result, implementation_key="rex.cell",
    preconditions=("grade and ordered-basis index occur in the bound relational complex",),
))
register(OperatorSignature(
    name="CELLS", source_kind=ValueKind.REX, inputs=(_GRADE, _INDICES),
    result=lambda args: _cell_result(args, plural=True), implementation_key="rex.cells",
    preconditions=("grade and every selected ordered-basis index occur in the bound relational complex",),
))
register(OperatorSignature(
    name="INDICATOR", source_kind=ValueKind.REX, inputs=(_CELL_OR_SET,),
    result=_indicator_result, implementation_key="rex.indicator",
    preconditions=(
        "keeps the selected primary cells distinct from their explicit 0/1 cochain",
        "materializing the full coefficient basis is proportional to its grade population",
    ),
))
register(OperatorSignature(
    name="BOUNDARY", source_kind=ValueKind.REX,
    inputs=(TypePattern("cell or grade", kind=(ValueKind.CELL, ValueKind.CELL_SET), literal=int,
                        source_bound=True, basis_bound=True),
            TypePattern("chain", kind=ValueKind.CHAIN, variance=Variance.CHAIN,
                        source_bound=True, basis_bound=True, optional=True)),
    result=_boundary_result, implementation_key="rex.boundary",
    preconditions=("a direct C1 boundary retains declared head/share coefficients",),
))
register(OperatorSignature(
    name="COBOUNDARY", source_kind=ValueKind.REX,
    inputs=(TypePattern("cell or grade", kind=(ValueKind.CELL, ValueKind.CELL_SET), literal=int,
                        source_bound=True, basis_bound=True),
            TypePattern("cochain", kind=ValueKind.COCHAIN, variance=Variance.COCHAIN,
                        source_bound=True, basis_bound=True, optional=True)),
    result=_coboundary_result, implementation_key="rex.coboundary",
    preconditions=("a direct C0 coboundary retains declared relation-share coefficients",),
))
register(OperatorSignature(
    name="COMPOSITE", source_kind=ValueKind.REX, inputs=(_C1_CELL,),
    result=_t("CompositeBinary", ValueKind.COMPOSITE_BINARY, grade=1, variance=Variance.CELL,
              domain=Domain.RATIONAL, exactness=Exactness.RATIONAL),
    implementation_key="rex.composite_binary",
    preconditions=("repeated C1 incidence refuses because vertex-basis binary masks would collapse occurrences",),
))
register(OperatorSignature(
    name="EXISTENCE", source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.INTEGER, exactness=Exactness.INTEGER),
    implementation_key="rex.composite_binary.existence",
))
register(OperatorSignature(
    name="HEAD", source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.INTEGER, exactness=Exactness.INTEGER),
    implementation_key="rex.composite_binary.head",
))
register(OperatorSignature(
    name="SHARE", source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.RATIONAL, exactness=Exactness.RATIONAL),
    implementation_key="rex.composite_binary.share",
))
register(OperatorSignature(
    name="SHARE_SUPPORT", source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_t("Chain", ValueKind.CHAIN, grade=0, variance=Variance.CHAIN,
              domain=Domain.INTEGER, exactness=Exactness.INTEGER),
    implementation_key="rex.composite_binary.share_support",
))
register(OperatorSignature(
    name="ARITY", source_kind=ValueKind.REX, inputs=(_C1_COMPOSITE,),
    result=_exact_integer(), implementation_key="rex.composite_binary.arity",
))
register(OperatorSignature(
    name="CORELATIONS", source_kind=ValueKind.REX, inputs=(_CELL_OR_SET,),
    result=_corelations_result, implementation_key="rex.corelations",
))
register(OperatorSignature(
    name="STAR", source_kind=ValueKind.REX, inputs=(_CELL_OR_SET,),
    result=_t("GradedCellPattern", ValueKind.CELL_PATTERN, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL), implementation_key="rex.star",
))
register(OperatorSignature(
    name="ENCLOSURE", source_kind=ValueKind.REX, inputs=(_CELL_OR_SET,),
    result=_t("GradedCellPattern", ValueKind.CELL_PATTERN, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL), implementation_key="rex.enclosure",
))

# Temporal relations are C1-primary values.  The delta carrier separates its structural
# axes from the optional measured amplitude, so amplitude is conservatively numerical at
# planning time even when an individual unweighted dataset happens to yield exact ones.
register(OperatorSignature(
    name="TEMPORAL_DELTA", source_kind=ValueKind.TEMPORAL_REX, inputs=(_GRADE,),
    result=_temporal_delta_result, implementation_key="rex.temporal_delta",
    preconditions=(
        "step lies in the source timeline transition range",
        "parallel equal-support C1 relations require stable relation identities and refuse otherwise",
    ),
))
register(OperatorSignature(
    name="SIGNAL_AT", source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _RELATION_KEY),
    result=_t("TemporalSignalEvent", ValueKind.TEMPORAL_EVENT, grade=1,
              variance=Variance.CELL, domain=Domain.METADATA,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rex.temporal_signal.event",
))
register(OperatorSignature(
    name="SIGNAL_SOURCE", source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _CHANNEL), result=_signal_source_result,
    implementation_key="rex.temporal_signal.source_field",
    preconditions=("signing is a retained gauge event channel and has an exact zero B1 source",),
))
register(OperatorSignature(
    name="RELATION_SIGNAL", source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _CHANNEL), result=_relation_signal_result,
    implementation_key="rex.temporal_signal.relation_field",
    preconditions=("the direct C1 field remains separate from its derived C0 boundary source",),
))
register(OperatorSignature(
    name="SIGNAL_FLOW", source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _CHANNEL), result=_signal_flow_result,
    implementation_key="rex.temporal_signal.flow",
    preconditions=("the local response is B1* followed by B1; it is not a vertex-path search",),
))
register(OperatorSignature(
    name="SIGNAL_HODGE", source_kind=ValueKind.TEMPORAL_REX,
    inputs=(_TEMPORAL_DELTA, _CHANNEL),
    result=_t("HodgeSplit", ValueKind.HODGE_SPLIT, grade=1, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE),
    implementation_key="rex.temporal_signal.hodge",
    preconditions=("the present Hodge action is numerical; direct C1 amplitude is not preprojected first",),
))
register(OperatorSignature(
    name="METRIC_CURVATURE", source_kind=_REX_OR_TEMPORAL, inputs=(_BOUND_C1_COCHAIN,),
    result=_metric_curvature_result, implementation_key="rex.metric_curvature",
    preconditions=(
        "uses declared C1 boundary incidences and exact rational shares",
        "preserves witnesses and repeated incidence; no pairwise projection is applied",
    ),
))

# ------------------------------------------------------------------ remaining native readings
#
# These operations already execute in the Rex adapter.  Their declarations make the
# whole-phrase planner useful for the Hodge/Green/character layer as well, while keeping
# numerical action distinct from structural addressing and exact rational geometry.

_BOOL = TypePattern("exact", literal=bool, optional=True)
_SCALAR = TypePattern("scalar", literal=(int, float), optional=True)
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
    "Green or grade-preserving Rex action", kind=(ValueKind.GREEN_ACTION, ValueKind.OPERATOR),
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


def _coefficient_carrier(value: RCType) -> RCType:
    """View a structural boundary/coboundary wrapper as its declared coefficients.

    ``BOUNDARY(CELL(...))`` retains participant cells and the composite-binary witness,
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
              exactness=Exactness.APPROXIMATE)


def _green_result(args: tuple[object, ...]) -> RCType:
    if not args:
        return _t("GreenAction", ValueKind.GREEN_ACTION, grade=0,
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    return _t("Field", ValueKind.FIELD, grade=0, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE)


def _apply_result(args: tuple[object, ...]) -> RCType:
    action, value = args
    assert isinstance(action, RCType) and isinstance(value, RCType)
    if action.kind is ValueKind.OPERATOR and action.name != "RexOperator":
        raise TypeError(
            "APPLY accepts only a grade-preserving HODGE_OPERATOR; use BOUNDARY or "
            "COBOUNDARY for a graded map"
        )
    if action.grade != value.grade:
        raise TypeError("APPLY requires a cochain at the Green action domain grade")
    return _t("Field", ValueKind.FIELD, grade=action.grade, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE)


def _geometry_result(args: tuple[object, ...], *, name: str) -> RCType:
    raw = args[0]
    assert isinstance(raw, RCType)
    value = _coefficient_carrier(raw)
    exact = len(args) > 1 and args[1] is True
    if exact:
        if value.domain not in {Domain.INTEGER, Domain.RATIONAL}:
            raise TypeError(f"{name} exact=True requires an integer or rational carrier")
        return _t("Rational", ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL,
                  exactness=Exactness.RATIONAL)
    return _t("Real", ValueKind.REAL, domain=Domain.REAL,
              exactness=Exactness.APPROXIMATE)


def _spread_result(args: tuple[object, ...]) -> RCType:
    raw_left, raw_right = args[:2]
    assert isinstance(raw_left, RCType) and isinstance(raw_right, RCType)
    left, right = _coefficient_carrier(raw_left), _coefficient_carrier(raw_right)
    if not left.same_space(right):
        raise TypeError("SPREAD requires matching grade, variance, basis, source, and temporal state")
    return _geometry_result((left, args[2]) if len(args) > 2 else (left,), name="SPREAD")


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
    return _t("StructuralDescription", ValueKind.STRUCTURAL_DESCRIPTION,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def _character_result(args: tuple[object, ...]) -> RCType:
    exact = bool(args[0]) if args else False
    return _t("Character", ValueKind.CHARACTER,
              domain=Domain.RATIONAL if exact else Domain.REAL,
              exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE)


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


register(OperatorSignature(
    name="GRADE", source_kind=ValueKind.REX, inputs=(_ANY_VALUE,),
    result=_exact_integer(), implementation_key="rex.grade",
))
register(OperatorSignature(
    name="DESCRIBE", source_kind=ValueKind.REX, inputs=(),
    result=_t("StructuralDescription", ValueKind.STRUCTURAL_DESCRIPTION,
              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="rex.describe",
))
register(OperatorSignature(
    name="HODGE_OPERATOR", source_kind=ValueKind.REX, inputs=(_GRADE, _SCALAR),
    result=lambda args: _t("RexOperator", ValueKind.OPERATOR, grade=int(args[0]),
                           domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
    implementation_key="rex.hodge_operator",
))
register(OperatorSignature(
    name="RANK", source_kind=ValueKind.REX, inputs=(_BOUND_OPERATOR_OR_GRADE,),
    result=_rank_result, implementation_key="rex.rank",
    preconditions=("the present sparse rank action is numerical even though its result is integral",),
))
register(OperatorSignature(
    name="NULLITY", source_kind=ValueKind.REX, inputs=(_BOUND_OPERATOR_OR_GRADE,),
    result=_rank_result, implementation_key="rex.nullity",
    preconditions=("the present sparse rank action is numerical even though its result is integral",),
))
register(OperatorSignature(
    name="BETTI", source_kind=ValueKind.REX, inputs=(_GRADE,),
    result=_t("Integer", ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
              exactness=Exactness.STRUCTURAL), implementation_key="rex.betti",
))
register(OperatorSignature(
    name="HODGE", source_kind=_REX_OR_TEMPORAL, inputs=(_C1_COCHAIN,),
    result=_t("HodgeSplit", ValueKind.HODGE_SPLIT, grade=1, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE),
    implementation_key="rex.hodge",
))
register(OperatorSignature(
    name="HARMONIC", source_kind=_REX_OR_TEMPORAL, inputs=(_C1_COCHAIN,),
    result=_t("Cochain", ValueKind.COCHAIN, grade=1, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE),
    implementation_key="rex.harmonic",
))
register(OperatorSignature(
    name="GREEN", source_kind=ValueKind.REX, inputs=(_OPTIONAL_C0_COCHAIN,),
    result=_green_result, implementation_key="rex.green",
    preconditions=("the unapplied Green action is structural; applying it is numerical",),
))
register(OperatorSignature(
    name="APPLY", source_kind=ValueKind.REX, inputs=(_ACTION, _BOUND_COCHAIN),
    result=_apply_result, implementation_key="rex.green.apply",
))
register(OperatorSignature(
    name="QUADRANCE", source_kind=_REX_OR_TEMPORAL, inputs=(_CHAIN_OR_COCHAIN, _BOOL),
    result=lambda args: _geometry_result(args, name="QUADRANCE"),
    implementation_key="rex.quadrance",
))
register(OperatorSignature(
    name="SPREAD", source_kind=ValueKind.REX,
    inputs=(_CHAIN_OR_COCHAIN, _CHAIN_OR_COCHAIN, _BOOL), result=_spread_result,
    implementation_key="rex.spread",
))
register(OperatorSignature(
    name="ACCUMULATE", source_kind=_REX_OR_TEMPORAL,
    inputs=(_ACCUMULATABLE, _ACCUMULATABLE), result=_accumulate_result,
    implementation_key="rex.accumulate",
    preconditions=(
        "combines only source, basis, and temporal-state aligned coefficient carriers",
        "cross-time accumulation requires an explicit transport or alignment action",
    ),
))
register(OperatorSignature(
    name="HODGE_COORDS", source_kind=_REX_OR_TEMPORAL, inputs=(_C1_COCHAIN,),
    result=_t("HodgeCoordinates", ValueKind.HODGE_COORDINATES, grade=1,
              variance=Variance.COCHAIN, domain=Domain.REAL,
              exactness=Exactness.APPROXIMATE), implementation_key="rex.hodge_coordinates",
))
register(OperatorSignature(
    name="WINDING", source_kind=_REX_OR_TEMPORAL, inputs=(_C1_COCHAIN,),
    result=_t("Winding", ValueKind.WINDING, grade=1, variance=Variance.COCHAIN,
              domain=Domain.REAL, exactness=Exactness.APPROXIMATE),
    implementation_key="rex.winding",
))
register(OperatorSignature(
    name="CLOSURE", source_kind=ValueKind.REX,
    inputs=(TypePattern("C0 seed", literal=int),
            TypePattern("maximum depth", literal=int, optional=True),
            TypePattern("grade", literal=int, optional=True)),
    result=_closure_result, implementation_key="rex.closure",
))
register(OperatorSignature(
    name="SIGNIFICANCE", source_kind=ValueKind.REX, inputs=(_INDEX,),
    result=_t("Real", ValueKind.REAL, domain=Domain.REAL,
              exactness=Exactness.APPROXIMATE), implementation_key="rex.significance",
))
register(OperatorSignature(
    name="CHARACTER", source_kind=ValueKind.REX, inputs=(_BOOL,),
    result=_character_result, implementation_key="rex.character",
))
register(OperatorSignature(
    name="ZERO", source_kind=ValueKind.REX,
    inputs=(_GRADE, TypePattern("kind", literal=str, optional=True)),
    result=_zero_result, implementation_key="rex.zero",
))

# Registered, catalogued, and impossible. RCDB_STATE_HASH calls source.state_digest(),
# which no RCStore defines and none of the nine registered backends provides, so it raises
# for every possible source rather than for a mistyped one. Declaring it unreachable lets
# the binder refuse it with a reason instead of letting an adapter raise a TypeError that
# reads like a caller error.
register(OperatorSignature(
    name="RCDB_STATE_HASH", source_kind=ValueKind.REX, inputs=(),
    result=_t("Digest", ValueKind.DIGEST, domain=Domain.BYTES,
              exactness=Exactness.STRUCTURAL),
    implementation_key="rcdb.state_digest",
    unreachable=(
        "no RCStore implements state_digest(), so this operator cannot execute against "
        "any registered backend; bind it to a real store-level digest before enabling it"
    ),
))
