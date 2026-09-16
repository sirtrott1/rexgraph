"""RCQL value types.

An RCQL type is not a Python class name. It is the answer to several independent
questions about one value, and the value is only usable where every answer agrees with
the operator receiving it:

    what kind of thing is it          ValueKind
    at which grade                    grade
    with which variance               Variance, chains and cochains are not interchangeable
    in whose basis                    BasisRef, equal length is not equal meaning
    over which coefficients           Domain
    under which contract              Exactness, an exact value and a rounded one differ
    read from which source and when   SourceRef, TemporalRef
    requiring what, causing what      capabilities, effects

The first three fields are ``name``, ``grade`` and ``exactness`` in that order, because
that was the whole of RCType before this and the extension has to be compatible rather
than a replacement. Everything added is optional and defaults to unknown.

Unknown is a real answer here and is distinct from a claim. ``exactness=None`` means the
contract has not been established, which is what the current post execution dtype
inspection produces; it must not be read as APPROXIMATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum


@dataclass(frozen=True)
class PredicateResult:
    """A checked condition, not a prose precondition or a numerical tolerance."""

    name: str
    status: str
    evidence: str


@dataclass(frozen=True)
class MetricDescriptor:
    """A contraction form, distinct from a channel scale or a type accession."""

    construction: str
    basis: BasisRef
    shape: tuple[int | None, int | None]
    coefficient_domain: Domain
    arithmetic: Exactness
    positive_definite: bool | None = None
    coefficient_digest: str | None = None


@dataclass(frozen=True)
class CoordinateDescriptor:
    """Explicit type coordinates, distinct from an equally sized cell basis."""

    name: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class AccessionDescriptor:
    """A type axis and declared measurement map, separate from the ambient basis."""

    name: str
    basis: BasisRef
    shape: tuple[int, int]
    coefficient_domain: Domain
    coefficient_digest: str
    nnz: int
    construction: str = "sparse-ambient-endomorphism"
    chain_preserving: bool | None = None
    coordinates: CoordinateDescriptor | None = None


@dataclass(frozen=True)
class CrossMetricDescriptor:
    """Ordered sparse cross block; no symmetry or positivity is asserted."""

    left: AccessionDescriptor
    right: AccessionDescriptor
    shape: tuple[int, int]
    coefficient_domain: Domain
    arithmetic: Exactness
    coefficient_digest: str
    nnz: int
    construction: str = "sparse-cross-pairing"


@dataclass(frozen=True)
class RealizationDescriptor:
    """Sparse map from accession coordinates into its declared ambient basis."""

    accession: AccessionDescriptor
    shape: tuple[int, int]
    coefficient_domain: Domain
    coefficient_digest: str
    nnz: int
    construction: str = "sparse-type-realization"
    injective: bool | None = None
    chain_preserving: bool | None = None


@dataclass(frozen=True)
class FamilyMetricDescriptor:
    """PSD direct sum form represented by E* M E, not stored cross blocks."""

    metric: MetricDescriptor
    realizations: tuple[RealizationDescriptor, ...]
    shape: tuple[int, int]
    coefficient_domain: Domain
    arithmetic: Exactness
    coefficient_digest: str
    construction: str = "factored-family-metric"
    psd: bool = True
    positive_definite: bool | None = None


@dataclass(frozen=True)
class GradedMapDescriptor:
    """Whole finite tower, with ordered spaces and boundary state digests."""

    domain: tuple[CoordinateDescriptor, ...]
    codomain: tuple[CoordinateDescriptor, ...]
    shapes: tuple[tuple[int, int], ...]
    domain_digest: str
    codomain_digest: str
    coefficient_digest: str
    nnz: tuple[int, ...]
    construction: str = "exact-sparse-graded-map"
    chain_preserving: bool | None = None


@dataclass(frozen=True)
class OperatorDescriptor:
    """The two spaces of a map, independent of its display/type name.

    Euclidean actions and explicit diagonal metric adjoints retain separate
    contracts. Metrics/accessions are never inferred from array shapes.
    ``arithmetic`` describes default application, not the structural handle;
    exact_action/transpose advertise separate certified Q capabilities.
    """

    construction: str
    domain: BasisRef
    codomain: BasisRef
    shape: tuple[int | None, int | None]
    coefficient_domain: Domain
    arithmetic: Exactness
    metric: str = "euclidean-cell"
    symmetric: bool | None = None
    psd: bool | None = None
    kernel_policy: str | None = None
    parameters: tuple[tuple[str, object], ...] = ()
    transpose_available: bool = False
    exact_action: bool = False
    exact_transpose: bool = False
    action_variance: str = "cochain"
    adjoint_domain_metric: MetricDescriptor | None = None
    adjoint_codomain_metric: MetricDescriptor | None = None
    grade_metrics: tuple[MetricDescriptor, ...] = ()
    metric_self_adjoint: bool | None = None
    metric_psd: bool | None = None
    primal_operator: OperatorDescriptor | None = None
    operands: tuple[OperatorDescriptor, ...] = ()
    metric_skew_adjoint: bool | None = None
    euclidean_skew_adjoint: bool | None = None


@dataclass(frozen=True)
class GradedOperatorDescriptor:
    """Direct sum spaces stay graded; total shape is not a single cell basis."""

    construction: str
    bases: tuple[BasisRef, ...]
    sizes: tuple[int, ...]
    grade_metrics: tuple[MetricDescriptor, ...]
    anti: bool | None
    active_boundaries: tuple[int, ...]
    exact_action: bool
    coefficient_domain: Domain
    arithmetic: Exactness
    action_variance: str = "chain"
    transpose_available: bool = True
    metric_self_adjoint: bool | None = None
    metric_skew_adjoint: bool | None = None
    symmetric: bool = False
    psd: bool = False
    operands: tuple[GradedOperatorDescriptor, ...] = ()
    bracket_kind: str | None = None


class Exactness(str, Enum):
    """The contract a value's arithmetic was produced under.

    These names are unchanged. INTEGER and RATIONAL assert a certified construction, not
    that a float happens to hold a whole number. ROUNDED names a specified rounding rule.
    APPROXIMATE is a floating action or numerical solve and carries method and residual
    where applicable. ANALYTIC is an unevaluated exact formula. STRUCTURAL is a certified
    symbolic fact or descriptor, and is not a precision claim about a scalar.
    """

    INTEGER = "integer"
    RATIONAL = "rational"
    ROUNDED = "rounded"
    APPROXIMATE = "approximate"
    ANALYTIC = "analytic"
    STRUCTURAL = "structural"


class Variance(str, Enum):
    """Chains and cochains are dual, not two spellings of an array.

    A raw array has NEUTRAL variance: it has neither, so an operator may not silently
    reclassify it. CELL is an addressing value rather than a coefficient tensor.
    """

    CHAIN = "chain"
    COCHAIN = "cochain"
    CELL = "cell"
    NEUTRAL = "neutral"


class Domain(str, Enum):
    """The coefficient system, which is a different question from the exactness contract.

    An integer domain computed by a floating action is INTEGER over APPROXIMATE, and a
    rational value rendered to a float is RATIONAL over ROUNDED. Collapsing the two loses
    exactly the distinction the typed contract asks for.
    """

    INTEGER = "integer"
    RATIONAL = "rational"
    REAL = "real"
    COMPLEX = "complex"
    SYMBOLIC = "symbolic"
    BYTES = "bytes"
    METADATA = "metadata"


class Effect(str, Enum):
    """What evaluating a value does besides producing it."""

    READ = "read"
    MUTATE = "mutate"
    SIGN = "sign"
    PUBLISH = "publish"
    FILESYSTEM = "filesystem"


class ValueKind(str, Enum):
    """The families in the native value schema, plus the metadata readings that exist today.

    A name here is a language slot. Its presence is not permission to invent a formula for
    it: an operator becomes executable only when current source and its direct tests
    support the contract.
    """

    # scalars
    BOOLEAN = "Boolean"
    EXACT_INTEGER = "ExactInteger"
    EXACT_RATIONAL = "ExactRational"
    REAL = "Real"
    COMPLEX = "Complex"
    TEXT = "Text"
    BYTES = "Bytes"

    # cells
    CELL = "Cell"
    CELL_SET = "CellSet"
    CELL_PATTERN = "GradedCellPattern"
    CELL_BOUNDARY = "CellBoundary"
    CELL_COBOUNDARY = "CellCoboundary"
    COMPOSITE_BINARY = "CompositeBinary"
    HYPERSLICE = "Hyperslice"
    COLUMN_EXPANSION = "ColumnExpansion"
    COLUMN_LEGS = "ColumnLegs"
    PRIMARY_COLUMN_LIFT = "PrimaryColumnLift"
    REX_PARTITION = "RexPartition"
    BOUNDARY_DIFFERENCE = "BoundaryDifference"
    ARTIFACT_BYTES = "ArtifactBytes"

    # algebra
    CHAIN = "Chain"
    GRADED_CHAIN = "GradedChain"
    GRADED_OPERATOR = "GradedOperator"
    COCHAIN = "Cochain"
    FIELD = "Field"
    OPERATOR = "Operator"
    GRAM = "Gram"
    METRIC = "Metric"
    TYPE_ACCESSION = "TypeAccession"
    CROSS_METRIC = "CrossMetric"
    FAMILY_METRIC = "FamilyMetric"
    GRADED_MAP = "GradedMap"
    CHAIN_MAP = "ChainMap"
    CHAIN_HOMOTOPY = "ChainHomotopy"
    ACCESSION_FAMILY = "AccessionFamily"
    TYPE_VIEW = "TypeView"
    TYPED_FAMILY = "TypedFamily"
    MOMENT_TENSOR = "TypedMomentTensor"
    EXACT_SHEAF = "ExactSheaf"
    EXACT_GLUE = "ExactGlueResult"
    EXACT_SECTION_CHECK = "ExactSectionCheck"

    # structure
    REX = "Rex"
    TEMPORAL_REX = "TemporalRex"
    TURN_FIELD = "TurnFieldSource"
    DELTA = "Delta"
    TEMPORAL_EVENT = "TemporalSignalEvent"
    SIGNAL_FLOW = "TemporalSignalFlow"
    METRIC_CURVATURE = "MetricCurvature"
    PARTITION = "Partition"
    SUBCOMPLEX = "Subcomplex"

    # hodge, green, character
    HODGE_SPLIT = "HodgeSplit"
    HODGE_COORDINATES = "HodgeCoordinates"
    HARMONIC_SPACE = "HarmonicSpace"
    GREEN_ACTION = "GreenAction"
    CHARACTER = "Character"
    WINDING = "Winding"

    # persistence and transport
    RECORD = "Record"
    RECORD_SET = "RecordSet"
    RCDB_STORE = "RCDBStore"
    COMMIT = "Commit"
    COMMIT_LINK = "CommitLink"
    HISTORY = "History"
    ARTIFACT = "Artifact"
    MODEL_STATE = "ModelState"
    DIGEST = "Digest"
    STORE_STATS = "StoreStats"
    SECURITY_STATUS = "SecurityStatus"

    # file catalog readings
    CATALOG_ENTRY = "CatalogEntry"
    CATALOG_ENTRY_SET = "CatalogEntrySet"
    TENSOR_MANIFEST = "TensorManifest"

    # compatibility and rendering
    QUERY_TABLE = "QueryTable"
    SEQUENCE = "Sequence"
    STRUCTURAL_DESCRIPTION = "StructuralDescription"
    OPERATOR_SIGNATURE_SET = "OperatorSignatureSet"

    # the absence of a determined kind, which is not a kind
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class BasisRef:
    """Which ordered cell basis a coefficient tensor is expressed in.

    Two arrays of equal length from different sources, grades, or orderings are not
    interchangeable, so combining unaligned bases is a type error rather than a reshape.
    """

    source_id: str
    grade: int
    ordering: str | tuple[object, ...] = "canonical"


@dataclass(frozen=True)
class SourceRef:
    """Which bound source a value came from, and in which state."""

    name: str
    state_digest: str | None = None
    policy_digest: str | None = None
    record_id: str | None = None
    record_version: int | None = None
    record_as_of: float | None = None
    record_valid_at: float | None = None
    contributors: tuple[SourceRef, ...] = ()


@dataclass(frozen=True)
class TemporalRef:
    """Which version or time a source was read at.

    Present on every value read from a temporal or RCDB source, so EXPLAIN cannot imply a
    field was derived from a state other than the one actually read.
    """

    version: int | None = None
    as_of: float | None = None
    valid_at: float | None = None


@dataclass(frozen=True)
class ShapeRef:
    """Declared shape, where a value has one. ``dims`` of None means unconstrained."""

    dims: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class RCType:
    """The RCQL type of one value.

    ``name`` remains the first field and stays free text so existing constructions keep
    working. ``kind`` is the classified form; where both are present, ``kind`` is what
    inference and signatures read.
    """

    name: str
    grade: int | None = None
    exactness: Exactness | None = None
    kind: ValueKind = ValueKind.UNKNOWN
    variance: Variance | None = None
    domain: Domain | None = None
    basis: BasisRef | None = None
    source: SourceRef | None = None
    temporal: TemporalRef | None = None
    shape: ShapeRef | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    effects: frozenset[Effect] = field(default_factory=frozenset)
    operator: OperatorDescriptor | None = None
    metric: MetricDescriptor | None = None
    accessions: tuple[AccessionDescriptor, ...] = ()
    cross_metric: CrossMetricDescriptor | None = None
    member_shapes: tuple[tuple[int | None, ...], ...] = ()
    family_metric: FamilyMetricDescriptor | None = None
    graded_map: GradedMapDescriptor | None = None
    graded_operator: GradedOperatorDescriptor | None = None
    graded_bases: tuple[BasisRef, ...] = ()

    def with_(self, **changes) -> RCType:
        """A copy with fields replaced, since a type is immutable once inferred."""
        return replace(self, **changes)

    def is_determined(self) -> bool:
        """Whether this type says anything a signature can check."""
        return self.kind is not ValueKind.UNKNOWN

    def same_space(self, other: RCType) -> bool:
        """Whether two values may be combined without an explicit alignment.

        Kind, grade, variance, source state, and temporal state must agree, and where both
        declare a basis the bases must be identical. A missing basis on either side is
        unknown rather than compatible, so it does not license the combination on its own;
        the caller decides whether an undetermined operand is acceptable.
        """
        if self.kind is not other.kind or self.grade != other.grade:
            return False
        if self.variance is not other.variance:
            return False
        if self.source != other.source:
            return False
        if self.temporal != other.temporal:
            return False
        if self.basis is not None and other.basis is not None:
            return self.basis == other.basis
        return True


# Compatibility constants. These predate the extension and are constructed positionally,
# which is why name, grade and exactness had to keep their order.
INTEGER = RCType("Integer", exactness=Exactness.INTEGER,
                 kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER)
RATIONAL = RCType("Rational", exactness=Exactness.RATIONAL,
                  kind=ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL)
BOOLEAN = RCType("Boolean", exactness=Exactness.STRUCTURAL,
                 kind=ValueKind.BOOLEAN, domain=Domain.METADATA)
REX = RCType("Rex", kind=ValueKind.REX, domain=Domain.METADATA)
TENSOR = RCType("Tensor", kind=ValueKind.UNKNOWN, variance=Variance.NEUTRAL)

UNKNOWN = RCType("Unknown")
