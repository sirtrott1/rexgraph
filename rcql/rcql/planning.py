"""Whole expression static plans for RCQL.

The parser currently supplies a compact call tree, but a relational tensor phrase has
meaning only after all of its modifiers and nested readings are considered together.
This module composes the existing operator signatures across that tree before an adapter
executes.  It deliberately does not choose a new query spelling: it is the semantic seam
on which a later phrase grammar, modifier syntax, and optimizer can depend.

Every planned call retains the source, grade, basis, coefficient domain, exactness, and
temporal state established by inference.  A literal remains a literal; a supplied
``RCType`` parameter remains an explicitly declared carrier rather than being guessed
from array length or a Python class.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from numbers import Integral

from .ast import (
    Alias,
    Call,
    Expr,
    LetBinding,
    ListExpr,
    Literal,
    Member,
    Parameter,
    Query,
    Reference,
    StructuralEdit,
    Comparison,
)
from .binding import Binding
from .inference import TypedCall, infer
from .types import (
    BasisRef,
    Domain,
    Exactness,
    OperatorDescriptor,
    PredicateResult,
    RCType,
    ShapeRef,
    SourceRef,
    ValueKind,
    Variance,
)

__all__ = ["PlannedExpression", "QueryPlan", "plan_query"]


def _plain_type(value: RCType) -> dict[str, object]:
    """Render one declared carrier without exposing a live source value."""
    result = {
        "kind": value.kind.value,
        "grade": value.grade,
        "variance": None if value.variance is None else value.variance.value,
        "domain": None if value.domain is None else value.domain.value,
        "exactness": None if value.exactness is None else value.exactness.value,
        "shape": None if value.shape is None else value.shape.dims,
        "operator": None if value.operator is None else asdict(value.operator),
        "metric": None if value.metric is None else asdict(value.metric),
        "accessions": [asdict(a) for a in value.accessions],
        "cross_metric": None if value.cross_metric is None else asdict(value.cross_metric),
        "family_metric": None if value.family_metric is None else asdict(value.family_metric),
        "graded_map": None if value.graded_map is None else asdict(value.graded_map),
        "graded_operator": None if value.graded_operator is None else asdict(value.graded_operator),
        "graded_bases": [asdict(b) for b in value.graded_bases],
        "member_shapes": [list(s) for s in value.member_shapes],
        "source": None if value.source is None else value.source.name,
        "temporal": None if value.temporal is None else {
            "version": value.temporal.version,
            "as_of": value.temporal.as_of,
            "valid_at": value.temporal.valid_at,
        },
        "basis": None if value.basis is None else {
            "source_id": value.basis.source_id,
            "grade": value.basis.grade,
            "ordering": value.basis.ordering,
        },
    }

    if value.tensor_axes is not None:
        result["tensor_axes"] = [asdict(axis) for axis in value.tensor_axes]
    if value.program_outputs is not None:
        result["program_outputs"] = [_plain_type(output) for output in value.program_outputs]
    if value.coordinate_action is not None:
        result["coordinate_action"] = asdict(value.coordinate_action)
    if value.coordinates is not None:
        result["coordinates"] = asdict(value.coordinates)
    if value.declaration_digest is not None:
        result["declaration_digest"] = value.declaration_digest
    return result


def _plain_source(ref: SourceRef) -> dict[str, object]:
    """Render source provenance without exposing its live store or carrier."""
    rendered = {
        "name": ref.name,
        "state_digest": ref.state_digest,
        "record_id": ref.record_id,
        "record_version": ref.record_version,
        "record_as_of": ref.record_as_of,
        "record_valid_at": ref.record_valid_at,
    }
    if ref.contributors:
        rendered["contributors"] = [_plain_source(item) for item in ref.contributors]
    return rendered


def _plain_literal(value: object) -> object:
    """Keep a plan renderable without leaking a live Python object."""
    if isinstance(value, RCType):
        return {"type": _plain_type(value)}
    if isinstance(value, Fraction):
        return {"rational": {"numerator": value.numerator, "denominator": value.denominator}}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain_literal(item) for item in value]
    return {"python_type": type(value).__name__}


def _coefficient_contract(value: object) -> tuple[Domain, Exactness]:
    """Classify an already declared carrier without running a tensor action.

    A typed Chain/Cochain is a legitimate RCQL literal.  Its coefficient array
    still matters to the phrase contract: an integer metric can retain rational
    curvature, while a measured float must not acquire an exact label merely
    because it happened to be integral in one observation.  This only reads
    the array protocol; it never calls an adapter or reconstructs a source.
    """
    dtype = getattr(value, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if kind in {"i", "u"}:
        return Domain.INTEGER, Exactness.INTEGER
    if kind == "O":
        # A native object carrier has no coefficient domain tag. Certifying one
        # explicit literal is necessary to distinguish a real Fraction tensor
        # from arbitrary Python objects before an exact functional is admitted.
        # This is O(nnz) only for externally supplied object literals; nested
        # RCQL structural operations already declare their exact contract and
        # never take this path.
        flat = getattr(value, "flat", None)
        seen, fractions, integers = False, True, True
        if flat is not None:
            for item in flat:
                seen = True
                fractions &= isinstance(item, (Fraction, Integral)) and not isinstance(item, bool)
                integers &= isinstance(item, Integral) and not isinstance(item, bool)
        if seen and integers:
            return Domain.INTEGER, Exactness.INTEGER
        if seen and fractions:
            return Domain.RATIONAL, Exactness.RATIONAL
        if flat is not None and not seen:
            # The empty coefficient tensor is the exact zero carrier; there is no
            # unexamined symbolic entry that could invalidate its rational action.
            return Domain.RATIONAL, Exactness.RATIONAL
        return Domain.SYMBOLIC, Exactness.STRUCTURAL
    if kind == "c":
        return Domain.COMPLEX, Exactness.APPROXIMATE
    return Domain.REAL, Exactness.APPROXIMATE


def _carrier_literal(binding: Binding, value: object) -> RCType | tuple | None:
    """Turn a core typed carrier literal into its non executing RCQL contract.

    RCQL accepts native typed values as expression literals for programmatic
    callers.  Planning must preserve their grade and variance rather than
    treating them as opaque Python objects, or ordinary execution would bypass
    the same source/basis checks that protect nested phrase results.
    """
    from .transformation_contracts import literal_type as transformation_literal
    transformation = transformation_literal(value)
    if transformation is not None:
        return transformation
    from .recursion_contracts import literal_type as relation_literal
    declared_relation = relation_literal(binding, value)
    if declared_relation is not None:
        return declared_relation
    from .program import Program, ProgramResult
    from .program_family import ProgramAssembly, ProgramFamily
    from .program_evolution import ProgramEvolution
    if isinstance(value, ProgramEvolution):
        return RCType("ProgramEvolution", kind=ValueKind.PROGRAM_EVOLUTION, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest)
    if isinstance(value, (ProgramAssembly, ProgramFamily)):
        kind = ValueKind.PROGRAM_ASSEMBLY if isinstance(value, ProgramAssembly) else ValueKind.PROGRAM_FAMILY
        return RCType(kind.value, kind=kind, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest)
    if isinstance(value, Program):
        return RCType("Program", kind=ValueKind.PROGRAM, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest, program_declaration=value.to_bytes())
    if isinstance(value, ProgramResult):
        from .program_contracts import result_types
        return RCType("ProgramResult", kind=ValueKind.PROGRAM_RESULT, exactness=Exactness.STRUCTURAL,
                      program_outputs=result_types(binding, value.values))
    if isinstance(value, (tuple, list)):
        return tuple(_carrier_literal(binding, item) or item for item in value)
    if value is None or isinstance(value, (RCType, bool, int, float, complex, str, bytes)):
        return None
    from .artifact_services import ArtifactServices
    if isinstance(value, ArtifactServices):
        raise TypeError("ArtifactServices belongs in Executor configuration, not query values")
    from rexgraph.cells import Cell, CellSet, GradedCellPattern
    if isinstance(value, GradedCellPattern):
        ref = binding.ref if value.source is binding.value else SourceRef("foreign")
        return RCType("GradedCellPattern", kind=ValueKind.CELL_PATTERN, domain=Domain.METADATA,
                      exactness=Exactness.STRUCTURAL, source=ref, temporal=binding.temporal)
    from rexgraph.io.partition_state import RexPartition
    from rexgraph.boundary_difference import BoundaryDifference
    if isinstance(value, BoundaryDifference):
        value.check_state()
        if value.reference is not binding.value:
            raise ValueError("difference requires its original reference source")
        return RCType("BoundaryDifference", kind=ValueKind.BOUNDARY_DIFFERENCE,
                      domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL, source=binding.ref)
    if isinstance(value, RexPartition):
        from .partition_contracts import partition_result
        value.check_state()
        if value.state.source_state != binding.ref.state_digest:
            raise ValueError("partition lineage differs from the bound source state")
        return partition_result().with_(source=binding.ref, temporal=binding.temporal)
    from rexgraph.cell_neighborhood import Hyperslice
    from rexgraph.column_expansion import ColumnExpansion, ColumnLegs, PrimaryBoundary, PrimaryColumnLift
    if isinstance(value, Hyperslice):
        return _carrier_literal(binding, value.cell).with_(name="Hyperslice", kind=ValueKind.HYPERSLICE)
    if isinstance(value, (ColumnExpansion, ColumnLegs, PrimaryColumnLift, PrimaryBoundary)):
        from .structure_contracts import expansion_result, lift_result
        expansion = value if isinstance(value, ColumnExpansion) else value.expansion
        expansion.check_state()
        desc = expansion_result((_carrier_literal(binding, expansion.boundary),))
        if isinstance(value, PrimaryBoundary):
            return lift_result((desc, desc))
        kind = (ValueKind.COLUMN_EXPANSION if isinstance(value, ColumnExpansion) else
                ValueKind.COLUMN_LEGS if isinstance(value, ColumnLegs) else ValueKind.PRIMARY_COLUMN_LIFT)
        return desc.with_(name=kind.value, kind=kind)
    from rexgraph.temporal_signal import TemporalSignal
    if isinstance(value, TemporalSignal):
        from .signatures import _temporal_delta_result
        ref = binding.ref if value.source is binding.value else SourceRef("foreign")
        return _temporal_delta_result((value.step,)).with_(source=ref, basis=BasisRef(ref.name, 1))
    if isinstance(value, (Cell, CellSet)):
        ref = binding.ref if value.source is binding.value else SourceRef("foreign")
        kind = ValueKind.CELL if isinstance(value, Cell) else ValueKind.CELL_SET
        return RCType(kind.value, kind=kind, grade=value.grade, variance=Variance.CELL,
                      domain=Domain.METADATA, exactness=Exactness.STRUCTURAL, source=ref,
                      basis=BasisRef(ref.name, value.grade), temporal=binding.temporal)
    from .calculus_operators import ComposedAction
    if isinstance(value, ComposedAction):
        from .calculus_contracts import chain_result, strain_result
        if hasattr(value, "strain_weight"):
            result = strain_result((value.strain_grade, _carrier_literal(binding, value.strain_weight)))
            return result.with_(shape=ShapeRef(value.shape), operator=replace(result.operator, shape=value.shape,
                                exact_action=value.exact_matvec is not None,
                                exact_transpose=value.exact_transpose_matvec is not None))
        return chain_result((tuple(_carrier_literal(binding, f) for f in value.factors),))
    from rexgraph.operator_bracket import GradedOperatorBracket, OperatorBracket
    if isinstance(value, (OperatorBracket, GradedOperatorBracket)):
        from .operator_algebra import bracket_result
        return bracket_result((_carrier_literal(binding, value.left), _carrier_literal(binding, value.right)), anti=value.anti)
    from rexgraph.weighted_dirac import GradedChain, WeightedDiracOperator
    if isinstance(value, (GradedChain, WeightedDiracOperator)):
        ref = binding.ref if value.source is binding.value else SourceRef("foreign")
        if isinstance(value, GradedChain):
            domain, arithmetic = _coefficient_contract(value.components[0].values)
            return RCType("GradedChain", kind=ValueKind.GRADED_CHAIN, variance=Variance.CHAIN,
                          domain=domain, exactness=arithmetic, source=ref, temporal=binding.temporal,
                          graded_bases=tuple(BasisRef(ref.name, k) for k in range(len(value.sizes))),
                          member_shapes=tuple(c.values.shape for c in value.components))
        from .graded_calculus import dirac_descriptor
        from .validation import metric_descriptor
        desc = dirac_descriptor(ref, value.sizes, tuple(metric_descriptor(ref, m) for m in value.grade_metrics),
                                value.anti, value.active_boundaries, value.exact)
        return RCType("GradedOperator", kind=ValueKind.GRADED_OPERATOR, domain=Domain.METADATA,
                      exactness=Exactness.STRUCTURAL, source=ref, temporal=binding.temporal,
                      graded_operator=desc, shape=ShapeRef(value.shape))
    from rexgraph.chain_map import ChainHomotopy, ChainMap, GradedMap, SymmetryGroup
    if isinstance(value, SymmetryGroup):
        from .symmetry_contracts import result_type
        value.check_state()
        if value.domain.source is not binding.value:
            raise ValueError("symmetry group requires its bound source")
        return result_type().with_(source=binding.ref, temporal=binding.temporal)
    from rexgraph.cochain import Chain, Cochain, Field
    from rexgraph.graded_metric import DiagonalMetric
    from rexgraph.green import GreenOperator
    from rexgraph.linear_operator import RexOperator
    from rexgraph.type_accession import (
        AccessionFamily,
        CrossMetric,
        FamilyMetric,
        TypeAccession,
        TypedFamily,
        TypeView,
    )

    from .model_contracts import model_literal
    model_type = model_literal(binding, value)
    if model_type is not None:
        return model_type
    from .section_contracts import section_literal
    section_type = section_literal(binding, value)
    if section_type is not None:
        return section_type
    from .tensor_contracts import tensor_literal
    tensor_type = tensor_literal(binding, value)
    if tensor_type is not None:
        return tensor_type
    from .coordinate_contracts import coordinate_literal
    coordinate = coordinate_literal(binding, value)
    if coordinate is not None:
        return coordinate

    if isinstance(value, ChainHomotopy):
        value.check_state()
        ref = binding.ref if value.left.declaration.domain.source is binding.value else SourceRef("foreign")
        return RCType("ChainHomotopy", kind=ValueKind.CHAIN_HOMOTOPY, domain=Domain.RATIONAL,
                      exactness=Exactness.STRUCTURAL, source=ref, temporal=binding.temporal)
    if isinstance(value, (GradedMap, ChainMap)):
        from .validation import graded_map_descriptor
        declaration = value.declaration if isinstance(value, ChainMap) else value
        ref = binding.ref if declaration.domain.source is binding.value else SourceRef("foreign")
        kind = ValueKind.CHAIN_MAP if isinstance(value, ChainMap) else ValueKind.GRADED_MAP
        return RCType(kind.value, kind=kind, domain=Domain.RATIONAL,
                      exactness=Exactness.STRUCTURAL, source=ref,
                      graded_map=graded_map_descriptor(declaration))

    if isinstance(value, FamilyMetric):
        from .types import FamilyMetricDescriptor, RealizationDescriptor
        from .validation import metric_descriptor
        ref = binding.ref if value.metric.source is binding.value else SourceRef("foreign")
        base = metric_descriptor(ref, value.metric)
        realizations = tuple(RealizationDescriptor(
            _carrier_literal(binding, e.accession).accessions[0], e.shape,
            Domain.RATIONAL if e.exact else Domain.REAL, e.coefficient_digest, len(e.entries))
            for e in value.realizations)
        domain = Domain.RATIONAL if value.exact else Domain.REAL
        desc = FamilyMetricDescriptor(base, realizations, value.shape, domain,
            Exactness.RATIONAL if value.exact else Exactness.APPROXIMATE, value.coefficient_digest)
        return RCType("FamilyMetric", grade=value.metric.grade, kind=ValueKind.FAMILY_METRIC,
            domain=domain, exactness=Exactness.STRUCTURAL, basis=base.basis, source=ref,
            shape=ShapeRef(value.shape), accessions=tuple(e.accession for e in realizations), family_metric=desc)

    if isinstance(value, (TypeAccession, AccessionFamily, TypeView, TypedFamily, CrossMetric)):
        from .types import AccessionDescriptor, CoordinateDescriptor, CrossMetricDescriptor
        views = value.views if isinstance(value, TypedFamily) else (value,) if isinstance(value, TypeView) else ()
        maps = (value.left, value.right) if isinstance(value, CrossMetric) else (
            tuple(v.accession for v in views) if views else (
                value.accessions if isinstance(value, AccessionFamily) else (value,)))
        first = maps[0]
        ref = binding.ref if first.source is binding.value else SourceRef("foreign")
        basis = BasisRef(ref.name, first.grade, "canonical" if first.cell_keys is None else first.cell_keys)
        descriptors = tuple(AccessionDescriptor(a.name, basis, a.shape,
            Domain.RATIONAL if a.exact else Domain.REAL, a.coefficient_digest, len(a.entries),
            construction="sparse-ambient-endomorphism" if a.coordinates is None else "sparse-type-coordinates",
            coordinates=None if a.coordinates is None else CoordinateDescriptor(a.coordinates.name, a.coordinates.keys))
            for a in maps)
        if isinstance(value, CrossMetric):
            domain = Domain.RATIONAL if value.exact else Domain.REAL
            desc = CrossMetricDescriptor(*descriptors, value.shape, domain,
                Exactness.RATIONAL if value.exact else Exactness.APPROXIMATE, value.coefficient_digest, len(value.entries))
            return RCType("CrossMetric", grade=first.grade, kind=ValueKind.CROSS_METRIC,
                domain=domain, exactness=Exactness.STRUCTURAL, basis=basis, source=ref,
                shape=ShapeRef(value.shape), accessions=descriptors, cross_metric=desc)
        member_shapes = ()
        if views:
            contracts = [_coefficient_contract(v.values) for v in views]
            domains = (Domain.COMPLEX, Domain.REAL, Domain.RATIONAL, Domain.INTEGER)
            domain = (next(d for d in domains if any(c[0] is d for c in contracts))
                      if all(c[0] in domains for c in contracts) else Domain.SYMBOLIC)
            arithmetic = Exactness.APPROXIMATE if domain in {Domain.REAL, Domain.COMPLEX} else (
                Exactness.RATIONAL if domain is Domain.RATIONAL else
                Exactness.INTEGER if domain is Domain.INTEGER else Exactness.STRUCTURAL)
            variance = Variance(views[0].variance)
            shape = views[0].values.shape
            kind = ValueKind.TYPED_FAMILY if isinstance(value, TypedFamily) else ValueKind.TYPE_VIEW
            if kind is ValueKind.TYPED_FAMILY:
                member_shapes = tuple(v.values.shape for v in views)
                rows = shape[0] if all(s[0] == shape[0] for s in member_shapes) else None
                shape = (len(views), rows, *shape[1:])
        else:
            domain = Domain.RATIONAL if all(a.exact for a in maps) else Domain.REAL
            arithmetic, variance = Exactness.STRUCTURAL, None
            kind = ValueKind.ACCESSION_FAMILY if isinstance(value, AccessionFamily) else ValueKind.TYPE_ACCESSION
            shape = (len(maps),) if kind is ValueKind.ACCESSION_FAMILY else first.shape
        return RCType(kind.value, grade=first.grade, kind=kind, domain=domain,
                      exactness=arithmetic, variance=variance, basis=basis, source=ref,
                      shape=ShapeRef(tuple(shape)), accessions=descriptors, member_shapes=member_shapes)

    if isinstance(value, DiagonalMetric):
        from .validation import metric_descriptor
        ref = binding.ref if value.source is binding.value else SourceRef("foreign")
        desc = metric_descriptor(ref, value)
        return RCType("Metric", grade=value.grade, kind=ValueKind.METRIC,
                      domain=desc.coefficient_domain, exactness=Exactness.STRUCTURAL,
                      source=ref, basis=desc.basis, shape=ShapeRef(desc.shape), metric=desc)

    if isinstance(value, (RexOperator, GreenOperator)):
        from rexgraph.adjugate_operator import AdjugateOperator
        if isinstance(value, AdjugateOperator):
            from .certificate_contracts import adjugate_result
            return adjugate_result((_carrier_literal(binding, value.primal),))
        from rexgraph.channel_operator import ChannelOperator
        green = isinstance(value, GreenOperator)
        op = value.operator if green else value
        source = binding.ref if op.source is binding.value else SourceRef("foreign")
        descriptor = OperatorDescriptor(
            "external-green" if green else op.construction,
            BasisRef(source.name, op.domain_grade), BasisRef(source.name, op.codomain_grade),
            op.shape, Domain.REAL, Exactness.APPROXIMATE, metric="unspecified",
            symmetric=op.symmetric, psd=op.psd, kernel_policy=value.kind if green else None,
            parameters=value.parameters if green else op.parameters,
            transpose_available=False if green else op.has_transpose,
            exact_action=False if green else op.exact_matvec is not None,
            exact_transpose=False if green else op.exact_transpose_matvec is not None,
            action_variance=op.variance,
        )
        if not green and getattr(getattr(op.matrix, "dtype", None), "kind", None) == "c":
            descriptor = replace(descriptor, coefficient_domain=Domain.COMPLEX)
        from rexgraph.linear_operator import MetricAdjointOperator
        if isinstance(value, MetricAdjointOperator):
            from .validation import adjoint_descriptor, metric_descriptor
            primal = _carrier_literal(binding, value.primal).operator
            descriptor = adjoint_descriptor(primal, metric_descriptor(source, value.domain_metric),
                                             metric_descriptor(source, value.codomain_metric))
        if isinstance(value, ChannelOperator):
            from .validation import channel_descriptor
            descriptor = channel_descriptor(source, op.shape, value.channel, value.g_channel, value.c_channel)
        from rexgraph.rational_operator import RationalOperator
        from rexgraph.markov import MarkovView
        if isinstance(value, MarkovView):
            from .markov_contracts import descriptor as markov_descriptor
            value.check_state()
            descriptor = markov_descriptor(source, value.shape[0])
        from rexgraph.text_overlap import TextOverlapView
        if isinstance(value, TextOverlapView):
            from .document_contracts import overlap_descriptor
            value.check_state()
            descriptor = overlap_descriptor(source, value.shape[0])
        if isinstance(value, RationalOperator):
            value.check_state()
            descriptor = replace(descriptor, coefficient_domain=Domain.RATIONAL,
                                 euclidean_skew_adjoint=value.euclidean_skew_adjoint)
        from rexgraph.weighted_hodge import WeightedHodgeOperator
        if isinstance(value, WeightedHodgeOperator):
            from .validation import metric_descriptor, weighted_hodge_descriptor
            metrics = tuple(metric_descriptor(source, m) for m in (
                value.grade_metric, value.lower_metric, value.upper_metric) if m is not None)
            descriptor = weighted_hodge_descriptor(source, value.domain_grade, value.shape[0], value.sector,
                                                   metrics, value.active_sectors, value.exact_matvec is not None)
        if green and value.kind in {"resolvent", "metric-resolvent"}:
            from .validation import resolvent_descriptor
            descriptor = resolvent_descriptor(_carrier_literal(binding, op).operator, **dict(value.parameters))
        return RCType(
            "GreenAction" if green else "RexOperator", grade=op.domain_grade,
            kind=ValueKind.GREEN_ACTION if green else ValueKind.OPERATOR,
            domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
            source=source, basis=descriptor.domain, shape=ShapeRef(op.shape), operator=descriptor,
        )

    # The value can only exist after its module has been loaded by the caller.  Match
    # that public carrier by its canonical defining type instead of importing sheaf.py
    # on every plan: EXPLAIN must remain able to type unrelated phrases without pulling
    # in another numerical implementation.
    if (type(value).__module__, type(value).__qualname__) in {
        ("rexgraph.sheaf", "ExactSheaf"),
        ("rcql.phrase", "PhraseSheaf"),
    }:
        # A phrase sheaf is a local section carrier, not an untyped Python object.
        # Its source and grade are part of the restriction contract: an identically
        # shaped sheaf over a different Rex cannot silently glue into this phrase.
        value.check_state()
        source = binding.ref if value.rex is binding.value else SourceRef("foreign")
        return RCType(
            "ExactSheaf", grade=value.grade, kind=ValueKind.EXACT_SHEAF,
            domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL,
            source=source, basis=BasisRef(source.name, value.grade),
        )

    from rexgraph.cells import CellBoundary, CellCoboundary, CompositeBinary
    if isinstance(value, CompositeBinary):
        ref = binding.ref if value.cell.source is binding.value else SourceRef("foreign")
        return RCType("CompositeBinary", grade=value.cell.grade, kind=ValueKind.COMPOSITE_BINARY,
                      variance=Variance.CELL, domain=Domain.RATIONAL, exactness=Exactness.RATIONAL,
                      source=ref, basis=BasisRef(ref.name, value.cell.grade))
    field = isinstance(value, Field)
    carrier = value.chain if isinstance(value, CellBoundary) else value.cochain if field or isinstance(value, CellCoboundary) else value
    if not isinstance(carrier, (Chain, Cochain)):
        return None

    # A foreign or unbound carrier must be rejected by a source bound
    # signature, not silently adopted because its shape happens to match.
    source = binding.ref if carrier.source is binding.value else SourceRef("foreign")
    domain, exactness = _coefficient_contract(carrier.values)
    if isinstance(value, CellBoundary):
        kind, variance, name = ValueKind.CELL_BOUNDARY, Variance.CHAIN, "CellBoundary"
    elif isinstance(value, CellCoboundary):
        kind, variance, name = ValueKind.CELL_COBOUNDARY, Variance.COCHAIN, "CellCoboundary"
    elif field:
        kind, variance, name = ValueKind.FIELD, Variance.COCHAIN, "Field"
    elif isinstance(carrier, Chain):
        kind, variance, name = ValueKind.CHAIN, Variance.CHAIN, "Chain"
    else:
        kind, variance, name = ValueKind.COCHAIN, Variance.COCHAIN, "Cochain"
    return RCType(
        name, grade=carrier.grade, kind=kind, variance=variance,
        domain=domain, exactness=exactness,
        source=source, basis=BasisRef(source.name, carrier.grade,
            "canonical" if carrier.cell_keys is None else tuple(carrier.cell_keys)),
        shape=ShapeRef(tuple(carrier.values.shape)),
    )


@dataclass(frozen=True)
class PlannedExpression:
    """One static phrase fragment and the carrier it establishes."""

    expr: Expr
    result: object
    call: TypedCall | None = None
    children: tuple[PlannedExpression, ...] = ()
    predicates: tuple[PredicateResult, ...] = ()

    def explain(self) -> object:
        """Return a plain, recursive structural account of this phrase fragment."""
        if isinstance(self.expr, Comparison):
            return {"comparison": self.expr.operation, "arguments": [child.explain() for child in self.children]}
        if isinstance(self.expr, StructuralEdit):
            return {"edit": self.expr.operation, "arguments": [child.explain() for child in self.children],
                    "requires": ["mutate"], "closure": "remove-cofaces", "basis": "current-C1-indices"}
        if isinstance(self.expr, Alias):
            return {"alias": self.expr.name, "expression": self.children[0].explain()}
        if isinstance(self.expr, Member):
            return {"member": self.expr.name, "record": self.children[0].explain(),
                    "result": _plain_literal(self.result),
                    "predicates": [asdict(item) for item in self.predicates]}
        if isinstance(self.expr, ListExpr):
            return {"list": [child.explain() for child in self.children]}
        if isinstance(self.expr, Reference):
            return {"reference": self.expr.name, "result": _plain_literal(self.result)}
        if self.call is None:
            return {"literal": _plain_literal(self.result)}
        detail = self.call.explain()
        detail["arguments"] = [child.explain() for child in self.children]
        detail["predicates"] = [asdict(item) for item in self.predicates]
        return detail


@dataclass(frozen=True)
class QueryPlan:
    """A fully typed, non executing plan for every returned phrase."""

    binding: Binding
    returns: tuple[PlannedExpression, ...]
    query: Query
    bindings: tuple[tuple[str, PlannedExpression], ...] = ()

    def dag(self):
        from .native_plan import lower
        return lower(self)

    @property
    def effects(self) -> frozenset:
        """The union of effects declared by all planned expression calls."""
        effects = frozenset()
        pending = list(self.returns) + [expr for _, expr in self.bindings]
        visited = set()
        while pending:
            expression = pending.pop()
            if id(expression) in visited:
                continue
            visited.add(id(expression))
            if expression.call is not None:
                effects |= expression.call.effects
            if isinstance(expression.expr, StructuralEdit):
                from .types import Effect
                effects |= frozenset({Effect.MUTATE})
            pending.extend(expression.children)
        return effects

    def explain(self) -> dict[str, object]:
        """Describe the full phrase without evaluating an operator or serializing a source."""
        return {
            "source": self.binding.ref.name,
            "source_kind": self.binding.schema.kind.value,
            "source_state": _plain_source(self.binding.ref),
            "policy_digest": self.binding.ref.policy_digest,
            "effects": sorted(effect.value for effect in self.effects),
            "bindings": [{"name": name, "expression": expr.explain()} for name, expr in self.bindings],
            "returns": [item.explain() for item in self.returns],
            "native_plan": self.dag().explain(),
        }


def _plan_expression(binding: Binding, expr: Expr, parameters: Mapping[str, object], context, locals_) -> PlannedExpression:
    if isinstance(expr, Comparison):
        from .comparison import validate
        children = tuple(_plan_expression(binding, item, parameters, context, locals_)
                         for item in (expr.left, expr.right))
        return PlannedExpression(expr, validate(expr.operation, tuple(child.result for child in children)), children=children)
    if isinstance(expr, StructuralEdit):
        from rexgraph.graph import RexGraph
        from rexgraph.structural_edit import validate_edit
        binding.source.require("mutate")
        children = tuple(_plan_expression(binding, item, parameters, context, locals_)
                         for item in (expr.state, expr.value))
        state, value = (child.result for child in children)
        if not isinstance(state, RexGraph) and not (isinstance(state, RCType) and state.kind is ValueKind.REX):
            raise TypeError("ADD/REMOVE requires a static native RexGraph state")
        validate_edit(expr.operation, value)
        return PlannedExpression(expr, RCType("Rex", kind=ValueKind.REX, domain=Domain.METADATA,
                                 exactness=Exactness.STRUCTURAL), children=children)
    if isinstance(expr, Alias):
        raise TypeError("AS aliases are permitted only on returned expressions")
    if isinstance(expr, ListExpr):
        children = tuple(_plan_expression(binding, item, parameters, context, locals_) for item in expr.items)
        return PlannedExpression(expr, tuple(child.result for child in children), children=children)
    if isinstance(expr, Member):
        from .members import member_type
        child = _plan_expression(binding, expr.value, parameters, context, locals_)
        result = member_type(child.result, expr.name, context)
        predicates = (PredicateResult("record-member-present", "deferred",
            f"runtime checks public Mapping key {expr.name!r}; no reader executes during EXPLAIN"),)
        return PlannedExpression(expr, _carrier_literal(binding, result) or result,
                                 children=(child,), predicates=predicates)
    if isinstance(expr, Reference):
        if expr.name not in locals_:
            raise ValueError(f"unbound local {expr.name!r}; LET may reference only earlier bindings")
        target = locals_[expr.name]
        if isinstance(target.expr, Literal) and target.expr.value is binding.value:
            binding.source.require("read")
        return PlannedExpression(expr, target.result, children=(target,))
    if isinstance(expr, Literal):
        return PlannedExpression(expr, _carrier_literal(binding, expr.value) or expr.value)
    if isinstance(expr, Parameter):
        try:
            value = parameters[expr.name]
        except KeyError as exc:
            raise KeyError(f"no static value declared for parameter ${expr.name}") from exc
        # Programmatic parameters are as capable of carrying a source bound native
        # value as literal builder inputs.  Leaving them raw would let an ExactSheaf,
        # Chain, or Cochain bypass the same source/basis proof merely because it was
        # named with ``$`` instead of embedded in an AST.
        return PlannedExpression(expr, _carrier_literal(binding, value) or value)
    if isinstance(expr, Call):
        children = tuple(_plan_expression(binding, item, parameters, context, locals_) for item in expr.args)
        typed = infer(binding, expr.name, tuple(item.result for item in children),
                      context=context, children=children)
        return PlannedExpression(expr, typed.result, typed, children, typed.predicates)
    raise TypeError(f"cannot statically plan {type(expr).__name__}")


def plan_query(binding: Binding, query: Query, *, parameters: Mapping[str, object] | None = None) -> QueryPlan:
    """Type every returned expression of a query before any runtime operator resolves.

    ``binding`` is supplied explicitly so the caller cannot accidentally plan one source
    and run another.  The source expression in ``query`` remains parser level syntax;
    this function checks the returned phrase against the already bound source.
    """
    if not isinstance(query, Query):
        raise TypeError("plan_query expects a Query")
    if query.matches:
        from .matching import plan_match
        return plan_match(binding, query, parameters=parameters)
    supplied = {} if parameters is None else parameters
    from .validation import ValidationContext
    context = ValidationContext(binding, supplied)
    locals_ = {}
    if query.source_alias is not None:
        # Returning a complete source is a read, not an ungoverned literal escape.
        if binding.schema.kind not in (ValueKind.REX, ValueKind.TEMPORAL_REX):
            # Store/catalog aliases still qualify methods but are not serializable
            # state values and cannot leak a write capable handle into a result.
            pass
        else:
            value = binding.value
            locals_[query.source_alias] = PlannedExpression(Literal(value),
                _carrier_literal(binding, value) or value)
    for item in query.bindings:
        if not isinstance(item, LetBinding):
            raise TypeError("query bindings must be LetBinding values")
        if item.name in locals_ or item.name == query.source_alias:
            raise ValueError(f"duplicate LET binding {item.name!r}")
        locals_[item.name] = _plan_expression(binding, item.value, supplied, context, locals_)
    returns, aliases = [], set()
    for expression in query.returns:
        if isinstance(expression, Alias):
            if expression.name in aliases:
                raise ValueError(f"duplicate return alias {expression.name!r}")
            aliases.add(expression.name)
            child = _plan_expression(binding, expression.value, supplied, context, locals_)
            returns.append(PlannedExpression(expression, child.result, children=(child,)))
        else:
            returns.append(_plan_expression(binding, expression, supplied, context, locals_))
    return QueryPlan(
        binding=binding,
        returns=tuple(returns),
        query=query,
        bindings=tuple((item.name, locals_[item.name]) for item in query.bindings),
    )
