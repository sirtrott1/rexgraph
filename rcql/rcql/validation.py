"""Source aware structural predicates and native operator space descriptors.

Checks inspect incidence/declared carriers, never evaluate expression adapters,
form channel/Laplacian Gram matrices, or solve. Deferred predicates are not proofs.
"""
from __future__ import annotations

from dataclasses import replace
from math import isfinite

from .types import (
    BasisRef,
    Domain,
    Exactness,
    MetricDescriptor,
    OperatorDescriptor,
    PredicateResult,
    RCType,
    ShapeRef,
    ValueKind,
)


class ValidationContext:
    def __init__(self, binding, parameters=None):
        self.binding = binding
        self.parameters = parameters or {}
        self._boundaries = None
        self._chain = None
        self._homology = None

    @property
    def native(self):
        return self.binding.schema.kind is ValueKind.REX and hasattr(self.binding.value, "_boundary_ptr")

    @property
    def boundaries(self):
        if self._boundaries is None:
            from rexgraph.native_sparse import boundary_carriers
            self._boundaries = boundary_carriers(self.binding.value)
        return self._boundaries

    @property
    def sizes(self):
        maps = self.boundaries
        return [maps[0].shape[0]] + [item.shape[1] for item in maps]

    def grade(self, grade, *, lower=0, allow_empty_upper=False):
        if allow_empty_upper and grade == len(self.sizes):
            return 0
        if not lower <= grade < len(self.sizes):
            raise ValueError(f"grade {grade} is not present")
        return self.sizes[grade]

    def chain(self):
        if self._chain is None:
            from rexgraph.graded_boundary import _exact_chain_residual
            residual = _exact_chain_residual(self.boundaries)
            self._chain = PredicateResult(
                "chain_condition", "unknown" if residual is None else "verified" if residual == 0 else "failed",
                "unsupported exact coefficient domain" if residual is None else f"exact sparse composition residual = {residual}",
            )
        return self._chain

    def known_value(self, expression):
        from .ast import Literal, Parameter, Reference
        while isinstance(expression.expr, Reference):
            expression = expression.children[0]
        if isinstance(expression.expr, Literal):
            return expression.expr.value
        if isinstance(expression.expr, Parameter):
            return self.parameters.get(expression.expr.name)
        return None

    def homology(self):
        if self._homology is None:
            from rexgraph.native_rank import exact_tower, tower_chain_residual
            shapes, columns = exact_tower(self.binding.value)
            residual = tower_chain_residual(shapes, columns)
            if residual:
                raise ValueError(f"homology requires the chain condition; exact residual = {residual}")
            self._homology = PredicateResult("chain_condition", "verified",
                "exact zero composition of the original rational/integer tower")
        return self._homology


def graded_map_descriptor(declaration):
    from .types import CoordinateDescriptor, GradedMapDescriptor
    return GradedMapDescriptor(
        tuple(CoordinateDescriptor(s.name, s.keys) for s in declaration.domain.spaces),
        tuple(CoordinateDescriptor(s.name, s.keys) for s in declaration.codomain.spaces),
        declaration.shapes, declaration.domain.coefficient_digest,
        declaration.codomain.coefficient_digest, declaration.coefficient_digest,
        tuple(len(p) for p in declaration.components))


def metric_descriptor(source_ref, metric):
    return MetricDescriptor(
        "positive-diagonal", BasisRef(source_ref.name, metric.grade,
            "canonical" if metric.cell_keys is None else metric.cell_keys),
        (len(metric.weights), len(metric.weights)),
        Domain.RATIONAL if metric.exact else Domain.REAL,
        Exactness.RATIONAL if metric.exact else Exactness.APPROXIMATE,
        positive_definite=True, coefficient_digest=metric.coefficient_digest,
    )


def channel_descriptor(source_ref, shape, channel, g_channel, c_channel):
    rational = channel != "G" or g_channel == "raw"
    return OperatorDescriptor(
        "channel", BasisRef(source_ref.name, 1), BasisRef(source_ref.name, 1), shape,
        Domain.REAL if channel == "G" and g_channel == "normalized" else Domain.RATIONAL,
        Exactness.APPROXIMATE, metric="relation-weighted" if channel != "C" else "unweighted",
        symmetric=True, psd=True,
        transpose_available=True,
        exact_action=rational, exact_transpose=rational,
        parameters=(("channel", channel), ("g_channel", g_channel), ("c_channel", c_channel),
                    ("trace_normalized", False), ("frustration_reference", "raw-G")),
    )


def adjoint_descriptor(primal, domain_metric=None, codomain_metric=None):
    """Endpoint metrics describe the primal map; result spaces are reversed."""
    def identity(basis, n):
        return MetricDescriptor("identity", basis, (n, n), Domain.RATIONAL,
                                Exactness.RATIONAL, positive_definite=True)
    md = domain_metric or identity(primal.domain, primal.shape[1])
    mc = codomain_metric or identity(primal.codomain, primal.shape[0])
    rational_metrics = all(m.coefficient_domain in {Domain.INTEGER, Domain.RATIONAL} for m in (md, mc))
    return OperatorDescriptor(
        "metric-adjoint", primal.codomain, primal.domain, primal.shape[::-1],
        Domain.RATIONAL if rational_metrics and primal.exact_transpose else (
            Domain.COMPLEX if primal.coefficient_domain is Domain.COMPLEX else Domain.REAL),
        Exactness.APPROXIMATE, metric="explicit-endpoint-diagonal", symmetric=False, psd=False,
        parameters=(("primal_construction", primal.construction), ("primal_parameters", primal.parameters)),
        transpose_available=True, exact_action=rational_metrics and primal.exact_transpose,
        exact_transpose=rational_metrics and primal.exact_action,
        action_variance=primal.action_variance, adjoint_domain_metric=md, adjoint_codomain_metric=mc,
        primal_operator=primal,
    )


def weighted_hodge_descriptor(source_ref, grade, n, sector, metrics, active, exact):
    basis = BasisRef(source_ref.name, grade)
    positive = all(m.positive_definite is True for m in metrics)
    return OperatorDescriptor(
        "weighted-hodge", basis, basis, (n, n), Domain.RATIONAL if exact else Domain.REAL,
        Exactness.APPROXIMATE, metric="explicit-graded-diagonal", symmetric=False, psd=False,
        parameters=(("sector", sector), ("active_sectors", tuple(active))),
        transpose_available=True, exact_action=exact, exact_transpose=exact, action_variance="chain",
        grade_metrics=tuple(metrics), metric_self_adjoint=True if positive else None,
        metric_psd=True if positive and sector != "difference" else None,
    )


def resolvent_descriptor(primal, alpha=1.0, tol=1e-10, maxiter=1000):
    """A numerical inverse handle never inherits its primal's exact/transpose hooks."""
    weighted = primal.construction == "weighted-hodge"
    return replace(primal, construction="metric-resolvent" if weighted else "resolvent",
        coefficient_domain=Domain.REAL, arithmetic=Exactness.APPROXIMATE,
        kernel_policy="inverse-I-plus-alpha-L", transpose_available=False,
        exact_action=False, exact_transpose=False, primal_operator=primal,
        parameters=(("base_construction", primal.construction), ("base_parameters", primal.parameters),
                    ("alpha", float(alpha)), ("tol", float(tol)), ("maxiter", int(maxiter))))


def _operator_tree(desc):
    yield desc
    for operand in desc.operands:
        yield from _operator_tree(operand)
    if desc.primal_operator is not None:
        yield from _operator_tree(desc.primal_operator)


def refine(typed, children, context):
    """Return the refined call and the predicates actually checked for it."""
    args = typed.args
    name = typed.operator
    result = typed.result
    facts = [PredicateResult("signature", "verified", "source kind, arity, input types and capabilities checked")]
    if name in {"TURN_FIELD", "PATH_CHANGE"}:
        from .turn_contracts import refine as refine_turn
        facts.extend(refine_turn(typed, children, context))
    if name == "CORPUS_FIELD":
        from .corpus_contracts import refine as refine_corpus
        facts.extend(refine_corpus(typed, children, context))
    from .value_contracts import ARGUMENTS, refine as refine_value
    if name in ARGUMENTS:
        result = refine_value(typed, context)
    from .calculus_contracts import ARGUMENTS as CALCULUS, refine as refine_calculus
    if name in CALCULUS:
        result = refine_calculus(typed, context)
    from .critical_contracts import ARGUMENTS as CRITICAL, refine as refine_critical
    if name in CRITICAL:
        result = refine_critical(typed, context)
    from .certificate_contracts import ARGUMENTS as CERTIFICATE, refine as refine_certificate
    if name in CERTIFICATE:
        result, certificates = refine_certificate(typed, children, context)
        facts.extend(certificates)
    from .temporal_contracts import ARGUMENTS as TEMPORAL, refine as refine_temporal
    if name in TEMPORAL:
        result = refine_temporal(typed, context)
    from .structure_contracts import ARGUMENTS as STRUCTURE, refine as refine_structure
    from .homology_contracts import ARGUMENTS as HOMOLOGY, refine as refine_homology
    from .partition_contracts import ARGUMENTS as PARTITION, refine as refine_partition
    from .artifact_contracts import ARGUMENTS as ARTIFACT, refine as refine_artifact
    from .filling_contracts import ARGUMENTS as FILLING, refine as refine_filling
    from .difference_contracts import ARGUMENTS as DIFFERENCE, refine as refine_difference
    from .document_contracts import ARGUMENTS as DOCUMENT, refine as refine_document
    from .rational_contracts import ARGUMENTS as RATIONAL, refine as refine_rational
    from .markov_contracts import ARGUMENTS as MARKOV, refine as refine_markov
    from .replay_contracts import ARGUMENTS as REPLAY, refine as refine_replay
    if name in REPLAY:
        facts.extend(refine_replay(typed, children, context))
    if name == "SYMMETRY":
        from .symmetry_contracts import refine as refine_symmetry
        facts.extend(refine_symmetry(typed, children, context))
    if name in MARKOV:
        result, markov_facts = refine_markov(typed, children, context)
        facts.extend(markov_facts)
    if name in RATIONAL:
        facts.extend(refine_rational(typed, context))
    if name in DOCUMENT or name == "CLOSURE":
        result, document_facts = refine_document(typed, children, context)
        facts.extend(document_facts)
    if name in DIFFERENCE:
        facts.extend(refine_difference(typed, children, context))
    if name in FILLING:
        facts.extend(refine_filling(typed, context))
    if name in ARTIFACT:
        facts.extend(refine_artifact(typed, children, context))
    if name in PARTITION:
        facts.extend(refine_partition(typed, context))
    if name in HOMOLOGY:
        facts.extend(refine_homology(typed, context))
    if name in STRUCTURE:
        result, structural_facts = refine_structure(typed, children, context)
        facts.extend(structural_facts)

    def verified(name, evidence):
        facts.append(PredicateResult(name, "verified", evidence))

    if name in {"DIRAC", "ANTI_DIRAC", "GRADED_CHAIN", "GRADE_COMPONENT", "COMMUTATOR", "ANTICOMMUTATOR"} and not context.native:
        raise TypeError(f"{name} requires a native graded source")

    if name == "HODGE_OPERATOR":
        alpha = args[1] if len(args) > 1 else 1
        if not isfinite(float(alpha)) or alpha < 0:
            raise ValueError("HODGE_OPERATOR alpha must be finite and >= 0")
        verified("nonnegative_hodge_coupling", "finite alpha >= 0; both Euclidean Gram sectors are PSD")
    if name == "SHOW_OPERATORS":
        limit = args[0] if args else 1000
        offset = args[1] if len(args) > 1 else 0
        if not 0 <= limit <= 1000 or offset < 0:
            raise ValueError("SHOW_OPERATORS requires limit in [0, 1000] and nonnegative offset")
        verified("pagination_bounds", "limit in [0,1000], offset >= 0")
    if name == "TEMPORAL_DELTA":
        if not 0 < args[0] < int(context.binding.value.T):
            raise ValueError("TEMPORAL_DELTA step must be an interior transition index")
        verified("transition_index", "0 < step < number of snapshots; no reconstruction performed")
    if name in {"SIGNAL_SOURCE", "SIGNAL_FLOW", "RELATION_SIGNAL", "SIGNAL_HODGE"} and len(args) > 1:
        allowed = ({"amplitude", "existence", "orientation", "signing"}
                   if name in {"RELATION_SIGNAL", "SIGNAL_HODGE"} else
                   {"structural", "existence", "geometry", "amplitude", "signing"})
        if args[1].lower() not in allowed:
            raise ValueError(f"{name} has no channel {args[1]!r}")
        verified("signal_channel", "declared temporal channel name")

    if context.native:
        if name in {"COMMUTATOR", "ANTICOMMUTATOR"}:
            verified("bracket_spaces", "same canonical source, grade/variance or full Chain tower; AB and BA both well typed")
            facts.append(PredicateResult("bracket_identities", "not-asserted",
                "ordinary AB +/- BA; no inferred commutation, chain-law cancellation, positivity or exact action beyond operand capabilities"))
            facts.append(PredicateResult("bracket_adjointness", "conditional",
                "derived only from operand adjoint declarations in a known common form; no matrix or numerical verification"))
        if name in {"DIRAC", "ANTI_DIRAC", "GRADED_CHAIN", "GRADE_COMPONENT"} or any(
            isinstance(v, RCType) and v.kind in {ValueKind.GRADED_CHAIN, ValueKind.GRADED_OPERATOR} for v in args
        ):
            from .graded_calculus import refine_graded
            result, graded_facts = refine_graded(name, args, result, context)
            facts.extend(graded_facts)
        if name in {"HODGE_DOWN", "HODGE_UP", "HODGE_SUM", "HODGE_DIFFERENCE"}:
            from rexgraph.graded_boundary import _integer_columns
            grade, sector = args[0], name.removeprefix("HODGE_").lower()
            n = context.grade(grade)
            padded = (*args[1:], None, None, None)
            supplied = {grade: padded[0]}
            if sector != "up" and grade > 0:
                supplied[grade-1] = padded[1]
            if sector != "down":
                supplied[grade+1] = padded[1 if sector == "up" else 2]
            metrics = {}
            for k, metric in supplied.items():
                size = context.grade(k, allow_empty_upper=True)
                metrics[k] = (metric.metric if metric is not None else MetricDescriptor(
                    "identity", BasisRef(typed.binding.ref.name, k), (size, size), Domain.RATIONAL,
                    Exactness.RATIONAL, positive_definite=True))
            active = []
            exact = True
            for label, k, enabled in (("down", grade, grade > 0 and sector != "up"),
                                      ("up", grade+1, grade < len(context.boundaries) and sector != "down")):
                if not enabled:
                    continue
                matrix = context.boundaries[k-1]
                if any(not isfinite(float(v)) for v in matrix.data):
                    raise ValueError("Hodge requires finite boundary coefficients")
                if not matrix.nnz:
                    continue
                active.append(label)
                exact &= (k == 1 or _integer_columns(matrix) is not None) and all(
                    metrics[g].coefficient_domain in {Domain.INTEGER, Domain.RATIONAL} for g in (k-1, k))
            desc = weighted_hodge_descriptor(typed.binding.ref, grade, n, sector,
                                             tuple(metrics.values()), active, exact)
            result = result.with_(operator=desc, shape=ShapeRef(desc.shape))
            verified("weighted_hodge_spaces", "canonical Chain grade; neighboring endpoint metrics; missing sectors are zero")
            facts.append(PredicateResult("metric_self_adjointness", "verified" if desc.metric_self_adjoint else "deferred",
                "factored metric adjoints; positive known metrics or runtime validation of computed diagonals"))
            facts.append(PredicateResult("hodge_sector_annihilation", "not-asserted",
                "requires B_k B_(k+1)=0; handle construction does not run a full chain-law verification"))
            facts.append(PredicateResult("euclidean_psd", "not-asserted",
                "metric self-adjointness does not grant Euclidean symmetry; no Euclidean solver dispatch"))
        if name == "CHAIN_MAP":
            desc = args[0].graded_map
            if tuple(s[1] for s in desc.shapes) != tuple(context.sizes):
                raise ValueError("CHAIN_MAP must cover the full present source tower")
            if any(s.name != f"C{k}" or s.keys != tuple(str(i) for i in range(n))
                   for k, (s, n) in enumerate(zip(desc.domain, context.sizes, strict=True))):
                raise ValueError("CHAIN_MAP requires canonical ordered source spaces")
            if args[0].temporal is not None and args[0].temporal != context.binding.temporal:
                raise ValueError("CHAIN_MAP requires the bound source temporal state")
            known = context.known_value(children[0])
            if known is not None and not isinstance(known, RCType):
                from rexgraph.chain_map import ChainMap, GradedMap
                declaration = known.declaration if isinstance(known, ChainMap) else known
                if not isinstance(declaration, GradedMap) or declaration.domain.source is not context.binding.value:
                    raise TypeError("CHAIN_MAP requires a map bound to this source")
                if graded_map_descriptor(declaration) != replace(desc, chain_preserving=None):
                    raise ValueError("graded-map descriptor does not match the supplied declaration")
                certificate = declaration.verify()
                verified("source_chain_law", f"exact sparse residual = {certificate.source_residual}")
                verified("target_chain_law", f"exact sparse residual = {certificate.target_residual}")
                verified("chain_map_squares", f"exact residuals at grades 1..{len(desc.shapes)-1} = {certificate.commutation_residuals}")
                verified("chain_map_boundary_state", "full captured source/target boundary states checked, including primary C1 incidence")
                desc = replace(desc, chain_preserving=True)
            else:
                facts.append(PredicateResult("chain_map_certificate", "deferred",
                    "computed declaration must pass both chain laws, all squares and current-boundary checks at execution"))
                desc = replace(desc, chain_preserving=None)
            result = result.with_(graded_map=desc)
        if name == "METRIC":
            size = context.grade(args[0], allow_empty_upper=True)
            weights = args[1] if len(args) > 1 else None
            desc = MetricDescriptor("identity" if weights is None else "positive-diagonal",
                result.basis, (size, size), result.domain,
                Exactness.RATIONAL if result.domain is Domain.RATIONAL else Exactness.APPROXIMATE,
                positive_definite=True if weights is None else None)
            known = context.known_value(children[1]) if len(children) > 1 else None
            if known is not None and not isinstance(known, RCType):
                from rexgraph.graded_metric import diagonal_metric
                desc = metric_descriptor(typed.binding.ref, diagonal_metric(context.binding.value, args[0], known))
            if weights is not None and weights.shape is not None and weights.shape.dims != (size,):
                raise ValueError("METRIC requires one diagonal coefficient per cell")
            result = result.with_(metric=desc, shape=ShapeRef(desc.shape))
            facts.append(PredicateResult("positive_metric", "verified" if desc.positive_definite else "deferred",
                "identity/declared coefficients checked" if desc.positive_definite else "computed diagonal checked when constructed"))
        if name == "RESOLVENT":
            result = result.with_(shape=ShapeRef(result.operator.shape))
            verified("resolvent_parameters", "alpha >= 0, 0 < tol < 1, maxiter > 0; numerical identity-plus-operator inverse")
            primal = args[0].operator
            if primal.construction == "weighted-hodge":
                facts.append(PredicateResult("resolvent_metric_psd",
                    "verified" if primal.metric_psd is True else "deferred",
                    "native Hodge sum/down/up is PSD in its positive grade metric; computed metrics checked at execution"))
                facts.append(PredicateResult("euclidean_psd", "not-asserted",
                    "solve M(I+alpha L)x=Mb with diagonal preconditioning; L itself need not be Euclidean symmetric"))
            else:
                facts.append(PredicateResult("resolvent_psd", "verified" if primal.construction in {"hodge_operator", "channel"} else "deferred",
                    "native PSD construction or external PSD declaration; no eigensolver run"))
        channel_read = name in {"CHANNEL", "STAR_CHARACTER"} or (
            name in {"SCALE_MOMENT", "CHARACTER_ENERGY", "APPLY", "RESOLVENT"}
            and args[0].operator is not None and args[0].operator.construction == "channel")
        if channel_read:
            from rexgraph.channel_operator import _require_channel_source
            _require_channel_source(context.binding.value)
            if name not in {"CHANNEL", "STAR_CHARACTER"}:
                selection = dict(args[0].operator.parameters)
                if (selection["g_channel"], selection["c_channel"]) != (
                        context.binding.value.g_channel, context.binding.value.c_channel):
                    raise ValueError("channel selection changed; bind a fresh operator")
            verified("channel_source", "distinct participants, unit vertex metric, selected G/C; normalized G requires nonnegative relation metric")
        if name == "CHANNEL":
            size = context.grade(1)
            desc = channel_descriptor(typed.binding.ref, (size, size), args[0].upper(),
                                      context.binding.value.g_channel, context.binding.value.c_channel)
            result = result.with_(operator=desc, shape=ShapeRef(desc.shape))
            verified("factored_channel", "incidence-factored C1 action; not a trace-normalized hat")
        if name == "STAR_CHARACTER":
            result = result.with_(shape=ShapeRef((4,)))
            verified("star_mean", "C0 star mean of C1 character with four fixed channel coordinates")
        if name in {"SCALE_MOMENT", "CHARACTER_ENERGY"}:
            desc = args[0].operator
            order = args[1] if name == "SCALE_MOMENT" else 2
            if result.kind is ValueKind.COCHAIN:
                result = result.with_(shape=ShapeRef((desc.shape[0],)))
            verified("symmetric_moment", f"order {order}; symmetric square descriptor; explicit numeric matrices checked at execution")
            if result.exactness is Exactness.RATIONAL:
                verified("exact_moment_domain", "identity moment or existing exact channel diagonal, not reconstructed floats")
        grade = None
        if name in {"CELL", "CELLS", "ZERO", "BETTI", "HODGE_OPERATOR"} or name in {"BOUNDARY", "COBOUNDARY", "RANK", "NULLITY"} and isinstance(args[0], int):
            grade = args[0]
        elif name == "GREEN":
            grade = 0
        if grade is not None:
            size = context.grade(grade, lower=1 if name in {"BOUNDARY", "RANK", "NULLITY"} else 0)
            verified("grade_present", f"C{grade} has {size} cells")
        if name == "HODGE_OPERATOR":
            maps = context.boundaries[max(grade - 1, 0):grade + 1]
            if any(not isfinite(float(x)) for matrix in maps for x in matrix.data):
                raise ValueError("HODGE_OPERATOR requires finite boundary coefficients")
            verified("euclidean_gram_psd", "finite boundary maps and nonnegative alpha certify the factored sum")
        if name in {"RANK", "NULLITY"} and grade is not None:
            from rexgraph.native_rank import boundary_columns
            boundary_columns(context.binding.value, grade, integer=True)
            verified("exact_rank_domain", "primary arity or stored integer higher coefficients; no rank solve during validation")
        if name == "BETTI":
            facts.append(context.homology())
            verified("exact_rank_domain", "integer rank representatives at every carried grade; no rank solve during validation")
        if name in {"CELL", "CELLS"} and len(args) > 1:
            indices = (args[1],) if name == "CELL" else args[1]
            if any(type(i) is not int or not 0 <= i < size for i in indices):
                raise ValueError(f"{name} indices must lie in the grade's ordered cell basis")
            verified("cell_indices", "every index lies in the ordered grade basis")

        # Check known coefficient axes before ANY expression adapter executes.
        for position, value in enumerate(args):
            if name == "ACCESSION_DELTA" and position == 1:
                # The dedicated contract has checked this actual target Rex and
                # its ordered axes against the explicit correspondence codomain.
                continue
            if (isinstance(value, RCType) and value.accessions and value.kind in {
                    ValueKind.TYPE_ACCESSION, ValueKind.ACCESSION_FAMILY, ValueKind.TYPE_VIEW,
                    ValueKind.TYPED_FAMILY, ValueKind.MOMENT_TENSOR, ValueKind.CROSS_METRIC, ValueKind.FAMILY_METRIC}):
                for accession in value.accessions:
                    expected = context.grade(accession.basis.grade)
                    rows = expected if accession.coordinates is None else len(accession.coordinates.keys)
                    if accession.shape != (rows, expected) or accession.basis.source_id != typed.binding.ref.name:
                        raise ValueError("accession axes differ from the bound ambient cell space")
                    if accession.basis != value.basis:
                        raise ValueError("accession descriptor must retain the declared ambient basis")
                if value.kind in {ValueKind.TYPE_VIEW, ValueKind.TYPED_FAMILY} and value.shape is not None:
                    dims = value.shape.dims
                    if value.kind is ValueKind.TYPED_FAMILY:
                        if not dims or dims[0] != len(value.accessions):
                            raise ValueError("typed family type axis differs from its declared accessions")
                        if len(value.member_shapes) != len(value.accessions):
                            raise ValueError("typed family requires explicit member shapes")
                        for a, shape in zip(value.accessions, value.member_shapes, strict=True):
                            if len(shape) not in (1, 2) or shape[0] != a.shape[0] or shape[1:] != dims[2:]:
                                raise ValueError("typed family member differs from its declared coordinate/block axes")
                        rows = value.member_shapes[0][0]
                        if any(s[0] != rows for s in value.member_shapes):
                            rows = None
                        dims = dims[1:]
                    else:
                        rows = value.accessions[0].shape[0]
                    if len(dims) not in (1, 2) or dims[0] != rows:
                        raise ValueError("type view axis differs from its declared output coordinates")
                if value.kind is ValueKind.CROSS_METRIC:
                    desc = value.cross_metric
                    if (desc is None or value.accessions != (desc.left, desc.right)
                            or desc.shape != (desc.left.shape[0], desc.right.shape[0])
                            or value.shape != ShapeRef(desc.shape)):
                        raise ValueError("cross-metric axes differ from its ordered endpoints")
                if value.kind is ValueKind.FAMILY_METRIC:
                    desc = value.family_metric
                    if (desc is None or value.accessions != tuple(e.accession for e in desc.realizations)
                            or desc.metric.basis != value.basis):
                        raise ValueError("family metric must retain its ambient basis and explicit realizations")
                    expected = context.grade(value.grade)
                    n = sum(e.accession.shape[0] for e in desc.realizations)
                    if (desc.metric.shape != (expected, expected) or desc.shape != (n, n)
                            or value.shape != ShapeRef(desc.shape)
                            or any(e.shape != (expected, e.accession.shape[0]) for e in desc.realizations)):
                        raise ValueError("family metric axes differ from its realization and ambient spaces")
            if isinstance(value, RCType) and value.metric is not None and value.kind is ValueKind.METRIC:
                expected = context.grade(value.grade, allow_empty_upper=True)
                if value.metric.shape != (expected, expected):
                    raise ValueError("metric axes differ from its source grade population")
            if isinstance(value, RCType) and value.operator is not None:
                for desc in _operator_tree(value.operator):
                    for basis, axis in ((desc.domain, desc.shape[1]), (desc.codomain, desc.shape[0])):
                        # A missing upper cochain grade is an explicit zero axis.
                        if not 0 <= basis.grade <= len(context.sizes):
                            raise ValueError(f"{name} operator grade is not present")
                        expected = context.sizes[basis.grade] if basis.grade < len(context.sizes) else 0
                        if basis.source_id != typed.binding.ref.name or (axis is not None and axis != expected):
                            raise ValueError(f"{name} operator space does not match its source grade basis")
                    if desc.construction == "channel":
                        from rexgraph.channel_operator import _require_channel_source
                        _require_channel_source(context.binding.value)
                        selection = dict(desc.parameters)
                        if (selection["g_channel"], selection["c_channel"]) != (
                                context.binding.value.g_channel, context.binding.value.c_channel):
                            raise ValueError("channel selection changed inside operator; bind a fresh handle")
            if not isinstance(value, RCType) or value.kind not in {
                ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.FIELD,
            }:
                continue
            if value.grade is None:
                continue
            expected = context.grade(value.grade, allow_empty_upper=name in {"APPLY", "ADJOINT", "METRIC"})
            if value.shape is not None:
                dims = value.shape.dims
                if len(dims) not in (1, 2) or (dims[0] is not None and dims[0] != expected):
                    raise ValueError(f"{name} input cell axis does not match grade {value.grade}")
                verified("cell_axis", f"input C{value.grade} axis = {expected}")
        if name in {"BOUNDARY", "COBOUNDARY"} and grade is not None and len(args) == 2:
            if args[1].basis.ordering != "canonical":
                raise TypeError(f"{name} requires the canonical ordered basis")
            verified("canonical_basis", "input uses the operator's canonical ordered basis")

        if name in {"BOUNDARY", "COBOUNDARY", "HODGE_OPERATOR", "GREEN"} and grade is not None:
            out_grade = (grade - 1 if name == "BOUNDARY" else grade + 1 if name == "COBOUNDARY" else grade)
            out_size = context.sizes[out_grade] if out_grade < len(context.sizes) else 0
            descriptor = OperatorDescriptor(
                name.lower(), BasisRef(typed.binding.ref.name, grade), BasisRef(typed.binding.ref.name, out_grade),
                (out_size, size), Domain.REAL, Exactness.APPROXIMATE,
                symmetric=True if name in {"HODGE_OPERATOR", "GREEN"} else None,
                psd=True if name in {"HODGE_OPERATOR", "GREEN"} else None,
                transpose_available=name != "GREEN",
                action_variance="chain" if name == "BOUNDARY" else "cochain",
                kernel_policy="moore-penrose" if name == "GREEN" else None,
                # The action is numerical; the exact declared scalar is retained
                # by its input expression, separately from this physical value.
                parameters=(("alpha", float(args[1]) if len(args) > 1 else 1.0),) if name == "HODGE_OPERATOR" else (),
            )
            if name in {"BOUNDARY", "COBOUNDARY"}:
                boundary_grade = grade if name == "BOUNDARY" else grade + 1
                if boundary_grade == 1:
                    descriptor = replace(descriptor, coefficient_domain=Domain.RATIONAL,
                                         exact_action=True, exact_transpose=True)
                else:
                    from rexgraph.graded_boundary import _integer_columns
                    integral = boundary_grade > len(context.boundaries) or (
                        _integer_columns(context.boundaries[boundary_grade - 1]) is not None
                    )
                    if integral:
                        descriptor = replace(descriptor, coefficient_domain=Domain.INTEGER,
                                             exact_action=True, exact_transpose=True)
                        verified("integral_higher_boundary", "literal integer higher coefficients or an absent zero sector")
                    elif len(args) == 2 and result.exactness in {Exactness.INTEGER, Exactness.RATIONAL}:
                        raise ValueError("exact boundary action requires integer higher boundary coefficients")
            if result.kind in {ValueKind.OPERATOR, ValueKind.GREEN_ACTION}:
                result = result.with_(operator=descriptor, shape=ShapeRef(descriptor.shape))
            else:
                operand = args[-1]
                trailing = operand.shape.dims[1:] if isinstance(operand, RCType) and operand.shape else ()
                result = result.with_(shape=ShapeRef((out_size,) + trailing))
            if out_size == 0:
                verified("empty_codomain", f"C{out_grade} is empty; exact zero map")
        if name in {"APPLY", "GREEN_SOLVE"} and args[0].kind is not ValueKind.GRADED_OPERATOR:
            descriptor, value = args[0].operator, args[1]
            dims = value.shape.dims if value.shape else ()
            if dims and descriptor.shape[1] is not None and dims[0] != descriptor.shape[1]:
                raise ValueError("APPLY input cell axis differs from operator domain")
            result = result.with_(shape=ShapeRef((descriptor.shape[0],) + dims[1:]))
            verified("operator_spaces", "declared domain grade, variance, ordered basis and cell axis match the input")
        if name == "ADJOINT":
            verified("adjoint_spaces", "reversed operator endpoints; original variance retained; explicit transpose capability")
            facts.append(PredicateResult("adjoint_metric_positivity",
                "verified" if all(m.positive_definite for m in (result.operator.adjoint_domain_metric,
                                                               result.operator.adjoint_codomain_metric)) else "deferred",
                "endpoint diagonals checked here if known; computed metrics checked at construction"))
            facts.append(PredicateResult("adjoint_euclidean_psd", "not-asserted",
                "weighted coordinate adjoint need not be Euclidean symmetric or PSD"))
        if (name in {"BOUNDARY", "COBOUNDARY"} and len(children) == 2
                and children[1].call is not None and children[1].call.operator == name):
            facts.append(context.chain())

        if name == "CHARACTER" and args and args[0] is True:
            rex = context.binding.value
            for attr in ("w_V", "vertex_weights"):
                weights = getattr(rex, attr, None)
                if weights is not None and any(w != 1 for w in weights):
                    raise ValueError("exact channel diagonals do not support vertex weighting")
            if rex.g_channel == "normalized":
                from rexgraph.sparse_character import _require_distinct_channel_participants
                _require_distinct_channel_participants(rex)
                metric = rex.edge_metric_exact
                if metric is not None and any(w < 0 for w in metric):
                    raise ValueError("exact normalized G requires nonnegative relation weights")
                verified("normalized_G_diagonal", "distinct participants and nonnegative metric; rational 1-Kee/row_sum")
            verified("channel_selection", f"G={rex.g_channel}; C={rex.c_channel}; no channel matrix assembled")

        if name in {"ACCESS", "ACCESS_TYPES", "CO_RELATE", "MOMENT_TENSOR"}:
            verified("common_ambient_accession", "declared sparse maps from a common ambient input basis; output coordinates retained separately")
            if name == "CO_RELATE" and result.cross_metric is not None:
                verified("cross_metric_endpoints", "ordered output coordinate spaces and block axes match the explicit cross block")
                facts.append(PredicateResult("cross_metric_positivity", "not-asserted",
                    "a cross block need not be symmetric or positive and is not a family Gram certificate"))
            if result.family_metric is not None:
                verified("family_metric_factorization", "explicit E_sigma* M E_tau factors into one positive diagonal ambient metric; full family form is PSD")
                facts.append(PredicateResult("realization_injectivity", "not-asserted",
                    "realizations may be noninjective; induced type forms need not be positive definite"))
            facts.append(PredicateResult("accession_chain_preservation", "not-asserted",
                "measurement maps need not commute with boundary; no induced boundary was supplied"))
        if name == "INTEGRATE":
            verified("dual_pairing", "Cochain evaluation on Chain; aligned source, grade, ordered basis and time; no metric identification")
        if name in {"ACCUMULATE", "SPREAD", "MOMENT", "INTEGRATE"} or (
                name == "CO_RELATE" and result.cross_metric is None and result.family_metric is None):
            if args[0].shape and args[1].shape and args[0].shape != args[1].shape:
                raise ValueError(f"{name} requires matching value shapes")
            if name == "ACCUMULATE":
                result = result.with_(shape=args[0].shape)
        if (result.kind in {ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.FIELD} and result.shape is None
                and result.grade is not None and result.grade < len(context.sizes)):
            result = result.with_(shape=ShapeRef((context.sizes[result.grade],)))
    else:
        if name in {"BOUNDARY", "COBOUNDARY", "HODGE_OPERATOR", "HODGE_DOWN", "HODGE_UP",
                    "HODGE_SUM", "HODGE_DIFFERENCE", "GREEN", "APPLY"}:
            facts.append(PredicateResult("source_structure", "deferred", "source has no native incidence available during planning"))
    return replace(typed, result=result), tuple(facts)
