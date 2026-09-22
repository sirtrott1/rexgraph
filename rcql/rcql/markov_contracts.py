"""The participation walk and the imported PageRank fixed point on it.

These are imported algorithms named for their implementation. The relational ranking
that replaces PageRank is RESOLVENT_RANK, in resolvent_rank_contracts.
"""
from fractions import Fraction

from .types import BasisRef, Domain, Exactness, OperatorDescriptor, PredicateResult, RCType, ShapeRef, ValueKind, Variance

ARGUMENTS = {"PARTICIPATION_WALK": (("grade",), (0,)),
             "PAGERANK_SOLVE": (("view", "damping", "seed"), (Fraction(17,20), None)),
             "PAGERANK_ITERATION": (("view", "damping", "seed", "tol", "maxiter"), (0.85, None, 1e-10, 1000))}


def descriptor(source, n):
    basis = BasisRef(source.name, 0)
    return OperatorDescriptor("participation-walk", basis, basis, (n, n), Domain.RATIONAL,
        Exactness.APPROXIMATE, metric="participation", parameters=(("participation", "abs(B1) W"),
        ("dangling", "uniform"), ("direction", "column-mass")),
        transpose_available=True, exact_action=True, exact_transpose=True)


def rank_result(args):
    view = args[0]
    if view.operator is None or view.operator.construction != "participation-walk":
        raise TypeError("the PageRank iteration requires an explicit PARTICIPATION_WALK")
    tail = (*args[1:], *ARGUMENTS["PAGERANK_ITERATION"][1][len(args)-1:])
    damping, seed, tol, maxiter = tail
    from rexgraph.core._standard import pagerank_controls
    pagerank_controls(damping, maxiter, tol)
    if seed is not None and (seed.grade != 0 or seed.basis != view.operator.domain
            or seed.variance is not Variance.COCHAIN or seed.shape.dims != (view.operator.shape[0],)):
        raise TypeError("PageRank restart requires one canonical C0 Cochain vector")
    return RCType("Cochain", kind=ValueKind.COCHAIN, grade=0, variance=Variance.COCHAIN,
        source=view.source, basis=view.operator.domain, domain=Domain.REAL,
        shape=ShapeRef((view.operator.shape[0],)), exactness=Exactness.APPROXIMATE)


def exact_rank_result(args):
    view = args[0]
    if view.operator is None or view.operator.construction != "participation-walk":
        raise TypeError("the PageRank solve requires an explicit PARTICIPATION_WALK")
    damping, seed = (*args[1:], *ARGUMENTS["PAGERANK_SOLVE"][1][len(args)-1:])
    from rexgraph.ranking_response import _damping
    _damping(damping)
    if seed is not None and (seed.grade != 0 or seed.basis != view.operator.domain
            or seed.variance is not Variance.COCHAIN or seed.shape.dims != (view.operator.shape[0],)):
        raise TypeError("the PageRank solve seed requires one canonical C0 Cochain vector")
    return RCType("Cochain", kind=ValueKind.COCHAIN, grade=0, variance=Variance.COCHAIN,
        source=view.source, basis=view.operator.domain, domain=Domain.RATIONAL,
        shape=ShapeRef((view.operator.shape[0],)), exactness=Exactness.RATIONAL)


def refine(typed, children, context):
    if not context.native:
        raise TypeError("the participation walk requires a native Rex source")
    result = typed.result
    if typed.operator == "PARTICIPATION_WALK":
        from rexgraph.markov import validate_markov_source
        args = (*typed.args, *ARGUMENTS["PARTICIPATION_WALK"][1][len(typed.args):])
        validate_markov_source(context.binding.value, *args)
        desc = descriptor(context.binding.ref, int(context.binding.value.nV))
        result = result.with_(operator=desc, shape=ShapeRef(desc.shape))
    if typed.operator == "PAGERANK_SOLVE":
        return result, [PredicateResult("ranking_equation", "deferred",
            "sparse rational solve with exact residual, mass and positivity checks")]
    return result, [PredicateResult("native_participation", "verified",
        "C0 action through the primary tensor; uniform dangling mass, relation metrics, complete source retained"),
        PredicateResult("pagerank_solve", "deferred",
        "compiled numerical power iteration with measured L1 fixed point bound; no exact solver claim")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    register(OperatorSignature(name="PARTICIPATION_WALK", source_kind=ValueKind.REX,
        inputs=(TypePattern("grade", literal=int, optional=True),),
        result=RCType("ParticipationWalk", kind=ValueKind.OPERATOR, grade=0, domain=Domain.METADATA,
                      exactness=Exactness.STRUCTURAL), memoizable=True,
        implementation_key="rexgraph.markov.ParticipationWalk",
        preconditions=("grade zero native participation; no branching expansion or execution mode switch",
                       "column stochastic mass action and observable transpose; uniform dangling columns")))
    real = (Domain.INTEGER, Domain.RATIONAL, Domain.REAL)
    scalar = (int, float, Fraction)
    register(OperatorSignature(name="PAGERANK_ITERATION", source_kind=ValueKind.REX,
        inputs=(TypePattern("view", kind=ValueKind.OPERATOR, source_bound=True, basis_bound=True),
                TypePattern("damping", literal=scalar, optional=True),
                TypePattern("seed", kind=(ValueKind.COCHAIN, ValueKind.FIELD), literal=type(None), source_bound=True,
                            basis_bound=True, domain=real, optional=True),
                TypePattern("tol", literal=scalar, optional=True),
                TypePattern("maxiter", literal=int, optional=True)),
        result=rank_result, memoizable=True, implementation_key="rexgraph.markov.pagerank_iteration",
        preconditions=("numerical C0 fixed point; restart is uniform or an explicit nonnegative Cochain",
                       "0<=damping<1; measured residual/(1-damping)<=tol or refusal")))

    register(OperatorSignature(name="PAGERANK_SOLVE", source_kind=ValueKind.REX,
        inputs=(TypePattern("view", kind=ValueKind.OPERATOR, source_bound=True, basis_bound=True),
                TypePattern("damping", literal=(int, Fraction), optional=True),
                TypePattern("seed", kind=(ValueKind.COCHAIN, ValueKind.FIELD), literal=type(None),
                            source_bound=True, basis_bound=True,
                            domain=(Domain.INTEGER, Domain.RATIONAL), optional=True)),
        result=exact_rank_result, memoizable=True,
        implementation_key="rexgraph.ranking_response.pagerank_solve",
        preconditions=("rational C0 fixed point on the explicitly selected native participation view",
                       "nonnegative rational seed and damping in [0, 1); no approximate fallback")))
