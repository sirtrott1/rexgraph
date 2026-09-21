"""Exact ranking responses and their finite temporal contributions."""
from __future__ import annotations

from fractions import Fraction as Q
import numpy as np

from rexgraph.coordinate_map import CoordinateMap, _exact_values, _identity
from rexgraph.exact_green import ExactSparse
from rexgraph.graded_metric import _fraction
from rexgraph.temporal_calculus import TemporalDelta
from rexgraph.type_accession import CoordinateSpace

__all__ = ["exact_pagerank", "pagerank_delta"]


def _damping(value):
    value = _fraction(value)
    if not 0 <= value < 1:
        raise ValueError("exact PageRank damping must lie in [0, 1)")
    return value


def _seed(values, n):
    values = np.full(n, Q(1, n), object) if values is None else _exact_values(values, n)
    if values.ndim != 1 or any(v < 0 for v in values):
        raise ValueError("PageRank seed must be a nonnegative vector")
    mass = sum(values, Q(0))
    if mass <= 0:
        raise ValueError("PageRank seed must have positive mass")
    return values / mass


class _RankingSystem:
    def __init__(self, view, damping):
        from rexgraph.markov import MarkovView, _participation_entries, validate_markov_source
        from rexgraph.markov_oracle import PairwiseMarkovOracle
        self.view, self.damping = view, _damping(damping)
        if isinstance(view, CoordinateMap):
            if view.domain != view.codomain:
                raise ValueError("a stochastic coordinate map must act on one named space")
            n = len(view.domain.keys)
            sums = [Q(0)] * n
            for _, j, v in view.entries:
                if v < 0:
                    raise ValueError("transition probabilities must be nonnegative")
                sums[j] += v
            if any(s != 1 for s in sums):
                raise ValueError("transition columns must sum exactly to one")
            self.space = view.domain
        elif isinstance(view, (MarkovView, PairwiseMarkovOracle)):
            view.check_state()
            n = view.shape[0]
            self.space = CoordinateSpace("C0", tuple(str(i) for i in range(n)))
        else:
            raise TypeError("exact PageRank requires an explicit Markov view or stochastic coordinate map")
        if n == 0:
            raise ValueError("PageRank requires a nonempty state space")
        self.n = n
        data = {(i, i): Q(1) for i in range(n)}

        def add(i, j, v):
            data[i, j] = data.get((i, j), Q(0)) + v

        alpha = self.damping
        if isinstance(view, CoordinateMap):
            for i, j, v in view.entries:
                add(i, j, -alpha * v)
            dimension = n
        elif isinstance(view, MarkovView):
            m = int(view.source.nE)
            entries = _participation_entries(view.source, validate_markov_source(view.source))
            dv, de = [Q(0)] * n, [Q(0)] * m
            for v, e, w in entries:
                dv[v] += w
                de[e] += w
            for e in range(m):
                add(n+e, n+e, Q(1))
            for v, e, w in entries:
                add(v, n+e, -alpha*w/de[e])
                add(n+e, v, -w/dv[v])
            h = n+m
            add(h, h, Q(1))
            for v in range(n):
                add(v, h, -alpha/n)
                if not dv[v]:
                    add(h, v, -Q(1))
            dimension = h+1
        else:
            ptr, indices, edges = view.source._adjacency_bundle
            weights = validate_markov_source(view.source)
            weights = tuple(Q(1) for _ in range(view.source.nE)) if weights is None else tuple(weights)
            h = n
            add(h, h, Q(1))
            for j in range(n):
                terms = [(int(indices[k]), weights[int(edges[k])])
                         for k in range(int(ptr[j]), int(ptr[j+1]))]
                degree = sum((w for _, w in terms), Q(0))
                if degree:
                    for i, w in terms:
                        add(i, j, -alpha*w/degree)
                else:
                    add(h, j, -Q(1))
                add(j, h, -alpha/n)
            dimension = n+1
        self.system = ExactSparse(dimension, dimension, data)
        self.check_state()

    def check_state(self):
        if not isinstance(self.view, CoordinateMap):
            self.view.check_state()

    def apply(self, values):
        self.check_state()
        return (self.view.apply(values) if isinstance(self.view, CoordinateMap)
                else self.view.apply(values, exact=True))

    @property
    def digest(self):
        if isinstance(self.view, CoordinateMap):
            view_id = self.view.coefficient_digest
        else:
            from rexgraph.io.catalog import object_digest
            view_id = _identity((object_digest(self.view.source), self.view.construction,
                                 repr(self.view.parameters)))
        return _identity(("exact-ranking-v1", view_id, hex(self.damping.numerator), hex(self.damping.denominator)))

    def solve(self, rhs):
        rhs = _exact_values(rhs, self.n)
        if rhs.ndim != 1:
            raise ValueError("ranking solve requires one vector")
        self.check_state()
        extended = tuple(rhs) + (Q(0),) * (self.system.nrows-self.n)
        result = np.asarray(self.system.solve(extended)[:self.n], dtype=object)
        if not np.array_equal(result-self.damping*self.apply(result), rhs):
            raise ArithmeticError("ranking solve has a nonzero exact residual")
        self.check_state()
        return result


def exact_pagerank(view, damping=Q(17, 20), seed=None, *, report=False):
    """Solve the selected stochastic equation over Q without constructing an inverse."""
    if not isinstance(report, bool):
        raise TypeError("report must be a boolean")
    system = _RankingSystem(view, damping)
    source = _seed(seed, system.n)
    result = system.solve((1-system.damping)*source)
    if sum(result, Q(0)) != 1 or any(v < 0 for v in result):
        raise ArithmeticError("exact PageRank failed its mass certificate")
    info = {"method": "rational-sparse-ranking-solve", "coefficient_domain": "Q",
            "residual": Q(0), "system_size": system.system.nrows,
            "system_entries": len(system.system.entries), "operator_digest": system.digest}
    return (result, info) if report else result


def pagerank_delta(old_view, new_view, correspondence, *, damping=Q(17, 20), new_damping=None,
                   seed=None, new_seed=None, old_rank=None, metric=None):
    """Separate transition, personalization and damping changes with all interactions."""
    from rexgraph.chain_map import ChainMap, GradedMap
    from rexgraph.field_delta import validate_correspondence
    old = _RankingSystem(old_view, damping)
    new = _RankingSystem(new_view, old.damping if new_damping is None else new_damping)
    if isinstance(correspondence, (GradedMap, ChainMap)):
        mapping = validate_correspondence(correspondence)
        if (getattr(old_view, "source", None) is not mapping.domain.source
                or getattr(new_view, "source", None) is not mapping.codomain.source):
            raise ValueError("ranking views must belong to the correspondence endpoints")
        j = CoordinateMap(old.space, new.space, mapping.components[0])
        correspondence_digest = mapping.coefficient_digest
    elif isinstance(correspondence, CoordinateMap):
        j = correspondence
        correspondence_digest = j.coefficient_digest
    else:
        raise TypeError("ranking change requires an explicit coordinate or graded correspondence")
    if j.domain != old.space or j.codomain != new.space:
        raise ValueError("ranking correspondence must match the named state spaces")
    v, vp = _seed(seed, old.n), _seed(new_seed, new.n)
    pi = old.solve((1-old.damping)*v) if old_rank is None else _exact_values(old_rank, old.n)
    if pi.ndim != 1 or not np.array_equal(pi-old.damping*old.apply(pi), (1-old.damping)*v):
        raise ValueError("stored old ranking fails the exact source equation")
    transported = j.apply(pi)
    transition = new.solve(new.damping*(new.apply(transported)-j.apply(old.apply(pi))))
    personalization = new.solve((1-new.damping)*(vp-j.apply(v)))
    damping_change = new.solve((new.damping-old.damping)*j.apply(old.apply(pi)-v))
    delta = TemporalDelta(new.space, ("transition", "personalization", "damping"),
                          (transition, personalization, damping_change),
                          _identity((old.digest, new.digest, correspondence_digest)))
    result = transported+delta.values
    if not np.array_equal(result-new.damping*new.apply(result), (1-new.damping)*vp):
        raise ArithmeticError("temporal ranking decomposition failed the new equation")
    record = delta.as_record(metric)
    record.update(old_rank=tuple(pi), new_rank=tuple(result),
                  correspondence_digest=correspondence_digest,
                  old_operator_digest=old.digest, new_operator_digest=new.digest)
    return record
