"""Small exact fixtures for the tensor field tests."""
from fractions import Fraction as Q

from rexgraph.chain_map import CoordinateComplex
from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric
from rexgraph.native_field import NativeFieldCalculus
from rexgraph.span import SpanAttachment, SpanBlock
from rexgraph.type_accession import CoordinateSpace


def space(name, n):
    return CoordinateSpace(name, tuple(str(i) for i in range(n)))


def mapping(domain, codomain, m):
    return CoordinateMap(domain, codomain,
                         tuple((i, j, Q(v)) for i, row in enumerate(m) for j, v in enumerate(row) if v))


def square(filled=False, metric=None):
    spaces = (space('vertices', 4), space('edges', 4), space('faces', int(filled)))
    b = [[-1, 0, 0, 1], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]
    b1 = mapping(spaces[1], spaces[0], b)
    b2 = CoordinateMap(spaces[2], spaces[1], tuple((i, 0, 1) for i in range(4)) if filled else ())
    tower = CoordinateComplex(spaces, (b1.entries, b2.entries))
    metrics = tuple(CoordinateMetric.identity(s) for s in spaces)
    if metric is not None:
        metrics = (metrics[0], CoordinateMetric(spaces[1], tuple(
            (i, j, Q(v)) for i, row in enumerate(metric) for j, v in enumerate(row) if v)), metrics[2])
    return NativeFieldCalculus(tower, metrics)


def block(name, components):
    return SpanBlock(name, 'event_time', 'abstract_year',
                     tuple((str(i), Q(a), Q(b)) for i, (a, b) in enumerate(components)))


def attachment(name='a', owner='event_a', role='time', support=None):
    return SpanAttachment(name, owner, role, 'doc',
                          time=support or block('span', [(2017, 2019), (2024, 2026)]))
