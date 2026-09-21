from fractions import Fraction as F
import random
import numpy as np
import pytest
import sympy as sp
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.section_calculus import SectionFamily, InconsistentSectionError
from rexgraph.tensor_field import TensorField
from rexgraph.type_accession import CoordinateSpace


@pytest.mark.parametrize('seed',range(25))
def test_rational_affine_family_against_independent_symbolic_oracle(seed):
    rng=random.Random(seed);m=rng.randrange(0,6);n=rng.randrange(0,6)
    data={(i,j):F(rng.randrange(-2,3),rng.randrange(1,5)) for i in range(m) for j in range(n)}
    domain=CoordinateSpace('variables',tuple(map(str,range(n))));codomain=CoordinateSpace('equations',tuple(map(str,range(m))))
    D=CoordinateMap(domain,codomain,tuple((i,j,v) for (i,j),v in data.items() if v))
    x=np.array([F(rng.randrange(-3,4),5) for _ in range(n)],object)
    b=TensorField(codomain,D.apply(x))
    family=SectionFamily.solve(D,b)
    oracle=sp.zeros(m,n)
    for (i,j),v in data.items():oracle[i,j]=sp.Rational(v.numerator,v.denominator)
    assert family.dimension==n-oracle.rank()
    assert D.apply(family.particular.values).tolist()==b.values.tolist()
    assert not family.directions.compose(D).entries
    image=family.observe(D)
    assert image.determined
    assert image.value().values.tolist()==b.values.tolist()


def test_multiple_rhs_inconsistency_certificate():
    d=CoordinateSpace('x',('x',));c=CoordinateSpace('eq',('a','b'));axis=CoordinateSpace('trials',('first','second'))
    D=CoordinateMap(d,c,((0,0,1),(1,0,1)))
    b=TensorField(c,[[2,3],[2,4]],(axis,))
    with pytest.raises(InconsistentSectionError) as err:SectionFamily.solve(D,b)
    w=err.value.witness.values
    assert not any(D.T.apply(w))
    assert err.value.contradiction.tolist()[0]==0
    assert err.value.contradiction.tolist()[1]!=0


def test_no_equations_retains_all_free_directions():
    d=CoordinateSpace('x',('x','y'));c=CoordinateSpace('eq',())
    f=SectionFamily.solve(CoordinateMap(d,c,()),TensorField(c,[]))
    assert f.dimension==2


def test_no_variables_inconsistent_constraint():
    d=CoordinateSpace('x',());c=CoordinateSpace('eq',('bad',))
    with pytest.raises(InconsistentSectionError):SectionFamily.solve(CoordinateMap(d,c,()),TensorField(c,[1]))


def test_retained_axes_dimension_and_empty_observation():
    from rexgraph.type_accession import CoordinateSpace
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.tensor_field import TensorField
    from rexgraph.section_calculus import SectionFamily
    import numpy as np
    source = CoordinateSpace('state', ('x', 'y'))
    target = CoordinateSpace('constraints', ())
    eq = CoordinateMap(source, target, ())
    for size in (0, 1, 3):
        axis = CoordinateSpace('fields', tuple(str(i) for i in range(size)))
        rhs = TensorField(target, np.empty((0, size), dtype=object), axes=(axis,))
        family = SectionFamily.solve(eq, rhs)
        assert family.direction_count == 2
        assert family.dimension == 2 * size
        observation = family.observe(CoordinateMap.identity(source))
        assert observation.determined is (size == 0)
        if size == 0:
            assert observation.value().values.shape == (2, 0)
