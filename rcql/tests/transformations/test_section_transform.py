from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.type_accession import CoordinateSpace
from rexgraph.tensor_field import TensorField, FieldSource
from rexgraph.section_calculus import SectionFamily, UnderdeterminedSectionError
from rcql import Executor, ProgramTransformation, NameRelation, parse, bind, SourcePolicy
from rcql.program_codec import dumps, loads
from rcql.readout_equivalence import ReadoutEquivalence


def fixture(rex=None, *, offset=0, axes=(), reference=None):
    rex = RexGraph.from_cells([2, [[0, 1]]]) if rex is None else rex
    space = CoordinateSpace('latent', ('x', 'y'))
    equations = CoordinateSpace('compatibility', ('difference',))
    result = CoordinateSpace('readout', ('value',))
    shape = (1, *(len(a.keys) for a in axes))
    rhs = np.full(shape, offset, dtype=object)
    family = SectionFamily.solve(CoordinateMap(space, equations, ((0, 0, 1), (0, 1, -1))),
        TensorField(equations, rhs, axes, reference or FieldSource(rex)))
    p = CoordinateMap(space, result, ((0, 0, Q(2**140+1, 7)),))
    q = CoordinateMap(space, result, ((0, 1, Q(2**140+1, 7)),))
    return rex, family, p, q


def runtime(rex, **params):
    return Executor(sources={'r': rex}, params=params)


def test_equivalent_does_not_mean_determined():
    r, f, p, q = fixture()
    t = ProgramTransformation.section(f, p, q)
    n = t.original()
    binding = bind('r', r, SourcePolicy.allow('read'))
    candidate = t.compile(binding, n, [f, p, q])
    a, b = n.apply(binding, [f, p, q]), candidate.apply(binding, [f, p, q])
    assert not a.determined and not b.determined
    assert a.directions.entries == b.directions.entries == ((0, 0, Q(2**140+1, 7)),)
    with pytest.raises(UnderdeterminedSectionError):
        b.value()
    proof = t.verify(n)
    assert not any(proof['chain_residuals'])
    assert proof['dependencies'] == (f.coefficient_digest, p.coefficient_digest, q.coefficient_digest)
    assert 'requires exact live inputs' in proof['readout_status']
    assert n.to_bytes() == t.original().to_bytes()


def test_real_sheaf_gluing():
    from rexgraph.sheaf import ExactSheaf
    from rexgraph.section_calculus import SectionSystem
    r = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    system = SectionSystem.from_sheaf(ExactSheaf(r, grade=1))
    f = system.complete()
    output = CoordinateSpace('shared', ('scalar',))
    p = CoordinateMap(system.space, output, ((0, 0, 1),))
    q = CoordinateMap(system.space, output, ((0, 1, 1),))
    assert f.dimension == 1
    t = ProgramTransformation.section(f, p, q)
    assert t.compile(bind('r', r, SourcePolicy.allow('read')), t.original(), [f, p, q])


@pytest.mark.parametrize('axes', [(), (CoordinateSpace('sample', ('a', 'b')),),
    (CoordinateSpace('sample', ('a', 'b')), CoordinateSpace('channel', ('c', 'd')))])
def test_retained_axes(axes):
    r, f, p, q = fixture(axes=axes)
    x = runtime(r, f=f, p=p, q=q).execute(parse('''FROM $r
        LET t = TRANSFORM_SECTION($f,$p,$q)
        LET n = TRANSFORM_COMPILE(t,TRANSFORM_SOURCE(t),[$f,$p,$q])
        RETURN NAME_APPLY(n,[$f,$p,$q])''')).values[0]
    assert x.particular.axes == axes and x.family.dimension == np.prod([len(a.keys) for a in axes])
    assert x.directions.entries == ((0, 0, Q(2**140+1, 7)),)


@pytest.mark.parametrize('reason', ['offset', 'variation', 'codomain', 'domain'])
def test_false_equivalence_refused(reason):
    r, f, p, q = fixture(offset=Q(1, 13) if reason == 'offset' else 0)
    if reason == 'variation':
        q = CoordinateMap(q.domain, q.codomain, ())
    elif reason == 'codomain':
        q = replace(q, codomain=CoordinateSpace('another', q.codomain.keys))
    elif reason == 'domain':
        q = replace(q, domain=CoordinateSpace('another', q.domain.keys))
    with pytest.raises(ValueError):
        ProgramTransformation.section(f, p, q)


def test_zero_kernel_nonzero_offset_refused():
    r, f, p, q = fixture()
    eq = CoordinateMap(f.particular.space, f.particular.space, ((0, 0, 1), (1, 1, 1)))
    unique = SectionFamily.solve(eq, TensorField(eq.codomain, [1, 2], source=FieldSource(r)))
    assert unique.dimension == 0
    with pytest.raises(ValueError, match='differ'):
        ProgramTransformation.section(unique, p, q)


def test_no_candidate_execution_during_compile(monkeypatch):
    from rcql.operators import _REGISTRY
    r, f, p, q = fixture()
    original = _REGISTRY['SECTION_CERTIFIED_OBSERVE']
    calls = []
    def counted(*args):
        calls.append(1)
        return original.fn(*args)
    monkeypatch.setitem(_REGISTRY, 'SECTION_CERTIFIED_OBSERVE', replace(original, fn=counted))
    e = runtime(r, f=f, p=p, q=q)
    query = '''FROM $r LET t=TRANSFORM_SECTION($f,$p,$q)
        RETURN TRANSFORM_COMPILE(t,TRANSFORM_SOURCE(t),[$f,$p,$q])'''
    assert e.execute(parse('EXPLAIN '+query)).execution == () and not calls
    candidate = e.execute(parse(query)).values[0]
    assert not calls
    runtime(r, f=f, p=p, q=q, n=candidate).execute(parse('FROM $r RETURN NAME_APPLY($n,[$f,$p,$q])'))
    assert calls == [1]


@pytest.mark.parametrize('change', ['family', 'left', 'right', 'source', 'record_version', 'dependency'])
def test_candidate_rechecks_actual_inputs(change):
    r, f, p, q = fixture()
    t = ProgramTransformation.section(f, p, q)
    b = bind('r', r, SourcePolicy.allow('read'))
    candidate = t.compile(b, t.original(), [f, p, q])
    if change == 'family':
        f = replace(f, declaration_digest='another system')
    elif change in ('left', 'right'):
        changed = CoordinateMap(p.domain, p.codomain, ((0, 0, 1), (0, 1, -1)))
        if change == 'left':
            p = changed
        else:
            q = changed
    elif change == 'source':
        r = RexGraph.from_cells([2, [[0, 1]]])
    elif change == 'record_version':
        f = replace(f, rhs=replace(f.rhs, source=FieldSource(r, 'r', 2)),
                    particular=replace(f.particular, source=FieldSource(r, 'r', 2)))
    else:
        ref = FieldSource(r, 'contributor', 1)
        f = replace(f, rhs=replace(f.rhs, dependencies=(ref,)),
                    particular=replace(f.particular, dependencies=(ref,)))
    with pytest.raises((ValueError, PermissionError)):
        runtime(r, n=candidate, f=f, p=p, q=q).execute(parse('FROM $r RETURN NAME_APPLY($n,[$f,$p,$q])'))


@pytest.mark.parametrize('field', ['family', 'left', 'right', 'certificate', 'equivalent', 'version', 'extra'])
def test_readout_claim_tamper(field):
    r, f, p, q = fixture()
    certificate = ReadoutEquivalence.check(f, p, q)
    payload = loads(certificate.to_bytes())
    payload[field] = True if field == 'version' else False if field == 'equivalent' else '0'*64
    with pytest.raises((ValueError, TypeError)):
        ReadoutEquivalence.verify_bytes(dumps(payload), f, p, q)


def test_forged_claim_is_never_execution_authority():
    from rcql.program_transformation import _section_name
    r, f, p, q = fixture()
    raw = loads(ReadoutEquivalence.check(f, p, q).to_bytes())
    raw['certificate'] = '0'*64
    forged = dumps(raw)
    t = ProgramTransformation._create(_section_name(forged, 'left'), _section_name(forged, 'right'),
                                      'section_readout', (forged,))
    assert 'requires exact live inputs' in t.verify()['readout_status']
    with pytest.raises(ValueError, match='does not match'):
        t.compile(bind('r', r, SourcePolicy.allow('read')), t.original(), [f, p, q])
    with pytest.raises(ValueError, match='does not match'):
        _section_name(forged, 'right').apply(bind('r', r, SourcePolicy.allow('read')), [f, p, q])


@pytest.mark.parametrize('format', ['bytes', 'rcbd', 'safetensors', 'rcdb'])
def test_transformation_roundtrip(tmp_path, format):
    r, f, p, q = fixture()
    t = ProgramTransformation.section(f, p, q)
    if format == 'bytes':
        restored = ProgramTransformation.from_bytes(t.to_bytes())
    else:
        record = t.to_record()
        if format == 'rcbd':
            from rexgraph.io.bundle import save_rcbd, load_rcbd
            path = tmp_path/'t.rcbd'
            save_rcbd(path, record)
            record = load_rcbd(path)
        elif format == 'safetensors':
            from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
            path = tmp_path/'t.safetensors'
            rex_to_safetensors(record, path)
            record = safetensors_to_rex(path)
        else:
            from rcdb import open_store
            uri = 'file://'+str(tmp_path/'store')
            db = open_store(uri)
            db.commit_mutation('t', record, expected_version=0, analytics=False)
            db.close()
            db = open_store(uri)
            record = db.read_record('t', version=1).value
            assert db.verify_commits('t')
            db.close()
        restored = ProgramTransformation.from_record(record)
    assert restored.coefficient_digest == t.coefficient_digest
    b = bind('r', r, SourcePolicy.allow('read'))
    n = restored.compile(b, restored.original(), [f, p, q])
    assert n.apply(b, [f, p, q]).action.coefficient_digest == q.coefficient_digest
    assert restored.topology().corelations() == t.topology().corelations()


def case(name, rex):
    _, f, p, q = fixture(rex)
    t = ProgramTransformation.section(f, p, q)
    return {'TRANSFORM_SECTION': (f, p, q), 'TRANSFORM_SOURCE': (t,),
            'SECTION_CERTIFIED_OBSERVE': (f, p, q, ReadoutEquivalence.check(f, p, q).to_bytes())}[name]


@pytest.mark.parametrize('name', ['TRANSFORM_SECTION', 'TRANSFORM_SOURCE', 'SECTION_CERTIFIED_OBSERVE'])
def test_operator_contract(name):
    from rcql import call, query, source
    from rcql.operators import get_operator
    r, _, _, _ = fixture()
    values = case(name, r)
    direct = get_operator(name).fn(r, *values)
    typed = runtime(r).execute(query(source('r'), call(name, *values)))
    assert typed.values[0].coefficient_digest == direct.coefficient_digest


def test_unique_determined_value():
    r, f, p, q = fixture()
    eq = CoordinateMap(f.particular.space, f.particular.space, ((0, 0, 1), (1, 1, 1)))
    f = SectionFamily.solve(eq, TensorField(eq.codomain, [Q(3, 11)]*2, source=FieldSource(r)))
    t = ProgramTransformation.section(f, p, q)
    b = bind('r', r, SourcePolicy.allow('read'))
    result = t.compile(b, t.original(), [f, p, q]).apply(b, [f, p, q])
    assert result.value().values[0] == Q(3*(2**140+1), 77)


def test_stored_family_requires_explicit_restore():
    from rexgraph.io.section_state import pack_section, unpack_section
    r, f, p, q = fixture()
    t = ProgramTransformation.section(f, p, q)
    detached = unpack_section(pack_section(f))
    with pytest.raises(ValueError, match='live selected'):
        t.compile(bind('r', r, SourcePolicy.allow('read')), t.original(), [detached, p, q])
    e = runtime(r, f=detached, p=p, q=q, t=t)
    result = e.execute(parse('''FROM $r LET f=SECTION_RESTORE($f)
        LET n=TRANSFORM_COMPILE($t,TRANSFORM_SOURCE($t),[f,$p,$q])
        RETURN NAME_APPLY(n,[f,$p,$q])''')).values[0]
    assert result.family.coefficient_digest == f.coefficient_digest


def test_meta_program_and_persistent_cache():
    from rcql import Program, ProgramInput, ProgramStep, OutputRef, QueryCache
    from rcdb import open_store
    r, f, p, q = fixture()
    query = parse('FROM $r RETURN TRANSFORM_SECTION($f,$p,$q)')
    program = Program('readout_replacement', (ProgramStep('proposal', query),),
        (ProgramInput('f','SectionFamily'), ProgramInput('p','CoordinateMap'),
         ProgramInput('q','CoordinateMap')), (('transformation',OutputRef('proposal')),))
    program = Program.from_record(program.to_record())
    e = runtime(r, f=f, p=p, q=q)
    proposal = e.execute_program(program).values[0]
    assert e.execute_program(program, explain=True)['evaluation'].startswith('none')
    db = open_store('memory://')
    try:
        cache = QueryCache(db)
        first, second = e.execute_cached(query, cache), e.execute_cached(query, cache)
        assert second.native_plan['cache']['hit']
        assert first.values[0].coefficient_digest == second.values[0].coefficient_digest == proposal.coefficient_digest
        e.params['f'] = replace(f, declaration_digest='new declaration')
        third = e.execute_cached(query, cache)
        assert not third.native_plan['cache']['hit']
        assert third.values[0].coefficient_digest != proposal.coefficient_digest
    finally:
        db.close()


def test_historical_cutoff_and_permissions():
    from rcql import SnapshotContext, SourceSelection, BoundSource
    from rcdb import open_store
    r, _, _, _ = fixture()
    db = open_store('memory://')
    try:
        db.commit_mutation('data', r, expected_version=0, tx_time=10, analytics=False)
        context = SnapshotContext.select(db, (SourceSelection('r','data'),), cutoff=20)
        selected = context.sources['r']
        _, f, p, q = fixture(selected.value, reference=FieldSource(selected.value,'data',1))
        e = Executor(sources=context.sources, evidence=context, params={'f':f,'p':p,'q':q})
        text = '''FROM $r LET t=TRANSFORM_SECTION($f,$p,$q)
            RETURN TRANSFORM_COMPILE(t,TRANSFORM_SOURCE(t),[$f,$p,$q])'''
        candidate = e.execute(parse(text)).values[0]
        db.commit_mutation('data', r, expected_version=1, tx_time=30, analytics=False)
        future = db.read_record('data',version=2)
        _, new, _, _ = fixture(future.value, reference=FieldSource(future.value,'data',2))
        e.params.update(n=candidate, f=new)
        with pytest.raises(ValueError):
            e.execute(parse('FROM $r RETURN NAME_APPLY($n,[$f,$p,$q])'))
        denied = Executor(sources={'r':BoundSource(selected.value, SourcePolicy.allow())},
                          params={'n':candidate,'f':f,'p':p,'q':q})
        with pytest.raises(PermissionError):
            denied.execute(parse('FROM $r RETURN NAME_APPLY($n,[$f,$p,$q])'))
    finally:
        db.close()


@pytest.mark.parametrize('side', ['left', 'right', 'other'])
def test_explicit_readout_side(side):
    r, f, p, q = fixture()
    raw = ReadoutEquivalence.check(f,p,q).to_bytes()
    e = runtime(r,f=f,p=p,q=q,proof=raw,side=side)
    text = 'FROM $r RETURN SECTION_CERTIFIED_OBSERVE($f,$p,$q,$proof,$side)'
    if side == 'other':
        with pytest.raises(ValueError, match='side'):
            e.execute(parse(text))
    else:
        result = e.execute(parse(text)).values[0]
        assert result.action.coefficient_digest == (p if side == 'left' else q).coefficient_digest


def test_certified_candidate_composes_with_existing_names():
    r, f, p, q = fixture()
    eq = CoordinateMap(f.particular.space, f.particular.space, ((0, 0, 1), (1, 1, 1)))
    f = SectionFamily.solve(eq, TensorField(eq.codomain, [1, 1], source=FieldSource(r)))
    b = bind('r', r, SourcePolicy.allow('read'))
    t = ProgramTransformation.section(f, p, q)
    n = t.compile(b, t.original(), [f, p, q])
    renamed = ProgramTransformation.name(n, 'port', ['family', 'sections'])
    candidate = renamed.compile(b, n, [f, p, q])
    chained = candidate.then(NameRelation.operator('SECTION_VALUE'), 'image', name='completed_readout')
    restored = NameRelation.from_record(chained.to_record())
    result = restored.apply(b, [f, p, q])
    assert result.values[0] == Q(2**140+1, 7)
