from fractions import Fraction as Q
from rexgraph.graph import RexGraph
from rexgraph.native_field import NativeFieldCalculus
from rexgraph.tensor_field import TensorField, FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric
from rexgraph.tensor_moment import CoordinatePairing
from rcql import (Executor, Program, ProgramStep, OutputRef, parse, bind, SourcePolicy)


def native():
    rex=RexGraph.from_graph([0,1,2],[1,2,3])
    calculus=NativeFieldCalculus.from_rex(rex)
    space=calculus.complex.spaces[1]
    field=TensorField(space,[1,0,0],source=FieldSource(rex),grade=1,variance='chain')
    return rex,field,CoordinatePairing.metric(CoordinateMetric.identity(space))


def tensor(rex=None):
    rex=rex or native()[0]
    space=CoordinateSpace('support',('x','y'))
    a=CoordinateSpace('left/sample',('0','1'))
    b=CoordinateSpace('right/sample',('0','1'))
    return TensorField(space,[[[1,2],[3,4]],[[5,6],[7,8]]],(a,b),FieldSource(rex))


def family(rex):
    from rexgraph.section_calculus import SectionFamily
    space=CoordinateSpace('local',('a','b','c','d'))
    equations=CoordinateSpace('equations',('agree','observe'))
    D=CoordinateMap(space,equations,((0,0,1),(0,2,-1),(1,0,1)))
    rhs=TensorField(equations,[0,Q(2,7)],source=FieldSource(rex))
    result=SectionFamily.solve(D,rhs,'fixture')
    out=CoordinateSpace('answer',('value',))
    return result, CoordinateMap(space,out,((0,0,1),)),CoordinateMap(space,out,((0,2,1),))


def count_program():
    return Program('counts',(
        ProgramStep('count',parse('FROM $r RETURN COUNT(CELLS(1))')),
        ProgramStep('read',parse('FROM $r RETURN $n'),(('n',OutputRef('count')),))))


def query_case(name,rex):
    from rexgraph.process_field import sampled_field
    if name=='TENSOR_SELECT':return tensor(rex),'left/sample','0'
    if name=='TENSOR_CONTRACT':return tensor(rex),'left/sample',(Q(1),Q(2))
    if name=='TENSOR_DIAGONAL':return tensor(rex),'left/sample','right/sample','sample'
    if name=='SAMPLE_RATES':
        f=TensorField(CoordinateSpace('values',('a',)),[0],source=FieldSource(rex))
        return (sampled_field((f,f.with_values([2]),f),(0,Q(1,3),1),axis='elapsed',unit='s'),)
    if name=='SECTION_EQUIVALENT':return family(rex)
    p=count_program()
    if name=='PROGRAM_READ':return (p.to_record(),)
    if name=='PROGRAM_RECORD':return (p,)
    if name in ('PROGRAM_RUN','PROGRAM_EXPLAIN','PROGRAM_TOPOLOGY'):
        return p,{'r':bind('r',rex,SourcePolicy.allow('*'))},{}
    if name=='PROGRAM_OUTPUT':return Executor(sources={'r':rex}).execute_program(p),0
    raise KeyError(name)
