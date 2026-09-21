from fractions import Fraction as Q
from rcql import NameRelation
from rcql.ast import Call,ListExpr,Literal,Parameter
from rexgraph.affine_feedback import FeedbackEquation,AffineFeedback
from rexgraph.type_accession import CoordinateSpace
from rexgraph.tensor_field import FieldSource,TensorField
from rexgraph.coordinate_map import CoordinateMap
from .helpers import counter,binding


def add(k):
    return NameRelation('add',('x',),Call('SUM',(ListExpr((Parameter('x'),Literal(k))),)))


def feedback(r):
    ref=FieldSource(r);s=CoordinateSpace('local',('x',))
    a=CoordinateMap(s,s,((0,0,Q(1,2)),));rhs=TensorField(s,[1],source=ref)
    return AffineFeedback('example',(('x',s),),(FeedbackEquation('relation','x',(('x',a),),rhs),),ref)


def query_case(name,r):
    base=NameRelation.operator('SUM'); p=counter()
    cases={
      'NAME':('SUM',), 'NAME_BIND':(base,'values',[1,2]),
      'NAME_REBIND':(base.bind('values',[1,2]),'values',[2,3]),
      'NAME_ALIAS':(base,'total'), 'NAME_PORT':(base,'values','items'),
      'NAME_CHAIN':(base,add(1),'x'), 'NAME_MODIFY':(base,add(1),'x','changed'),
      'NAME_APPLY':(base,[[1,2,3]]), 'NAME_EXPLAIN':(base,[[1,2,3]]),
      'NAME_RECORD':(base,), 'NAME_READ':(base.to_record(),), 'NAME_TOPOLOGY':(base,),
      'NAME_ITERATE':(add(1),0,3,'x'),
      'RECURSIVE_RUN':(p,'count',[3]), 'RECURSIVE_EXPLAIN':(p,'count',[3]),
      'RECURSIVE_RECORD':(p,), 'RECURSIVE_READ':(p.to_record(),), 'RECURSIVE_TOPOLOGY':(p,)}
    if name in cases:return cases[name]
    if name=='RECURSIVE_FIELDS':
        from .advanced_helpers import field_program, field
        op=field_program()
        out=op.execute(binding(r),'resolve',[1,field(r)])
        return (out,)
    if name.startswith('RECURSIVE_'):
        out=p.execute(binding(r),'count',[2])
        return (out.to_record(),) if name=='RECURSIVE_RESULT_READ' else (out,)
    if name.startswith('FEEDBACK_'):
        f=feedback(r)
        if name=='FEEDBACK_SELECT':return (f,'x')
        if name=='FEEDBACK_ITERATE':return (f,TensorField(f.space,[0],source=f.source),3)
        if name=='FEEDBACK_READ':
            from rcql.recursive_state import feedback_record
            return (feedback_record(f),)
        return (f,)
    raise AssertionError(name)
