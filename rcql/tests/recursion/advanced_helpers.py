from rcql import RecursiveProgram,RecursiveDefinition,ProgramInput,ValueKind,recur
from rcql.ast import Parameter,Literal,Call,Comparison
from rexgraph.native_field import NativeFieldCalculus
from rexgraph.tensor_field import TensorField,FieldSource
from rexgraph.type_accession import CoordinateSpace
from .helpers import integer,minus


def field(r,record_id=None,version=None):
    space=NativeFieldCalculus.from_rex(r).complex.spaces[1]
    return TensorField(space,[[int(i==0),int(i==r.nE-1)] for i in range(r.nE)],
        (CoordinateSpace('sources',('left','right')),),FieldSource(r,record_id,version),1,'chain')


def field_program():
    n,x=Parameter('n'),Parameter('x')
    t=ProgramInput('x',ValueKind.TENSOR_FIELD.value,1,'chain')
    d=RecursiveDefinition('resolve',(integer('n'),t),Comparison('==',n,Literal(0)),x,
         recur('resolve',minus(n),Call('NATIVE_RESPONSE',(x,))),
         ProgramInput('out',ValueKind.TENSOR_FIELD.value,1,'chain'),result_like='x',decreases='n')
    return RecursiveProgram('field_evolution',(d,))
