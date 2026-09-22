"""Direct inventory cases for the coordinate, temporal word and exact ranking operators."""
from fractions import Fraction as Q
from pathlib import Path
import numpy as np


def coordinate_case(rex,name):
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.temporal_calculus import TemporalWord,TemporalOperation
    from rexgraph.type_accession import CoordinateSpace,CoordinateField
    from rcql import call
    if name=='PAGERANK_SOLVE':return (call('PARTICIPATION_WALK'),)
    s=CoordinateSpace('inventory',('x',));i=CoordinateMap.identity(s)
    x=CoordinateField(rex,1,s,np.array([Q(1)],object),'chain')
    w=TemporalWord((i,),(i,),(i,i));op=TemporalOperation(i,i,i,i)
    return {'COORDINATE_APPLY':(i,x),'OPERATION_DELTA':(op,x,x),
            'INJECTION_DELTA':(op,x,x),'WORD_DELTA':(w,x),'KERNEL_MOMENTS':(w.moment_kernel(),x)}[name]


ADDITIONS=('COORDINATE_APPLY','OPERATION_DELTA','INJECTION_DELTA','WORD_DELTA','KERNEL_MOMENTS','PAGERANK_SOLVE')


def pytest_collection_modifyitems(session,config,items):
    modules={item.module for item in items if getattr(item,'module',None) is not None
             and Path(getattr(item.module,'__file__','')).name=='test_operator_inventory.py'}
    for module in modules:
        for name in ADDITIONS:
            if name in module.NATIVE_CASES:
                raise RuntimeError('a direct case would replace an existing inventory case')
            module.NATIVE_CASES[name]=lambda rex,name=name:coordinate_case(rex,name)
