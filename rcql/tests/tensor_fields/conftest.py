"""Direct inventory cases for the tensor field and moment operators."""
from pathlib import Path
from fractions import Fraction as Q
import pytest


def tensor_cases(rex):
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.tensor_field import FieldSource, TensorField
    from rexgraph.tensor_moment import TensorMomentKernel, CoordinatePairing
    from rexgraph.temporal_field import TensorEvolution, SectorTransport
    from rexgraph.temporal_calculus import TemporalOperation
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.attachment_field import AttachmentField
    from rexgraph.span import SpanAttachment, SpanBlock
    c=NativeFieldCalculus.from_rex(rex);s=c.complex.spaces[1];state=FieldSource(rex)
    x=TensorField(s,[Q(1)]*len(s.keys),source=state,grade=1,variance='chain')
    i=CoordinateMap.identity(s);k=TensorMomentKernel(('identity',),(i,),common_metric=c.metrics[1])
    moments=k.evaluate(x);moment=moments.pair('identity','identity')
    evolution=TensorEvolution(TemporalOperation(i,i,i,i),state,state)
    channels=evolution.delta(x,x)
    one=moment.contracted_field()
    attachment=SpanAttachment('a','e','time','doc',time=SpanBlock('time','event','year',(('a',1,2),)))
    spans=AttachmentField((attachment,),state)
    pairing=CoordinatePairing.metric(c.metrics[1])
    return {
        'TENSOR_APPLY':(c.hodge(1),x),'NATIVE_RESPONSE':(x,1,c),'TENSOR_MOMENTS':(k,x),
        'FIELD_PAIR':(x,x,pairing),'MOMENT_PAIR':(moments,'identity','identity'),
        'MOMENT_SUPPORT':(moment,),'MOMENT_CONTRACT':(moment,),'TENSOR_SCALAR':(one,),
        'TENSOR_DELTA':(evolution,x,x),'CHANNEL_FIELD':(channels,'operation'),
        'CHANNEL_TOTAL':(channels,),'ATTACHMENTS':(), 'SPAN_FIELD':(spans,), 'SPAN_DELTA':(spans,spans),
        'SECTOR_FIELDS':(SectorTransport(c,c,1,i,state,state),x,x),
        'MOMENT_CHANGE':(x,x,i,x,x,i,pairing,pairing),
    }


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session,config,items):
    from rcql.tensor_contracts import ARGUMENTS
    modules={item.module for item in items if getattr(item,'module',None) is not None
             and Path(getattr(item.module,'__file__','')).name=='test_operator_inventory.py'}
    for module in modules:
        for name in ARGUMENTS:
            if name in module.NATIVE_CASES:
                raise RuntimeError('a direct case would replace an existing inventory case')
            module.NATIVE_CASES[name]=lambda rex,name=name:tensor_cases(rex)[name]
