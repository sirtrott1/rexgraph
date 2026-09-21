import numpy as np
from rexgraph.graph import RexGraph
from rexgraph.nn.lifecycle import create_checkpoint
from rexgraph.model_state import ModelBatch


def rex(ids=(11,12,13)):
    return RexGraph.from_hypergraph([0,2,4,6],[0,1,1,2,2,3], relation_ids=np.array(ids))


def state(source, **kwargs):
    return create_checkpoint(source, configuration={'n_classes':2}, optimizer={'name':'auto','lr':0.1}, seed=7, **kwargs)


def batch(model, labels=(0,999,1)):
    return ModelBatch(model.source, model.space, np.array(labels,dtype=np.int64), np.array([True,False,True]))


def equal_tree(a,b):
    from collections.abc import Mapping
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape
        assert np.array_equal(a,b)
        if not a.dtype.hasobject: assert a.tobytes()==b.tobytes()
    elif isinstance(a,Mapping):
        assert set(a)==set(b)
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True): equal_tree(x,y)
    else: assert a==b
