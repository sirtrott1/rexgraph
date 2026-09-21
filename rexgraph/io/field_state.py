"""Versioned exact tensor encoding for retained fields and moment spans."""
from __future__ import annotations

import json
import numpy as np

from rexgraph.tensor_field import FieldSource,TensorField,TensorChannels
from rexgraph.tensor_moment import CoordinatePairing,MomentSpan
from rexgraph.temporal_field import MomentChange
from rexgraph.type_accession import CoordinateSpace

FIELD_VALUES=(TensorField,TensorChannels,MomentSpan,MomentChange)


def _space(space):
    return {"name":space.name,"keys":space.keys}


def _source(source):
    return None if source is None else source.as_record()


def pack_field(value):
    """Encode retained exact values without serializing live source objects."""
    tensors={}
    def put(node,path):
        if isinstance(node,TensorField):
            tensors[path+"values"]=np.array(node.values,dtype=object,copy=True)
            return {"kind":"field","space":_space(node.space),"axes":[_space(a) for a in node.axes],
                    "source":_source(node.source),"grade":node.grade,"variance":node.variance,
                    "provenance":node.provenance,"tensor":path+"values",
                    "dependencies":[s.as_record() for s in node.dependencies]}
        if isinstance(node,TensorChannels):
            return {"kind":"channels","names":node.names,"declaration":node.declaration_digest,
                    "sources":[s.as_record() for s in node.endpoint_sources],
                    "fields":[put(f,path+f"field/{i}/") for i,f in enumerate(node.fields)]}
        if isinstance(node,MomentSpan):
            if not isinstance(node.pairing,CoordinatePairing):
                raise TypeError("persist an explicit realized MomentSpan, not a live realization action")
            form=node.pairing
            tensors[path+"indices"]=np.asarray([(i,j) for i,j,_ in form.entries],dtype=np.int64).reshape((-1,2))
            tensors[path+"coefficients"]=np.asarray([v for _,_,v in form.entries],dtype=object)
            return {"kind":"moment","left":put(node.left,path+"left/"),"right":put(node.right,path+"right/"),
                    "form":{"left":_space(form.left),"right":_space(form.right),"indices":path+"indices",
                            "coefficients":path+"coefficients"},"channels":node.channel_keys,"kernel":node.kernel_digest,
                    "sources":[s.as_record() for s in node.endpoint_sources]}
        if isinstance(node,MomentChange):
            tensors[path+"weights"]=np.asarray([w for _,w,_ in node.terms],dtype=object)
            return {"kind":"change","old":put(node.old,path+"old/"),"new":put(node.new,path+"new/"),
                    "names":[n for n,_,_ in node.terms],"weights":path+"weights",
                    "terms":[put(v,path+f"term/{i}/") for i,(_,_,v) in enumerate(node.terms)]}
        raise TypeError("unsupported retained exact value")
    spec={"version":1,"root":put(value,"value/")}
    tensors["spec"]=np.frombuffer(json.dumps(spec,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode(),dtype=np.uint8).copy()
    return tensors


def unpack_field(tensors):
    """Restore exact values with detached native state references."""
    def space(s):
        return CoordinateSpace(s["name"],tuple(s["keys"]))
    def source(s):
        return None if s is None else FieldSource(None,s["record_id"],s["version"],s["state_digest"])
    def take(node):
        kind=node["kind"]
        if kind=="field":
            return TensorField(space(node["space"]),tensors[node["tensor"]],tuple(space(a) for a in node["axes"]),
                               source(node["source"]),node["grade"],node["variance"],tuple(node["provenance"]),
                               tuple(source(s) for s in node["dependencies"]))
        if kind=="channels":
            return TensorChannels(tuple(node["names"]),tuple(take(v) for v in node["fields"]),node["declaration"],
                                  tuple(source(s) for s in node["sources"]))
        if kind=="moment":
            form=node["form"]
            indices=np.asarray(tensors[form["indices"]])
            coefficients=np.asarray(tensors[form["coefficients"]],dtype=object)
            if indices.shape!=(len(coefficients),2) or indices.dtype.kind not in "iu" or coefficients.ndim!=1:
                raise ValueError("invalid retained cross form arrays")
            pairing=CoordinatePairing(space(form["left"]),space(form["right"]),
                tuple((int(i),int(j),v) for (i,j),v in zip(indices,coefficients,strict=True)))
            return MomentSpan(take(node["left"]),take(node["right"]),pairing,tuple(node["channels"]),node["kernel"],tuple(source(s) for s in node["sources"]))
        if kind=="change":
            values=np.asarray(tensors[node["weights"]],dtype=object)
            if values.ndim!=1:
                raise ValueError("invalid moment change weights")
            return MomentChange(take(node["old"]),take(node["new"]),
                tuple((name,w,take(v)) for name,w,v in zip(node["names"],values,node["terms"],strict=True)))
        raise ValueError("unknown retained field schema kind")
    try:
        raw=np.asarray(tensors["spec"])
        if raw.dtype!=np.uint8 or raw.ndim!=1:
            raise ValueError("invalid retained field specification")
        spec=json.loads(raw.tobytes().decode())
        if spec.get("version")!=1:
            raise ValueError("unsupported retained field schema version")
        return take(spec["root"])
    except (KeyError,UnicodeError,json.JSONDecodeError,TypeError) as exc:
        raise ValueError("invalid retained field tensor payload") from exc
