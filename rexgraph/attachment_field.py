"""Exact nested observations of persisted annotation attachments."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import numpy as np

from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric, CoordinateWord, _exact_values, _identity
from rexgraph.graded_metric import _fraction
from rexgraph.span import SpanAttachment, SpanRefinement
from rexgraph.tensor_field import FieldSource, TensorField, apply_tensor
from rexgraph.type_accession import CoordinateSpace

__all__ = ["IntervalAction", "AttachmentField", "AttachmentObservation", "AttachmentAction",
           "common_attachment_observations"]


def _key(*parts):
    return json.dumps(parts,ensure_ascii=False,separators=(",",":"))


@dataclass(frozen=True)
class IntervalAction:
    """Range incidence evaluated by endpoint accumulation and prefix sums."""
    domain: CoordinateSpace
    codomain: CoordinateSpace
    ranges: tuple
    transposed: bool = False

    def __post_init__(self):
        if not isinstance(self.domain,CoordinateSpace) or not isinstance(self.codomain,CoordinateSpace):
            raise TypeError("interval action requires named endpoints")
        if not isinstance(self.transposed,bool):
            raise TypeError("transpose flag must be boolean")
        rows,columns=self.shape if not self.transposed else self.shape[::-1]
        ranges=[]
        for start,stop,column,weight in self.ranges:
            if (any(isinstance(i,bool) or not isinstance(i,int) for i in (start,stop,column))
                    or not 0<=start<=stop<=rows or not 0<=column<columns):
                raise ValueError("range lies outside the declared interval coordinates")
            ranges.append((start,stop,column,_fraction(weight)))
        object.__setattr__(self,"ranges",tuple(ranges))

    @property
    def shape(self):
        return len(self.codomain.keys),len(self.domain.keys)

    @property
    def coefficient_digest(self):
        return _identity(("interval_action_v1",(self.domain.name,self.domain.keys),(self.codomain.name,self.codomain.keys),
                          tuple((a,b,j,hex(w.numerator),hex(w.denominator)) for a,b,j,w in self.ranges),self.transposed))

    @property
    def T(self):
        return IntervalAction(self.codomain,self.domain,self.ranges,not self.transposed)

    def apply(self,values):
        values=_exact_values(values,self.shape[1])
        shape=(self.shape[0],*values.shape[1:])
        if self.transposed:
            prefix=np.full((self.shape[1]+1,*values.shape[1:]),Fraction(0),dtype=object)
            for i in range(self.shape[1]):
                prefix[i+1]=prefix[i]+values[i]
            result=np.full(shape,Fraction(0),dtype=object)
            for start,stop,column,weight in self.ranges:
                result[column]+=weight*(prefix[stop]-prefix[start])
            return result
        difference=np.full((self.shape[0]+1,*values.shape[1:]),Fraction(0),dtype=object)
        for start,stop,column,weight in self.ranges:
            value=weight*values[column]
            difference[start]+=value
            difference[stop]-=value
        result=np.full(shape,Fraction(0),dtype=object)
        running=np.full(values.shape[1:],Fraction(0),dtype=object)
        for i in range(self.shape[0]):
            running=running+difference[i]
            result[i]=running
        return result

    def transpose_apply(self,values):
        return self.T.apply(values)

    def then(self,following):
        return CoordinateWord((self,following))


def _merged(components):
    result=[]
    for start,stop in sorted((a,b) for _,a,b in components):
        if result and start<=result[-1][1]:
            result[-1]=(result[-1][0],max(result[-1][1],stop))
        else:
            result.append((start,stop))
    return tuple(result)


@dataclass(frozen=True)
class AttachmentField:
    """Identified attachments with several roles allowed on one primary event."""
    attachments: tuple[SpanAttachment,...]
    source: FieldSource | None = None
    addresses: tuple = ()

    def __post_init__(self):
        attachments=tuple(self.attachments)
        if any(not isinstance(a,SpanAttachment) for a in attachments):
            raise TypeError("attachment field requires native SpanAttachments")
        keys=tuple(_key(a.source_id,a.annotation_id) for a in attachments)
        if len(set(keys))!=len(keys):
            raise ValueError("each attachment needs a distinct assertion identity within its source")
        if self.source is not None and not isinstance(self.source,FieldSource):
            raise TypeError("attachment source requires FieldSource")
        if self.addresses and len(self.addresses)!=len(attachments):
            raise ValueError("addresses must match the selected attachments")
        object.__setattr__(self,"attachments",attachments)
        object.__setattr__(self,"addresses",tuple(self.addresses))

    @classmethod
    def from_source(cls,source,*,annotation_ids=None,roles=None,record_id=None,version=None):
        from rexgraph.graph import RexGraph
        selected=[]
        wanted=None if annotation_ids is None else set(annotation_ids)
        wanted_roles=None if roles is None else set(roles)
        def visit(rex,path,active):
            if id(rex) in active:
                raise ValueError("nested native source contains a cycle")
            active=active|{id(rex)}
            for grade,cells in sorted(getattr(rex,"_cell_metadata",{}).items()):
                for cell,values in sorted(cells.items()):
                    for slot,value in sorted(values.items()):
                        address=(*path,(int(grade),int(cell),slot))
                        if isinstance(value,SpanAttachment):
                            if ((wanted is None or value.annotation_id in wanted)
                                    and (wanted_roles is None or value.role in wanted_roles)):
                                selected.append((value,address))
                        elif isinstance(value,RexGraph):
                            visit(value,address,active)
        visit(source,(),set())
        selected.sort(key=lambda v:(v[0].source_id,v[0].annotation_id))
        return cls(tuple(a for a,_ in selected),FieldSource(source,record_id,version),
                   tuple(address for _,address in selected))

    @property
    def space(self):
        return CoordinateSpace("attachment_assertions",tuple(_key(a.source_id,a.annotation_id) for a in self.attachments))

    @property
    def coefficient_digest(self):
        return _identity(("attachment_field_v1",tuple(a.coefficient_digest for a in self.attachments),self.addresses,
                          None if self.source is None else self.source.coefficient_digest))

    def check_state(self):
        if self.source is not None:
            self.source.check()

    def amplitudes(self,values=None,*,axes=()):
        values=[Fraction(1)]*len(self.attachments) if values is None else values
        return TensorField(self.space,values,axes,self.source,provenance=(self.coefficient_digest,))

    def correspondence(self,other):
        if not isinstance(other,AttachmentField):
            raise TypeError("attachment correspondence requires AttachmentField")
        index={key:i for i,key in enumerate(other.space.keys)}
        return CoordinateMap(self.space,other.space,tuple((index[key],j,1)
                             for j,key in enumerate(self.space.keys) if key in index))

    def observe(self,*,support="time",local=True,mode="sum",refinement=None,scopes=None):
        return AttachmentObservation(self,support,local,mode,refinement,scopes)

    def as_record(self):
        return {"source":None if self.source is None else self.source.as_record(),"keys":self.space.keys,
                "attachment_digests":tuple(a.coefficient_digest for a in self.attachments),
                "addresses":self.addresses,"coefficient_digest":self.coefficient_digest}


@dataclass(frozen=True)
class AttachmentObservation:
    """Retained amplitude, fragment grounding and interval realization factors."""
    field: AttachmentField
    support: str = "time"
    local: bool = True
    mode: str = "sum"
    refinement: SpanRefinement | None = None
    scopes: tuple | None = None

    def __post_init__(self):
        if not isinstance(self.field,AttachmentField):
            raise TypeError("observation requires AttachmentField")
        if self.support not in {"text","time","grounded_time"} or self.mode not in {"sum","union"}:
            raise ValueError("unknown support observation")
        if not isinstance(self.local,bool):
            raise TypeError("local observation flag must be boolean")
        if self.support=="grounded_time" and self.mode!="sum":
            raise ValueError("grounded component fields retain a signed sum, not a union")
        blocks=tuple(getattr(a,"text" if self.support=="text" else "time") for a in self.field.attachments)
        if any(b is None for b in blocks):
            raise ValueError("every selected attachment must declare the requested support")
        if any(b.interpretation!="joint" for b in blocks):
            raise ValueError("alternative supports require an explicit choice before coverage")
        if self.support=="grounded_time" and any(a.grounding is None for a in self.field.attachments):
            raise ValueError("grounded time requires every fragment grounding map")
        if not blocks and self.refinement is None:
            raise ValueError("an empty attachment field requires an explicit comparison refinement")
        refinement=SpanRefinement.from_blocks(blocks) if self.refinement is None else self.refinement
        if not isinstance(refinement,SpanRefinement) or any(
                (b.axis,b.unit)!=(refinement.axis,refinement.unit) or
                not {p for _,a,z in b.components for p in (a,z)}.issubset(refinement.points) for b in blocks):
            raise ValueError("refinement must retain every requested support endpoint")
        own=tuple(self.scope(a) for a in self.field.attachments)
        scopes=tuple(sorted(set(own))) if self.scopes is None else tuple(self.scopes)
        if len(set(scopes))!=len(scopes) or any(s not in scopes for s in own):
            raise ValueError("local scopes must retain the attachment identities")
        object.__setattr__(self,"refinement",refinement)
        object.__setattr__(self,"scopes",scopes)

    @staticmethod
    def scope(a):
        return _key(a.source_id,a.annotation_id,a.owner_id,a.role)

    @property
    def domain(self):
        return self.field.space

    @property
    def codomain(self):
        if not self.local:
            return self.refinement.space
        return CoordinateSpace("attachment_support/"+self.refinement.coefficient_digest,
            tuple(_key(scope,key) for scope in self.scopes for key in self.refinement.space.keys))

    @property
    def coefficient_digest(self):
        return _identity(("attachment_observation_v1",self.field.coefficient_digest,self.support,self.local,
                          self.mode,self.refinement.coefficient_digest,self.scopes))

    @property
    def metric(self):
        weights=tuple(b-a for a,b in self.refinement.intervals)
        return CoordinateMetric.diagonal(self.codomain,weights*len(self.scopes) if self.local else weights)

    @property
    def factors(self):
        self.field.check_state()
        points={p:i for i,p in enumerate(self.refinement.points)}
        n=len(self.refinement.intervals)
        keys,injection,ranges,grounding,text_keys,text_injection=[],[],[],[],[],[]
        for j,a in enumerate(self.field.attachments):
            block=a.text if self.support=="text" else a.time
            offset=self.scopes.index(self.scope(a))*n if self.local else 0
            base=len(keys)
            components=(tuple(("union:"+str(i),start,stop) for i,(start,stop) in enumerate(_merged(block.components)))
                        if self.mode=="union" else block.components)
            for key,start,stop in components:
                column=len(keys)
                keys.append(_key(a.source_id,a.annotation_id,key))
                injection.append((column,j,1))
                ranges.append((offset+points[start],offset+points[stop],column,1))
            if self.support=="grounded_time":
                if a.text.interpretation!="joint":
                    raise ValueError("alternative text grounding requires a declared choice")
                text_base=len(text_keys)
                for key,_,_ in a.text.components:
                    text_keys.append(_key(a.source_id,a.annotation_id,key))
                    text_injection.append((len(text_keys)-1,j,1))
                grounding.extend((base+i,text_base+k,v) for i,k,v in a.grounding.entries)
        components=CoordinateSpace("attachment_components/"+self.support,tuple(keys))
        realization=IntervalAction(components,self.codomain,tuple(ranges))
        if self.support=="grounded_time":
            fragments=CoordinateSpace("attachment_fragments",tuple(text_keys))
            return (CoordinateMap(self.domain,fragments,tuple(text_injection)),
                    CoordinateMap(fragments,components,tuple(grounding)),realization)
        return CoordinateMap(self.domain,components,tuple(injection)),realization

    @property
    def action(self):
        return AttachmentAction(self)

    @property
    def fragment_action(self):
        return AttachmentAction(self,start=1)

    def evaluate(self,values=None):
        self.field.check_state()
        tensor=self.field.amplitudes() if values is None else values
        if not isinstance(tensor,TensorField):
            tensor=self.field.amplitudes(tensor)
        if self.field.source is not None and (tensor.source is None or not self.field.source.matches(tensor.source)):
            raise ValueError("annotation tensor belongs to another selected state")
        return apply_tensor(self.action,tensor)

    def as_record(self):
        return {"attachment_field":self.field.as_record(),"support":self.support,"local":self.local,
                "mode":self.mode,"refinement":self.refinement.points,"scopes":self.scopes,
                "coefficient_digest":self.coefficient_digest}


@dataclass(frozen=True)
class AttachmentAction:
    """A native attachment observation retaining its source and internal factors."""
    observation: AttachmentObservation
    start: int = 0
    transposed: bool = False

    def __post_init__(self):
        if not isinstance(self.observation,AttachmentObservation) or self.start not in (0,1) or isinstance(self.start,bool):
            raise TypeError("attachment action requires a declared observation and factor entry")
        if not isinstance(self.transposed,bool):
            raise TypeError("transpose flag must be boolean")

    @property
    def word(self):
        factors = self.observation.factors[self.start:]
        word = factors[0] if len(factors) == 1 else CoordinateWord(factors)
        return word.T if self.transposed else word

    @property
    def domain(self):
        return self.word.domain

    @property
    def codomain(self):
        return self.word.codomain

    @property
    def shape(self):
        return len(self.codomain.keys),len(self.domain.keys)

    @property
    def coefficient_digest(self):
        return _identity(("attachment_action_v1",self.observation.coefficient_digest,self.start,self.transposed))

    @property
    def T(self):
        return AttachmentAction(self.observation,self.start,not self.transposed)

    def apply(self,values):
        self.observation.field.check_state()
        return self.word.apply(values)

    def transpose_apply(self,values):
        return self.T.apply(values)

    def then(self,following):
        return CoordinateWord((self,following))


def common_attachment_observations(old,new,*,support="time",local=True,mode="sum"):
    """Compare support coordinates without identifying endpoint assertion states."""
    blocks=tuple(getattr(a,"text" if support=="text" else "time")
                 for f in (old,new) for a in f.attachments)
    if any(b is None for b in blocks):
        raise ValueError("all selected attachments require the requested support")
    refinement=SpanRefinement.from_blocks(blocks)
    scopes=tuple(sorted({AttachmentObservation.scope(a) for f in (old,new) for a in f.attachments}))
    return tuple(f.observe(support=support,local=local,mode=mode,refinement=refinement,scopes=scopes) for f in (old,new))
