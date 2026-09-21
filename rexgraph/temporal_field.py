"""Native temporal tensor fields, resolved injections and retained moment changes."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod
import numpy as np

from rexgraph.coordinate_map import CoordinateMap, CoordinateWord, CoordinateDifference, CoordinateMetric, _identity, _is_action
from rexgraph.graded_metric import _fraction
from rexgraph.native_field import NativeFieldCalculus, ActionSum
from rexgraph.temporal_calculus import TemporalOperation
from rexgraph.tensor_field import FieldSource, TensorField, TensorChannels, _native_sources
from rexgraph.tensor_moment import TensorMomentKernel, MomentSpan
from rexgraph.type_accession import CoordinateSpace

__all__ = ["TensorEvolution", "ResolvedEvolution", "SectorTransport", "MomentChange",
           "moment_change", "factor_change_channels", "NativeFieldEvolution"]


def _evaluate(action, tensor, source=None, grade=None):
    if tensor.space != action.domain:
        raise ValueError("temporal action and field coordinates disagree")
    block=tensor.values.reshape((len(tensor.space.keys),prod(len(a.keys) for a in tensor.axes)))
    values=action.apply(block).reshape((len(action.codomain.keys),*tensor.values.shape[1:]))
    references = {s.coefficient_digest: s for s in tensor.dependencies}
    for s in (tensor.source, source):
        if s is not None:
            references[s.coefficient_digest] = s
    return TensorField(action.codomain,values,tensor.axes,source,grade,tensor.variance,
                       (*tensor.provenance,action.coefficient_digest), tuple(references.values()))


def _check_source(declared, tensor):
    tensor.check_state()
    if declared is None:
        if tensor.source is not None:
            raise ValueError("a source bound temporal field requires explicit endpoint references")
    else:
        declared.check()
        if tensor.source is None or not declared.matches(tensor.source):
            raise ValueError("temporal field differs from its declared endpoint state")


def _check_native(action, source):
    sources=_native_sources(action)
    if sources and (source is None or any(s is not source.source for s in sources)):
        raise ValueError("temporal native action requires its actual endpoint source binding")


def _operation_digest(operation):
    return operation.coefficient_digest


def _diagnostic_space(operation):
    return CoordinateSpace("temporal_input/"+_operation_digest(operation),
        tuple("old/"+k for k in operation.old.domain.keys)+tuple("innovation/"+k for k in operation.new.domain.keys))


def _selectors(operation):
    space=_diagnostic_space(operation)
    n=len(operation.old.domain.keys)
    return (CoordinateMap(space,operation.old.domain,tuple((i,i,1) for i in range(n))),
            CoordinateMap(space,operation.new.domain,tuple((i,n+i,1) for i in range(len(operation.new.domain.keys)))))


@dataclass(frozen=True)
class TensorEvolution:
    """One operation at two declared states without collapsing tensor axes."""
    operation: TemporalOperation
    old_source: FieldSource | None = None
    new_source: FieldSource | None = None
    names: tuple = ("operation","innovation")

    def __post_init__(self):
        if not isinstance(self.operation,TemporalOperation):
            raise TypeError("tensor evolution requires a TemporalOperation")
        if (self.old_source is None)!=(self.new_source is None):
            raise ValueError("both temporal endpoint references must be declared together")
        if any(s is not None and not isinstance(s,FieldSource) for s in (self.old_source,self.new_source)):
            raise TypeError("temporal endpoints require FieldSources")
        names=tuple(self.names)
        if len(names)!=2 or len(set(names))!=2 or any(not isinstance(n,str) or not n for n in names):
            raise ValueError("two distinct temporal channel names are required")
        object.__setattr__(self,"names",names)
        _check_native(self.operation.old,self.old_source)
        _check_native(self.operation.new,self.new_source)

    @classmethod
    def from_observations(cls,old,new):
        from rexgraph.attachment_field import AttachmentObservation
        if not isinstance(old,AttachmentObservation) or not isinstance(new,AttachmentObservation):
            raise TypeError("evolution requires native attachment observations")
        if old.codomain!=new.codomain:
            raise ValueError("attachment observations require explicit common support coordinates")
        return cls(TemporalOperation(old.action,new.action,old.field.correspondence(new.field),
                                     CoordinateMap.identity(old.codomain)),old.field.source,new.field.source,
                   ("attachment","amplitude"))

    @property
    def coefficient_digest(self):
        return _identity(("tensor_evolution_v1",_operation_digest(self.operation),self.names,
                          None if self.old_source is None else self.old_source.coefficient_digest,
                          None if self.new_source is None else self.new_source.coefficient_digest))

    def validate(self,old,new):
        if not isinstance(old,TensorField) or not isinstance(new,TensorField):
            raise TypeError("temporal values must be TensorFields")
        if old.space!=self.operation.old.domain or new.space!=self.operation.new.domain:
            raise ValueError("temporal input coordinates disagree with their endpoint operations")
        if old.axes!=new.axes or old.variance!=new.variance or old.grade!=new.grade:
            raise ValueError("temporal inputs require matching field axes, grade and variance")
        _check_source(self.old_source,old)
        _check_source(self.new_source,new)
        _check_native(self.operation.old,self.old_source)
        _check_native(self.operation.new,self.new_source)

    def diagnostic(self,old,new):
        self.validate(old,new)
        moved=_evaluate(self.operation.input_map,old,self.new_source,new.grade)
        values=np.concatenate((old.values,new.values-moved.values),axis=0)
        return TensorField(_diagnostic_space(self.operation),values,old.axes,
                           variance=old.variance,provenance=(self.coefficient_digest,old.coefficient_digest,new.coefficient_digest),
                           dependencies=tuple({ref.coefficient_digest: ref for ref in
                               (*old.dependencies, *new.dependencies,
                                *(() if self.old_source is None else (self.old_source, self.new_source)))}.values()))

    @property
    def channels(self):
        left,innovation=_selectors(self.operation)
        return (CoordinateWord((left,self.operation.defect)),CoordinateWord((innovation,self.operation.new)))

    def delta(self,old,new):
        diagnostic=self.diagnostic(old,new)
        grade=getattr(self.operation.new,"codomain_grade",None)
        fields=tuple(_evaluate(a,diagnostic,self.new_source,grade) for a in self.channels)
        refs=() if self.old_source is None else (self.old_source,self.new_source)
        result=TensorChannels(self.names,fields,self.coefficient_digest,refs)
        direct_new=_evaluate(self.operation.new,new,self.new_source,grade)
        direct_old=_evaluate(CoordinateWord((self.operation.old,self.operation.output_map)),old,self.new_source,grade)
        if not np.array_equal(result.total().values,direct_new.values-direct_old.values):
            raise ArithmeticError("temporal channel reconstruction failed")
        return result

    def kernel(self,metric):
        return TensorMomentKernel(self.names,self.channels,common_metric=metric,
                                  endpoint_sources=() if self.old_source is None else (self.old_source,self.new_source))


@dataclass(frozen=True)
class ResolvedEvolution:
    """Changes of native Green fields through injections and optional observations."""
    old_calculus: NativeFieldCalculus
    new_calculus: NativeFieldCalculus
    grade: int
    old_injection: object
    new_injection: object
    input_map: CoordinateMap
    ambient_map: CoordinateMap
    parameter: Fraction = Fraction(1)
    old_source: FieldSource | None = None
    new_source: FieldSource | None = None
    old_observation: object = None
    new_observation: object = None
    observation_map: CoordinateMap | None = None

    def __post_init__(self):
        if not isinstance(self.old_calculus,NativeFieldCalculus) or not isinstance(self.new_calculus,NativeFieldCalculus):
            raise TypeError("resolved evolution requires two native field calculi")
        k=self.old_calculus._grade(self.grade)
        self.new_calculus._grade(k)
        if (not _is_action(self.old_injection) or not _is_action(self.new_injection)
                or self.old_injection.codomain!=self.old_calculus.complex.spaces[k]
                or self.new_injection.codomain!=self.new_calculus.complex.spaces[k]):
            raise ValueError("injections must realize fields on the declared native grades")
        TemporalOperation(self.old_injection,self.new_injection,self.input_map,self.ambient_map)
        parameter=_fraction(self.parameter)
        if parameter<0:
            raise ValueError("resolved evolution requires a nonnegative rational parameter")
        object.__setattr__(self,"parameter",parameter)
        object.__setattr__(self,"grade",k)
        if self.old_observation is None:
            if self.new_observation is not None or self.observation_map is not None:
                raise ValueError("both observations and their correspondence are required")
        else:
            if not _is_action(self.old_observation) or not _is_action(self.new_observation):
                raise TypeError("observations must be declared exact actions")
            TemporalOperation(self.old_observation,self.new_observation,self.ambient_map,self.observation_map)
        self._injection_evolution()
        for calculus,source in ((self.old_calculus,self.old_source),(self.new_calculus,self.new_source)):
            _check_native(calculus.hodge(k),source)

    def _injection_evolution(self):
        return TensorEvolution(TemporalOperation(self.old_injection,self.new_injection,self.input_map,self.ambient_map),
                               self.old_source,self.new_source,("injection","amplitude"))

    @property
    def coefficient_digest(self):
        return _identity(("resolved_evolution_v1",self._injection_evolution().coefficient_digest,
                          self.old_calculus.coefficient_digest,self.new_calculus.coefficient_digest,self.grade,
                          (hex(self.parameter.numerator),hex(self.parameter.denominator)),
                          tuple(None if a is None else a.coefficient_digest
                                for a in (self.old_observation,self.new_observation,self.observation_map))))

    @property
    def names(self):
        return (("accession",) if self.old_observation is not None else ())+("injection","amplitude","operator")

    @property
    def channels(self):
        evolution=self._injection_evolution()
        left,innovation=_selectors(evolution.operation)
        gold=self.old_calculus.green(self.grade,self.parameter)
        gnew=self.new_calculus.green(self.grade,self.parameter)
        old_response=CoordinateWord((left,self.old_injection,gold))
        defect=TemporalOperation(self.old_calculus.hodge(self.grade),self.new_calculus.hodge(self.grade),
                                 self.ambient_map,self.ambient_map).defect
        result=[CoordinateWord((left,evolution.operation.defect,gnew)),
                CoordinateWord((innovation,self.new_injection,gnew)),
                ActionSum(((-self.parameter,CoordinateWord((old_response,defect,gnew))),))]
        if self.old_observation is not None:
            accession=TemporalOperation(self.old_observation,self.new_observation,self.ambient_map,self.observation_map).defect
            result=[CoordinateWord((old_response,accession)),
                    *(CoordinateWord((a,self.new_observation)) for a in result)]
        return tuple(result)

    def diagnostic(self,old,new):
        return self._injection_evolution().diagnostic(old,new)

    def delta(self,old,new):
        diagnostic=self.diagnostic(old,new)
        output_grade=self.grade if self.new_observation is None else None
        fields=tuple(_evaluate(a,diagnostic,self.new_source,output_grade) for a in self.channels)
        refs=() if self.old_source is None else (self.old_source,self.new_source)
        result=TensorChannels(self.names,fields,self.coefficient_digest,refs)
        old_factors=(self.old_injection,self.old_calculus.green(self.grade,self.parameter))
        new_factors=(self.new_injection,self.new_calculus.green(self.grade,self.parameter))
        if self.old_observation is not None:
            old_factors=(*old_factors,self.old_observation,self.observation_map)
            new_factors=(*new_factors,self.new_observation)
        else:
            old_factors=(*old_factors,self.ambient_map)
        expected=_evaluate(CoordinateWord(new_factors),new,self.new_source).values-_evaluate(CoordinateWord(old_factors),old,self.new_source).values
        if not np.array_equal(result.total().values,expected):
            raise ArithmeticError("resolved temporal channels failed their exact response identity")
        return result

    def kernel(self,metric):
        return TensorMomentKernel(self.names,self.channels,common_metric=metric,
                                  endpoint_sources=() if self.old_source is None else (self.old_source,self.new_source))


@dataclass(frozen=True)
class SectorTransport:
    """Hodge sector transport and separate field innovations at declared endpoints."""
    old: NativeFieldCalculus
    new: NativeFieldCalculus
    grade: int
    correspondence: CoordinateMap
    old_source: FieldSource | None = None
    new_source: FieldSource | None = None

    def __post_init__(self):
        if not isinstance(self.old,NativeFieldCalculus) or not isinstance(self.new,NativeFieldCalculus):
            raise TypeError("sector transport requires native field calculi")
        k=self.old._grade(self.grade)
        self.new._grade(k)
        if not isinstance(self.correspondence,CoordinateMap) or (
                self.correspondence.domain!=self.old.complex.spaces[k] or self.correspondence.codomain!=self.new.complex.spaces[k]):
            raise ValueError("sector transport requires the declared grade correspondence")
        object.__setattr__(self,"grade",k)
        _check_native(self.old.hodge(k),self.old_source)
        _check_native(self.new.hodge(k),self.new_source)

    @property
    def coefficient_digest(self):
        return _identity(("sector_transport_v1",self.old.coefficient_digest,self.new.coefficient_digest,self.grade,
                          self.correspondence.coefficient_digest,
                          None if self.old_source is None else self.old_source.coefficient_digest,
                          None if self.new_source is None else self.new_source.coefficient_digest))

    def channel(self,destination,origin):
        return CoordinateWord((self.old.sector(self.grade,origin),self.correspondence,self.new.sector(self.grade,destination)))

    def typed_channel(self, destination, origin, source_family, target_family, source_type, target_type):
        """Retain type reconstruction around one exact sector transition."""
        from rexgraph.reconstruction import ReconstructionFamily
        if not isinstance(source_family, ReconstructionFamily) or not isinstance(target_family, ReconstructionFamily):
            raise TypeError("typed sector transport requires certified reconstruction families")
        return source_family.typed_action(self.channel(destination, origin), source_type, target_type,
                                          target=target_family)

    def transport(self,old_field):
        _check_source(self.old_source,old_field)
        names=tuple(a+"/"+b for a in ("gradient","curl","harmonic") for b in ("gradient","curl","harmonic"))
        fields=tuple(_evaluate(self.channel(*name.split("/")),old_field,self.new_source,self.grade) for name in names)
        result=TensorChannels(names,fields,self.coefficient_digest)
        if not np.array_equal(result.total().values,_evaluate(self.correspondence,old_field,self.new_source,self.grade).values):
            raise ArithmeticError("sector channels do not reconstruct temporal transport")
        return result

    def reconstruct(self,old_field,new_field):
        _check_source(self.old_source,old_field)
        _check_source(self.new_source,new_field)
        if old_field.axes!=new_field.axes or old_field.variance!=new_field.variance:
            raise ValueError("sector fields require matching named field axes and variance")
        if new_field.space!=self.correspondence.codomain:
            raise ValueError("new field is outside the declared target coordinates")
        moved=_evaluate(self.correspondence,old_field,self.new_source,self.grade)
        innovation=new_field.with_values(new_field.values-moved.values)
        transported=self.transport(old_field)
        names=transported.names+tuple(a+"/innovation" for a in ("gradient","curl","harmonic"))
        fields=transported.fields+tuple(_evaluate(self.new.sector(self.grade,a),innovation,self.new_source,self.grade)
                                        for a in ("gradient","curl","harmonic"))
        result=TensorChannels(names,fields,self.coefficient_digest,
                              () if self.old_source is None else (self.old_source,self.new_source))
        if not np.array_equal(result.total().values,new_field.values):
            raise ArithmeticError("sector innovations do not reconstruct the new field")
        return result


@dataclass(frozen=True)
class MomentChange:
    """A finite moment difference retaining both supports and every interaction."""
    old: MomentSpan
    new: MomentSpan
    terms: tuple

    def __post_init__(self):
        terms=tuple((name,_fraction(weight),span) for name,weight,span in self.terms)
        if not isinstance(self.old,MomentSpan) or not isinstance(self.new,MomentSpan) or any(
                not isinstance(span,MomentSpan) for _,_,span in terms):
            raise TypeError("moment changes require retained MomentSpans")
        if len({name for name,_,_ in terms})!=len(terms):
            raise ValueError("moment change terms require unique names")
        object.__setattr__(self,"terms",terms)

    def contract_support(self):
        expected=np.asarray(self.new.contract_support()-self.old.contract_support(), dtype=object)
        result=np.full(expected.shape,Fraction(0),dtype=object)
        for _,weight,span in self.terms:
            value=np.asarray(span.contract_support(), dtype=object)
            if value.shape!=expected.shape:
                raise ValueError("moment changes require matching retained field axes")
            result+=weight*value
        if not np.array_equal(result,expected):
            raise ArithmeticError("finite moment change identity failed")
        result.flags.writeable=False
        return result

    def as_record(self):
        return {"old":self.old.as_record(),"new":self.new.as_record(),
                "terms":tuple((name,weight,span.as_record()) for name,weight,span in self.terms),
                "coefficient_domain":"Q"}


def moment_change(old_left,new_left,left_map,old_right,new_right,right_map,old_form,new_form):
    """Retain the transported form change and the two field innovations."""
    for old,new,mapping in ((old_left,new_left,left_map),(old_right,new_right,right_map)):
        if old.axes!=new.axes or old.variance!=new.variance or old.space!=mapping.domain or new.space!=mapping.codomain:
            raise ValueError("moment correspondence must match both endpoints and retained field axes")
        old.check_state();new.check_state()
    lx=_evaluate(left_map,old_left,new_left.source,new_left.grade)
    ry=_evaluate(right_map,old_right,new_right.source,new_right.grade)
    eta=new_left.with_values(new_left.values-lx.values)
    zeta=new_right.with_values(new_right.values-ry.values)
    old=MomentSpan(old_left,old_right,old_form)
    new=MomentSpan(new_left,new_right,new_form)
    return MomentChange(old,new,(("transported_form",1,MomentSpan(lx,ry,new_form)),
                                ("old_form",-1,old),("left_innovation",1,MomentSpan(eta,ry,new_form)),
                                ("right_innovation",1,MomentSpan(lx,zeta,new_form)),
                                ("interaction",1,MomentSpan(eta,zeta,new_form))))


def factor_change_channels(old,new,subsets):
    """Construct only the explicitly requested changed factor subsets."""
    old,new=tuple(old),tuple(new)
    CoordinateWord(old);CoordinateWord(new)
    if len(old)!=len(new) or any(a.domain!=b.domain or a.codomain!=b.codomain for a,b in zip(old,new,strict=True)):
        raise ValueError("subset attribution requires already aligned factor coordinates")
    subsets=tuple(tuple(s) for s in subsets)
    if len(set(subsets))!=len(subsets):
        raise ValueError("changed factor subsets must be distinct")
    result=[]
    for subset in subsets:
        if (not subset or len(set(subset))!=len(subset) or any(isinstance(i,bool) or not isinstance(i,int)
                                                              or not 0<=i<len(old) for i in subset)):
            raise ValueError("invalid changed factor subset")
        result.append(CoordinateWord(tuple(CoordinateDifference(b,a) if i in subset else a
                                           for i,(a,b) in enumerate(zip(old,new,strict=True)))))
    return tuple(result)


def _adjoint(calculus,grade):
    from rexgraph.native_field import MetricAction
    return CoordinateWord((MetricAction(calculus.metrics[grade-1]),calculus.boundary(grade).T,
                           MetricAction(calculus.metrics[grade],True)))


def _adjacent_action(calculus,grade):
    source=calculus.complex.spaces[grade]
    parts=[]
    if grade>0:
        parts.append(("down",calculus.boundary(grade),calculus.metrics[grade-1]))
    if grade+1<len(calculus.metrics):
        parts.append(("up",_adjoint(calculus,grade+1),calculus.metrics[grade+1]))
    target=CoordinateSpace("adjacent/"+source.name,
                           tuple(name+"/"+key for name,action,_ in parts for key in action.codomain.keys))
    offset=0
    terms=[]
    entries=[]
    for name,action,metric in parts:
        injection=CoordinateMap(action.codomain,target,tuple((offset+i,i,1) for i in range(len(action.codomain.keys))))
        terms.append((1,CoordinateWord((action,injection))))
        entries.extend((offset+i,offset+j,w) for i,j,w in metric.entries)
        offset+=len(action.codomain.keys)
    action=ActionSum(tuple(terms)) if terms else CoordinateMap(source,target,())
    return action,CoordinateMetric(target,tuple(entries))


@dataclass(frozen=True)
class NativeFieldEvolution:
    """Adjacent field changes and their cross grade Hodge defect factors."""
    old: NativeFieldCalculus
    new: NativeFieldCalculus
    grade: int
    correspondence: object
    old_source: FieldSource | None = None
    new_source: FieldSource | None = None

    def __post_init__(self):
        from rexgraph.chain_map import GradedMap
        if not isinstance(self.old,NativeFieldCalculus) or not isinstance(self.new,NativeFieldCalculus):
            raise TypeError("native field evolution requires two declared calculi")
        k=self.old._grade(self.grade);self.new._grade(k)
        if (not isinstance(self.correspondence,GradedMap) or self.correspondence.domain is not self.old.complex
                or self.correspondence.codomain is not self.new.complex):
            raise ValueError("graded correspondence must use the actual endpoint complexes")
        object.__setattr__(self,"grade",k)
        self.adjacent_evolution()

    @property
    def coefficient_digest(self):
        return _identity(("native_field_evolution_v1",self.old.coefficient_digest,self.new.coefficient_digest,
                          self.grade,self.correspondence.coefficient_digest,
                          None if self.old_source is None else self.old_source.coefficient_digest,
                          None if self.new_source is None else self.new_source.coefficient_digest))

    def map(self,grade):
        return CoordinateMap(self.old.complex.spaces[grade],self.new.complex.spaces[grade],
                             self.correspondence.components[grade])

    def adjacent_evolution(self):
        a,_=_adjacent_action(self.old,self.grade)
        b,_=_adjacent_action(self.new,self.grade)
        entries=[]
        old_offset=new_offset=0
        for grade in (self.grade-1,self.grade+1):
            if 0<=grade<len(self.old.metrics) and 0<=grade<len(self.new.metrics):
                mapping=self.map(grade)
                entries.extend((new_offset+i,old_offset+j,w) for i,j,w in mapping.entries)
                old_offset+=len(mapping.domain.keys);new_offset+=len(mapping.codomain.keys)
        output=CoordinateMap(a.codomain,b.codomain,tuple(entries))
        return TensorEvolution(TemporalOperation(a,b,self.map(self.grade),output),self.old_source,self.new_source,
                               ("structure","field"))

    @property
    def metric(self):
        return _adjacent_action(self.new,self.grade)[1]

    def validate(self,old,new):
        if (not isinstance(old,TensorField) or not isinstance(new,TensorField)
                or old.grade != self.grade or new.grade != self.grade
                or old.variance != "chain" or new.variance != "chain"):
            raise ValueError("native field evolution requires chain fields on its declared grade")
        self.adjacent_evolution().validate(old,new)

    def diagnostic(self,old,new):
        self.validate(old,new)
        return self.adjacent_evolution().diagnostic(old,new)

    def delta(self,old,new):
        self.validate(old,new)
        return self.adjacent_evolution().delta(old,new)

    def kernel(self):
        return self.adjacent_evolution().kernel(self.metric)

    def down_defect(self,grade):
        return TemporalOperation(self.old.boundary(grade),self.new.boundary(grade),
                                 self.map(grade),self.map(grade-1)).defect

    def up_defect(self,grade):
        return TemporalOperation(_adjoint(self.old,grade+1),_adjoint(self.new,grade+1),
                                 self.map(grade),self.map(grade+1)).defect

    def hodge_terms(self):
        k=self.grade
        names=[];actions=[]
        if k>0:
            names.extend(("lower_boundary","lower_adjoint"))
            actions.extend((CoordinateWord((self.down_defect(k),_adjoint(self.new,k))),
                            CoordinateWord((self.old.boundary(k),self.up_defect(k-1)))))
        if k+1<len(self.old.metrics) and k+1<len(self.new.metrics):
            names.extend(("upper_adjoint","upper_boundary"))
            actions.extend((CoordinateWord((self.up_defect(k),self.new.boundary(k+1))),
                            CoordinateWord((_adjoint(self.old,k+1),self.down_defect(k+1)))))
        return tuple(names),tuple(actions)

    def hodge_delta(self,old):
        if not isinstance(old,TensorField) or old.grade != self.grade or old.variance != "chain":
            raise ValueError("Hodge transport requires a chain field on the declared grade")
        _check_source(self.old_source,old)
        names,actions=self.hodge_terms()
        if not actions:
            zero=CoordinateMap(self.map(self.grade).domain,self.map(self.grade).codomain,())
            names,actions=("zero",),(zero,)
        result=TensorChannels(names,tuple(_evaluate(a,old,self.new_source,self.grade) for a in actions),
                              self.coefficient_digest,() if self.old_source is None else (self.old_source,self.new_source))
        direct=TemporalOperation(self.old.hodge(self.grade),self.new.hodge(self.grade),
                                  self.map(self.grade),self.map(self.grade)).defect
        if not np.array_equal(result.total().values,_evaluate(direct,old,self.new_source,self.grade).values):
            raise ArithmeticError("cross grade Hodge defect factors failed their exact identity")
        return result

    def compatibility(self,grade,field):
        if grade<2:
            raise ValueError("adjacent boundary compatibility requires grade at least two")
        if not isinstance(field,TensorField) or field.grade != grade or field.variance != "chain":
            raise ValueError("boundary compatibility requires a chain field on the selected grade")
        _check_source(self.old_source,field)
        action=ActionSum(((1,CoordinateWord((self.down_defect(grade),self.new.boundary(grade-1)))),
                          (1,CoordinateWord((self.old.boundary(grade),self.down_defect(grade-1))))))
        result=_evaluate(action,field,self.new_source,grade-2)
        if any(result.values.flat):
            raise ArithmeticError("temporal boundary defects violate endpoint chain compatibility")
        return result
