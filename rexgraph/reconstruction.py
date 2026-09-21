"""Certified reconstruction of overlapping coordinate observations."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import numpy as np

from rexgraph.coordinate_map import CoordinateMap, CoordinateWord, _identity, _is_action
from rexgraph.native_field import ActionSum
from rexgraph.type_accession import CoordinateSpace

__all__ = ["ReconstructionFamily"]


@dataclass(frozen=True)
class ReconstructionFamily:
    """Observation maps and reconstruction maps with an exact identity certificate."""
    names: tuple[str,...]
    observations: tuple
    reconstructions: tuple

    def __post_init__(self):
        names,observations,reconstructions=tuple(self.names),tuple(self.observations),tuple(self.reconstructions)
        if (not names or len(set(names))!=len(names) or len(names)!=len(observations)
                or len(names)!=len(reconstructions) or any(not isinstance(n,str) or not n for n in names)
                or any(not _is_action(a) for a in (*observations,*reconstructions))):
            raise ValueError("reconstruction requires named observation and reconstruction maps")
        space=observations[0].domain
        if any(a.domain!=space or r.codomain!=space or a.codomain!=r.domain
               for a,r in zip(observations,reconstructions,strict=True)):
            raise ValueError("reconstruction maps must use the original named ambient and view spaces")
        object.__setattr__(self,"names",names)
        object.__setattr__(self,"observations",observations)
        object.__setattr__(self,"reconstructions",reconstructions)
        for j in range(len(space.keys)):
            value=np.asarray([Fraction(int(i==j)) for i in range(len(space.keys))],dtype=object)
            total=np.full(value.shape,Fraction(0),dtype=object)
            for a,r in zip(observations,reconstructions,strict=True):
                total+=r.apply(a.apply(value))
            if not np.array_equal(total,value):
                raise ValueError("view reconstruction does not resolve the ambient identity")

    @property
    def space(self):
        return self.observations[0].domain

    @property
    def coefficient_digest(self):
        return _identity(("reconstruction_family_v1",self.names,tuple(a.coefficient_digest for a in self.observations),
                          tuple(r.coefficient_digest for r in self.reconstructions)))

    @property
    def view_space(self):
        import json
        return CoordinateSpace("view_family/"+self.coefficient_digest,
            tuple(json.dumps((name,key),ensure_ascii=False) for name,a in zip(self.names,self.observations,strict=True)
                  for key in a.codomain.keys))

    @property
    def selectors(self):
        offset=0
        result=[]
        for a in self.observations:
            result.append(CoordinateMap(self.view_space,a.codomain,
                                        tuple((j,offset+j,1) for j in range(len(a.codomain.keys)))))
            offset+=len(a.codomain.keys)
        return tuple(result)

    @property
    def observe(self):
        return ActionSum(tuple((1,CoordinateWord((a,s.T))) for a,s in zip(self.observations,self.selectors,strict=True)))

    @property
    def reconstruct(self):
        return ActionSum(tuple((1,CoordinateWord((s,r))) for s,r in zip(self.selectors,self.reconstructions,strict=True)))

    @property
    def consistency(self):
        return CoordinateWord((self.reconstruct,self.observe))

    def typed_action(self,action,origin,destination,*,target=None):
        target=self if target is None else target
        if not isinstance(target,ReconstructionFamily) or action.domain!=self.space or action.codomain!=target.space:
            raise ValueError("typed action must match the two ambient family spaces")
        return CoordinateWord((self.reconstructions[self.names.index(origin)],action,
                               target.observations[target.names.index(destination)]))

    def representation_pairing(self,metric):
        from rexgraph.tensor_moment import RealizedPairing
        return RealizedPairing(self.reconstruct,self.reconstruct,metric)
