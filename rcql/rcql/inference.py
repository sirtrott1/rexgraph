"""Decide a call's result type before the call happens.

``executor.value_exactness`` answers the same question by looking at a finished value's
dtype. That is not inference: it runs the work first, it cannot answer for a plan that has
not executed, and a dtype cannot distinguish a rational value that happens to be stored as
a float from a float that was never exact. It also cannot say a request is impossible,
because there is no result to inspect when the answer is a refusal.

This module reads the signature instead, and attaches the source and temporal reference
from the binding so a result can always say which state produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .binding import Binding, resolve
from .signatures import OperatorSignature
from .types import BasisRef, PredicateResult, RCType


def _plain_source(ref):
    """Render complete source provenance, including all phrase contributors."""
    rendered = {
        "name": ref.name,
        "state_digest": ref.state_digest,
        "record_id": ref.record_id,
        "record_version": ref.record_version,
        "record_as_of": ref.record_as_of,
        "record_valid_at": ref.record_valid_at,
    }
    if ref.contributors:
        rendered["contributors"] = [_plain_source(item) for item in ref.contributors]
    return rendered


@dataclass(frozen=True)
class TypedCall:
    """A call that has been checked and typed, but not run.

    ``signature.implementation_key`` is deliberately not resolved here. Deciding the type
    must not require the adapter to exist or the numeric stack to be importable, which is
    what lets EXPLAIN answer for a plan without paying for it.
    """

    operator: str
    binding: Binding
    args: tuple[object, ...]
    signature: OperatorSignature
    result: RCType
    predicates: tuple[PredicateResult, ...] = ()

    @property
    def effects(self) -> frozenset:
        return self.signature.effects

    def explain(self) -> dict:
        """A structural account of the call, with no source values and no key material."""
        return {
            "operator": self.operator,
            "source": self.binding.ref.name,
            "source_kind": self.binding.schema.kind.value,
            "source_state": _plain_source(self.binding.ref),
            "policy_digest": self.binding.ref.policy_digest,
            "requires": sorted(self.signature.requires),
            "source_methods": sorted(self.signature.source_methods),
            "effects": sorted(effect.value for effect in self.signature.effects),
            "memoizable": self.signature.memoizable,
            "preconditions": list(self.signature.preconditions),
            "predicates": [asdict(item) for item in self.predicates],
            "result": {
                "kind": self.result.kind.value,
                "grade": self.result.grade,
                "variance": None if self.result.variance is None else self.result.variance.value,
                "domain": None if self.result.domain is None else self.result.domain.value,
                "exactness": None if self.result.exactness is None else self.result.exactness.value,
                "shape": None if self.result.shape is None else self.result.shape.dims,
                "operator": None if self.result.operator is None else asdict(self.result.operator),
                "metric": None if self.result.metric is None else asdict(self.result.metric),
                "accessions": [asdict(a) for a in self.result.accessions],
                "cross_metric": None if self.result.cross_metric is None else asdict(self.result.cross_metric),
                "family_metric": None if self.result.family_metric is None else asdict(self.result.family_metric),
                "graded_map": None if self.result.graded_map is None else asdict(self.result.graded_map),
                "graded_operator": None if self.result.graded_operator is None else asdict(self.result.graded_operator),
                "graded_bases": [asdict(b) for b in self.result.graded_bases],
                "member_shapes": [list(s) for s in self.result.member_shapes],
                "coordinates": None if self.result.coordinates is None else asdict(self.result.coordinates),
                "tensor_axes": None if self.result.tensor_axes is None else [asdict(a) for a in self.result.tensor_axes],
                "coordinate_action": None if self.result.coordinate_action is None else asdict(self.result.coordinate_action),
                "declaration_digest": self.result.declaration_digest,
                # The state the result was read at, and the basis it is expressed in. Both
                # are part of what the value means: an identical looking reading from
                # another version or another ordered basis is a different value, so an
                # account that omitted them would let EXPLAIN imply a state and a frame it
                # never established. Rendered as plain fields rather than the descriptors
                # themselves, since this payload carries no objects.
                "temporal": None if self.result.temporal is None else {
                    "version": self.result.temporal.version,
                    "as_of": self.result.temporal.as_of,
                    "valid_at": self.result.temporal.valid_at,
                },
                "basis": None if self.result.basis is None else {
                    "source_id": self.result.basis.source_id,
                    "grade": self.result.basis.grade,
                    "ordering": self.result.basis.ordering,
                },
            },
            "implementation_key": self.signature.implementation_key,
        }


def infer(binding: Binding, operator: str, args: tuple[object, ...] = (), *,
          context=None, children=()) -> TypedCall:
    """Resolve, check and type one call without executing it."""
    signature = resolve(binding, operator, args)
    result = signature.result_type(args)

    # Every value carries where it came from and when.  A result rule may have refined
    # the temporal state itself (TEMPORAL_DELTA(step), for example), so binding metadata
    # fills only absent fields rather than erasing that more specific declaration.
    source = binding.ref if result.source is None else result.source
    temporal = result.temporal
    if temporal is None:
        carried_times = {
            value.temporal for value in args
            if isinstance(value, RCType) and value.temporal is not None
        }
        # An operator such as SIGNAL_SOURCE inherits the transition it was handed.
        # More than one distinct input time needs an explicit alignment operator; no
        # present signature accepts that combination, so retain the binding fallback.
        temporal = carried_times.pop() if len(carried_times) == 1 else binding.temporal
    basis = result.basis
    if basis is None and result.grade is not None:
        basis = BasisRef(source.name, result.grade)
    result = result.with_(source=source, temporal=temporal, basis=basis)

    from .validation import ValidationContext, refine
    typed = TypedCall(operator=operator.upper(), binding=binding, args=tuple(args),
                      signature=signature, result=result)
    refined, predicates = refine(typed, children, context or ValidationContext(binding))
    return replace(refined, predicates=predicates)
