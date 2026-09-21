"""Finite affine feedback over named local fields and declared primary relations."""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np

from rexgraph.coordinate_map import CoordinateMap, _identity
from rexgraph.type_accession import CoordinateSpace
from rexgraph.tensor_field import TensorField, FieldSource
from rexgraph.section_calculus import SectionFamily


@dataclass(frozen=True)
class FeedbackEquation:
    """One output and its ordered input maps, with an explicit forcing field."""
    name: str
    output: str
    terms: tuple[tuple[str, CoordinateMap], ...]
    forcing: TensorField

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (self.name, self.output)):
            raise ValueError("feedback equations require output and relation names")
        if not isinstance(self.forcing, TensorField) or self.forcing.variance != "coordinate":
            raise TypeError("feedback forcing requires a coordinate TensorField")
        terms = tuple(self.terms)
        for key, action in terms:
            if not isinstance(key, str) or not isinstance(action, CoordinateMap) or action.codomain != self.forcing.space:
                raise ValueError("feedback input maps must arrive at the output space")
        object.__setattr__(self, "terms", terms)


@dataclass(frozen=True)
class AffineFeedback:
    """The complete solution family of declared feedback equations."""
    name: str
    spaces: tuple[tuple[str, CoordinateSpace], ...]
    equations: tuple[FeedbackEquation, ...]
    source: FieldSource

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or not isinstance(self.source, FieldSource):
            raise TypeError("feedback requires a name and native source reference")
        spaces, equations = tuple(self.spaces), tuple(self.equations)
        if not spaces or len(dict(spaces)) != len(spaces) or any(not isinstance(k, str) or not k or not isinstance(s, CoordinateSpace) for k, s in spaces):
            raise ValueError("feedback requires distinct named local spaces")
        if not equations or any(not isinstance(e, FeedbackEquation) for e in equations) or len({e.name for e in equations}) != len(equations):
            raise ValueError("feedback equations require distinct primary identities")
        known = dict(spaces)
        axes = equations[0].forcing.axes
        for equation in equations:
            if equation.output not in known or equation.forcing.space != known[equation.output]:
                raise ValueError("feedback output disagrees with its declared local space")
            if equation.forcing.axes != axes or equation.forcing.source is None or not self.source.matches(equation.forcing.source):
                raise ValueError("feedback forcing fields require the same source and retained axes")
            for key, action in equation.terms:
                if key not in known or action.domain != known[key]:
                    raise ValueError("feedback input disagrees with its declared local space")
        object.__setattr__(self, "spaces", spaces)
        object.__setattr__(self, "equations", equations)
        self.check_state()
        if self.source.source is not None:
            from rexgraph.chain_map import CoordinateComplex, _chain_residual
            if _chain_residual(CoordinateComplex.from_rex(self.source.source)):
                raise ValueError("feedback source violates its boundary chain condition")

    def check_state(self):
        self.source.check()
        for equation in self.equations:
            equation.forcing.check_state()

    @property
    def dependencies(self):
        refs = {v.coefficient_digest: v for e in self.equations for v in e.forcing.dependencies}
        return tuple(refs.values())

    @property
    def coefficient_digest(self):
        return _identity(("affine_feedback_v1", self.name,
            tuple((k, s.name, s.keys) for k, s in self.spaces),
            tuple((e.name, e.output, tuple((k, a.coefficient_digest) for k, a in e.terms), e.forcing.coefficient_digest)
                  for e in self.equations), self.source.coefficient_digest))

    @property
    def space(self):
        return CoordinateSpace(self.name+"/variables", tuple(json.dumps((k, c), separators=(",", ":"))
                            for k, s in self.spaces for c in s.keys))

    def assemble(self):
        self.check_state()
        offsets, offset = {}, 0
        for key, space in self.spaces:
            offsets[key] = offset
            offset += len(space.keys)
        entries, keys, blocks = [], [], []
        row = 0
        for equation in self.equations:
            n = len(equation.forcing.space.keys)
            keys.extend(json.dumps((equation.name, key), separators=(",", ":")) for key in equation.forcing.space.keys)
            entries.extend((row+i, offsets[equation.output]+i, 1) for i in range(n))
            for key, action in equation.terms:
                entries.extend((row+i, offsets[key]+j, -value) for i, j, value in action.entries)
            blocks.append(equation.forcing.values)
            row += n
        target = CoordinateSpace(self.name+"/constraints", tuple(keys))
        action = CoordinateMap(self.space, target, tuple(entries))
        rhs = TensorField(target, np.concatenate(blocks, axis=0), self.equations[0].forcing.axes,
                          self.source, None, "coordinate", (self.coefficient_digest,), self.dependencies)
        return action, rhs

    def complete(self):
        action, rhs = self.assemble()
        return SectionFamily.solve(action, rhs, self.coefficient_digest)

    def selection(self, variable):
        offset = 0
        for key, space in self.spaces:
            if key == variable:
                return CoordinateMap(self.space, space, tuple((i, offset+i, 1) for i in range(len(space.keys))))
            offset += len(space.keys)
        raise ValueError("unknown feedback variable")

    def iterate(self, initial, steps):
        """Evaluate the declared recurrence without identifying it with its fixed point."""
        from rexgraph.tensor_field import apply_tensor
        if type(steps) is not int or steps < 0:
            raise ValueError("the recurrence horizon must be a nonnegative integer")
        if (not isinstance(initial, TensorField) or initial.space != self.space or initial.source is None
                or not initial.source.matches(self.source) or initial.axes != self.equations[0].forcing.axes
                or initial.variance != "coordinate" or initial.grade is not None):
            raise ValueError("initial recurrence field does not match the declared variables")
        outputs = [e.output for e in self.equations]
        if len(set(outputs)) != len(outputs) or set(outputs) != set(dict(self.spaces)):
            raise ValueError("iteration requires exactly one defining equation per variable")
        ordered = {e.output: e for e in self.equations}
        current = initial
        history = [current]
        for _ in range(steps):
            self.check_state()
            current.check_state()
            blocks = []
            for key, space in self.spaces:
                equation = ordered[key]
                value = equation.forcing.values.copy()
                for input_key, action in equation.terms:
                    selected = apply_tensor(self.selection(input_key), current)
                    value += apply_tensor(action, selected).values
                blocks.append(value)
            current = TensorField(self.space, np.concatenate(blocks), initial.axes, self.source,
                                  None, "coordinate", (self.coefficient_digest,),
                                  tuple({v.coefficient_digest: v for v in (*initial.dependencies, *self.dependencies)}.values()))
            history.append(current)
        return tuple(history)
