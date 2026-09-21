"""Exact section equations, completion families and retained observations."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import prod
from numbers import Integral
import json
import numpy as np

from rexgraph.coordinate_map import CoordinateMap, _identity, _is_action
from rexgraph.exact_green import ExactSparse
from rexgraph.graded_metric import _fraction
from rexgraph.tensor_field import FieldSource, TensorField, apply_tensor
from rexgraph.type_accession import CoordinateSpace

__all__ = ["SectionRecipe", "SectionSystem", "SectionFamily", "SectionImage",
           "InconsistentSectionError", "UnderdeterminedSectionError"]


def _key(*parts):
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def _references(values):
    result = {}
    for value in values:
        if not isinstance(value, FieldSource):
            raise TypeError("section dependencies require native FieldSources")
        result[value.coefficient_digest] = value
    return tuple(result.values())


def _spaces(values, names, widths, prefix):
    if values is None:
        values = tuple(CoordinateSpace(prefix + "/" + name,
                       tuple(str(i) for i in range(width)))
                       for name, width in zip(names, widths, strict=True))
    values = tuple(values)
    if len(values) != len(names) or any(not isinstance(v, CoordinateSpace)
            or len(v.keys) != width for v, width in zip(values, widths, strict=True)):
        raise ValueError("section spaces must match each declared local dimension")
    return values


@dataclass(frozen=True)
class SectionRecipe:
    """A finite declaration of local spaces and exact incidence maps."""
    name: str
    grade: int
    cells: tuple[str, ...]
    mediators: tuple[str, ...]
    stalk_spaces: tuple[CoordinateSpace, ...]
    mediator_spaces: tuple[CoordinateSpace, ...]
    incidences: tuple[tuple[int, ...], ...]
    restrictions: tuple[tuple[int, int, CoordinateMap], ...]
    source: FieldSource
    dependencies: tuple[FieldSource, ...] = ()
    stalk_sources: tuple[FieldSource | None, ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a section recipe requires a name")
        if isinstance(self.grade, bool) or not isinstance(self.grade, Integral) or self.grade not in (0, 1, 2):
            raise ValueError("the source sheaf grade must be 0, 1 or 2")
        cells, mediators = tuple(self.cells), tuple(self.mediators)
        for names in (cells, mediators):
            if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
                raise ValueError("section cell and mediator names must be unique")
        stalks, targets = tuple(self.stalk_spaces), tuple(self.mediator_spaces)
        if (len(stalks) != len(cells) or len(targets) != len(mediators)
                or any(not isinstance(s, CoordinateSpace) for s in (*stalks, *targets))):
            raise ValueError("section spaces must match the named cells")
        incidences = tuple(tuple(ms) for ms in self.incidences)
        if len(incidences) != len(cells):
            raise ValueError("incidence rows must match the stalks")
        for row in incidences:
            if (any(isinstance(m, bool) or not isinstance(m, int) or not 0 <= m < len(mediators) for m in row)
                    or tuple(sorted(set(row))) != row):
                raise ValueError("incidences require sorted distinct mediator indices")
        restrictions = tuple(self.restrictions)
        expected = {(c, m) for c, row in enumerate(incidences) for m in row}
        seen = set()
        for c, m, action in restrictions:
            if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (c, m)):
                raise TypeError("restriction incidence indices must be integers")
            if (c, m) not in expected or (c, m) in seen:
                raise ValueError("restriction does not name one distinct source incidence")
            if not isinstance(action, CoordinateMap) or action.domain != stalks[c] or action.codomain != targets[m]:
                raise ValueError("restriction spaces disagree with their incidence")
            seen.add((c, m))
        if seen != expected:
            raise ValueError("every section incidence requires a restriction")
        if not isinstance(self.source, FieldSource):
            raise TypeError("a section recipe requires its native source reference")
        stalk_sources = tuple(self.stalk_sources) or (None,) * len(cells)
        if len(stalk_sources) != len(cells):
            raise ValueError("stalk provenance must match the declared cell order")
        dependencies = _references(self.dependencies)
        dependency_keys = {d.coefficient_digest for d in dependencies}
        if any(s is not None and (not isinstance(s, FieldSource) or s.coefficient_digest not in dependency_keys)
               for s in stalk_sources):
            raise ValueError("stalk provenance requires a recorded contributor")
        object.__setattr__(self, "stalk_sources", stalk_sources)
        for attr, value in (("cells", cells), ("mediators", mediators), ("stalk_spaces", stalks),
                            ("mediator_spaces", targets), ("incidences", incidences),
                            ("restrictions", tuple(sorted(restrictions, key=lambda x: (x[0], x[1])))),
                            ("dependencies", dependencies)):
            object.__setattr__(self, attr, value)

    @property
    def coefficient_digest(self):
        return _identity(("section_recipe_v1", self.name, self.grade, self.cells, self.mediators,
            tuple((s.name, s.keys) for s in self.stalk_spaces),
            tuple((s.name, s.keys) for s in self.mediator_spaces), self.incidences,
            tuple((c, m, a.coefficient_digest) for c, m, a in self.restrictions),
            self.source.coefficient_digest, tuple(s.coefficient_digest for s in self.dependencies),
            tuple(None if s is None else s.coefficient_digest for s in self.stalk_sources)))

    def detached(self):
        def detach(s):
            return FieldSource(None, s.record_id, s.version, s.state_digest)
        return replace(self, source=detach(self.source), dependencies=tuple(detach(s) for s in self.dependencies),
                       stalk_sources=tuple(None if s is None else detach(s) for s in self.stalk_sources))

    def bind(self, source, dependencies=()):
        """Bind all recorded inputs explicitly before executing the recipe."""
        dependencies = tuple(dependencies)
        if not isinstance(source, FieldSource) or source.source is None or not source.matches(self.source):
            raise ValueError("section recipe requires its exact live source version")
        if len(dependencies) != len(self.dependencies):
            raise ValueError("section recipe requires every recorded contributor")
        for expected, actual in zip(self.dependencies, dependencies, strict=True):
            if not isinstance(actual, FieldSource) or actual.source is None or not expected.matches(actual):
                raise ValueError("section contributor does not match its recorded version")
        from rexgraph.sheaf import ExactSheaf
        probe = ExactSheaf(source.source, grade=self.grade,
                          stalk_dims=tuple(len(s.keys) for s in self.stalk_spaces),
                          mediator_dims=tuple(len(s.keys) for s in self.mediator_spaces))
        if tuple(tuple(ms) for ms in probe._inc) != self.incidences:
            raise ValueError("section recipe does not match the native incidence structure")
        refs = {s.coefficient_digest: s for s in dependencies}
        bound = replace(self, source=source, dependencies=dependencies,
                        stalk_sources=tuple(None if s is None else refs[s.coefficient_digest]
                                            for s in self.stalk_sources))
        result = SectionSystem(bound)
        result.check_state()
        return result


class SectionSystem:
    """Compatibility equations over the retained primary sheaf incidences."""

    def __init__(self, recipe, *, _sheaf=None):
        if not isinstance(recipe, SectionRecipe):
            raise TypeError("a section system requires a SectionRecipe")
        self.recipe = recipe
        self._sheaf = _sheaf
        if _sheaf is None and recipe.source.source is not None:
            from rexgraph.sheaf import ExactSheaf
            recipe.source.check()
            probe = ExactSheaf(recipe.source.source, grade=recipe.grade,
                stalk_dims=tuple(len(s.keys) for s in recipe.stalk_spaces),
                mediator_dims=tuple(len(s.keys) for s in recipe.mediator_spaces))
            if tuple(tuple(ms) for ms in probe._inc) != recipe.incidences:
                raise ValueError("section recipe does not match the native incidence structure")
        offsets, size = [], 0
        for space in recipe.stalk_spaces:
            offsets.append(size)
            size += len(space.keys)
        self._offsets = tuple(offsets)
        self.space = CoordinateSpace(recipe.name + "/stalks", tuple(
            _key(name, key) for name, space in zip(recipe.cells, recipe.stalk_spaces, strict=True)
            for key in space.keys))
        anchors, comparisons, keys, entries = {}, [], [], []
        restrictions = {(c, m): a for c, m, a in recipe.restrictions}
        for cell, mediators in enumerate(recipe.incidences):
            for mediator in mediators:
                if mediator not in anchors:
                    anchors[mediator] = cell
                    continue
                anchor = anchors[mediator]
                offset = len(keys)
                comparisons.append((anchor, cell, mediator))
                keys.extend(_key(recipe.mediators[mediator], recipe.cells[anchor], recipe.cells[cell], k)
                            for k in recipe.mediator_spaces[mediator].keys)
                for owner, sign in ((anchor, 1), (cell, -1)):
                    entries.extend((offset + i, offsets[owner] + j, sign * v)
                                   for i, j, v in restrictions[owner, mediator].entries)
        self.comparisons = tuple(comparisons)
        self.residual_space = CoordinateSpace(recipe.name + "/residuals", tuple(keys))
        self.action = CoordinateMap(self.space, self.residual_space, tuple(entries))

    @classmethod
    def from_sheaf(cls, section, *, name="section", stalk_spaces=None, mediator_spaces=None, source=None):
        from rexgraph.sheaf import ExactSheaf, UndeclaredRestrictionError
        if not isinstance(section, ExactSheaf):
            raise TypeError("section equations require an ExactSheaf")
        section.check_state()
        if hasattr(section, "_require_section"):
            section._require_section()
        missing = tuple((c, m) for c, row in enumerate(section._inc) for m in row
                        if (c, m) not in section._R and (section.require_declared_restrictions
                            or section.stalk_dimensions[c] != section.mediator_dimensions[m]))
        if missing:
            raise UndeclaredRestrictionError(missing)
        if source is None:
            source = FieldSource(section.rex)
        if not isinstance(source, FieldSource) or source.source is not section.rex:
            raise ValueError("section source must be the actual native sheaf source")
        source.check()
        if hasattr(section, "correspondences") and hasattr(section, "stalks"):
            cells = tuple(s.name for s in section.stalks)
            mediators = tuple(c.name for c in section.correspondences)
            stalk_sources = tuple(FieldSource(s.source.value, s.source.ref.record_id,
                          s.source.ref.record_version, s.source.ref.state_digest) for s in section.stalks)
            dependencies = _references(stalk_sources)
        else:
            ids = section.rex.relation_ids
            cells = tuple(str(int(ids[i])) if section.grade == 1 and ids is not None else str(i)
                          for i in range(section.n_cells))
            mediators = tuple(str(int(ids[i])) if section.grade != 1 and ids is not None else str(i)
                              for i in range(len(section.mediator_dimensions)))
            dependencies = ()
            stalk_sources = ()
        stalks = _spaces(stalk_spaces, cells, section.stalk_dimensions, name + "/cell")
        targets = _spaces(mediator_spaces, mediators, section.mediator_dimensions, name + "/mediator")
        restrictions = []
        for c, row in enumerate(section._inc):
            for m in row:
                matrix = section._R.get((c, m))
                entries = (tuple((i, i, 1) for i in range(len(stalks[c].keys))) if matrix is None else
                           tuple((i, j, v) for i, r in enumerate(matrix) for j, v in enumerate(r) if v))
                restrictions.append((c, m, CoordinateMap(stalks[c], targets[m], entries)))
        recipe = SectionRecipe(name, section.grade, cells, mediators, stalks, targets,
                   tuple(tuple(ms) for ms in section._inc), tuple(restrictions), source, dependencies, stalk_sources)
        return cls(recipe, _sheaf=section)

    @property
    def source(self):
        return self.recipe.source

    @property
    def coefficient_digest(self):
        return self.recipe.coefficient_digest

    def check_state(self):
        if self.source.source is None or any(s.source is None for s in self.recipe.dependencies):
            raise ValueError("bind the complete section recipe before evaluation")
        self.source.check()
        for ref in self.recipe.dependencies:
            ref.check()
        if self._sheaf is not None:
            current = SectionSystem.from_sheaf(self._sheaf, name=self.recipe.name,
                stalk_spaces=self.recipe.stalk_spaces, mediator_spaces=self.recipe.mediator_spaces,
                source=self.source)
            if current.coefficient_digest != self.coefficient_digest:
                raise ValueError("section restrictions changed; compile a fresh section system")

    def field(self, values=None, *, axes=()):
        self.check_state()
        if values is None:
            if self._sheaf is None:
                raise ValueError("restored equations require explicitly supplied section values")
            values = tuple(v for row in self._sheaf._stalks for v in row)
        return TensorField(self.space, values, tuple(axes), self.source, None, "coordinate",
                           (self.coefficient_digest,), self.recipe.dependencies)

    def check_field(self, field):
        self.check_state()
        if not isinstance(field, TensorField) or field.space != self.space or field.variance != "coordinate":
            raise ValueError("section field requires its named local coordinate space")
        if field.source is None or not self.source.matches(field.source):
            raise ValueError("section field belongs to another native source version")
        field.check_state()

    def residual(self, field):
        self.check_field(field)
        result = apply_tensor(self.action, field)
        return replace(result, dependencies=_references((*field.dependencies, *self.recipe.dependencies)),
                       provenance=(*result.provenance, self.coefficient_digest))

    def selection(self, cell):
        index = self.recipe.cells.index(cell)
        local = self.recipe.stalk_spaces[index]
        return CoordinateMap(self.space, local, tuple((i, self._offsets[index] + i, 1)
                                                     for i in range(len(local.keys))))

    def pins(self, observations):
        """Build observation coordinates only for explicitly supplied local values."""
        entries, keys, values = [], [], []
        for (cell, coordinate), value in observations.items():
            i = self.recipe.cells.index(cell)
            j = self.recipe.stalk_spaces[i].keys.index(coordinate)
            keys.append(_key(cell, coordinate))
            entries.append((len(values), self._offsets[i] + j, 1))
            values.append(_fraction(value))
        space = CoordinateSpace(self.recipe.name + "/observations", tuple(keys))
        return (CoordinateMap(self.space, space, tuple(entries)),
                TensorField(space, values, source=self.source, dependencies=self.recipe.dependencies))

    def complete(self, observation=None, observed=None):
        """Retain all sections satisfying compatibility and the supplied observations."""
        self.check_state()
        if observation is None and observed is None:
            observation, observed = self.pins({})
        if not isinstance(observation, CoordinateMap) or observation.domain != self.space:
            raise ValueError("section observations require an explicit map on the stalk coordinates")
        if not isinstance(observed, TensorField) or observed.space != observation.codomain:
            raise ValueError("observed field and observation coordinates disagree")
        if observed.source is None or not self.source.matches(observed.source):
            raise ValueError("section observations require the same declared native source")
        if observed.variance != "coordinate":
            raise ValueError("section observations require coordinate variance")
        observed.check_state()
        n = len(self.residual_space.keys)
        space = CoordinateSpace(self.recipe.name + "/equations", tuple(
            [_key("compatibility", k) for k in self.residual_space.keys] +
            [_key("observation", k) for k in observation.codomain.keys]))
        equations = CoordinateMap(self.space, space, self.action.entries +
                      tuple((n + i, j, v) for i, j, v in observation.entries))
        values = np.concatenate((np.full((n, *observed.values.shape[1:]), Fraction(0), dtype=object),
                                 observed.values), axis=0)
        rhs = TensorField(space, values, observed.axes, self.source, None, "coordinate",
                          (self.coefficient_digest, observation.coefficient_digest),
                          _references((*observed.dependencies, *self.recipe.dependencies)))
        return SectionFamily.solve(equations, rhs, self.coefficient_digest)

    def delta(self, new, input_map, output_map, old_field, new_field):
        """Retain restriction change and local field innovation at two states."""
        if not isinstance(new, SectionSystem):
            raise TypeError("a new section system is required")
        self.check_field(old_field)
        new.check_field(new_field)
        from rexgraph.temporal_calculus import TemporalOperation
        from rexgraph.temporal_field import TensorEvolution
        evolution = TensorEvolution(TemporalOperation(self.action, new.action, input_map, output_map),
                                    self.source, new.source, ("restrictions", "section"))
        return evolution.delta(old_field, new_field)


class InconsistentSectionError(ValueError):
    """A left null witness separates the supplied data from every completion."""
    def __init__(self, witness, contradiction):
        self.witness = witness
        self.contradiction = contradiction
        super().__init__("no section satisfies the exact observations and compatibility equations")


class UnderdeterminedSectionError(ValueError):
    """The requested observation varies across the admissible sections."""


@dataclass(frozen=True)
class SectionFamily:
    """A certified affine family, not a selected prediction among free directions."""
    equations: CoordinateMap
    rhs: TensorField
    particular: TensorField
    directions: CoordinateMap
    declaration_digest: str

    def __post_init__(self):
        if not isinstance(self.equations, CoordinateMap) or not isinstance(self.directions, CoordinateMap):
            raise TypeError("section families require explicit exact equation and kernel maps")
        if not isinstance(self.rhs, TensorField) or not isinstance(self.particular, TensorField):
            raise TypeError("section families require exact tensor fields")
        if (self.equations.domain != self.particular.space or self.equations.codomain != self.rhs.space
                or self.directions.codomain != self.particular.space or self.rhs.axes != self.particular.axes
                or self.rhs.variance != "coordinate" or self.particular.variance != "coordinate"):
            raise ValueError("section family coordinates disagree")
        left, right = self.rhs.source, self.particular.source
        if (left is None) != (right is None) or left is not None and not left.matches(right):
            raise ValueError("section family fields have different source states")
        if tuple(s.coefficient_digest for s in self.rhs.dependencies) != tuple(
                s.coefficient_digest for s in self.particular.dependencies):
            raise ValueError("section family fields have different contributor dependencies")
        if not isinstance(self.declaration_digest, str):
            raise TypeError("section declaration identity must be a string")
        n = self.equations.shape[1]
        p = prod(len(a.keys) for a in self.rhs.axes)
        if not np.array_equal(self.equations.apply(self.particular.values.reshape(n, p)),
                              self.rhs.values.reshape(self.equations.shape[0], p)):
            raise ValueError("section particular field does not solve its equations")
        if self.directions.compose(self.equations).entries:
            raise ValueError("section directions are not closed under the equations")
        rank = len(self.equations.as_sparse().rref()[1])
        rank_n = len(self.directions.as_sparse().rref()[1])
        if rank_n != self.directions.shape[1] or rank + rank_n != n:
            raise ValueError("section directions do not span the complete kernel")

    @classmethod
    def solve(cls, equations, rhs, declaration_digest=""):
        if not isinstance(equations, CoordinateMap) or not isinstance(rhs, TensorField):
            raise TypeError("exact completion requires named equations and right hand sides")
        if rhs.space != equations.codomain or rhs.variance != "coordinate":
            raise ValueError("completion right hand side coordinates disagree")
        rhs.check_state()
        m, n = equations.shape
        p = prod(len(a.keys) for a in rhs.axes)
        block = rhs.values.reshape(m, p)
        entries = {(i, j): v for i, j, v in equations.entries}
        entries.update({(i, n + j): v for (i, j), v in np.ndenumerate(block) if v})
        reduced, pivots = ExactSparse(m, n + p, entries).rref()
        if any(j >= n for j in pivots):
            witnesses = equations.as_sparse().T.kernel_frame()
            for col in range(witnesses.ncols):
                vector = tuple(witnesses.entries.get((i, col), Fraction(0)) for i in range(m))
                contradiction = np.asarray([sum((vector[i] * block[i, j] for i in range(m)), Fraction(0))
                                             for j in range(p)], dtype=object).reshape(tuple(len(a.keys) for a in rhs.axes))
                if any(v for v in contradiction.flat):
                    witness = TensorField(equations.codomain, vector, source=rhs.source,
                                          dependencies=rhs.dependencies, provenance=(equations.coefficient_digest,))
                    if any(equations.T.apply(vector)):
                        raise ArithmeticError("section contradiction certificate failed")
                    raise InconsistentSectionError(witness, contradiction)
            raise ArithmeticError("inconsistent system has no contradiction certificate")
        values = np.full((n, p), Fraction(0), dtype=object)
        for row, col in enumerate(pivots):
            for j in range(p):
                values[col, j] = reduced.entries.get((row, n + j), Fraction(0))
        free = [j for j in range(n) if j not in pivots]
        data = []
        for a, j in enumerate(free):
            data.append((j, a, Fraction(1)))
            for row, col in enumerate(pivots):
                coefficient = -reduced.entries.get((row, j), Fraction(0))
                if coefficient:
                    data.append((col, a, coefficient))
        parameters = CoordinateSpace("section_parameters/" + equations.coefficient_digest,
                                     tuple(equations.domain.keys[j] for j in free))
        directions = CoordinateMap(parameters, equations.domain, tuple(data))
        particular = TensorField(equations.domain, values.reshape((n, *rhs.values.shape[1:])), rhs.axes,
            rhs.source, None, "coordinate", (*rhs.provenance, equations.coefficient_digest), rhs.dependencies)
        return cls(equations, rhs, particular, directions, declaration_digest)

    @property
    def coefficient_digest(self):
        return _identity(("section_family_v1", self.equations.coefficient_digest, self.rhs.coefficient_digest,
                          self.particular.coefficient_digest, self.directions.coefficient_digest,
                          self.declaration_digest))

    @property
    def source(self):
        return self.particular.source

    @property
    def dependencies(self):
        return self.particular.dependencies

    @property
    def dimension(self):
        """Number of free scalar parameters across all retained field axes."""
        return self.direction_count * prod(len(a.keys) for a in self.particular.axes)

    @property
    def direction_count(self):
        """Independent local directions for each retained field component."""
        return self.directions.shape[1]

    def check_state(self):
        self.particular.check_state()
        self.rhs.check_state()

    def bind(self, source, dependencies=()):
        """Bind a restored exact family to its source and contributor versions."""
        if self.particular.source is None or not isinstance(source, FieldSource):
            raise ValueError("this family has no bindable native source")
        if source.source is None or not source.matches(self.particular.source):
            raise ValueError("section family requires its exact source version")
        dependencies = tuple(dependencies)
        expected = self.particular.dependencies
        if len(expected) != len(dependencies) or any(not isinstance(b, FieldSource)
                or b.source is None or not a.matches(b) for a, b in zip(expected, dependencies, strict=True)):
            raise ValueError("section family requires its exact contributor versions")
        source.check()
        for dependency in dependencies:
            dependency.check()
        return replace(self, particular=replace(self.particular, source=source, dependencies=dependencies),
                       rhs=replace(self.rhs, source=source, dependencies=dependencies))

    def observe(self, action):
        return SectionImage(self, action)

    def evaluate(self, parameters):
        self.check_state()
        if not isinstance(parameters, TensorField) or parameters.space != self.directions.domain:
            raise ValueError("section parameters require the named free coordinates")
        if parameters.axes != self.particular.axes or parameters.variance != "coordinate":
            raise ValueError("section parameters must retain the observed field axes")
        parameters.check_state()
        n = self.directions.shape[1]
        p = prod(len(a.keys) for a in parameters.axes)
        value = self.particular.values + self.directions.apply(parameters.values.reshape(n, p)).reshape(self.particular.values.shape)
        return replace(self.particular, values=value,
                       provenance=(*self.particular.provenance, parameters.coefficient_digest),
                       dependencies=_references((*self.particular.dependencies, *parameters.dependencies,
                                                 *((parameters.source,) if parameters.source is not None else ()))))


@dataclass(frozen=True)
class SectionImage:
    """An observation of every admissible section with its remaining variation."""
    family: SectionFamily
    action: object
    particular: TensorField = field(init=False)
    directions: CoordinateMap = field(init=False)

    def __post_init__(self):
        if not isinstance(self.family, SectionFamily) or not _is_action(self.action):
            raise TypeError("section observation requires an exact family and action")
        if self.action.domain != self.family.particular.space:
            raise ValueError("section observation domain disagrees with the family")
        self.family.check_state()
        observed = apply_tensor(self.action, self.family.particular)
        columns = self.family.directions.shape[1]
        entries = []
        for j in range(columns):
            value = np.full(self.action.shape[1], Fraction(0), dtype=object)
            for i, a, coefficient in self.family.directions.entries:
                if a == j:
                    value[i] = coefficient
            result = self.action.apply(value)
            entries.extend((i, j, _fraction(v)) for i, v in enumerate(result) if v)
        object.__setattr__(self, "particular", observed)
        object.__setattr__(self, "directions", CoordinateMap(self.family.directions.domain,
                                                           self.action.codomain, tuple(entries)))

    @property
    def determined(self):
        return self.family.dimension == 0 or not self.directions.entries

    @property
    def coefficient_digest(self):
        return _identity(("section_image_v1", self.family.coefficient_digest, self.action.coefficient_digest))

    def value(self):
        self.family.check_state()
        self.particular.check_state()
        if not self.determined:
            raise UnderdeterminedSectionError("this observation varies over the retained section family")
        return self.particular
