"""Native section completion through existing source and query contracts."""
from .execution_trace import current_binding, record_method
from .capabilities import SourcePolicy
from .binding import Binding


def _selected(source, reference):
    if reference is None or reference.source is not source:
        raise ValueError("section operation requires its actual selected native source")


def _reference(binding, expected=None):
    from rexgraph.tensor_field import FieldSource
    if expected is not None:
        if expected.record_id is not None and expected.record_id != binding.ref.record_id:
            raise ValueError("section source record identity differs")
        if expected.version is not None and expected.version != binding.ref.record_version:
            raise ValueError("section source version differs")
        return FieldSource(binding.value, expected.record_id, expected.version, expected.state_digest)
    return FieldSource(binding.value, binding.ref.record_id, binding.ref.record_version)


def _policy(binding, contributors):
    if binding is None:
        if contributors:
            raise PermissionError("a phrase section requires a policy bearing query binding")
        return
    binding.source.require("read")
    for contributor in contributors:
        if not isinstance(contributor, Binding):
            raise TypeError("section contributors must be explicit source Bindings")
        contributor.source.require("read")
    intersection = SourcePolicy.intersection(binding.source.policy,
                                            *(v.source.policy for v in contributors))
    if intersection.digest != binding.source.policy.digest:
        raise PermissionError("bind the section source under the contributor intersection policy")


def authorize_system(system, binding):
    system.check_state()
    if binding is not None:
        if system.source.source is not binding.value:
            raise ValueError("section system belongs to another selected source")
        _reference(binding, system.source)
    contributors = getattr(system, "_query_contributors", None)
    if contributors is None:
        sheaf = system._sheaf
        contributors = tuple(s.source for s in sheaf.stalks) if hasattr(sheaf, "correspondences") else ()
    if len(contributors) != len(system.recipe.dependencies):
        # A phrase may use the same selected state in several named stalks.
        contributors = tuple({(_reference(c).coefficient_digest): c for c in contributors}.values())
    if len(contributors) != len(system.recipe.dependencies):
        raise ValueError("bind every section contributor before querying")
    for expected, actual in zip(system.recipe.dependencies, contributors, strict=True):
        ref = _reference(actual, expected)
        if not ref.matches(expected):
            raise ValueError("section contributor changed")
    _policy(binding, contributors)


def authorize_family(family, binding):
    family.check_state()
    source = family.particular.source
    if binding is not None:
        if source is None or source.source is not binding.value:
            raise ValueError("section family requires its live selected source")
        _reference(binding, source)
    contributors = getattr(family, "_query_contributors", ())
    _policy(binding, contributors)
    if family.particular.dependencies and not contributors:
        raise ValueError("a persisted section family needs explicit contributor bindings")
    for ref in family.particular.dependencies:
        found = False
        for candidate in contributors:
            try:
                found = _reference(candidate, ref).matches(ref)
            except ValueError:
                continue
            if found:
                break
        if not found:
            raise ValueError("section family contributor binding is missing")


def validate_restore(recipe, contributors, binding):
    if binding is None:
        raise PermissionError("section restoration requires an explicit source binding")
    contributors = tuple(contributors)
    _policy(binding, contributors)
    if len(contributors) != len(recipe.dependencies):
        raise ValueError("every recorded section contributor must be supplied")
    source = _reference(binding, recipe.source)
    dependencies = tuple(_reference(actual, expected) for expected, actual in
                         zip(recipe.dependencies, contributors, strict=True))
    return source, dependencies


def section_system(source, section, name="section", stalk_spaces=None, mediator_spaces=None):
    from rexgraph.section_calculus import SectionSystem
    from rexgraph.tensor_field import FieldSource
    if section.rex is not source:
        raise ValueError("section belongs to another source")
    binding = current_binding()
    reference = FieldSource(source) if binding is None else _reference(binding)
    result = SectionSystem.from_sheaf(section, name=name, stalk_spaces=stalk_spaces,
                                     mediator_spaces=mediator_spaces, source=reference)
    authorize_system(result, binding)
    record_method("exact-incidence-section-system")
    return result


def section_field(source, system, values=None, axes=()):
    _selected(source, system.source)
    authorize_system(system, current_binding())
    record_method("exact-local-section-field")
    return system.field(values, axes=axes)


def section_residual(source, system, field):
    _selected(source, system.source)
    authorize_system(system, current_binding())
    record_method("exact-retained-section-residual")
    return system.residual(field)


def section_complete(source, system, observation=None, observed=None):
    _selected(source, system.source)
    authorize_system(system, current_binding())
    result = system.complete(observation, observed)
    contributors = getattr(system, "_query_contributors", None)
    if contributors is None:
        contributors = tuple(s.source for s in system._sheaf.stalks) if hasattr(system._sheaf, "correspondences") else ()
    object.__setattr__(result, "_query_contributors", contributors)
    record_method("rational-affine-section-completion")
    return result


def section_observe(source, family, action):
    _selected(source, family.source)
    authorize_family(family, current_binding())
    record_method("exact-section-family-observation")
    return family.observe(action)


def section_value(source, image):
    _selected(source, image.family.source)
    authorize_family(image.family, current_binding())
    record_method("exact-invariant-section-value")
    return image.value()


def section_recipe(source, system):
    _selected(source, system.source)
    authorize_system(system, current_binding())
    record_method("exact-section-recipe")
    return system.recipe.detached()


def section_restore(source, recipe, contributors=()):
    binding = current_binding()
    if binding is None:
        from .binding import bind
        binding = bind("section_source", source, SourcePolicy.allow("read"))
    reference, dependencies = validate_restore(recipe, contributors, binding)
    result = recipe.bind(reference, dependencies)
    object.__setattr__(result, "_query_contributors", tuple(contributors))
    record_method("verified-section-recipe-binding")
    return result


def authorize_destination(system, binding):
    if binding is None:
        authorize_system(system, None)
        return
    from .binding import bind
    from .types import SourceRef
    destination = bind("section_destination", system.source.source, binding.source.policy,
        source_ref=SourceRef(name="section_destination", record_id=system.source.record_id,
                             record_version=system.source.version, state_digest=system.source.state_digest))
    authorize_system(system, destination)


def section_delta(source, old, new, input_map, output_map, old_field, new_field):
    _selected(source, old.source)
    binding = current_binding()
    authorize_system(old, binding)
    authorize_destination(new, binding)
    record_method("exact-temporal-section-fields")
    return old.delta(new, input_map, output_map, old_field, new_field)


def install(register):
    for name, fn in (("SECTION_SYSTEM", section_system), ("SECTION_FIELD", section_field),
                     ("SECTION_RESIDUAL", section_residual), ("SECTION_COMPLETE", section_complete),
                     ("SECTION_OBSERVE", section_observe), ("SECTION_VALUE", section_value),
                     ("SECTION_RECIPE", section_recipe), ("SECTION_RESTORE", section_restore),
                     ("SECTION_DELTA", section_delta)):
        register(name)(fn)
