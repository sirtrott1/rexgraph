"""Direct inventory cases for the section operators."""
from pathlib import Path
import pytest


def section_cases(rex):
    from rexgraph.sheaf import ExactSheaf
    from rexgraph.coordinate_map import CoordinateMap
    sheaf = ExactSheaf(rex, grade=0)
    system = sheaf.section_system()
    observation, field = system.pins({(system.recipe.cells[0], '0'): 3})
    family = system.complete(observation, field)
    action = system.selection(system.recipe.cells[0])
    image = family.observe(action)
    x = system.field()
    return {
        'SECTION_SYSTEM': (sheaf,), 'SECTION_FIELD': (system,),
        'SECTION_RESIDUAL': (system, x),
        'SECTION_COMPLETE': (system, observation, field),
        'SECTION_OBSERVE': (family, action), 'SECTION_VALUE': (image,),
        'SECTION_RECIPE': (system,), 'SECTION_RESTORE': (system.recipe.detached(),),
        'SECTION_DELTA': (system, system, CoordinateMap.identity(system.space),
                          CoordinateMap.identity(system.residual_space), x, x),
    }


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    from rcql.section_contracts import ARGUMENTS
    modules = {item.module for item in items if getattr(item, 'module', None) is not None
               and Path(getattr(item.module, '__file__', '')).name == 'test_operator_inventory.py'}
    for module in modules:
        for name in ARGUMENTS:
            if name in module.NATIVE_CASES:
                raise RuntimeError('a section case would replace an existing inventory case')
            module.NATIVE_CASES[name] = lambda rex, name=name: section_cases(rex)[name]
