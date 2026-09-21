from pathlib import Path
import pytest


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(session, config, items):
    yield
    from rcql.transformation_contracts import ARGUMENTS
    module = next((i.module for i in items if Path(i.module.__file__).name == 'test_transformation.py'), None)
    if module is None:
        return
    inventories = {i.module for i in items if Path(i.module.__file__).name == 'test_operator_inventory.py'}
    sections = next((i.module for i in items if Path(i.module.__file__).name == 'test_section_transform.py'), None)
    constructions = next((i.module for i in items if Path(i.module.__file__).name == 'test_construction_transform.py'), None)
    programs = next((i.module for i in items if Path(i.module.__file__).name == 'test_finite_transform.py'), None)
    families = next((i.module for i in items if Path(i.module.__file__).name == 'test_program_family.py'), None)
    evolutions = next((i.module for i in items if Path(i.module.__file__).name == 'test_program_evolution.py'), None)
    for inventory in inventories:
        for name in ARGUMENTS:
            assert name not in inventory.NATIVE_CASES
            owner = sections if name in {'TRANSFORM_SECTION', 'TRANSFORM_SOURCE', 'SECTION_CERTIFIED_OBSERVE'} else module
            if name in {'TRANSFORM_SPECIALIZE', 'TRANSFORM_COMPOSE'}:
                owner = constructions
            if name in {'TRANSFORM_PROGRAM','TRANSFORM_PROGRAM_COMPILE','TRANSFORM_PROGRAM_TOPOLOGY'}:
                owner = programs
            inventory.NATIVE_CASES[name] = lambda rex, name=name, owner=owner: owner.case(name, rex)
        from rcql.program_contracts import ARGUMENTS as PROGRAM_ARGUMENTS
        for name in PROGRAM_ARGUMENTS:
            if name in {'PROGRAM_ASSEMBLY','PROGRAM_GLUE'} or name.startswith(('PROGRAM_FAMILY_', 'PROGRAM_ASSEMBLY_')):
                owner=families
            elif name=='PROGRAM_COMPARE' or name=='PROGRAM_BOUNDARY_CHANGE' or name.startswith('PROGRAM_EVOLUTION_'):
                owner=evolutions
            else:
                continue
            previous = inventory.NATIVE_CASES.get(name)
            if previous is not None:
                # The finite program hook enumerates every program contract, but its
                # fixture dispatcher predates these operations, so their owners replace it.
                assert previous.__module__.endswith('programs.conftest')
            inventory.NATIVE_CASES[name] = lambda rex, name=name, owner=owner: owner.case(name,rex)
