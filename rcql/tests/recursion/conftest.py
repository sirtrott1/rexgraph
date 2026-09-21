from pathlib import Path
import pytest

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session,config,items):
    from rcql.recursion_contracts import ARGUMENTS
    from .query_cases import query_case
    modules={i.module for i in items if Path(getattr(i.module,'__file__','')).name=='test_operator_inventory.py'}
    for module in modules:
        for name in ARGUMENTS:
            if name in module.NATIVE_CASES:
                raise AssertionError('new case overlaps an original contract')
            module.NATIVE_CASES[name]=lambda rex,name=name:query_case(name,rex)
