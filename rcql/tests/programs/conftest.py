"""Direct inventory cases for the finite program operators."""
from pathlib import Path
import pytest

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    from rcql.program_contracts import ARGUMENTS
    from .helpers import query_case
    modules={i.module for i in items if getattr(i,'module',None) and Path(getattr(i.module,'__file__','')).name=='test_operator_inventory.py'}
    for module in modules:
        for name in ARGUMENTS:
            if name in module.NATIVE_CASES:raise AssertionError('new case would overwrite existing coverage')
            module.NATIVE_CASES[name]=lambda rex,name=name:query_case(name,rex)
