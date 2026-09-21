"""Direct inventory cases for the molecular operators."""
from pathlib import Path
import tempfile
import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    from rcql.molecular_contracts import ARGUMENTS
    from .test_storage_queries import prepared
    modules = {item.module for item in items if getattr(item, 'module', None) is not None
               and Path(getattr(item.module, '__file__', '')).name == 'test_operator_inventory.py'}
    def arguments(name):
        with tempfile.TemporaryDirectory() as directory:
            return prepared(name, Path(directory))[1]
    for module in modules:
        for name in ARGUMENTS:
            if name in module.NATIVE_CASES:
                raise RuntimeError('new molecular case would replace an existing case')
            module.NATIVE_CASES[name] = lambda rex, name=name: arguments(name)
