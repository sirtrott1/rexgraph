"""Direct inventory cases for the model lifecycle operators."""
from pathlib import Path
import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    from importlib.util import find_spec
    from rcql.model_contracts import ARGUMENTS
    if find_spec("torch") is None:
        # The lifecycle cases train a torch model; without the optional runtime they skip.
        def prepared(name):
            pytest.skip("model lifecycle cases require torch")
    else:
        from .test_rcql_model import prepared
    modules={item.module for item in items if getattr(item,'module',None) is not None
             and Path(getattr(item.module,'__file__','')).name=='test_operator_inventory.py'}
    for module in modules:
        for name in ARGUMENTS:
            if name in module.NATIVE_CASES:raise RuntimeError('new model case would replace an existing case')
            module.NATIVE_CASES[name]=lambda rex,name=name:prepared(name)[1]
