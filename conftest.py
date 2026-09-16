"""Repo root pytest shim for the editable (meson-python) install.

Editable installs serve `rexgraph` (and its compiled extensions) through the meson-python import
finder, which exposes only INSTALLED sources. The test package `rexgraph.tests` is deliberately not
shipped, so it is not in that map and `import rexgraph.tests` fails during collection. Point the
package's __path__ at the physical source dir so the tests (and their reference fixtures) resolve
from the tree, while the package itself and the .so still come from the finder / build dir.
"""
import pathlib
import os
import importlib.util
import sys

import rexgraph

_src = str(pathlib.Path(__file__).parent / "rexgraph")
if os.environ.get("REXGRAPH_TEST_INSTALLED") == "1":
    # Expose only test modules. Never let an omitted runtime module resolve
    # from the checkout through the installed package's search path.
    if "rexgraph.tests" not in sys.modules:
        _tests = pathlib.Path(_src) / "tests"
        _spec = importlib.util.spec_from_file_location("rexgraph.tests", _tests / "__init__.py",
                                                      submodule_search_locations=[str(_tests)])
        _module = importlib.util.module_from_spec(_spec)
        sys.modules["rexgraph.tests"] = _module
        rexgraph.tests = _module
        _spec.loader.exec_module(_module)
elif _src not in rexgraph.__path__:
    rexgraph.__path__.append(_src)


# The sibling distributions shadow themselves from this directory. `rex-agent/rcql` is a
# DIRECTORY holding the `rcql` package and its tests, so with the repo root on sys.path --
# which pytest puts there to find this file -- `import rcql` binds that directory as a
# namespace package: importable, `__file__` None, and carrying none of the real module's
# names. A test guarded by `pytest.importorskip("rcql")` therefore does not skip; it
# proceeds and fails on the first attribute.
#
# Bind each one to its installed location instead, and only when it is genuinely shadowed,
# so an environment without the distribution still raises ImportError and still skips.
for _dist in ("rcql", "rcdb", "system"):
    _shadowed = sys.modules.get(_dist)
    if _shadowed is not None and getattr(_shadowed, "__file__", None) is not None:
        continue
    _init = pathlib.Path(__file__).parent / _dist / _dist / "__init__.py"
    if not _init.is_file():
        continue
    sys.modules.pop(_dist, None)
    _spec = importlib.util.spec_from_file_location(
        _dist, _init, submodule_search_locations=[str(_init.parent)])
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_dist] = _module
    try:
        _spec.loader.exec_module(_module)
    except Exception:            # not installed, or its own deps absent: let it skip
        sys.modules.pop(_dist, None)
