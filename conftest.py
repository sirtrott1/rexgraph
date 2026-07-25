"""Repo-root pytest shim for the editable (meson-python) install.

Editable installs serve `rexgraph` (and its compiled extensions) through the meson-python import
finder, which exposes only INSTALLED sources. The test package `rexgraph.tests` is deliberately not
shipped, so it is not in that map and `import rexgraph.tests` fails during collection. Point the
package's __path__ at the physical source dir so the tests (and their reference fixtures) resolve
from the tree, while the package itself and the .so still come from the finder / build dir.
"""
import pathlib

import rexgraph

_src = str(pathlib.Path(__file__).parent / "rexgraph")
if _src not in rexgraph.__path__:
    rexgraph.__path__.append(_src)
