"""The suite runs against real packages, not the directories that shadow them.

<repo>/<name>/ has no __init__.py, so from the repository root it resolves as a namespace
package of unknown location. Everything imported from it then fails, and the failure looks
like broken behaviour rather than a broken import path. conftest.py prevents that; these
prove it stayed prevented, because the shim is silent by construction.
"""
from __future__ import annotations

import sys

import rcdb


def test_the_package_under_test_is_not_a_namespace_shim():
    assert rcdb.__file__ is not None, (
        "rcdb resolved to a namespace package, so the tests below are running against "
        "an empty shim rather than the real package"
    )
    assert "rcdb" in rcdb.__file__


def test_no_sibling_distribution_resolved_to_a_shim():
    """Any sibling this suite pulled in must be real, not only the package under test.

    This is the case that actually escaped. A test added an rcdb dependency to the rcql
    suite, the conftest at the time de-shadowed only rcql, and pytest.importorskip('rcdb')
    PASSED against the namespace shim, so the guard read as satisfied and six tests failed
    on a missing attribute instead. Checking every sibling that was imported catches that
    without anyone remembering to update a list.
    """
    shims = [
        name
        for name in ("agent", "rcdb", "rcql", "rexgraph", "system")
        if name in sys.modules and getattr(sys.modules[name], "__file__", None) is None
    ]
    assert not shims, f"these resolved to namespace shims rather than real packages: {shims}"
