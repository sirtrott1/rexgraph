"""Every advertised cache group has to write something.

A cache table is a promise about what `cache="X"` does. The writers resolve names by
`getattr` inside `try/except: pass`, so a name that no longer exists costs nothing and
says nothing: the call succeeds, the file is written, and the thing you asked for is
absent. That is how these drifted.

What this found when it was written:

    wave, quotient, persistence   groups with no writer, or a writer that created an
                                  empty group and returned. `cache="wave"` was a no-op.
    overlap_adjacency             named by all four formats; not a property of RexGraph
                                  at all, so all four skipped it silently.
    safetensors                   a comment claiming it mirrored bundle's table exactly,
                                  next to a table that had already diverged from it.

So the rule is: a declared group must reach a writer, and a declared name must be
resolvable on the object it is written from. Both are checked here rather than trusted.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io._cache_layout import _CACHE_GROUPS

ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> RexGraph:
    """A tetrahedron with weights: cycles, a metric, and every channel non-trivial."""
    rex = RexGraph(sources=np.array([0, 1, 2, 0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0, 3, 3, 3], np.int32),
                   w_E=np.linspace(1.0, 2.0, 6))
    rex._ensure_clean()
    return rex


def _writers() -> set[str]:
    """The group each `_write_<group>_cache` covers, across the array backends."""
    found = set()
    for name in ("_cache_layout.py", "hdf5_format.py", "zarr_format.py"):
        tree = ast.parse((ROOT / "rexgraph" / "io" / name).read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name.startswith("_write_")
                    and node.name.endswith("_cache")):
                found.add(node.name[len("_write_"):-len("_cache")])
    return found


def test_every_declared_group_has_a_writer():
    """`wave` was declared, documented, and written by nothing at all."""
    missing = sorted(set(_CACHE_GROUPS) - _writers())
    assert not missing, f"groups nothing writes: {missing}"


def test_no_writer_covers_a_group_nobody_declares():
    """The other direction: a writer with no table entry is unreachable."""
    # `_write_cache` is the dispatcher, so its parsed name is empty; `rex_graph` and
    # `temporal_rex` write the object itself rather than a cache group.
    structural = {"", "rex_graph", "temporal_rex", "cache"}
    orphan = sorted(_writers() - set(_CACHE_GROUPS) - structural)
    assert not orphan, f"writers no group reaches: {orphan}"


@pytest.mark.parametrize("group", sorted(_CACHE_GROUPS))
def test_each_group_writes_something(group):
    """Ask for exactly one group and check the file grew.

    Counts attributes as well as datasets: a scalar such as betti lands as an HDF5
    attribute, and a check that only walks datasets calls it dead when it is not.
    """
    h5py = pytest.importorskip("h5py")
    from rexgraph.io.hdf5_format import RexHDF5Format

    # temporal entries describe a TemporalRex, so a RexGraph has nothing to write there
    if group == "temporal":
        pytest.skip("temporal caches a TemporalRex, not a RexGraph")

    rex, fmt = _fixture(), RexHDF5Format()
    with tempfile.TemporaryDirectory() as d:
        def contents(path):
            out = set()

            # a named function, not a lambda: visititems STOPS as soon as the callback
            # returns anything other than None, and a lambda whose body is a tuple
            # returns a truthy one, so the walk halts after the first entry and every
            # group looks empty. That is what this test reported before it was fixed.
            def visit(name, obj):
                out.add(name)
                out.update(f"{name}@{k}" for k in obj.attrs)

            with h5py.File(path) as f:
                f.visititems(visit)
                out.update(f"@{k}" for k in f.attrs)
            return out

        base = f"{d}/base.h5"
        fmt.write(base, rex, cache=None)
        one = f"{d}/one.h5"
        fmt.write(one, rex, cache=group)
        assert contents(one) - contents(base), f"cache={group!r} wrote nothing"


def test_the_bundle_formats_agree_by_construction():
    """safetensors derives its table from bundle, so the two cannot drift apart."""
    from rexgraph.io import bundle, safetensors_bridge

    shared = set(safetensors_bridge._CACHE_GROUPS)
    assert shared == set(bundle._CACHE_GROUPS) - safetensors_bridge._UNSUPPORTED_GROUPS
    for name in shared:
        assert (safetensors_bridge._CACHE_GROUPS[name]
                == bundle._CACHE_GROUPS[name]), name


@pytest.mark.parametrize("group", ["algebra", "spectral", "topology", "hodge"])
def test_the_bundle_groups_write_something(group):
    from rexgraph.io import bundle

    rex = _fixture()
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/x.rex"
        bundle.save_rex(path, rex, cache=group)
        import json
        manifest = json.loads(Path(path, "MANIFEST.json").read_text())
        wrote = set(manifest.get("cached_arrays") or [])
        wrote |= set(manifest.get("cache_scalars") or {})
        assert wrote, f"bundle cache={group!r} wrote nothing"


def test_standard_metrics_writes_the_metrics_and_not_just_a_marker():
    """It wrote an empty group with a type marker for as long as it has existed.

    Two faults stacked. The kernel call passed `(nV, nE, src, tgt)` where
    `build_standard_metrics` takes the adjacency CSR and at least six arguments, so it
    raised TypeError on every call and the surrounding `except Exception: pass` hid it.
    Fixing that surfaced the second: the kernel returns a DICT, `write_namedtuple` wants
    a namedtuple, and writing the dict produced a group containing `_type_name` and
    nothing else. `StandardMetrics._fields` is exactly the dict's keys.
    """
    h5py = pytest.importorskip("h5py")
    from rexgraph.io.hdf5_format import RexHDF5Format
    from rexgraph.rextypes import StandardMetrics

    rex = _fixture()
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/sm.h5"
        RexHDF5Format().write(path, rex, cache="standard_metrics")
        with h5py.File(path) as f:
            group = f["standard_metrics"]
            written = set(group.keys())
            pagerank = group["pagerank"][:]

    # every array field lands; the two scalars ride as attributes
    arrays = {n for n in StandardMetrics._fields
              if n not in ("n_communities", "modularity")}
    assert arrays <= written, f"missing: {sorted(arrays - written)}"
    assert pagerank.shape == (int(rex.nV),)
    assert pagerank.sum() == pytest.approx(1.0, abs=1e-6), pagerank.sum()
