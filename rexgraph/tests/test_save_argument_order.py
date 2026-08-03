"""The save/load surface: every `save_*` takes (path, obj).

A uniform order lets the format registry hold each saver directly, so adding a format
means registering a function rather than writing an adapter for it.

`load_*` returns the object, with `load_safetensors` the documented exception: it
returns the object together with its cached arrays and metadata.
"""

import warnings

import numpy as np
import pytest

import rexgraph.io as rio
from rexgraph.faces import autoface
from rexgraph.graph import RexGraph

# a branching relation and a face, so the round-trip carries arity and grade 2
BP = np.array([0, 3, 5, 7, 9], np.int32)
BI = np.array([0, 1, 2, 0, 1, 1, 2, 2, 3], np.int32)

SAVERS = [
    (".rex", "save_rex", "load_rex"),
    (".zarr", "save_zarr", "load_zarr"),
    (".h5", "save_hdf5", "load_hdf5"),
]


def _rex():
    r = RexGraph.from_hypergraph(BP, BI)
    autoface(r, 3)
    return r


def _shape(g):
    return (int(g.nV), int(g.nE), int(g.nF_hodge), tuple(int(b) for b in g.betti))


@pytest.mark.parametrize(("ext", "save", "load"), SAVERS)
def test_save_takes_path_then_obj(tmp_path, ext, save, load):
    r = _rex()
    p = str(tmp_path / ("t" + ext))
    getattr(rio, save)(p, r)
    assert _shape(getattr(rio, load)(p)) == _shape(r)


def test_safetensors_takes_path_then_obj(tmp_path):
    r = _rex()
    p = str(tmp_path / "t.safetensors")
    rio.save_safetensors(p, r)
    assert _shape(rio.safetensors_to_rex(p)) == _shape(r)


def test_the_swapped_order_warns_and_still_works(tmp_path):
    """RexGraph and path have disjoint types, so the swap is detected, not inferred."""
    r = _rex()
    p = str(tmp_path / "swapped.safetensors")
    with pytest.warns(DeprecationWarning, match=r"\(path, obj\)"):
        rio.save_safetensors(r, p)
    assert _shape(rio.safetensors_to_rex(p)) == _shape(r)


def test_the_registry_holds_the_saver_directly():
    """No argument-swapping adapter stands between the registry and the function."""
    assert not hasattr(rio, "_save_safetensors")


@pytest.mark.parametrize("ext", [".rex", ".json", ".zarr", ".h5", ".safetensors"])
def test_generic_save_load_round_trips_every_format(tmp_path, ext):
    r = _rex()
    p = str(tmp_path / ("g" + ext))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rio.save(p, r)
    assert _shape(rio.load(p)) == _shape(r)


def test_load_safetensors_returns_object_and_cache(tmp_path):
    """The asymmetric path, for callers that want the cached arrays without recomputing
    them. `load` and `safetensors_to_rex` return the object alone."""
    r = _rex()
    p = str(tmp_path / "d.safetensors")
    rio.save_safetensors(p, r)
    got = rio.load_safetensors(p)
    assert set(got) >= {"object", "tensors", "metadata", "scalars"}
    assert _shape(got["object"]) == _shape(r)
