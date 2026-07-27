import numpy as np
import pytest
from rexgraph.graph import RexGraph


def _rich():
    g = RexGraph(sources=np.array([0, 1, 2, 0], np.int32),
                 targets=np.array([1, 2, 3, 3], np.int32),
                 w_E=np.array([1.5, 2.0, 0.5, 3.0]), signs=[1, -1, 1, -1],
                 g_channel="normalized")
    g.set_vertex_attribution(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))
    g.attach_metadata(1, 0, "kind", "activation")
    g._agent_meta = {"vertex_labels": ["a", "b", "c", "d"], "source": "unit"}
    return g


def _assert_full_roundtrip(orig, r):
    assert np.array_equal(np.asarray(r.B1), np.asarray(orig.B1))
    assert np.array_equal(np.asarray(r._signs), np.asarray(orig._signs))
    assert r._g_channel == orig._g_channel
    assert r._w_boundary and set(r._w_boundary) == set(orig._w_boundary)
    assert r.get_metadata(1, 0, "kind") == "activation"
    assert r._agent_meta["vertex_labels"] == ["a", "b", "c", "d"]


def test_arrow_full_roundtrip():
    from rexgraph.io.arrow_bridge import rex_to_arrow, arrow_to_rex
    g = _rich()
    _assert_full_roundtrip(g, arrow_to_rex(rex_to_arrow(g)))


def test_hdf5_full_roundtrip(tmp_path):
    from rexgraph.io.hdf5_format import RexHDF5Format
    g = _rich()
    p = str(tmp_path / "g.h5")
    fmt = RexHDF5Format()
    fmt.write(p, g)
    _assert_full_roundtrip(g, fmt.read(p))


def test_zarr_full_roundtrip(tmp_path):
    from rexgraph.io import save, load
    g = _rich()
    p = str(tmp_path / "g.zarr")
    save(p, g)
    _assert_full_roundtrip(g, load(p))


def test_all_formats_agree_on_a_rich_complex(tmp_path):
    # every full-object format reconstructs the SAME rich complex (edge primacy + attribution + signs
    # + g_channel + labels + cell metadata), via the one canonical rex-state encoder.
    from rexgraph.io.bundle import save_rex, load_rex
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
    from rexgraph.io.arrow_bridge import rex_to_arrow, arrow_to_rex
    from rexgraph.io.hdf5_format import RexHDF5Format
    from rexgraph.io import save as zsave, load as zload
    g = _rich()
    rex_p = str(tmp_path / "g.rex"); save_rex(rex_p, g)
    st_p = str(tmp_path / "g.safetensors"); rex_to_safetensors(g, st_p)
    h5_p = str(tmp_path / "g.h5"); RexHDF5Format().write(h5_p, g)
    z_p = str(tmp_path / "g.zarr"); zsave(z_p, g)
    for r in (load_rex(rex_p), safetensors_to_rex(st_p), arrow_to_rex(rex_to_arrow(g)),
              RexHDF5Format().read(h5_p), zload(z_p)):
        _assert_full_roundtrip(g, r)
        assert list(r.betti) == list(g.betti)
