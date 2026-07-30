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


def test_generic_save_supports_safetensors(tmp_path):
    """io.load routed .safetensors but io.save had no branch for it, so the flagship
    format was load-only through the generic entry point and save() raised
    'Unknown format'."""
    import numpy as np
    from rexgraph import io
    from rexgraph.graph import RexGraph

    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    p = tmp_path / "g.safetensors"
    io.save(str(p), rex)
    assert p.exists()
    back = io.load(str(p))
    assert (int(back.nV), int(back.nE)) == (int(rex.nV), int(rex.nE))


def test_generic_save_load_round_trips_every_dependency_free_format(tmp_path):
    """save and load must accept the same set of formats: an asymmetry means a format
    you can read is one you cannot write."""
    import numpy as np
    from rexgraph import io
    from rexgraph.graph import RexGraph

    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    for name in ("g.rex", "g.safetensors", "g.json"):
        p = tmp_path / name
        io.save(str(p), rex)
        back = io.load(str(p))
        assert back is not None, f"{name} did not round-trip"
        assert int(back.nE) == int(rex.nE), f"{name} lost edges"


def test_unknown_extension_is_an_error_not_a_silent_zarr_write(tmp_path):
    """_detect_format fell through to 'zarr' for anything unrecognized, so a typo like
    'graph.saftensors' or 'graph.txt' silently wrote a Zarr store under that name. Same
    failure class as a secret store writing a file named 'vault://team/prod'."""
    import numpy as np
    import pytest
    from rexgraph import io
    from rexgraph.graph import RexGraph

    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    for bad in ("g.saftensors", "g.txt", "g.parquet"):
        with pytest.raises(ValueError) as ei:
            io.save(str(tmp_path / bad), rex)
        assert "format" in str(ei.value).lower()


def test_directory_and_extensionless_heuristics_still_work(tmp_path):
    """The legitimate heuristics stay: an existing .rex bundle dir, an existing Zarr
    dir, and an explicit format override."""
    import numpy as np
    from rexgraph import io
    from rexgraph.graph import RexGraph

    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    # explicit override needs no extension at all (save_rex appends .rex itself)
    p = tmp_path / "explicit"
    io.save(str(p), rex, format="rex")
    assert io.load(str(p), format="rex") is not None
    # and the resulting bundle directory is detected without an override
    assert io.load(str(p) + ".rex") is not None
