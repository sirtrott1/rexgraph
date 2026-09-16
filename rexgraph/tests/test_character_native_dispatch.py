"""Fresh character reads must not ask an unrelated spectral bundle for a mode."""
import builtins

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.sparse_character import build_sparse_character_cheap


@pytest.mark.parametrize("cells", [[[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2], [0], [1, 2]], []])
@pytest.mark.parametrize("normalized", [False, True])
def test_character_labels_and_star_never_read_spectral_or_dense_bundle(cells, normalized, monkeypatch):
    rex = RexGraph.from_cells([4, cells], g_channel="normalized" if normalized else "raw")
    def forbidden(*args, **kw):
        pytest.fail("native character reached a spectral or dense oracle")
    for name in ("spectral_bundle", "_dense_rcf_bundle", "_rcf_bundle", "_vertex_bundle", "_rl_eigen"):
        monkeypatch.setattr(RexGraph, name, property(forbidden))
    original_import = builtins.__import__
    def native_import(name, *args, **kw):
        if name == "scipy" or name.startswith("scipy."):
            pytest.fail("native character imported SciPy")
        return original_import(name, *args, **kw)
    monkeypatch.setattr(builtins, "__import__", native_import)
    expected = build_sparse_character_cheap(rex)
    assert rex._use_sparse_character is True
    assert rex.nhats == 4 and rex.hat_names == expected["hat_names"]
    np.testing.assert_array_equal(rex.structural_character, expected["chi"])
    np.testing.assert_array_equal(rex.star_character, expected["chi_star"])
    assert rex.structural_character.shape == (len(cells), 4)
    assert not {"spectral_bundle", "_rcf_bundle", "_dense_rcf_bundle"} & rex.__dict__.keys()
