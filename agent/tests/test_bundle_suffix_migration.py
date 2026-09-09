"""Writes land on the canonical suffix; bundles written before the rename still load.

The rename is only half a migration if the legacy half is untested: `doc_path` resolves
through a DEFAULT suffix, so moving that default is exactly the change that could make
every pre-rename document unreachable without any test noticing.
"""
import numpy as np
import pytest
from agent.formats import BUNDLE_SUFFIX, BUNDLE_SUFFIXES, LEGACY_BUNDLE_SUFFIXES, is_bundle_suffix


@pytest.fixture
def rex():
    from rexgraph import RexGraph
    return RexGraph(
        boundary_ptr=np.asarray([0, 2, 4], dtype=np.int64),
        boundary_idx=np.asarray([0, 1, 1, 2], dtype=np.int64),
    )


def test_the_canonical_suffix_is_rcbd_and_rex_is_legacy():
    assert BUNDLE_SUFFIX == ".rcbd"
    assert LEGACY_BUNDLE_SUFFIXES == (".rex",)
    # Canonical first: `existing_doc_path` returns the first hit, and while both exist
    # for one id the newer one has to win.
    assert BUNDLE_SUFFIXES[0] == BUNDLE_SUFFIX
    assert is_bundle_suffix(".rcbd") and is_bundle_suffix(".rex")
    assert is_bundle_suffix(".REX"), "suffix comparison is case-insensitive"
    assert not is_bundle_suffix(".zarr")


def test_a_document_is_written_under_the_canonical_suffix(tmp_path, monkeypatch, rex):
    from agent.server import persistence

    monkeypatch.setattr(persistence, "_docs_dir", lambda ws: tmp_path)
    path = persistence.save_document_rex("ws", "doc", rex)
    assert path.endswith(BUNDLE_SUFFIX), path
    assert (tmp_path / f"doc{BUNDLE_SUFFIX}").exists()
    assert not (tmp_path / "doc.rex").exists(), "nothing should still write the old suffix"


@pytest.mark.parametrize("suffix", BUNDLE_SUFFIXES)
def test_a_document_loads_from_either_suffix(tmp_path, monkeypatch, rex, suffix):
    """The legacy read path. A bundle saved before the rename carries `.rex`, and the
    core identifies it by its manifest rather than its name, so the only thing that can
    lose it is the agent resolving a path with the canonical suffix alone."""
    from agent.server import persistence

    from rexgraph.io import save_rcbd

    monkeypatch.setattr(persistence, "_docs_dir", lambda ws: tmp_path)
    # Build the legacy case the way one actually exists on disk: written by the older
    # version and therefore named `.rex`. Writing it through today's writer and asking
    # for `.rex` would test the current write path, which is not what is under test
    # here, and would couple this to whatever that path does with an explicit suffix.
    save_rcbd(str(tmp_path / f"doc{BUNDLE_SUFFIX}"), rex)
    if suffix != BUNDLE_SUFFIX:
        (tmp_path / f"doc{BUNDLE_SUFFIX}").rename(tmp_path / f"doc{suffix}")
    assert (tmp_path / f"doc{suffix}").exists()

    back = persistence.load_document_rex("ws", "doc")
    assert back is not None, f"a {suffix} bundle was not found"
    assert (int(back.nV), int(back.nE)) == (int(rex.nV), int(rex.nE))
    assert persistence.existing_doc_path("ws", "doc").suffix == suffix


def test_the_canonical_bundle_wins_when_both_are_present(tmp_path, monkeypatch, rex):
    from agent.server import persistence

    from rexgraph.io import save_rcbd

    monkeypatch.setattr(persistence, "_docs_dir", lambda ws: tmp_path)
    save_rcbd(str(tmp_path / f"doc{BUNDLE_SUFFIX}"), rex)
    (tmp_path / "doc.rex").mkdir()
    assert persistence.existing_doc_path("ws", "doc").suffix == BUNDLE_SUFFIX


def test_one_document_under_both_suffixes_is_listed_once(tmp_path, monkeypatch, rex):
    from agent.server import persistence

    from rexgraph.io import save_rcbd

    monkeypatch.setattr(persistence, "_docs_dir", lambda ws: tmp_path)
    save_rcbd(str(tmp_path / f"doc{BUNDLE_SUFFIX}"), rex)
    (tmp_path / "doc.rex").mkdir()
    assert persistence.list_document_bundles("ws") == ["doc"]


def test_the_export_format_key_did_not_move_with_the_file_suffix():
    """`rex` names the export contract in the HTTP surface, not a filename. Renaming it
    would be an unrelated API break, so the key stays put while the extension moves."""
    from agent.server.artifacts import COMPLEX_FORMATS

    assert "rex" in COMPLEX_FORMATS
    assert COMPLEX_FORMATS["rex"] == BUNDLE_SUFFIX
    assert "rcbd" not in COMPLEX_FORMATS
