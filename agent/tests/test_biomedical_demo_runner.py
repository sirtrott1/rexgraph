"""The agent-layer runner supplies what core deliberately does not import.

``rexgraph.biomedical_demo`` builds the complexes and refuses to reach upward for storage
or rendering, so the demonstration is only complete when a layer that is allowed to depend
on both provides them. This checks that seam end to end: the store really receives both
complexes, the panel really carries a verifiable governed transition, and the SVG really
comes from the agent renderer rather than from a stub.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("rcdb")
pytest.importorskip("rexgraph.biomedical_demo")


def _runner():
    """Load the example by path, anchored to this file rather than the caller's directory.

    examples/ sits beside the agent package rather than inside it, so it is a script
    directory and not importable as agent.examples. That is the existing convention for
    the other examples here, and it keeps demonstration code out of the shipped wheel.
    """
    path = Path(__file__).resolve().parents[1] / "examples" / "biomedical_demo_runner.py"
    spec = importlib.util.spec_from_file_location("biomedical_demo_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _runner()
PANEL_ID, PRIMARY_ID, run = runner.PANEL_ID, runner.PRIMARY_ID, runner.run


def _portal_row(*, accession: str, name: str, participants: str, disease: str, refs: str) -> str:
    values = [
        accession, name, "-", "9606", participants, "ECO:0000353", "-", "-", refs,
        "-", "-", "-", "-", disease, "-", "-", "-", "psi-mi:MI:0486(UniProt)", participants,
    ]
    return "\t".join(values)


def _fixtures(root: Path) -> None:
    """The same shape as the core fixture, so both layers exercise one case."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "complexportal").mkdir()
    (root / "bindingdb_kd.raw").write_text(
        "ID1\tX1\tID2\tX2\tY\n"
        "101\tC\tP42345\tsequence\t1.0\n"
        "102\tCC\tP42345\tsequence\t2.0\n"
        "103\tCCC\tP42345\tsequence\t4.0\n"
        "999\tN\tP00000\tother\t1.0\n",
        encoding="utf-8",
    )
    header = [
        "#Complex ac", "Recommended name", "Aliases for complex", "Taxonomy identifier",
        "Identifiers (and stoichiometry) of molecules in complex", "Evidence Code",
        "Experimental evidence", "Go Annotations", "Cross references", "Description",
        "Complex properties", "Complex assembly", "Ligand", "Disease", "Agonist",
        "Antagonist", "Comment", "Source", "Expanded participant list",
    ]
    rows = [
        _portal_row(accession="CPX-503", name="mTORC1 complex", participants="P42345(1)|Q8TAE8(1)",
                    disease="Cancer [MONDO:0004992]", refs="reactome:R-HSA-377400(identity)"),
        _portal_row(accession="CPX-4402", name="mTORC2 complex", participants="P42345(1)|O43516(1)",
                    disease="Cancer [MONDO:0004992]", refs="reactome:R-HSA-198626(identity)"),
    ]
    (root / "complexportal" / "human_complexes.tsv").write_text(
        "\t".join(header) + "\n" + "\n".join(rows) + "\n", encoding="utf-8",
    )


def test_the_runner_stores_both_complexes_and_a_verifiable_panel_transition(tmp_path):
    _fixtures(tmp_path / "data")
    written = run(tmp_path / "data", tmp_path / "out", max_compounds=3)

    summary = json.loads(Path(written["summary.json"]).read_text())
    persistence = summary["rcdb"]

    # two records: the measured primary structure, and the derived panel
    assert set(persistence["records"]) == {PRIMARY_ID, PANEL_ID}
    assert persistence["records"][PRIMARY_ID]["versions"] == [1]

    # the panel is written and then committed, so its lineage has a transition to verify
    assert persistence["records"][PANEL_ID]["versions"] == [1, 2]
    assert persistence["records"][PANEL_ID]["committed_version"] == 2
    assert persistence["panel_commit_chain_verified"] is True
    assert persistence["stats"]["n_records"] == 2
    assert persistence["stats"]["n_versions"] == 3


def test_the_svg_comes_from_the_agent_renderer(tmp_path):
    _fixtures(tmp_path / "data")
    written = run(tmp_path / "data", tmp_path / "out", max_compounds=3)

    svg = Path(written["affinity_panel.svg"]).read_text()
    assert svg.lstrip().startswith("<svg")
    assert "</svg>" in svg


def test_core_alone_writes_no_store_and_no_svg(tmp_path):
    """Without this layer the demonstration is still valid, and says so rather than faking it.

    This is the property that makes the seam worth having: core runs standalone and reports
    that nothing was persisted, instead of importing RCDB to avoid an awkward gap.
    """
    from rexgraph.biomedical_demo import build_mtor_demo, write_demo_artifacts

    _fixtures(tmp_path / "data")
    case = build_mtor_demo(tmp_path / "data", max_compounds=3)
    written = write_demo_artifacts(case, tmp_path / "out")

    summary = json.loads(Path(written["summary.json"]).read_text())
    assert "not persisted" in summary["rcdb"]["status"]
    assert not (Path(tmp_path / "out") / "affinity_panel.svg").exists()
    assert not (Path(tmp_path / "out") / "rcdb").exists()
