"""The local biomedical demo is provenance-first and needs no network to construct."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rexgraph.biomedical_demo import build_mtor_demo, write_demo_artifacts


def _portal_row(*, accession: str, name: str, participants: str, disease: str, refs: str) -> str:
    values = [
        accession, name, "-", "9606", participants, "ECO:0000353", "-", "-", refs,
        "-", "-", "-", "-", disease, "-", "-", "-", "psi-mi:MI:0486(UniProt)", participants,
    ]
    return "\t".join(values)


def _fixtures(root: Path) -> None:
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
        "Identifiers (and stoichiometry) of molecules in complex", "Evidence Code", "Experimental evidence",
        "Go Annotations", "Cross references", "Description", "Complex properties", "Complex assembly",
        "Ligand", "Disease", "Agonist", "Antagonist", "Comment", "Source", "Expanded participant list",
    ]
    rows = [
        _portal_row(accession="CPX-503", name="mTORC1 complex", participants="P42345(1)|Q8TAE8(1)",
                    disease="Cancer [MONDO:0004992]|Obesity [EFO:0001073]", refs="reactome:R-HSA-377400(identity)"),
        _portal_row(accession="CPX-4402", name="mTORC2 complex", participants="P42345(1)|O43516(1)",
                    disease="Cancer [MONDO:0004992]", refs="reactome:R-HSA-198626(identity)"),
    ]
    (root / "complexportal" / "human_complexes.tsv").write_text(
        "\t".join(header) + "\n" + "\n".join(rows) + "\n", encoding="utf-8",
    )


def test_mtor_demo_preserves_primary_relations_and_constructs_declared_faces(tmp_path):
    _fixtures(tmp_path)
    case = build_mtor_demo(tmp_path, max_compounds=3)

    assert [row["compound_id"] for row in case["case"]["compounds"]] == ["101", "102", "103"]
    assert case["case"]["source_manifest"]["join_key"] == "UniProt accession P42345"

    primary = case["primary"]
    membership = [r for r in primary["relations"] if r["kind"] == "complex_membership"]
    assert len(membership) == 2
    assert all(len(relation["members"]) == 3 for relation in membership)
    assert primary["rex"].chain_valid
    # The k-ary primary-relation support is connected; that projection diagnostic is
    # deliberately not substituted for the complex's own beta_0 calculation.
    from rexgraph.biomedical_demo import _shape
    assert _shape(primary["rex"])["support_projection_components"] == 1

    panel = case["affinity_panel"]
    assert panel["rex"].chain_valid
    assert panel["rex"].nF == 2
    assert panel["face_rule"]["face_count"] == 2
    assert panel["hodge"]["reconstruction_error"] < 1.0e-8
    assert np.isclose(sum(panel["hodge"][f"{part}_norm"] ** 2
                          for part in ("gradient", "curl", "harmonic")),
                      panel["hodge"]["flow_norm"] ** 2, rtol=1.0e-7)


def test_artifact_writer_receives_upper_layer_capabilities_as_callbacks(tmp_path):
    _fixtures(tmp_path)
    case = build_mtor_demo(tmp_path, max_compounds=3)

    def persist(value, output):
        assert value is case
        assert output.name == "artifacts"
        return {"backend": "test-double", "chain_valid": True}

    def render(rex, labels):
        assert rex is case["affinity_panel"]["rex"]
        assert labels[0] == "mTOR (P42345)"
        return "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>"

    artifacts = write_demo_artifacts(case, tmp_path / "artifacts", persist=persist, render=render)
    assert {"summary.json", "primary_relations.json", "affinity_panel.json", "report.html", "affinity_panel.svg"} == set(artifacts)
    assert '"backend": "test-double"' in artifacts["summary.json"].read_text(encoding="utf-8")
