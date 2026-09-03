"""JUMP Cell Painting profiles become temporal C1 metric fields, not causal edges."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from rexgraph.jump_cell_painting import (
    analyze_jump_delta,
    build_jump_cell_painting_temporal,
    load_jump_plate,
)

_HEADER = [
    "Metadata_gene", "Metadata_pert_type",
    "Cells_Intensity_mean", "Cells_Intensity_max",
    "Nuclei_Texture_mean", "Nuclei_Texture_max",
]


def _write_profile(path: Path, rows: list[dict[str, str]]) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _row(gene: str, cells: tuple[float, float], nuclei: tuple[float, float], *, trt="trt"):
    return {
        "Metadata_gene": gene,
        "Metadata_pert_type": trt,
        "Cells_Intensity_mean": str(cells[0]),
        "Cells_Intensity_max": str(cells[1]),
        "Nuclei_Texture_mean": str(nuclei[0]),
        "Nuclei_Texture_max": str(nuclei[1]),
    }


def test_plate_reader_uses_finite_section_means_and_omits_non_treatment_rows(tmp_path):
    path = tmp_path / "day1.csv.gz"
    _write_profile(path, [
        _row("GENE_A", (1, 3), (2, 4)),
        _row("GENE_A", (3, 5), (4, 6)),
        _row("GENE_B", (2, 2), (10, 10)),
        _row("CONTROL", (100, 100), (100, 100), trt="control"),
    ])

    plate = load_jump_plate(path)
    assert plate.sections == ("Cells_Intensity", "Nuclei_Texture")
    assert plate.wells_per_gene == {"GENE_A": 2, "GENE_B": 1}
    assert plate.gene_sections["GENE_A"] == {"Cells_Intensity": 3.0, "Nuclei_Texture": 4.0}
    assert plate.gene_sections["GENE_B"] == {"Cells_Intensity": 2.0, "Nuclei_Texture": 10.0}


def test_day_comparison_keeps_a_stable_c1_basis_and_reports_field_readings(tmp_path):
    day1 = tmp_path / "day1.csv.gz"
    day4 = tmp_path / "day4.csv.gz"
    _write_profile(day1, [
        _row("GENE_A", (1, 3), (2, 4)),
        _row("GENE_A", (3, 5), (4, 6)),
        _row("GENE_B", (2, 2), (10, 10)),
    ])
    _write_profile(day4, [
        _row("GENE_A", (5, 7), (1, 3)),
        _row("GENE_B", (1, 1), (12, 12)),
    ])

    study = build_jump_cell_painting_temporal(
        day1, day4, day1_time=1.0, day4_time=4.0,
        field_sections=("Cells_Intensity", "Nuclei_Texture"),
    )
    assert study.temporal.T == 2
    np.testing.assert_allclose(study.temporal.times, [1.0, 4.0])
    assert study.genes == ("GENE_A", "GENE_B")
    assert study.sections == ("Cells_Intensity", "Nuclei_Texture")
    assert (study.day1.nV, study.day1.nE, study.day1.nF) == (4, 4, 0)
    assert study.day1.chain_valid and study.day4.chain_valid

    result = analyze_jump_delta(study, top=2)
    assert result["shape"] == {
        "nV": 4, "nE": 4, "nF": 0, "chain_valid": True,
        "genes": 2, "field_sections": 2, "weight_only_c1_events": 4,
    }
    assert result["direct_c1_hodge"]["curl_l2"] == 0.0
    assert result["direct_c1_hodge"]["reconstruction_residual"] < 1.0e-10
    assert np.isclose(result["metric_curvature"]["total"], result["metric_curvature"]["c0_total"])
    assert np.isclose(result["metric_curvature"]["total"], result["metric_curvature"]["c1_total"])
    assert len(result["metric_curvature"]["top_relation_contributions"]) == 2
    assert "not assert gene regulation" in result["assay"]["interpretation_guardrail"]
    assert all(len(item["sha256"]) == 64 for item in result["source_manifest"].values())
