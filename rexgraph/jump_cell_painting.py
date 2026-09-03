"""Temporal relational-tensor readings for JUMP Cell Painting plate profiles.

The adapter keeps a perturbation identifier and a named CellProfiler field section as
the boundary participants of one primary C1 relation.  Its measured profile value is a
separate C1 metric coefficient.  A Day 1 to Day 4 comparison therefore keeps the same
relation basis while exposing an amplitude delta directly on C1, before deriving a C0
source or any Green response.

This is a morphology-assay adapter.  Its deterministic orientation names the
perturbation as the distinguished boundary participant only so the relation tensor has
a stable convention; it does not assert activation, inhibition, or a causal mechanism.
No C2 cells are fabricated from the profile table, so a zero curl sector is an honest
description of this assay model rather than missing biological evidence.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .graph import RexGraph, TemporalRex
from .green import vertex_green
from .metric_field import MetricCurvature, relation_metric_curvature
from .temporal_signal import TemporalSignal, signal_flow, temporal_signal

__all__ = [
    "CellPaintingPlate",
    "DEFAULT_JUMP_SECTIONS",
    "JumpCellPaintingStudy",
    "analyze_jump_delta",
    "build_jump_cell_painting_temporal",
    "load_jump_plate",
]


# The standard analysis view keeps the six morphology families present for every
# compartment.  Neighbors and Cytoplasm_Location remain available to callers through
# ``field_sections`` but are not silently mixed into this matched 18-section field.
DEFAULT_JUMP_SECTIONS = (
    "Cells_AreaShape", "Cells_Correlation", "Cells_Granularity", "Cells_Intensity",
    "Cells_RadialDistribution", "Cells_Texture",
    "Cytoplasm_AreaShape", "Cytoplasm_Correlation", "Cytoplasm_Granularity",
    "Cytoplasm_Intensity", "Cytoplasm_RadialDistribution", "Cytoplasm_Texture",
    "Nuclei_AreaShape", "Nuclei_Correlation", "Nuclei_Granularity", "Nuclei_Intensity",
    "Nuclei_RadialDistribution", "Nuclei_Texture",
)


@dataclass(frozen=True)
class CellPaintingPlate:
    """One plate's per-gene, per-CellProfiler-section profile means."""

    path: Path
    sections: tuple[str, ...]
    gene_sections: Mapping[str, Mapping[str, float]]
    wells_per_gene: Mapping[str, int]


@dataclass(frozen=True)
class JumpCellPaintingStudy:
    """Two time-aligned Cell Painting snapshots over a stable C1 relation basis."""

    temporal: TemporalRex
    day1: RexGraph
    day4: RexGraph
    genes: tuple[str, ...]
    sections: tuple[str, ...]
    vertex_labels: tuple[str, ...]
    relation_labels: tuple[tuple[str, str], ...]
    source_manifest: Mapping[str, Mapping[str, str]]
    orientation: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _section_name(column: str) -> str | None:
    """Map one feature column to its named CellProfiler field section."""
    if column.startswith("Metadata_"):
        return None
    pieces = column.split("_", 2)
    if len(pieces) < 3 or not pieces[0] or not pieces[1]:
        return None
    return "_".join(pieces[:2])


def _finite_float(value: str | None) -> float | None:
    try:
        result = float(value or "")
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def load_jump_plate(path: str | Path, *, perturbation: str = "trt") -> CellPaintingPlate:
    """Read one gzipped JUMP profile CSV without adding a dataframe dependency.

    A well contributes its finite feature values to a named section such as
    ``Cells_Intensity``.  The plate value for a gene/section is the arithmetic mean of
    the available well-section means.  Both levels are explicit so a missing feature or
    well is omitted rather than coerced to zero.
    """
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)

    by_gene: dict[str, dict[str, list[float]]] = {}
    wells: dict[str, int] = {}
    with gzip.open(profile_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{profile_path} has no CSV header")
        required = {"Metadata_gene", "Metadata_pert_type"}
        if not required <= set(reader.fieldnames):
            raise ValueError(f"{profile_path} is missing {sorted(required)}")
        groups = {
            column: section
            for column in reader.fieldnames
            if (section := _section_name(column)) is not None
        }
        if not groups:
            raise ValueError(f"{profile_path} has no CellProfiler feature sections")

        for row in reader:
            if str(row.get("Metadata_pert_type") or "").strip() != perturbation:
                continue
            gene = str(row.get("Metadata_gene") or "").strip()
            if not gene:
                continue
            per_section: dict[str, list[float]] = {}
            for column, section in groups.items():
                value = _finite_float(row.get(column))
                if value is not None:
                    per_section.setdefault(section, []).append(value)
            if not per_section:
                continue
            target = by_gene.setdefault(gene, {})
            for section, values in per_section.items():
                target.setdefault(section, []).append(float(np.mean(values)))
            wells[gene] = wells.get(gene, 0) + 1

    gene_sections = {
        gene: {section: float(np.mean(values)) for section, values in section_values.items()}
        for gene, section_values in by_gene.items()
    }
    sections = tuple(sorted({section for values in gene_sections.values() for section in values}))
    if not sections or not gene_sections:
        raise ValueError(f"{profile_path} has no usable {perturbation!r} profiles")
    return CellPaintingPlate(profile_path, sections, gene_sections, wells)


def _stable_genes(day1: CellPaintingPlate, day4: CellPaintingPlate,
                  sections: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(
        gene for gene in set(day1.gene_sections) & set(day4.gene_sections)
        if all(section in day1.gene_sections[gene] and section in day4.gene_sections[gene]
               for section in sections)
    ))


def _snapshot(plate: CellPaintingPlate, genes: tuple[str, ...],
              sections: tuple[str, ...]) -> RexGraph:
    n_genes = len(genes)
    sources = np.repeat(np.arange(n_genes, dtype=np.int64), len(sections))
    targets = np.tile(np.arange(n_genes, n_genes + len(sections), dtype=np.int64), len(genes))
    weights = np.asarray(
        [plate.gene_sections[gene][section] for gene in genes for section in sections],
        dtype=np.float64,
    )
    # This orientation is an address convention: perturbation -> measured morphology
    # section.  It is explicitly not a biological sign or a directed causal claim.
    return RexGraph(sources=sources, targets=targets, w_E=weights,
                    signs=np.ones(weights.size, dtype=np.int32))


def build_jump_cell_painting_temporal(
    day1_path: str | Path,
    day4_path: str | Path,
    *,
    day1_time: float = 1.0,
    day4_time: float = 4.0,
    perturbation: str = "trt",
    field_sections: tuple[str, ...] | None = None,
) -> JumpCellPaintingStudy:
    """Build the Day 1/Day 4 C1 assay field over a declared section vocabulary.

    The default is the standard matched 18-section morphology field.  Supplying
    ``field_sections`` is an explicit change of the metric space, useful for a focused
    analysis but never hidden in a generic profile import.
    """
    if day4_time < day1_time:
        raise ValueError("day4_time must not precede day1_time")
    day1_plate = load_jump_plate(day1_path, perturbation=perturbation)
    day4_plate = load_jump_plate(day4_path, perturbation=perturbation)
    available = set(day1_plate.sections) & set(day4_plate.sections)
    requested = DEFAULT_JUMP_SECTIONS if field_sections is None else tuple(field_sections)
    sections = tuple(section for section in requested if section in available)
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"requested Cell Painting sections are absent: {missing}")
    genes = _stable_genes(day1_plate, day4_plate, sections)
    if not genes or not sections:
        raise ValueError("the two plates have no shared complete perturbation/section profile")

    day1 = _snapshot(day1_plate, genes, sections)
    day4 = _snapshot(day4_plate, genes, sections)
    temporal = TemporalRex([])
    temporal.append_snapshot(day1, at=day1_time)
    temporal.append_snapshot(day4, at=day4_time)
    labels = tuple([f"gene:{gene}" for gene in genes] + [f"field:{section}" for section in sections])
    relation_labels = tuple((gene, section) for gene in genes for section in sections)
    manifest = {
        "day1": {"path": str(day1_plate.path), "sha256": _sha256(day1_plate.path)},
        "day4": {"path": str(day4_plate.path), "sha256": _sha256(day4_plate.path)},
    }
    return JumpCellPaintingStudy(
        temporal=temporal, day1=day1, day4=day4, genes=genes, sections=sections,
        vertex_labels=labels, relation_labels=relation_labels, source_manifest=manifest,
        orientation=(
            "canonical assay convention: perturbation identifier is the C1 head and "
            "CellProfiler field section is its boundary share; not activation/inhibition"
        ),
    )


def _top(values: np.ndarray, labels: tuple[str, ...], *, count: int) -> list[dict[str, float | str]]:
    order = sorted(range(values.size), key=lambda index: (-float(values[index]), labels[index]))
    return [
        {"label": labels[index], "value": float(values[index])}
        for index in order[:count]
    ]


def analyze_jump_delta(study: JumpCellPaintingStudy, *, top: int = 10) -> dict:
    """Read direct C1 delta, derived Green response, and C1-metric curvature.

    The returned C0 source is a derived boundary reading.  The Hodge split starts from
    the direct C1 amplitude delta, where non-potential sectors can exist; the Green
    action starts from that derived C0 source.  The two are intentionally reported as
    different mathematical readings rather than competing descriptions of one path.
    """
    if top < 1:
        raise ValueError("top must be positive")
    signal: TemporalSignal = temporal_signal(study.temporal, 1)
    current = signal.current
    direct = signal.relation_field("amplitude")
    source = signal.source_field("amplitude")
    gradient, curl, harmonic = current.hodge(np.asarray(direct.values, dtype=np.float64))
    reconstructed = np.asarray(gradient) + np.asarray(curl) + np.asarray(harmonic)
    residual = float(np.linalg.norm(reconstructed - direct.values))
    flow = signal_flow(signal, "amplitude")
    green = vertex_green(current)
    potential = green.solve(np.asarray(source.values, dtype=np.float64))
    green_energy = float(np.dot(np.asarray(source.values, dtype=np.float64), potential))
    curvature: MetricCurvature = relation_metric_curvature(current, direct)

    n_genes = len(study.genes)
    c0_curvature = np.asarray(curvature.curvature.values, dtype=np.float64)
    c1_contribution = np.asarray(curvature.relation_contribution.values, dtype=np.float64)
    gene_labels = study.vertex_labels[:n_genes]
    section_labels = study.vertex_labels[n_genes:]
    relation_labels = tuple(f"gene:{gene} -> field:{section}" for gene, section in study.relation_labels)
    direct_norm = float(np.linalg.norm(direct.values))
    return {
        "source_manifest": study.source_manifest,
        "assay": {
            "kind": "Cell Painting morphology profile",
            "orientation": study.orientation,
            "interpretation_guardrail": (
                "The observed C1 amplitudes are morphology profile changes. This model does not "
                "assert gene regulation, activation, inhibition, causal propagation, or C2 mechanisms."
            ),
        },
        "shape": {
            "nV": int(current.nV), "nE": int(current.nE), "nF": int(current.nF),
            "chain_valid": bool(current.chain_valid),
            "genes": n_genes, "field_sections": len(study.sections),
            "weight_only_c1_events": len(signal.events),
        },
        "direct_c1_hodge": {
            "l2": direct_norm,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "curl_l2": float(np.linalg.norm(curl)),
            "harmonic_l2": float(np.linalg.norm(harmonic)),
            "reconstruction_residual": residual,
        },
        "green": {
            "source_l2": float(np.linalg.norm(source.values)),
            "potential_l2": float(np.linalg.norm(potential)),
            "energy": green_energy,
            "local_response_l2": float(np.linalg.norm(flow.relation_response.values)),
        },
        "metric_curvature": {
            "total": float(curvature.total),
            "c0_total": float(c0_curvature.sum()),
            "c1_total": float(c1_contribution.sum()),
            "top_gene_centers": _top(c0_curvature[:n_genes], gene_labels, count=top),
            "top_field_centers": _top(c0_curvature[n_genes:], section_labels, count=top),
            "top_relation_contributions": _top(c1_contribution, relation_labels, count=top),
        },
    }
