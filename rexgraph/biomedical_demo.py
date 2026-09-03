"""Reproducible BindingDB + Complex Portal demonstration.

The demo deliberately keeps provenance separate from derived mathematics.
BindingDB supplies measured compound--protein affinities.  Complex Portal supplies
protein-complex membership plus its own disease and Reactome annotations.  Their
shared UniProt accession is the join key; a displayed disease edge therefore means
"Complex Portal annotates this complex with this condition", never that a compound
has been clinically shown to treat it.

``build_mtor_demo`` makes two related relational complexes:

* a native, primary-relation complex, preserving a protein complex as one k-ary
  relation rather than expanding it into an unlabelled clique;
* an affinity-panel 2-complex, whose faces are explicitly derived by an adaptive
  affinity-band rule.  It is the object used for the Hodge reading.

The module has no RCDB or Agent dependency. An upper-layer caller injects optional
persistence and rendering callbacks into :func:`write_demo_artifacts`; that preserves
the distribution direction while keeping the core data model reusable.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .graph import RexGraph

DEFAULT_TARGET = "P42345"
DEFAULT_TARGET_LABEL = "mTOR (P42345)"
DEFAULT_COMPLEXES = ("CPX-503", "CPX-4402")

# BindingDB's local export gives compound identifiers and structures, not display
# names.  These two public CID labels make the opening slide readable while retaining
# the source identifier and an explicit PubChem provenance link; every other compound
# remains an identifier rather than being guessed from its SMILES.
_PUBCHEM_DISPLAY_NAMES = {
    "49836027": ("Torin 1", "https://pubchem.ncbi.nlm.nih.gov/compound/Torin_1"),
    "44516953": ("Gedatolisib", "https://pubchem.ncbi.nlm.nih.gov/compound/Gedatolisib"),
}

_ACCESSION = re.compile(
    r"(?<![A-Z0-9])((?:[OPQ][0-9][A-Z0-9]{3}[0-9])|"
    r"(?:[A-NR-Z][0-9][A-Z0-9]{3}[0-9])|(?:A0A[0-9A-Z]{7}))(?![A-Z0-9])"
)
_CONCEPT = re.compile(r"\[((?:MONDO|EFO|MIM|Orphanet):[^\]]+)\]")
_REACTOME = re.compile(r"reactome:(R-HSA-\d+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_compound_id(value: str) -> str:
    """Keep an integer-looking source identifier readable without asserting its scheme."""
    value = str(value).strip()
    return value[:-2] if re.fullmatch(r"\d+\.0", value) else value


def _compound_label(record: dict[str, Any]) -> str:
    name = record.get("display_name")
    if name:
        return f"{name} (PubChem CID {record['compound_id']})"
    return f"BindingDB compound {record['compound_id']}"


def _binding_records(path: Path, target: str) -> list[dict[str, Any]]:
    """One best positive Kd record per BindingDB compound identifier for a target."""
    chosen: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        needed = {"ID1", "X1", "ID2", "Y"}
        if not reader.fieldnames or not needed <= set(reader.fieldnames):
            raise ValueError(f"{path} must contain tab-separated {sorted(needed)} columns")
        for row in reader:
            if str(row.get("ID2") or "").strip() != target:
                continue
            try:
                kd_nM = float(str(row.get("Y") or ""))
            except ValueError:
                continue
            if not math.isfinite(kd_nM) or kd_nM <= 0.0:
                continue
            compound_id = _normalise_compound_id(row["ID1"])
            record = {
                "compound_id": compound_id,
                "smiles": str(row.get("X1") or "").strip(),
                "target_accession": target,
                "kd_nM": kd_nM,
                # This is the convention used by the existing BindingDB scripts in
                # this checkout.  It is recorded below so the transformation is not
                # mistaken for a source measurement.
                "pKd": -math.log10(kd_nM * 1.0e-9),
            }
            if compound_id in _PUBCHEM_DISPLAY_NAMES:
                record["display_name"], record["display_name_source"] = _PUBCHEM_DISPLAY_NAMES[compound_id]
            previous = chosen.get(compound_id)
            if previous is None or kd_nM < float(previous["kd_nM"]):
                chosen[compound_id] = record
    return sorted(chosen.values(), key=lambda item: (item["kd_nM"], item["compound_id"]))


def _complex_records(path: Path, complex_ids: tuple[str, ...]) -> list[dict[str, str]]:
    """Read the requested Complex Portal records, retaining their named columns."""
    wanted = set(complex_ids)
    found: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "#Complex ac", "Recommended name", "Identifiers (and stoichiometry) of molecules in complex",
            "Cross references", "Disease",
        }
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"{path} does not have the expected Complex Portal export columns")
        for row in reader:
            accession = str(row.get("#Complex ac") or "").strip()
            if accession in wanted:
                found[accession] = {key: str(value or "").strip() for key, value in row.items()}
    missing = [accession for accession in complex_ids if accession not in found]
    if missing:
        raise ValueError(f"Complex Portal export does not contain {', '.join(missing)}")
    return [found[accession] for accession in complex_ids]


def _components(row: dict[str, str]) -> list[str]:
    values = _ACCESSION.findall(row["Identifiers (and stoichiometry) of molecules in complex"])
    # Stable de-duplication preserves Complex Portal's published participant order.
    return list(dict.fromkeys(values))


def _diseases(row: dict[str, str]) -> list[dict[str, str]]:
    values = [value.strip() for value in row.get("Disease", "").split("|") if value.strip() and value.strip() != "-"]
    result = []
    for ordinal, value in enumerate(values, start=1):
        match = _CONCEPT.search(value)
        concept = match.group(1) if match else f"{row['#Complex ac']}:disease:{ordinal}"
        name = value.split("[")[0].strip().rstrip(":") or concept
        result.append({"id": concept, "label": name, "source_text": value})
    return result


def _pathways(row: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(_REACTOME.findall(row.get("Cross references", ""))))


def _relation(relation_id: str, kind: str, members: list[str], **details: Any) -> dict[str, Any]:
    if len(members) < 2:
        raise ValueError(f"relation {relation_id!r} has fewer than two boundary vertices")
    return {"id": relation_id, "kind": kind, "members": members, **details}


def _make_rex(entities: list[dict[str, Any]], relations: list[dict[str, Any]]) -> tuple[RexGraph, list[str]]:
    index = {entity["id"]: position for position, entity in enumerate(entities)}
    if len(index) != len(entities):
        raise ValueError("biomedical entities must have unique identifiers")
    pointer = [0]
    values: list[int] = []
    for relation in relations:
        support = list(dict.fromkeys(relation["members"]))
        try:
            values.extend(index[member] for member in support)
        except KeyError as exc:
            raise ValueError(f"relation {relation['id']!r} references an unknown entity {exc.args[0]!r}") from exc
        pointer.append(len(values))
    rex = RexGraph.from_hypergraph(np.asarray(pointer, dtype=np.int64), np.asarray(values, dtype=np.int64))
    return rex, [str(entity["label"]) for entity in entities]


def _affinity_panel(records: list[dict[str, Any]], target_label: str) -> dict[str, Any]:
    """A transparent 2-complex derived from adjacent values inside an affinity band."""
    if len(records) < 3:
        raise ValueError("an affinity-panel Hodge demo needs at least three compounds")
    labels = [target_label] + [_compound_label(record) for record in records]
    edge_index: dict[tuple[int, int], int] = {}
    sources: list[int] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []

    def add_edge(a: int, b: int, detail: dict[str, Any]) -> int:
        key = (min(a, b), max(a, b))
        existing = edge_index.get(key)
        if existing is not None:
            return existing
        edge_index[key] = len(sources)
        sources.append(key[0])
        targets.append(key[1])
        metadata.append(detail)
        return edge_index[key]

    flow = []
    for position, record in enumerate(records, start=1):
        add_edge(0, position, {
            "kind": "binding_assay",
            "compound_id": record["compound_id"],
            "kd_nM": record["kd_nM"],
            "pKd": record["pKd"],
        })
        flow.append(float(record["pKd"]))

    # Records arrive strongest-first.  The adaptive fence is recorded in the artifact;
    # it makes a face an explicit analysis relation rather than an unstated clique rule.
    pkd = np.asarray([record["pKd"] for record in records], dtype=float)
    gaps = np.abs(np.diff(pkd))
    fence = float(np.median(gaps) + np.median(np.abs(gaps - np.median(gaps))) + 1.0e-12)
    triangles: list[tuple[int, int, int]] = []
    for offset, gap in enumerate(gaps, start=1):
        if float(gap) <= fence:
            left, right = offset, offset + 1
            add_edge(left, right, {
                "kind": "affinity_band_coparticipation",
                "rule": "adjacent pKd values within median_gap + MAD(gap)",
                "pKd_gap": float(gap),
            })
            triangles.append((0, left, right))

    # Binding relations were added first; added coparticipation relations carry no
    # measured assay flow.  This assignment follows the relation metadata instead of
    # relying on that construction order.
    edge_flow = np.asarray([float(item.get("pKd", 0.0)) for item in metadata], dtype=float)
    triangle_array = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    rex = RexGraph.from_simplicial(
        np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64), triangle_array,
    )
    gradient, curl, harmonic = rex.hodge(edge_flow)
    reconstructed = np.asarray(gradient) + np.asarray(curl) + np.asarray(harmonic)
    if not np.allclose(reconstructed, edge_flow, atol=1.0e-8, rtol=1.0e-8):
        raise RuntimeError("Hodge decomposition did not reconstruct the affinity signal")
    return {
        "rex": rex,
        "vertex_labels": labels,
        "relations": metadata,
        "face_rule": {
            "description": "Adjacent pKd-ranked compounds are joined when their gap is at most median(gaps) + MAD(gaps).",
            "gap_fence": fence,
            "face_count": len(triangles),
        },
        "flow": edge_flow,
        "hodge": {
            "flow_norm": float(np.linalg.norm(edge_flow)),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "curl_norm": float(np.linalg.norm(curl)),
            "harmonic_norm": float(np.linalg.norm(harmonic)),
            "reconstruction_error": float(np.linalg.norm(reconstructed - edge_flow)),
        },
    }


def build_mtor_demo(data_root: str | Path, *, max_compounds: int = 12) -> dict[str, Any]:
    """Build the local mTOR BindingDB + Complex Portal case study.

    ``data_root`` must contain ``bindingdb_kd.raw`` and
    ``complexportal/human_complexes.tsv``.  The first source provides the measured
    Kd values; the second supplies mTORC1/mTORC2 membership and its annotations.
    """
    if max_compounds < 3:
        raise ValueError("max_compounds must be at least 3")
    root = Path(data_root).expanduser().resolve()
    binding_path = root / "bindingdb_kd.raw"
    portal_path = root / "complexportal" / "human_complexes.tsv"
    if not binding_path.is_file() or not portal_path.is_file():
        raise FileNotFoundError("data_root must contain bindingdb_kd.raw and complexportal/human_complexes.tsv")

    records = _binding_records(binding_path, DEFAULT_TARGET)[:max_compounds]
    if len(records) < 3:
        raise ValueError(f"BindingDB contains fewer than three usable records for {DEFAULT_TARGET}")
    complexes = _complex_records(portal_path, DEFAULT_COMPLEXES)

    entities: list[dict[str, Any]] = [{
        "id": f"protein:{DEFAULT_TARGET}", "label": DEFAULT_TARGET_LABEL,
        "type": "protein_target", "accession": DEFAULT_TARGET,
    }]
    entities.extend({
        "id": f"compound:{record['compound_id']}",
        "label": _compound_label(record),
        "type": "compound", "smiles": record["smiles"],
    } for record in records)
    entity_ids = {entity["id"] for entity in entities}

    def add_entity(entity: dict[str, Any]) -> None:
        if entity["id"] not in entity_ids:
            entity_ids.add(entity["id"])
            entities.append(entity)

    relations: list[dict[str, Any]] = []
    for record in records:
        relations.append(_relation(
            f"binding:{record['compound_id']}:{DEFAULT_TARGET}", "binding_assay",
            [f"compound:{record['compound_id']}", f"protein:{DEFAULT_TARGET}"],
            source="BindingDB", kd_nM=record["kd_nM"], pKd=record["pKd"],
            measurement_transform="pKd = -log10(Kd_nM * 1e-9)",
        ))

    context: list[dict[str, Any]] = []
    for row in complexes:
        complex_id = row["#Complex ac"]
        complex_label = f"{row['Recommended name']} [{complex_id}]"
        add_entity({"id": f"complex:{complex_id}", "label": complex_label, "type": "protein_complex"})
        components = _components(row)
        for accession in components:
            add_entity({
                "id": f"protein:{accession}",
                "label": DEFAULT_TARGET_LABEL if accession == DEFAULT_TARGET else accession,
                "type": "protein_component", "accession": accession,
            })
        relations.append(_relation(
            f"complex-membership:{complex_id}", "complex_membership",
            [f"complex:{complex_id}"] + [f"protein:{accession}" for accession in components],
            source="Complex Portal", complex_accession=complex_id,
            evidence=row.get("Evidence Code", ""),
            participant_text=row["Identifiers (and stoichiometry) of molecules in complex"],
        ))
        disease_rows = _diseases(row)
        pathway_ids = _pathways(row)
        for disease in disease_rows:
            add_entity({"id": f"disease:{disease['id']}", "label": disease["label"], "type": "disease"})
            relations.append(_relation(
                f"disease-annotation:{complex_id}:{disease['id']}", "complex_disease_annotation",
                [f"complex:{complex_id}", f"disease:{disease['id']}"],
                source="Complex Portal", source_text=disease["source_text"],
            ))
        for pathway_id in pathway_ids:
            add_entity({"id": f"pathway:{pathway_id}", "label": pathway_id, "type": "reactome_pathway"})
            relations.append(_relation(
                f"pathway-reference:{complex_id}:{pathway_id}", "complex_reactome_reference",
                [f"complex:{complex_id}", f"pathway:{pathway_id}"], source="Complex Portal",
            ))
        context.append({
            "complex_accession": complex_id,
            "complex_name": row["Recommended name"],
            "components": components,
            "diseases": disease_rows,
            "reactome_pathways": pathway_ids,
        })

    primary_rex, primary_labels = _make_rex(entities, relations)
    panel = _affinity_panel(records, DEFAULT_TARGET_LABEL)
    source_manifest = {
        "bindingdb": {"path": str(binding_path), "sha256": _sha256(binding_path)},
        "complex_portal": {"path": str(portal_path), "sha256": _sha256(portal_path)},
        "join_key": f"UniProt accession {DEFAULT_TARGET}",
        "compound_display_name_enrichment": {
            "source": "PubChem CID pages; used only for the two named display labels",
            "records": [
                {"compound_id": record["compound_id"], "name": record["display_name"],
                 "url": record["display_name_source"]}
                for record in records if record.get("display_name")
            ],
        },
    }
    return {
        "case": {
            "title": "mTOR BindingDB affinity context",
            "target_accession": DEFAULT_TARGET,
            "target_label": DEFAULT_TARGET_LABEL,
            "compounds": records,
            "complex_context": context,
            "source_manifest": source_manifest,
            "interpretation_guardrail": (
                "BindingDB records measured compound--protein affinity. Complex Portal disease annotations "
                "belong to the listed protein complex; the composed path is contextual provenance, not a "
                "claim of clinical efficacy or causality."
            ),
        },
        "primary": {
            "rex": primary_rex,
            "vertex_labels": primary_labels,
            "entities": entities,
            "relations": relations,
        },
        "affinity_panel": panel,
    }


def _shape(rex: RexGraph) -> dict[str, Any]:
    # This is intentionally a *support projection* diagnostic, kept apart from beta_0.
    # One k-ary primary relation is one B1 column, whereas its ordinary support drawing
    # would fan it into k-1 or more pairwise links.  Those two readings need not agree.
    parent = list(range(int(rex.nV)))

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for support in rex.relation_supports():
        if len(support) < 2:
            continue
        anchor = int(support[0])
        for value in support[1:]:
            left, right = find(anchor), find(int(value))
            if left != right:
                parent[left] = right
    return {
        "nV": int(rex.nV), "nE": int(rex.nE), "nF": int(rex.nF),
        "betti": [int(value) for value in rex.betti],
        "chain_valid": bool(rex.chain_valid),
        "euler_characteristic": int(rex.euler_characteristic),
        "support_projection_components": len({find(vertex) for vertex in range(int(rex.nV))}),
    }


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing demo artifact: {path}")
    path.write_text(content, encoding="utf-8")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=lambda item: item.item() if hasattr(item, "item") else str(item))


def write_demo_artifacts(
    case: dict[str, Any], output_dir: str | Path, *,
    persist: Callable[[dict[str, Any], Path], dict[str, Any]] | None = None,
    render: Callable[[RexGraph, list[str]], str] | None = None,
) -> dict[str, Path]:
    """Write core provenance artifacts with optional, injected higher-layer outputs.

    The function never clears or replaces an artifact.  Use a fresh output directory
    for each demonstrated source snapshot, keeping an auditable relation between a
    report and the exact source checksums that made it. ``persist`` and ``render`` are
    supplied by an upper layer: this core module neither knows nor imports RCDB or the
    Agent application.
    """
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("summary.json", "primary_relations.json", "affinity_panel.json", "report.html", "affinity_panel.svg")
    exists = [name for name in names if (output / name).exists()]
    if exists:
        raise FileExistsError(f"refusing to overwrite existing demo artifact(s): {', '.join(exists)}")
    primary = case["primary"]
    panel = case["affinity_panel"]
    persistence = persist(case, output) if persist is not None else {
        "status": "not persisted; supply an upper-layer persistence callback",
    }
    svg = render(panel["rex"], panel["vertex_labels"]) if render is not None else None
    if svg is not None:
        _write_new(output / "affinity_panel.svg", svg)

    summary = {
        "case": case["case"],
        "primary_relational_complex": _shape(primary["rex"]),
        "affinity_panel_2complex": {
            **_shape(panel["rex"]), "face_rule": panel["face_rule"], "hodge": panel["hodge"],
        },
        "rcdb": persistence,
        "renderer": "Agent SVG renderer" if svg is not None else "not available; JSON report only",
    }
    _write_new(output / "summary.json", _json(summary) + "\n")
    _write_new(output / "primary_relations.json", _json({
        "entities": primary["entities"], "relations": primary["relations"],
    }) + "\n")
    _write_new(output / "affinity_panel.json", _json({
        "vertex_labels": panel["vertex_labels"], "relations": panel["relations"],
        "face_rule": panel["face_rule"], "hodge": panel["hodge"],
    }) + "\n")

    source_rows = "".join(
        f"<li><code>{html.escape(name)}</code>: <code>{html.escape(item['sha256'])}</code></li>"
        for name, item in case["case"]["source_manifest"].items()
        if isinstance(item, dict) and "sha256" in item
    )
    compound_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('display_name') or 'BindingDB compound'))}<br><code>{html.escape(str(item['compound_id']))}</code></td>"
        f"<td>{float(item['kd_nM']):.4g}</td><td>{float(item['pKd']):.4f}</td>"
        f"<td><code>{html.escape(str(item['smiles']))}</code></td></tr>"
        for item in case["case"]["compounds"]
    )
    image = ("<img src=\"affinity_panel.svg\" alt=\"mTOR affinity-panel relational complex\">"
             if svg is not None else "<p>The Agent renderer was not supplied; inspect the JSON artifacts.</p>")
    report = f"""<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><title>{html.escape(case['case']['title'])}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px;color:#18212b}} code{{word-break:break-word}} table{{border-collapse:collapse;width:100%;font-size:.86rem}}td,th{{border:1px solid #cbd5df;padding:.4rem;text-align:left;vertical-align:top}}th{{background:#edf3f8}}img{{width:100%;border:1px solid #cbd5df}}.note{{background:#fff6dc;padding:1rem;border-left:4px solid #cf8d00}}</style>
<h1>{html.escape(case['case']['title'])}</h1>
<p>Local, provenance-preserving demonstration of measured BindingDB compound--protein affinity joined on <strong>{html.escape(DEFAULT_TARGET)}</strong> to Complex Portal mTOR complex annotations.</p>
<p class=\"note\">{html.escape(case['case']['interpretation_guardrail'])}</p>
<h2>Primary relational complex</h2><pre>{html.escape(_json(_shape(primary['rex'])))}</pre>
<p>Complex assembly is retained as a k-ary primary relation. It is not silently expanded into pairwise protein edges. Accordingly, <code>support_projection_components</code> is only a drawing diagnostic; it is kept distinct from the relational-complex Betti reading.</p>
<h2>Affinity-panel 2-complex and Hodge reading</h2>{image}
<pre>{html.escape(_json({'shape': _shape(panel['rex']), 'face_rule': panel['face_rule'], 'hodge': panel['hodge']}))}</pre>
<p>Faces are derived only by the reported adaptive affinity-band rule; they are not asserted as source biomedical triples.</p>
<h2>Measured BindingDB rows</h2><table><tr><th>compound display label / identifier</th><th>Kd (nM)</th><th>pKd</th><th>SMILES</th></tr>{compound_rows}</table>
<h2>RCDB</h2><pre>{html.escape(_json(persistence))}</pre>
<h2>Source checksums</h2><ul>{source_rows}</ul>
</html>"""
    _write_new(output / "report.html", report)
    return {name: output / name for name in names if (output / name).exists()}
