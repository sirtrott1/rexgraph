"""Agent-layer runner for the core mTOR biomedical demonstration.

``rexgraph.biomedical_demo`` builds the relational complexes and writes the core
provenance artifacts. It deliberately imports neither RCDB nor the agent, because
rexgraph sits below both and a core module that reached upward would stop being
installable on its own. It exposes the two places an upper layer is needed as
injected callbacks instead:

    persist(case, output_dir) -> dict      what was stored, for the summary
    render(rex, vertex_labels) -> str      an SVG for the affinity panel

This module supplies both. It owns the RCDB import, the governed two-version
commit, the agent renderer, and the command line. Nothing here belongs in core.

The store keeps two records. The primary complex is written once, since it is the
measured source structure and does not change. The affinity panel is written and
then committed a second time through ``commit_mutation`` so the demonstration
carries a real governed transition rather than a single opaque write, and so
``verify_commits`` has a chain with something in it to verify.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rexgraph.biomedical_demo import build_mtor_demo, write_demo_artifacts
from rexgraph.graph import RexGraph

PRIMARY_ID = "mtor-primary"
PANEL_ID = "mtor-affinity-panel"


def persist(case: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Store both complexes in an RCDB store beside the report, and verify the chain.

    Returns the description that core folds into ``summary.json``. The store URI is
    returned rather than a handle: the summary is provenance, so it records where the
    records went and what the store said about them, not a live object.
    """
    from rcdb import open_store

    root = Path(output_dir) / "rcdb"
    store = open_store(f"rex://{root}")
    try:
        primary = case["primary"]
        panel = case["affinity_panel"]
        manifest = case["case"]["source_manifest"]

        store.put(
            PRIMARY_ID,
            primary["rex"],
            meta={
                "role": "primary relational complex",
                "target": case["case"]["target_accession"],
                # the manifest mixes per-file entries with plain descriptive strings, so
                # only the entries that actually carry a checksum are recorded as sources
                "sources": {
                    name: item["sha256"]
                    for name, item in manifest.items()
                    if isinstance(item, dict) and "sha256" in item
                },
            },
            tags=["biomedical", "bindingdb", "complexportal"],
        )

        # Written, then committed again. The second write is what makes this a governed
        # transition with a lineage rather than a single store call, which is the part
        # worth demonstrating: the panel is derived, so its provenance is the interesting
        # one. Same complex both times, because the demonstration is about the transition
        # being recorded and verifiable, not about the panel changing.
        store.put(
            PANEL_ID,
            panel["rex"],
            meta={"role": "affinity band 2-complex", "face_rule": panel["face_rule"]},
            tags=["biomedical", "derived"],
        )
        committed = store.commit_mutation(
            PANEL_ID,
            panel["rex"],
            meta={
                "role": "affinity band 2-complex",
                "face_rule": panel["face_rule"],
                "hodge": panel["hodge"],
            },
            tags=["biomedical", "derived"],
            actor="biomedical-demo-runner",
        )

        chain = [int(entry.version) for entry in store.history(PANEL_ID)]
        return {
            "store_uri": f"rex://{root}",
            "records": {
                PRIMARY_ID: {"versions": [int(e.version) for e in store.history(PRIMARY_ID)]},
                PANEL_ID: {"versions": chain, "committed_version": int(committed.version)},
            },
            "panel_commit_chain_verified": bool(store.verify_commits(PANEL_ID)),
            "stats": store.stats(),
        }
    finally:
        store.close()


def render(rex: RexGraph, vertex_labels: list[str]) -> str:
    """Render the affinity panel through the agent's own view and SVG writer."""
    from agent.graph_view import render_payload
    from agent.render_svg import render_svg

    payload = render_payload(rex, labels=vertex_labels)
    return render_svg(payload)


def run(data_root: str | Path, output_dir: str | Path, *, max_compounds: int = 12) -> dict[str, Path]:
    """Build the case in core, then write it with this layer's persistence and renderer."""
    case = build_mtor_demo(data_root, max_compounds=max_compounds)
    return write_demo_artifacts(case, output_dir, persist=persist, render=render)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the mTOR biomedical demonstration with RCDB persistence and agent rendering.",
    )
    parser.add_argument("--data-root", required=True,
                        help="directory holding bindingdb_kd.raw and complexportal/human_complexes.tsv")
    parser.add_argument("--output", required=True,
                        help="output directory; must not already hold these artifacts")
    parser.add_argument("--max-compounds", type=int, default=12,
                        help="how many compounds to carry into the panel (default 12)")
    args = parser.parse_args(argv)

    written = run(args.data_root, args.output, max_compounds=args.max_compounds)
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
