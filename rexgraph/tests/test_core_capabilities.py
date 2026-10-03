"""Every compiled extension has a declared production/optional/oracle owner."""
from pathlib import Path

import numpy as np

import rexgraph.core as core
from rexgraph.identity import canonical_json, digest_parts, manifest_digest, tensor_digest
from rexgraph.io.manifest import (
    canonical_json as io_canonical_json,
    digest_parts as io_digest_parts,
    manifest_digest as io_manifest_digest,
)
from rexgraph.io.rex_state import state_digest


def test_native_capability_inventory_has_no_unowned_or_dead_color_kernel():
    status = core.core_status()
    groups = (status["required"], status["optional"], status["oracle"])
    declared = [name for group in groups for name in group["modules"]]
    assert len(declared) == len(set(declared))
    assert "_color" not in declared
    assert "_harmonic" in status["oracle"]["modules"]
    assert "_recordlog" in status["optional"]["modules"]
    assert status["required"]["missing"] == ()
    core.require_core()


def test_dead_color_target_is_removed_and_temporal_twin_is_reference_only():
    root = Path(__file__).resolve().parents[1]
    core_build = (root / "core" / "meson.build").read_text(encoding="utf-8")
    assert "'_color'" not in core_build
    assert not (root / "core" / "_color.pyx").exists()
    assert not (root / "core" / "_temporal_entity_py.py").exists()
    assert (root / "reference" / "temporal_entity.py").is_file()
    assert (root / "reference" / "harmonic.py").is_file()


def test_retired_harmonic_runtime_path_is_only_a_compatibility_surface():
    import rexgraph.harmonic as legacy
    import rexgraph.reference.harmonic as reference

    assert legacy.get_harmonic_basis is reference.get_harmonic_basis
    assert legacy.harmonic_product_structure is reference.harmonic_product_structure


def test_identity_primitives_live_below_io_with_compatibility_exports():
    payload = {"z": [3, 2, 1], "a": "value"}
    assert canonical_json(payload) == io_canonical_json(payload)
    assert manifest_digest(payload) == io_manifest_digest(payload)
    parts = [("a", "00" * 32), ("b", "11" * 32)]
    assert digest_parts("test", parts) == io_digest_parts("test", parts)

    tensors = {
        "x": np.arange(8, dtype=np.int64),
        "y": np.asarray([[1.5, 2.5]], dtype=np.float64),
    }
    assert tensor_digest(tensors) == state_digest(tensors)
    assert tensor_digest(tensors, algo=1) == state_digest(tensors, algo=1)
