"""One asymmetric exact/present fixture across real containers and stores."""
from fractions import Fraction as Q
import json
import struct
import zlib

import numpy as np
import pytest

from rexgraph import Absent, Approx, Relations, RexGraph, VertexTable
from rexgraph.native_rank import primary_columns
from rexgraph.state import from_state, to_state
from rexgraph import protocol


def fixture():
    supports = [[0, 1, 2, 3], [0, 1], [0, 2], [0, 3], [2], [3, 3]]
    source = Relations.from_supports(
        supports, vertices=VertexTable(("a", "b", "c", "d", "isolate"), ("A", "B", "C", "D", "Isolate"), (("alias:a",), (), (), (), ())),
        shares=[0, Q(1,4), Q(1,2), Q(1,4)]+[Absent]*9,
        weights=[Q(2,3), Absent, 3, 1, Absent, 1], signs=[-1, 1, 1, 1, 1, -1],
        relation_ids=["group", "pair/a", "pair/b", "pair/c", "witness", "loop"],
        relation_types=["group", "pair", "pair", "pair", "witness", "loop"],
        embedding=[[Q(i,10), 0] for i in range(5)], provenance={"source": "memory:conformance"},
        attributes={1: {0: {"absent": Absent, "integer": 2**16000, "rational": Q(1,10), "approximate": Approx(.1,"sensor")}}},
    )
    r = RexGraph.from_relations(source, c_channel="count")
    r.add_faces([[0, 1, 2, 3]])
    return r


def equal_source(back, original):
    assert (back.nV, back.nE, back.nF) == (5, 6, 1)
    assert primary_columns(back) == primary_columns(original)
    assert back._exact_column_norms_B1()[0] == Q(11,8)
    assert back.relations.weight.values().tolist() == [Q(2,3), Absent, 3, 1, Absent, 1]
    assert back.relations.sign.tolist() == [-1, 1, 1, 1, 1, -1]
    assert back.relation_keys == original.relation_keys
    assert back.relations.vertices.ids == original.relations.vertices.ids
    assert back.relations.vertices.aliases == original.relations.vertices.aliases
    assert back.embedding == original.embedding
    assert back.get_metadata(1,0,"integer") == 2**16000
    assert back.get_metadata(1,0,"absent") is Absent
    assert back.get_metadata(1,0,"approximate") == Approx(.1,"sensor")
    assert back.chain_valid
    assert to_state(back).header["digest"] == to_state(original).header["digest"]


@pytest.mark.parametrize("container", ["rcbd", "safetensors", "hdf5", "zarr", "arrow", "wire", "state"])
def test_all_containers_reconstruct_the_same_declared_state(container, tmp_path):
    original = fixture()
    if container in {"state", "wire"}:
        back = from_state(to_state(original)) if container == "state" else protocol.to_complex(protocol.decode(protocol.encode(original)))
    elif container == "arrow":
        pytest.importorskip("pyarrow")
        from rexgraph.io.arrow_bridge import rex_to_arrow, arrow_to_rex
        back = arrow_to_rex(rex_to_arrow(original))
    else:
        dependency = {"safetensors":"safetensors", "hdf5":"h5py", "zarr":"zarr"}.get(container)
        if dependency:
            pytest.importorskip(dependency)
        from rexgraph.io import save, load
        suffix = {"hdf5":"h5"}.get(container, container)
        path = tmp_path / ("source."+suffix)
        save(str(path), original)
        back = load(str(path))
    equal_source(back, original)


@pytest.mark.parametrize("container", ["rcbd", "safetensors", "hdf5", "zarr", "arrow", "wire", "state"])
def test_containers_preserve_an_explicit_empty_face_basis(container, tmp_path):
    from rexgraph import Selection, restrict
    from rexgraph.partition_state import partition_tower
    original = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]],
        [[(0, 1), (1, 1), (2, -1)]]])
    part = restrict(original, Selection(original, {1: [1]}))
    if container == "state":
        back = from_state(to_state(part.rex))
    elif container == "wire":
        back = protocol.to_complex(protocol.decode(protocol.encode(part.rex)))
    elif container == "arrow":
        pytest.importorskip("pyarrow")
        from rexgraph.io.arrow_bridge import rex_to_arrow, arrow_to_rex
        back = arrow_to_rex(rex_to_arrow(part.rex))
    else:
        dependency = {"safetensors": "safetensors", "hdf5": "h5py", "zarr": "zarr"}.get(container)
        if dependency:
            pytest.importorskip(dependency)
        from rexgraph.io import save, load
        path = tmp_path / ("empty-face."+{"hdf5": "h5"}.get(container, container))
        save(str(path), part.rex)
        back = load(str(path))
    assert [boundary.shape for boundary in partition_tower(back)[0]] == [(2, 1), (1, 0)]
    assert back.betti_tower == (1, 0, 0)
    assert part.lineage.verify(original, back)
    with pytest.raises(ValueError, match="legacy state"):
        to_state(back, _native=False)


@pytest.mark.parametrize("backend", ["memory", "file", "sql", "rex"])
def test_store_roundtrip_preserves_declared_state_after_reopen(backend, tmp_path):
    rcdb = pytest.importorskip("rcdb")
    if not hasattr(rcdb, "open_store"):
        pytest.skip("RCDB distribution is not installed")
    uri = {"memory":"memory://", "file":"file://"+str(tmp_path / "files"),
           "sql":"sqlite:///"+str(tmp_path / "source.db"), "rex":str(tmp_path / "source.rexdb")}[backend]
    original = fixture()
    store = rcdb.open_store(uri, **({"read_only": False} if "://" not in uri or uri.startswith(("file://", "rex://")) else {}))
    try:
        saved = store.commit_mutation("fixture", original, expected_version=0)
        if backend != "memory":
            store.close(); store = rcdb.open_store(uri, **({"read_only": False} if "://" not in uri or uri.startswith(("file://", "rex://")) else {}))
        equal_source(store.read_record("fixture", version=saved.version).value, original)
        assert store.verify_commits("fixture")
    finally:
        store.close()


def test_safetensors_extras_have_explicit_ownership_and_integrity(tmp_path):
    safetensors = pytest.importorskip("safetensors.numpy")
    from safetensors import safe_open
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex, load_safetensors
    p = rex_to_safetensors(fixture(), tmp_path / "aux.safetensors", cache="topology", extra_tensors={"custom/field":np.array([1.,2.])})
    loaded = load_safetensors(p)
    assert loaded["tensors"]["custom/field"].tolist() == [1.,2.]
    equal_source(loaded["object"], fixture())
    with safe_open(str(p), framework="numpy") as opened:
        meta = opened.metadata()
    raw = safetensors.load_file(str(p)); raw["custom/field"][0] = 99
    safetensors.save_file(raw, str(p), metadata=meta)
    with pytest.raises(ValueError, match="auxiliary payload digest"):
        safetensors_to_rex(p)


@pytest.mark.parametrize("container", ["rcbd", "hdf5", "zarr", "safetensors"])
def test_unclaimed_physical_state_entries_are_refused(container, tmp_path):
    from rexgraph.io import save, load
    suffix = {"hdf5":"h5"}.get(container, container)
    path = tmp_path / ("source."+suffix)
    if container != "rcbd":
        pytest.importorskip({"hdf5":"h5py", "safetensors":"safetensors", "zarr":"zarr"}[container])
    save(str(path), fixture())
    if container == "rcbd":
        np.save(path / "unclaimed.npy", np.zeros(1))
    elif container == "hdf5":
        import h5py
        with h5py.File(path,"a") as f:
            f.create_dataset("unclaimed",data=np.zeros(1))
    elif container == "zarr":
        import zarr
        g = zarr.open_group(str(path),mode="a")
        g.create_array("unclaimed",data=np.zeros(1))
    else:
        from safetensors import safe_open
        import safetensors.numpy as st
        with safe_open(str(path),framework="numpy") as opened:
            meta = opened.metadata()
        arrays = st.load_file(str(path)); arrays["unclaimed"] = np.zeros(1)
        st.save_file(arrays,str(path),metadata=meta)
    with pytest.raises(ValueError, match="unclaimed|mismatched"):
        load(str(path))


@pytest.mark.parametrize("change", ["duplicate", "negative", "overflow", "overlap", "flags", "zipbomb"])
def test_wire_framing_bounds_and_tensor_inventory(change):
    original = protocol.encode(fixture(), compress=False)
    version, flags, hsize, size = struct.unpack("<HHII", original[4:16])
    head = json.loads(original[16:16+hsize]); body = original[16+hsize:]
    if change == "duplicate":
        head["tensors"].append(head["tensors"][0])
    elif change == "negative":
        head["tensors"][0]["shape"] = [-1]
    elif change == "overflow":
        head["tensors"][0]["shape"] = [2**63, 2**63]
    elif change == "overlap":
        head["tensors"][1]["offset"] = 0
    elif change == "flags":
        flags = 2
    else:
        body, flags = zlib.compress(bytes(1_000_000)), 1
    raw_head = json.dumps(head).encode()
    malformed = protocol.MAGIC+struct.pack("<HHII",version,flags,len(raw_head),len(body))+raw_head+body
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(malformed, max_frame=100_000)
