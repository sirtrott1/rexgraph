"""The wire contract, and the flush bug it uncovered.

A frame is checked before it is trusted: the chain condition is exact over the
integers, so a payload whose boundary data was altered fails it. What the structure
does NOT give is identity (a well-formed complex can be built by anyone), so nothing
here is treated as authentication.
"""
from __future__ import annotations

import numpy as np
import pytest

import rexgraph.protocol as P
from rexgraph.faces import find_cycles
from rexgraph.graph import RexGraph


def _filled():
    """A complex with faces attached and NOTHING read since, so B2 is still pending."""
    rex = RexGraph(sources=np.asarray([0, 1, 2, 2, 3, 4], np.int32),
                   targets=np.asarray([1, 2, 0, 3, 4, 2], np.int32))
    rex.add_faces(find_cycles(rex, 3), signs=None)
    return rex


#### the flush: faces attached and immediately serialised


def test_a_complex_saved_straight_after_add_faces_keeps_them():
    """`add_faces` QUEUES faces; `_ensure_clean` writes them into the boundary arrays.
    Reading `_B2_col_ptr` without flushing serialises an empty B2 under a header that
    declares nF, so every face is lost, in every container, silently. Nothing here
    touches the complex before saving it, which is what makes the test bite."""
    import json

    from rexgraph.io.rex_state import CODEC_TENSOR, decode_tensors, to_state
    state = to_state(_filled())
    assert state.header["nF"] == 2
    # a CSR pointer is stored as its first difference, so read it back through the codec
    # rather than around it: the array the format REPRESENTS is what this is about
    t = dict(state.tensors)
    if CODEC_TENSOR in t:
        decode_tensors(t, json.loads(
            bytes(np.asarray(t.pop(CODEC_TENSOR)).tobytes()).decode("utf-8")))
    assert np.asarray(t["B2_col_ptr"]).tolist() == [0, 3, 6], \
        "B2 was serialised empty while the header declared two faces"


def test_the_cold_save_round_trips_through_every_container(tmp_path):
    from rexgraph.io import load_rex, save_rex
    from rexgraph.io.safetensors_bridge import load_safetensors, rex_to_safetensors

    path = str(tmp_path / "cold.rex")
    save_rex(path, _filled())
    back = load_rex(path)
    assert int(back.nF) == 2
    assert tuple(back.betti) == (1, 0, 0), "the faces did not survive the save"

    st = str(tmp_path / "cold.safetensors")
    rex_to_safetensors(_filled(), st)
    assert int(load_safetensors(st)["object"].nF) == 2


#### the frame


def test_a_complex_round_trips_through_a_frame():
    back = P.to_complex(P.decode(P.encode(_filled())))
    assert (int(back.nV), int(back.nE), int(back.nF)) == (5, 6, 2)
    assert tuple(back.betti) == (1, 0, 0)


def test_meta_travels_with_the_frame():
    frame = P.decode(P.encode(_filled(), meta={"workspace": "lab", "op": "join"}))
    assert frame.header["meta"] == {"workspace": "lab", "op": "join"}


def test_a_frame_is_self_describing():
    frame = P.decode(P.encode(_filled()))
    assert frame.object_type == "RexGraph"
    assert frame.n_bytes > 0


def test_compression_is_transparent():
    a = P.encode(_filled(), compress=True)
    b = P.encode(_filled(), compress=False)
    assert P.to_complex(P.decode(a)).nE == P.to_complex(P.decode(b)).nE


#### refusals, which are the point


@pytest.mark.parametrize("name,mangle", [
    ("not a frame", lambda f: b"hello world, definitely not a frame"),
    ("bad magic", lambda f: b"XXXX" + f[4:]),
    ("truncated", lambda f: f[:len(f) // 2]),
    ("length lie", lambda f: f[:12] + b"\xff\xff\xff\xff" + f[16:]),
    ("empty", lambda f: b""),
])
def test_a_malformed_frame_is_refused(name, mangle):
    frame = P.encode(_filled())
    with pytest.raises(P.ProtocolError):
        P.decode(mangle(frame))


def test_an_oversized_frame_is_refused_before_it_is_read():
    frame = P.encode(_filled())
    with pytest.raises(P.ProtocolError, match="over the"):
        P.decode(frame, max_frame=16)


def test_a_frame_declaring_too_many_cells_is_refused():
    frame = P.encode(_filled())
    with pytest.raises(P.ProtocolError, match="cell limit|over the"):
        P.decode(frame, max_cells=2)


def test_a_wrong_wire_version_says_so():
    frame = bytearray(P.encode(_filled()))
    frame[4:6] = (P.WIRE_VERSION + 1).to_bytes(2, "little")
    with pytest.raises(P.ProtocolError, match="wire version"):
        P.decode(bytes(frame))


def test_a_tensor_pointing_outside_the_payload_is_refused():
    import json
    import struct
    rex = _filled()
    frame = P.encode(rex, compress=False)
    head_len, body_len = struct.unpack("<II", frame[8:16])
    header = json.loads(frame[16:16 + head_len])
    header["tensors"][0]["offset"] = 10 ** 9
    new_head = json.dumps(header).encode()
    rebuilt = (P.MAGIC
               + struct.pack("<HHII", P.WIRE_VERSION, 0, len(new_head), body_len)
               + new_head + frame[16 + head_len:])
    with pytest.raises(P.ProtocolError, match="outside the payload"):
        P.decode(rebuilt)


#### the chain condition is the STRUCTURAL check


def test_a_tampered_boundary_fails_verification():
    """One flipped coefficient. `B_d B_d+1 = 0` is exact over the rationals, so this is
    not a tolerance question, and the refusal names the face rather than the file."""
    rex = _filled()
    frame = P.decode(P.encode(rex))
    vals = frame.tensors["B2_vals"]
    vals[0] = vals[0] + 1.0
    frame.header.pop("digest", None)             # isolate the structural check
    with pytest.raises(P.ProtocolError, match="do not bound"):
        P.to_complex(frame)


def test_the_report_names_which_face_failed(_=None):
    """A boolean and "face 0 does not bound" are different amounts of help, and the
    caller deciding whether to store this wants the second."""
    rex = _filled()
    frame = P.decode(P.encode(rex))
    frame.tensors["B2_vals"][0] += 1.0
    frame.header.pop("digest", None)
    from rexgraph.io.rex_state import RexState, from_state
    bad = from_state(RexState(tensors=frame.tensors, header=frame.header))
    report = P.chain_report(bad)
    assert report["valid"] is False
    assert report["unbounded"] == [0]
    assert report["n_faces"] == bad._nF


def test_an_intact_frame_verifies():
    ok, residual = P.verify_complex(P.to_complex(P.decode(P.encode(_filled()))))
    assert ok and residual == 0.0


def test_verification_can_be_skipped_only_deliberately():
    rex = _filled()
    frame = P.decode(P.encode(rex))
    frame.tensors["B2_vals"][0] += 1.0
    assert P.to_complex(frame, verify=False) is not None, \
        "verify=False is for a caller that already checked this exact frame"


#### addressing


def test_the_fingerprint_survives_relabeling():
    """Betti and the cell counts are invariants, so the same complex under different
    vertex numbering addresses the same."""
    src = np.asarray([0, 1, 2, 2, 3, 4], np.int32)
    tgt = np.asarray([1, 2, 0, 3, 4, 2], np.int32)
    perm = np.asarray([2, 0, 1, 4, 3, 5])
    a = RexGraph(sources=src, targets=tgt)
    b = RexGraph(sources=perm[src].astype(np.int32),
                 targets=perm[tgt].astype(np.int32))
    fa, fb = P.fingerprint(a), P.fingerprint(b)
    assert fa["betti"] == fb["betti"]
    assert (fa["nV"], fa["nE"]) == (fb["nV"], fb["nE"])


def test_the_fingerprint_reports_whether_the_chain_holds():
    f = P.fingerprint(_filled())
    assert f["chain_valid"] is True and f["chain_residual"] == 0.0
