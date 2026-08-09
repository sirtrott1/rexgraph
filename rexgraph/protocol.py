"""
rexgraph.protocol: the wire contract, anchored in the chain condition.

A complex crosses a boundary as the same layered binary it is stored as, and two
different questions get asked on the far side: are these the bytes that were sent, and
do they describe a complex. Neither answers the other, and the second one does less
than it looks like it does.

    transmission  a sha256 over the payload, recorded in the header. Catches a flipped
                  bit, a truncation, a wrong-length read. This is what actually makes
                  a frame trustworthy against corruption.
    structure     ``B_d B_{d+1} = 0``, exact over the integers. Catches a payload that
                  arrived intact and still is not a complex: faces whose boundary does
                  not close. It constrains the relationship BETWEEN grades, so on a
                  complex with no faces there is no product to take and it constrains
                  NOTHING. A 1-skeleton is checked by the digest alone. Treating the
                  chain condition as an integrity check is a mistake it will not
                  announce, because it returns valid.
    addressing    the structural signature. Betti and the cell counts are exact and
                  survive relabeling, so two complexes that differ in homology are
                  provably different objects.
    restriction   a subcomplex is itself a complex, with boundary operators that do
                  not reference cells outside it. An operation handed one cannot
                  reach past it, which is a property of the object rather than a
                  check the caller must remember.

    NOT identity  a digest detects corruption, not forgery. Anyone who can rewrite the
                  payload can recompute the digest over it, so an active attacker in
                  the path is not covered by framing alone. `sign`/`verify_signature`
                  add an HMAC over the frame bytes for deployments with a shared key;
                  otherwise identity is TLS's job.
    NOT secrecy   framing is not encryption. Confidentiality is the transport's job.

Frames are self-describing and bounded. Every length is read before any allocation is
made against it, so a frame that claims more than the caller allows is refused rather
than reserved for.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import struct
import zlib
from dataclasses import dataclass

import numpy as np

#: identifies a rexgraph frame and its wire version
MAGIC = b"REXW"
WIRE_VERSION = 1

#: the media type a frame travels under
CONTENT_TYPE = "application/x-rex"

#: refuse a frame larger than this before reading it (bytes). A caller on a trusted
#: socket can raise it; a public listener should lower it.
DEFAULT_MAX_FRAME = 256 * 1024 * 1024

#: refuse a header larger than this. The header is meant to be kilobytes; a large one
#: is either a mistake or an attempt to make the decoder allocate.
MAX_HEADER_BYTES = 1 << 20

#: refuse a complex larger than this many cells at any grade, before it is built
DEFAULT_MAX_CELLS = 5_000_000

_DTYPES = {
    "f8": np.float64, "f4": np.float32,
    "i8": np.int64, "i4": np.int32, "i2": np.int16, "i1": np.int8,
    "u8": np.uint64, "u4": np.uint32, "u2": np.uint16, "u1": np.uint8,
    "b1": np.bool_,
}
_CODES = {np.dtype(v).str[1:]: k for k, v in _DTYPES.items()}


class ProtocolError(ValueError):
    """A frame that cannot be trusted: malformed, oversized, or not a valid complex.

    One exception type for every rejection, because the caller's response is the same
    in each case and distinguishing them in a reply tells a prober which of its
    guesses was closer.
    """


@dataclass
class Frame:
    """A decoded frame: what was sent, and what it says about itself."""

    header: dict
    tensors: dict
    n_bytes: int = 0

    @property
    def object_type(self) -> str:
        return str(self.header.get("object_type", ""))


def encode(rex, *, meta: dict | None = None, compress: bool = True) -> bytes:
    """A complex as a frame.

    The payload is the canonical layered state, so what crosses the wire and what the
    store holds are the same bytes rather than two encodings that can drift.
    """
    from rexgraph.io.rex_state import to_state

    state = to_state(rex)
    header = dict(state.header)
    if meta:
        header["meta"] = meta

    names = sorted(state.tensors)
    index = []
    body = io.BytesIO()
    for name in names:
        arr = np.ascontiguousarray(state.tensors[name])
        code = _CODES.get(arr.dtype.str[1:])
        if code is None:
            raise ProtocolError(f"tensor {name!r} has unsupported dtype {arr.dtype}")
        offset = body.tell()
        body.write(arr.tobytes())
        index.append({"name": name, "dtype": code, "shape": list(arr.shape),
                      "offset": offset, "nbytes": arr.nbytes})
    header["tensors"] = index

    payload = body.getvalue()
    # `to_state` already recorded the content digest in the header, over the tensors
    # themselves rather than over this packing, so the wire and every container agree
    # on one value. `wire_digest` covers the packed bytes as well, which is what catches
    # a truncation that lands between tensors.
    header["wire_digest"] = hashlib.sha256(payload).hexdigest()

    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    flags = 0
    if compress:
        squeezed = zlib.compress(payload, 6)
        if len(squeezed) < len(payload):
            payload, flags = squeezed, 1

    return b"".join([
        MAGIC,
        struct.pack("<HHII", WIRE_VERSION, flags, len(head), len(payload)),
        head, payload,
    ])


def decode(data: bytes, *, max_frame: int = DEFAULT_MAX_FRAME,
           max_cells: int = DEFAULT_MAX_CELLS) -> Frame:
    """Read a frame, refusing anything it cannot account for.

    Every length is checked against the limits and against the bytes actually present
    before it is used to allocate, so a frame claiming a gigabyte of tensor is refused
    rather than reserved for.
    """
    if len(data) > max_frame:
        raise ProtocolError(f"frame is {len(data)} bytes, over the {max_frame} limit")
    if len(data) < 16 or not data.startswith(MAGIC):
        raise ProtocolError("not a rexgraph frame")

    version, flags, head_len, body_len = struct.unpack("<HHII", data[4:16])
    if version != WIRE_VERSION:
        raise ProtocolError(
            f"wire version {version} is not {WIRE_VERSION}; the sender and this "
            "reader do not share a format")
    if head_len > MAX_HEADER_BYTES:
        raise ProtocolError(f"header claims {head_len} bytes, over the limit")
    if 16 + head_len + body_len != len(data):
        raise ProtocolError(
            "the frame's declared lengths do not match the bytes present")

    try:
        header = json.loads(data[16:16 + head_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError("header is not readable") from e
    if not isinstance(header, dict):
        raise ProtocolError("header is not an object")

    for grade in ("nV", "nE", "nF"):
        n = header.get(grade)
        if n is not None and int(n) > max_cells:
            raise ProtocolError(
                f"{grade}={n} is over the {max_cells}-cell limit for one frame")

    payload = data[16 + head_len:]
    if flags & 1:
        try:
            payload = zlib.decompress(payload)
        except zlib.error as e:
            raise ProtocolError("payload does not decompress") from e

    declared = header.get("wire_digest")
    if declared is not None:
        actual = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(str(declared), actual):
            raise ProtocolError(
                "the payload does not match the digest in its header; it was "
                "altered or truncated in transit")

    index = header.get("tensors")
    if not isinstance(index, list):
        raise ProtocolError("header declares no tensor index")
    tensors = {}
    for entry in index:
        try:
            name = str(entry["name"])
            dtype = _DTYPES[entry["dtype"]]
            shape = tuple(int(x) for x in entry["shape"])
            offset, nbytes = int(entry["offset"]), int(entry["nbytes"])
        except (KeyError, TypeError, ValueError) as e:
            raise ProtocolError("a tensor entry is malformed") from e
        if offset < 0 or nbytes < 0 or offset + nbytes > len(payload):
            raise ProtocolError(f"tensor {name!r} points outside the payload")
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize if shape else \
            np.dtype(dtype).itemsize
        if nbytes != expected:
            raise ProtocolError(
                f"tensor {name!r} declares {nbytes} bytes for shape {shape}")
        tensors[name] = np.frombuffer(
            payload[offset:offset + nbytes], dtype=dtype).reshape(shape).copy()

    header.pop("tensors", None)
    frame = Frame(header=header, tensors=tensors, n_bytes=len(data))

    # the content digest `to_state` recorded, checked on the unpacked tensors. This is
    # the same value a `.rex`, hdf5, zarr or safetensors reader checks, so a payload
    # that survived the wire and a payload that survived a disk are held to one rule.
    from rexgraph.io.rex_state import RexState, verify_state
    if not verify_state(RexState(tensors=tensors, header=header)):
        raise ProtocolError(
            "the tensors do not match the content digest recorded with them")
    return frame


def to_complex(frame: Frame, *, verify: bool = True):
    """Rebuild the complex a frame carries, checking the chain condition first.

    `verify=False` exists only for a caller that has already verified this exact frame
    and is rebuilding it a second time. On a boundary, leave it on: the check is a
    sparse matmul over stored nonzeros and it is what makes a frame trustworthy at
    all.
    """
    from rexgraph.io.rex_state import RexState, from_state

    try:
        rex = from_state(RexState(tensors=frame.tensors, header=frame.header),
                         verify=verify)
    except Exception as e:                       # noqa: BLE001 - any failure is refusal
        raise ProtocolError(f"frame does not rebuild a complex: {e}") from e
    if verify:
        report = chain_report(rex)
        if not report["valid"]:
            raise ProtocolError(
                f"{report['n_unbounded']} of {report['n_faces']} faces do not bound "
                f"(indices {report['unbounded'][:8]}); the boundary data was altered "
                "or was never valid")
    return rex


def chain_report(rex) -> dict:
    """Which cells of the frame AS SENT fail to bound, exactly.

    `rex._chain_col_bounds` is the complex's own predicate, adjudicated over the
    rationals: a face bounds or it does not, with no tolerance deciding the difference.
    It runs on every declared face, not the Hodge-filtered slice, because the filter
    exists to keep a non-bounding face out of the homology and a receiver that checks
    the filtered view reports success on a payload it has already discarded part of.

    Returns the offending face indices rather than a boolean, because "invalid" and
    "face 7 does not bound" are different amounts of help, and the caller deciding
    whether to store this needs the second one.
    """
    n_faces = int(getattr(rex, "_nF", 0) or 0)
    out = {"n_faces": n_faces, "n_unbounded": 0, "unbounded": [], "valid": True}
    if n_faces == 0:
        # A complex with no faces has no product to take: the chain condition relates
        # two grades and there is only one here, so it holds vacuously and says nothing
        # about the payload. The frame's digest is what covers that case.
        return out
    try:
        bounds = np.asarray(rex._chain_col_bounds, dtype=bool)
    except Exception:                            # noqa: BLE001 - unreadable is invalid
        return {"n_faces": n_faces, "n_unbounded": n_faces,
                "unbounded": list(range(n_faces)), "valid": False}
    bad = [int(f) for f in np.nonzero(~bounds)[0]]
    out["n_unbounded"] = len(bad)
    out["unbounded"] = bad[:64]
    out["valid"] = not bad
    return out


def verify_complex(rex) -> tuple[bool, float]:
    """Whether the frame AS SENT is a complex, and by how much it misses.

    The boolean comes from the exact predicate (`chain_report`); the magnitude is the
    curvature of the miss, `max_f ||B_1 B_2[:, f]||`, which is what
    `rcf_torch.chain_residual` calls the deviation from the `d d = 0` ideal. Kept as a
    pair because the caller wants both: whether to refuse, and how far off it was.
    """
    report = chain_report(rex)
    if report["valid"]:
        return True, 0.0
    try:
        colmax = np.asarray(rex._chain_col_maxabs, dtype=np.float64)
        bad = report["unbounded"]
        worst = float(np.max(colmax[bad])) if bad and colmax.size else 0.0
    except Exception:                            # noqa: BLE001
        worst = float("inf")
    return False, worst


def sign(frame_bytes: bytes, key: bytes) -> str:
    """An HMAC over a whole frame, for a deployment that shares a key.

    The digest in the header covers accidents: bits that flipped, a body that arrived
    short. It cannot cover an attacker in the path, who recomputes it over whatever
    they substituted. This can, because producing it needs the key.

    Detached rather than folded into the frame, so signing does not change the bytes
    being signed and an unsigned reader can still parse a signed frame. Travels beside
    it, conventionally in the `X-Rex-Signature` header.
    """
    return hmac.new(key, frame_bytes, hashlib.sha256).hexdigest()


def verify_signature(frame_bytes: bytes, signature: str, key: bytes) -> bool:
    """Whether a signature matches, compared in constant time.

    A comparison that returns as soon as two bytes differ tells anyone who can time it
    how much of a guess was right, which is enough to find the rest a byte at a time.
    """
    if not signature:
        return False
    return hmac.compare_digest(sign(frame_bytes, key), str(signature))


def fingerprint(rex) -> dict:
    """What this complex is, by invariants rather than by bytes.

    Betti and the cell counts survive relabeling, so two frames carrying the same
    complex under different vertex numbering agree here while two genuinely different
    complexes do not. Useful for addressing and for noticing that a stored object is
    not the one that was sent; NOT a signature, since anyone can build a complex with
    a chosen fingerprint.
    """
    out = {"nV": int(rex.nV), "nE": int(rex.nE), "nF": int(rex.nF_hodge),
           "nF_declared": int(getattr(rex, "_nF", 0) or 0)}
    try:
        out["betti"] = [int(b) for b in rex.betti]
    except Exception:                            # noqa: BLE001
        out["betti"] = None
    ok, residual = verify_complex(rex)
    out["chain_valid"] = bool(ok)
    out["chain_residual"] = float(residual)
    # a face the Hodge filter drops is a face that arrived and will not be used, so
    # the gap between what was declared and what survives is worth reporting rather
    # than absorbing
    out["faces_dropped"] = out["nF_declared"] - out["nF"]
    return out
