# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._recordlog: the append only record log's frame codec.

One frame is an op, an id, the fixed scalar row, a packed string table, the residual
leaf descriptors, and optionally a backend's own int64 row. The layout is written by
`agent.rcdb_index.log_append`; this reads it.

The scan and the string table are here; assembling the record stays in Python, where
the shape of a record is defined.

An incomplete final frame stops the scan. Malformed declarations and invalid UTF-8
raise instead of being replaced with empty values. The compatibility adapter can ask
for exact complete frame boundaries to distinguish explicit recovery from replay.
"""

from __future__ import annotations

import numpy as np

cimport cython
cimport numpy as np
from cpython.unicode cimport PyUnicode_DecodeUTF8
from libc.stdint cimport int8_t, int32_t, int64_t
from libc.string cimport memcpy

np.import_array()


cdef inline int8_t _i8(const unsigned char *p) noexcept nogil:
    cdef int8_t v
    memcpy(&v, p, 1)
    return v


cdef inline int32_t _i32(const unsigned char *p) noexcept nogil:
    cdef int32_t v
    memcpy(&v, p, 4)
    return v


cdef inline object _split_terms(list strings, list rest):
    """The leading `(kind, [term])` section of a frame's string table.

    The table starts with a count, then per kind a code, a length and that many terms.
    Everything after it belongs to the residual leaves and is handed back in `rest`.
    """
    cdef Py_ssize_t n = len(strings)
    cdef Py_ssize_t i = 0, nk, cnt, k, j
    cdef list terms = []
    cdef set seen = set()
    if n == 0:
        raise ValueError("missing legacy journal term declaration")
    nk = int(strings[0])
    if nk < 0 or nk > 6 or str(nk) != strings[0]:
        raise ValueError("invalid legacy journal term count")
    i = 1
    for k in range(nk):
        if i + 1 >= n:
            raise ValueError("incomplete legacy journal term declaration")
        code = int(strings[i])
        cnt = int(strings[i + 1])
        if (code < 0 or code >= 6 or code in seen or str(code) != strings[i]
                or cnt <= 0 or str(cnt) != strings[i + 1]):
            raise ValueError("invalid legacy journal term declaration")
        seen.add(code)
        i += 2
        if cnt > n - i:
            raise ValueError("legacy journal terms exceed the string table")
        terms.append((code, strings[i:i + cnt]))
        i += cnt
    for j in range(i, n):
        rest.append(strings[j])
    return terms


cdef inline object _result(list out, Py_ssize_t valid_end, bint return_offsets):
    return (out, valid_end) if return_offsets else out


def read_frames(const unsigned char[::1] buf, Py_ssize_t start, int nscal,
                bint return_offsets=False):
    """Scan the log from `start`, one tuple per whole frame.

    Returns a list of `(op, rid, scal, terms, rest, leaves, extra)`, where `op` is 1 for
    a put and 2 for a delete, `scal` is a float64 array of width `nscal` or None,
    `terms` is `[(kind code, [term])]`, `rest` is the strings the residual leaves index,
    and `extra` is an int64 array or None. With `return_offsets`, returns
    `(entries, valid_end)`, with each entry `(end, frame)`; only short reads leave
    `valid_end` below the input length. Negative counts, invalid discriminators,
    noncanonical term declarations and invalid text always raise.
    """
    cdef Py_ssize_t n = buf.shape[0]
    cdef Py_ssize_t o = start
    cdef const unsigned char *base = &buf[0] if n else NULL
    cdef int8_t op, has, scope, kind
    cdef int32_t ln, no, bl, nl, ns, nvals, ne
    cdef Py_ssize_t i, j, lo, hi, valid_end = start
    cdef list out = []
    cdef list strings
    cdef list leaves
    cdef object scal, extra, isidx, terms, rest, code
    cdef np.int64_t[::1] soffs_v
    cdef const unsigned char *sblob

    if nscal <= 0 or nscal > 4096:
        raise ValueError("invalid legacy journal scalar width")
    if start < 0 or start > n:
        raise ValueError("invalid legacy journal start offset")
    if base == NULL:
        return _result(out, valid_end, return_offsets)

    while o < n:
        # op, id length, id
        if o + 1 > n:
            break
        op = _i8(base + o); o += 1
        if op != 1 and op != 2:
            raise ValueError("unknown legacy journal operation")
        if o + 4 > n:
            break
        ln = _i32(base + o); o += 4
        if ln <= 0 or ln > 67108864:
            raise ValueError("invalid legacy journal record identity length")
        if o + ln + 1 > n:
            break
        rid = PyUnicode_DecodeUTF8(<const char *>(base + o), ln, "strict")
        o += ln
        has = _i8(base + o); o += 1
        if has < 0 or has > 2 or (op == 1 and has == 0) or (op == 2 and has != 0):
            raise ValueError("invalid legacy journal operation presence flag")

        scal = None
        extra = None
        strings = []
        leaves = []

        if has:
            # the fixed scalar row
            if o + 8 * nscal + 4 > n:
                break
            scal = np.empty(nscal, dtype=np.float64)
            memcpy(np.PyArray_DATA(scal), base + o, 8 * nscal)
            o += 8 * nscal

            # string offsets, then the blob they index
            no = _i32(base + o); o += 4
            if no < 2 or no > 8388608:
                raise ValueError("invalid legacy journal string offset count")
            if o + 8 * <Py_ssize_t>no + 4 > n:
                break
            soffs = np.empty(no, dtype=np.int64)
            memcpy(np.PyArray_DATA(soffs), base + o, 8 * <Py_ssize_t>no)
            o += 8 * <Py_ssize_t>no
            bl = _i32(base + o); o += 4
            if bl < 0 or bl > 67108864:
                raise ValueError("invalid legacy journal string blob length")
            if o + bl + 4 > n:
                break
            sblob = base + o
            soffs_v = soffs
            if soffs_v[0] != 0 or soffs_v[no - 1] != bl:
                raise ValueError("legacy journal string offsets do not span the blob")
            for i in range(<Py_ssize_t>no - 1):
                lo = <Py_ssize_t>soffs_v[i]
                hi = <Py_ssize_t>soffs_v[i + 1]
                if lo < 0 or hi < lo or hi > bl:
                    raise ValueError("invalid legacy journal string offsets")
                strings.append(PyUnicode_DecodeUTF8(
                    <const char *>(sblob + lo), hi - lo, "strict"))
            o += bl

            # residual leaf descriptors
            nl = _i32(base + o); o += 4
            if nl < 0 or nl > 6710886:
                raise ValueError("invalid legacy journal leaf count")
            for j in range(nl):
                if o + 6 > n:
                    return _result(out, valid_end, return_offsets)
                scope = _i8(base + o); o += 1
                kind = _i8(base + o); o += 1
                if scope < 0 or scope > 2 or kind < 0 or kind > 11:
                    raise ValueError("unknown legacy journal leaf scope or kind")
                ns = _i32(base + o); o += 4
                if ns < 0 or ns > 67108864:
                    raise ValueError("invalid legacy journal leaf path length")
                if o + ns + 4 > n:
                    return _result(out, valid_end, return_offsets)
                isidx = np.empty(ns, dtype=np.int8)
                if ns:
                    memcpy(np.PyArray_DATA(isidx), base + o, ns)
                for i in range(ns):
                    if _i8(base + o + i) != 0 and _i8(base + o + i) != 1:
                        raise ValueError("invalid legacy journal leaf path flag")
                o += ns
                nvals = _i32(base + o); o += 4
                if nvals < 0 or nvals > 67108864:
                    raise ValueError("invalid legacy journal leaf value count")
                leaves.append((int(scope), int(kind), isidx, int(nvals)))

            if has == 2:
                if o + 4 > n:
                    break
                ne = _i32(base + o); o += 4
                if ne < 0 or ne > 8388608:
                    raise ValueError("invalid legacy journal extra row length")
                if o + 8 * <Py_ssize_t>ne > n:
                    break
                extra = np.empty(ne, dtype=np.int64)
                memcpy(np.PyArray_DATA(extra), base + o, 8 * <Py_ssize_t>ne)
                o += 8 * <Py_ssize_t>ne

        rest = []
        terms = _split_terms(strings, rest) if has else []
        frame = (int(op), rid, scal, terms, rest, leaves, extra)
        if o - valid_end > 67108864:
            raise ValueError("legacy journal frame exceeds its byte limit")
        out.append((o, frame) if return_offsets else frame)
        valid_end = o
    return _result(out, valid_end, return_offsets)
