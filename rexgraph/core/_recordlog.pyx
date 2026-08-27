# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._recordlog: the append-only record log's frame codec.

One frame is an op, an id, the fixed scalar row, a packed string table, the residual
leaf descriptors, and optionally a backend's own int64 row. The layout is written by
`agent.rcdb_index.log_append`; this reads it.

The scan and the string table are here; assembling the record stays in Python, where
the shape of a record is defined.

A torn tail is where a process died mid append. The scan stops at the first field that
would read past the end and returns what came before it, so a partial frame is never
half interpreted.
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
    if n == 0:
        return terms
    try:
        nk = int(strings[0])
    except (TypeError, ValueError):
        return terms
    i = 1
    for k in range(nk):
        if i + 1 >= n:
            break
        try:
            code = int(strings[i])
            cnt = int(strings[i + 1])
        except (TypeError, ValueError):
            break
        i += 2
        if cnt < 0 or i + cnt > n:
            break
        terms.append((code, strings[i:i + cnt]))
        i += cnt
    for j in range(i, n):
        rest.append(strings[j])
    return terms


def read_frames(const unsigned char[::1] buf, Py_ssize_t start, int nscal):
    """Scan the log from `start`, one tuple per whole frame.

    Returns a list of `(op, rid, scal, terms, rest, leaves, extra)`, where `op` is 1 for
    a put and 2 for a delete, `scal` is a float64 array of width `nscal` or None,
    `terms` is `[(kind code, [term])]`, `rest` is the strings the residual leaves index,
    and `extra` is an int64 array or None.
    """
    cdef Py_ssize_t n = buf.shape[0]
    cdef Py_ssize_t o = start
    cdef const unsigned char *base = &buf[0] if n else NULL
    cdef int8_t op, has, scope, kind
    cdef int32_t ln, no, bl, nl, ns, nvals, ne
    cdef Py_ssize_t i, j, lo, hi
    cdef list out = []
    cdef list strings
    cdef list leaves
    cdef object scal, extra, isidx, terms, rest, code
    cdef np.int64_t[::1] soffs_v
    cdef const unsigned char *sblob

    if base == NULL or start < 0:
        return out

    while o < n:
        # op, id length, id
        if o + 5 > n:
            break
        op = _i8(base + o); o += 1
        ln = _i32(base + o); o += 4
        if ln < 0 or o + ln + 1 > n:
            break
        rid = PyUnicode_DecodeUTF8(<const char *>(base + o), ln, "replace")
        o += ln
        has = _i8(base + o); o += 1

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
            if no < 0 or o + 8 * <Py_ssize_t>no + 4 > n:
                break
            soffs = np.empty(no, dtype=np.int64)
            memcpy(np.PyArray_DATA(soffs), base + o, 8 * <Py_ssize_t>no)
            o += 8 * <Py_ssize_t>no
            bl = _i32(base + o); o += 4
            if bl < 0 or o + bl + 4 > n:
                break
            sblob = base + o
            soffs_v = soffs
            for i in range(<Py_ssize_t>no - 1):
                lo = <Py_ssize_t>soffs_v[i]
                hi = <Py_ssize_t>soffs_v[i + 1]
                if lo < 0 or hi < lo or hi > bl:
                    strings.append("")
                    continue
                strings.append(PyUnicode_DecodeUTF8(
                    <const char *>(sblob + lo), hi - lo, "replace"))
            o += bl

            # residual leaf descriptors
            nl = _i32(base + o); o += 4
            if nl < 0:
                break
            for j in range(nl):
                if o + 6 > n:
                    return out
                scope = _i8(base + o); o += 1
                kind = _i8(base + o); o += 1
                ns = _i32(base + o); o += 4
                if ns < 0 or o + ns + 4 > n:
                    return out
                isidx = np.empty(ns, dtype=np.int8)
                if ns:
                    memcpy(np.PyArray_DATA(isidx), base + o, ns)
                o += ns
                nvals = _i32(base + o); o += 4
                leaves.append((int(scope), int(kind), isidx, int(nvals)))

            if has == 2:
                if o + 4 > n:
                    break
                ne = _i32(base + o); o += 4
                if ne < 0 or o + 8 * <Py_ssize_t>ne > n:
                    break
                extra = np.empty(ne, dtype=np.int64)
                memcpy(np.PyArray_DATA(extra), base + o, 8 * <Py_ssize_t>ne)
                o += 8 * <Py_ssize_t>ne

        rest = []
        terms = _split_terms(strings, rest) if has else []
        out.append((int(op), rid, scal, terms, rest, leaves, extra))
    return out
