# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._spectral - Spectral layout and force-directed refinement.

Computes 2D vertex positions in two phases:

  1. Spectral embedding from L_0 eigenvectors (Fiedler vector and
     third eigenvector as x/y coordinates).
  2. Force-directed refinement via Fruchterman-Reingold with Coulomb
     repulsion, Hooke attraction, and centering forces.

For nV > 200, all-pairs O(nV^2) repulsion is replaced by Barnes-Hut
quadtree approximation at O(nV log nV) with opening angle 0.5.
"""

from __future__ import annotations

from libc.stdlib cimport malloc, free, calloc
from libc.math cimport sqrt, pow, fabs
from libc.string cimport memset

cimport cython

import numpy as np
cimport numpy as np

from rexgraph.core._common cimport (
    i32, i64, f64,

    get_EPSILON_NORM,
)

np.import_array()

cdef Py_ssize_t _BARNES_HUT_THRESHOLD = 200
cdef enum:
    _QT_EMPTY = -1
cdef double _MIN_DIST_SQ = 1.0


# Spectral embedding

def spectral_layout(evecs_in, Py_ssize_t nV,
                    double width=700.0, double height=500.0,
                    double pad=0.10, evals_in=None):
    """
    Spectral embedding using L_0 eigenvectors.

    Uses Fiedler vector (x) and third eigenvector (y), rescaled to
    the canvas. Falls back to deterministic placement if fewer than
    3 eigenvectors are available. If evals_in is provided,
    eigenvectors are sorted by ascending eigenvalue first.

    Parameters
    ----------
    evecs_in : array-like, shape (nV, k)
        Eigenvectors of L_0.
    nV : int
        Number of vertices.
    width, height : float
        Canvas dimensions in pixels.
    pad : float
        Fractional padding on each side (default 10%).
    evals_in : array-like, shape (k,), optional
        Eigenvalues of L_0 for sorting eigenvectors.

    Returns
    -------
    px, py : f64[nV]
        Vertex positions in pixel coordinates.
    """
    cdef f64[::1] pxv
    cdef f64[::1] pyv
    cdef f64[::1] ev_col1
    cdef f64[::1] ev_col2
    cdef Py_ssize_t i, kcols

    px_arr = np.empty(nV, dtype=np.float64)
    py_arr = np.empty(nV, dtype=np.float64)
    pxv = px_arr
    pyv = py_arr

    if nV == 0:
        return px_arr, py_arr

    if nV == 1:
        pxv[0] = width * 0.5
        pyv[0] = height * 0.5
        return px_arr, py_arr

    if nV == 2:
        pxv[0] = width / 3.0
        pxv[1] = width * 2.0 / 3.0
        pyv[0] = height * 0.5
        pyv[1] = height * 0.5
        return px_arr, py_arr

    # Force C-contiguous copy (eigh returns F-order)
    evecs = np.ascontiguousarray(evecs_in, dtype=np.float64)

    if evals_in is not None:
        evals = np.asarray(evals_in, dtype=np.float64)
        sorted_idx = np.argsort(evals)
        evecs = np.ascontiguousarray(evecs[:, sorted_idx])

    kcols = evecs.shape[1]

    if kcols < 3:
        _deterministic_placement(pxv, pyv, nV, width, height, pad)
        return px_arr, py_arr

    # Columns 1 and 2 (skip constant column 0)
    col1 = np.ascontiguousarray(evecs[:, 1])
    col2 = np.ascontiguousarray(evecs[:, 2])
    ev_col1 = col1
    ev_col2 = col2

    with nogil:
        for i in range(nV):
            pxv[i] = ev_col1[i]
            pyv[i] = ev_col2[i]

        _rescale_to_canvas(pxv, pyv, nV, width, height, pad)

    return px_arr, py_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _deterministic_placement(f64[::1] px, f64[::1] py,
                                   Py_ssize_t nV,
                                   double width, double height,
                                   double pad) noexcept nogil:
    """Place vertices on a grid-like pattern seeded by index."""
    cdef Py_ssize_t i
    cdef double x_lo = pad * width, x_hi = (1.0 - pad) * width
    cdef double y_lo = pad * height, y_hi = (1.0 - pad) * height
    cdef double x_range = x_hi - x_lo
    cdef double y_range = y_hi - y_lo
    cdef double phi_inv = 0.6180339887498949
    cdef double xf, yf

    for i in range(nV):
        xf = ((<double>i * phi_inv) % 1.0)
        yf = ((<double>(i * i + i) * phi_inv * phi_inv) % 1.0)
        px[i] = x_lo + xf * x_range
        py[i] = y_lo + yf * y_range


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _rescale_to_canvas(f64[::1] px, f64[::1] py,
                             Py_ssize_t nV,
                             double width, double height,
                             double pad) noexcept nogil:
    """
    Linearly rescale coordinate arrays to [pad*dim, (1-pad)*dim].
    """
    cdef Py_ssize_t i
    cdef double xmin, xmax, ymin, ymax
    cdef double x_lo, x_hi, y_lo, y_hi
    cdef double xscale, yscale, xoff, yoff

    xmin = px[0]
    xmax = px[0]
    ymin = py[0]
    ymax = py[0]

    for i in range(1, nV):
        if px[i] < xmin: xmin = px[i]
        if px[i] > xmax: xmax = px[i]
        if py[i] < ymin: ymin = py[i]
        if py[i] > ymax: ymax = py[i]

    x_lo = pad * width
    x_hi = (1.0 - pad) * width
    y_lo = pad * height
    y_hi = (1.0 - pad) * height

    if xmax - xmin < get_EPSILON_NORM():
        for i in range(nV):
            px[i] = (x_lo + x_hi) * 0.5
    else:
        xscale = (x_hi - x_lo) / (xmax - xmin)
        xoff = x_lo - xmin * xscale
        for i in range(nV):
            px[i] = px[i] * xscale + xoff

    if ymax - ymin < get_EPSILON_NORM():
        for i in range(nV):
            py[i] = (y_lo + y_hi) * 0.5
    else:
        yscale = (y_hi - y_lo) / (ymax - ymin)
        yoff = y_lo - ymin * yscale
        for i in range(nV):
            py[i] = py[i] * yscale + yoff


# Naive force-directed refinement

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fd_naive_iteration(f64* px, f64* py,
                              const i32* edge_src,
                              const i32* edge_tgt,
                              Py_ssize_t nV, Py_ssize_t nE,
                              f64* dx, f64* dy,
                              double repel, double attract_k,
                              double ideal, double centering,
                              double step, double cx, double cy,
                              double x_lo, double x_hi,
                              double y_lo, double y_hi) noexcept nogil:
    """
    Single iteration of O(nV^2) force-directed layout.

    Repulsion (Coulomb), attraction (Hooke), and centering forces.
    """
    cdef Py_ssize_t i, j, e
    cdef double ddx, ddy, dist_sq, dist, force
    cdef i32 s, t

    for i in range(nV):
        dx[i] = 0.0
        dy[i] = 0.0

    # Repulsion
    for i in range(nV):
        for j in range(i + 1, nV):
            ddx = px[i] - px[j]
            ddy = py[i] - py[j]
            dist_sq = ddx * ddx + ddy * ddy
            if dist_sq < _MIN_DIST_SQ:
                dist_sq = _MIN_DIST_SQ
            dist = sqrt(dist_sq)
            force = repel / dist_sq
            dx[i] += ddx * force / dist
            dy[i] += ddy * force / dist
            dx[j] -= ddx * force / dist
            dy[j] -= ddy * force / dist

    # Attraction
    for e in range(nE):
        s = edge_src[e]
        t = edge_tgt[e]
        if s < 0 or s >= nV or t < 0 or t >= nV:
            continue
        ddx = px[t] - px[s]
        ddy = py[t] - py[s]
        dist = sqrt(ddx * ddx + ddy * ddy)
        if dist < 0.01:
            continue
        force = attract_k * (dist - ideal) / dist
        dx[s] += ddx * force
        dy[s] += ddy * force
        dx[t] -= ddx * force
        dy[t] -= ddy * force

    # Centering, update, clamp
    cdef double mean_x = 0.0, mean_y = 0.0
    for i in range(nV):
        mean_x += px[i]
        mean_y += py[i]
    mean_x /= <double>nV
    mean_y /= <double>nV
    for i in range(nV):
        dx[i] += centering * (mean_x - px[i])
        dy[i] += centering * (mean_y - py[i])
        px[i] += dx[i] * step
        py[i] += dy[i] * step
        if px[i] < x_lo: px[i] = x_lo
        elif px[i] > x_hi: px[i] = x_hi
        if py[i] < y_lo: py[i] = y_lo
        elif py[i] > y_hi: py[i] = y_hi


# Barnes-Hut quadtree


cdef struct QuadTree:
    f64* cx
    f64* cy
    f64* mass
    f64* size
    f64* cell_x
    f64* cell_y
    i32* children
    i32* body
    i32 n_nodes
    i32 max_nodes


cdef QuadTree* _qt_alloc(Py_ssize_t max_nodes) noexcept nogil:
    """Allocate quadtree arrays."""
    cdef QuadTree* qt = <QuadTree*>malloc(sizeof(QuadTree))
    if qt == NULL:
        return NULL
    qt.max_nodes = <i32>max_nodes
    qt.n_nodes = 0
    qt.cx = <f64*>calloc(<size_t>max_nodes, sizeof(f64))
    qt.cy = <f64*>calloc(<size_t>max_nodes, sizeof(f64))
    qt.mass = <f64*>calloc(<size_t>max_nodes, sizeof(f64))
    qt.size = <f64*>malloc(<size_t>max_nodes * sizeof(f64))
    qt.cell_x = <f64*>malloc(<size_t>max_nodes * sizeof(f64))
    qt.cell_y = <f64*>malloc(<size_t>max_nodes * sizeof(f64))
    qt.children = <i32*>malloc(<size_t>(max_nodes * 4) * sizeof(i32))
    qt.body = <i32*>malloc(<size_t>max_nodes * sizeof(i32))

    if (qt.cx == NULL or qt.cy == NULL or qt.mass == NULL or
            qt.size == NULL or qt.cell_x == NULL or qt.cell_y == NULL or
            qt.children == NULL or qt.body == NULL):
        _qt_free(qt)
        return NULL
    return qt


cdef void _qt_free(QuadTree* qt) noexcept nogil:
    """Free all quadtree arrays."""
    if qt == NULL:
        return
    if qt.cx != NULL: free(qt.cx)
    if qt.cy != NULL: free(qt.cy)
    if qt.mass != NULL: free(qt.mass)
    if qt.size != NULL: free(qt.size)
    if qt.cell_x != NULL: free(qt.cell_x)
    if qt.cell_y != NULL: free(qt.cell_y)
    if qt.children != NULL: free(qt.children)
    if qt.body != NULL: free(qt.body)
    free(qt)


cdef i32 _qt_new_node(QuadTree* qt, double cell_x, double cell_y,
                      double cell_size) noexcept nogil:
    """Allocate a new node, return its index. Returns _QT_EMPTY on overflow."""
    if qt.n_nodes >= qt.max_nodes:
        return _QT_EMPTY
    cdef i32 idx = qt.n_nodes
    qt.n_nodes += 1
    qt.cx[idx] = 0.0
    qt.cy[idx] = 0.0
    qt.mass[idx] = 0.0
    qt.size[idx] = cell_size
    qt.cell_x[idx] = cell_x
    qt.cell_y[idx] = cell_y
    qt.body[idx] = _QT_EMPTY
    cdef i32 base = idx * 4
    qt.children[base] = _QT_EMPTY
    qt.children[base + 1] = _QT_EMPTY
    qt.children[base + 2] = _QT_EMPTY
    qt.children[base + 3] = _QT_EMPTY
    return idx


@cython.inline
cdef i32 _qt_quadrant(double px, double py,
                      double cell_x, double cell_y) noexcept nogil:
    """Determine which quadrant (0=NW, 1=NE, 2=SW, 3=SE) a point falls in."""
    if py < cell_y:
        return 0 if px < cell_x else 1   # NW or NE
    else:
        return 2 if px < cell_x else 3   # SW or SE


cdef void _qt_child_center(double cell_x, double cell_y, double half,
                           i32 quadrant,
                           double* out_cx, double* out_cy) noexcept nogil:
    """Compute the center of a child cell given parent center and quadrant."""
    cdef double q = half * 0.5
    if quadrant == 0:    # NW
        out_cx[0] = cell_x - q
        out_cy[0] = cell_y - q
    elif quadrant == 1:  # NE
        out_cx[0] = cell_x + q
        out_cy[0] = cell_y - q
    elif quadrant == 2:  # SW
        out_cx[0] = cell_x - q
        out_cy[0] = cell_y + q
    else:                # SE
        out_cx[0] = cell_x + q
        out_cy[0] = cell_y + q


cdef void _qt_insert(QuadTree* qt, i32 node, double px, double py,
                     i32 vertex_idx, i32 depth) noexcept nogil:
    """
    Insert a vertex into the quadtree rooted at node.

    Depth limit prevents infinite recursion from coincident points.
    """
    cdef double m

    if depth > 50:
        qt.mass[node] += 1.0
        m = qt.mass[node]
        qt.cx[node] = qt.cx[node] * (m - 1.0) / m + px / m
        qt.cy[node] = qt.cy[node] * (m - 1.0) / m + py / m
        return

    cdef i32 base = node * 4
    cdef i32 q, child, old_body
    cdef double half = qt.size[node] * 0.5
    cdef double ccx, ccy
    cdef double old_px, old_py

    if qt.mass[node] == 0.0:
        qt.body[node] = vertex_idx
        qt.cx[node] = px
        qt.cy[node] = py
        qt.mass[node] = 1.0
        return

    if qt.body[node] != _QT_EMPTY:
        old_body = qt.body[node]
        old_px = qt.cx[node]
        old_py = qt.cy[node]
        qt.body[node] = _QT_EMPTY

        q = _qt_quadrant(old_px, old_py, qt.cell_x[node], qt.cell_y[node])
        if qt.children[base + q] == _QT_EMPTY:
            _qt_child_center(qt.cell_x[node], qt.cell_y[node], half, q,
                             &ccx, &ccy)
            qt.children[base + q] = _qt_new_node(qt, ccx, ccy, half)
        child = qt.children[base + q]
        if child != _QT_EMPTY:
            _qt_insert(qt, child, old_px, old_py, old_body, depth + 1)

    cdef double m_new = qt.mass[node] + 1.0
    qt.cx[node] = (qt.cx[node] * qt.mass[node] + px) / m_new
    qt.cy[node] = (qt.cy[node] * qt.mass[node] + py) / m_new
    qt.mass[node] = m_new

    q = _qt_quadrant(px, py, qt.cell_x[node], qt.cell_y[node])
    if qt.children[base + q] == _QT_EMPTY:
        _qt_child_center(qt.cell_x[node], qt.cell_y[node], half, q,
                         &ccx, &ccy)
        qt.children[base + q] = _qt_new_node(qt, ccx, ccy, half)
    child = qt.children[base + q]
    if child != _QT_EMPTY:
        _qt_insert(qt, child, px, py, vertex_idx, depth + 1)


cdef void _qt_repulsion(QuadTree* qt, i32 node,
                        double px, double py,
                        double* out_fx, double* out_fy,
                        double theta, double repel) noexcept nogil:
    """
    Repulsive force on vertex at (px, py) from subtree at node.

    Uses Barnes-Hut criterion: if cell_size / distance < theta,
    treat the node as a single body.
    """
    if qt.mass[node] == 0.0:
        return

    cdef double ddx = px - qt.cx[node]
    cdef double ddy = py - qt.cy[node]
    cdef double dist_sq = ddx * ddx + ddy * ddy
    if dist_sq < _MIN_DIST_SQ:
        dist_sq = _MIN_DIST_SQ
    cdef double dist = sqrt(dist_sq)

    cdef double s = qt.size[node]
    cdef i32 base = node * 4
    cdef i32 c

    cdef double force

    if (qt.body[node] != _QT_EMPTY) or (s / dist < theta):
        force = repel * qt.mass[node] / dist_sq
        out_fx[0] += ddx * force / dist
        out_fy[0] += ddy * force / dist
        return

    for c in range(4):
        if qt.children[base + c] != _QT_EMPTY:
            _qt_repulsion(qt, qt.children[base + c], px, py,
                          out_fx, out_fy, theta, repel)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fd_bh_iteration(f64* px, f64* py,
                           const i32* edge_src,
                           const i32* edge_tgt,
                           Py_ssize_t nV, Py_ssize_t nE,
                           f64* dx_arr, f64* dy_arr,
                           double repel, double attract_k,
                           double ideal, double centering,
                           double step, double cx, double cy,
                           double x_lo, double x_hi,
                           double y_lo, double y_hi,
                           double theta) noexcept nogil:
    """
    Single iteration of Barnes-Hut O(nV log nV) force-directed layout.
    """
    cdef Py_ssize_t i, e
    cdef i32 s, t
    cdef double ddx, ddy, dist, force, fx, fy

    cdef double bx_min = px[0], bx_max = px[0]
    cdef double by_min = py[0], by_max = py[0]
    for i in range(1, nV):
        if px[i] < bx_min: bx_min = px[i]
        if px[i] > bx_max: bx_max = px[i]
        if py[i] < by_min: by_min = py[i]
        if py[i] > by_max: by_max = py[i]

    cdef double span_x = bx_max - bx_min
    cdef double span_y = by_max - by_min
    cdef double span = span_x if span_x > span_y else span_y
    if span < 1.0:
        span = 1.0
    span *= 1.01

    cdef double root_cx = (bx_min + bx_max) * 0.5
    cdef double root_cy = (by_min + by_max) * 0.5

    cdef Py_ssize_t max_nodes = 4 * nV + 64
    cdef QuadTree* qt = _qt_alloc(max_nodes)
    if qt == NULL:
        for i in range(nV):
            dx_arr[i] = centering * (cx - px[i])
            dy_arr[i] = centering * (cy - py[i])
            px[i] += dx_arr[i] * step
            py[i] += dy_arr[i] * step
        return

    _qt_new_node(qt, root_cx, root_cy, span)

    for i in range(nV):
        _qt_insert(qt, 0, px[i], py[i], <i32>i, 0)

    for i in range(nV):
        dx_arr[i] = 0.0
        dy_arr[i] = 0.0

    for i in range(nV):
        fx = 0.0
        fy = 0.0
        _qt_repulsion(qt, 0, px[i], py[i], &fx, &fy, theta, repel)
        dx_arr[i] += fx
        dy_arr[i] += fy

    _qt_free(qt)

    # Attraction
    for e in range(nE):
        s = edge_src[e]
        t = edge_tgt[e]
        if s < 0 or s >= nV or t < 0 or t >= nV:
            continue
        ddx = px[t] - px[s]
        ddy = py[t] - py[s]
        dist = sqrt(ddx * ddx + ddy * ddy)
        if dist < 0.01:
            continue
        force = attract_k * (dist - ideal) / dist
        dx_arr[s] += ddx * force
        dy_arr[s] += ddy * force
        dx_arr[t] -= ddx * force
        dy_arr[t] -= ddy * force

    # Centering, update, clamp
    for i in range(nV):
        dx_arr[i] += centering * (cx - px[i])
        dy_arr[i] += centering * (cy - py[i])
        px[i] += dx_arr[i] * step
        py[i] += dy_arr[i] * step
        if px[i] < x_lo: px[i] = x_lo
        elif px[i] > x_hi: px[i] = x_hi
        if py[i] < y_lo: py[i] = y_lo
        elif py[i] > y_hi: py[i] = y_hi


# Public entry points

def force_directed_refine(px_in, py_in,
                          edge_src_in, edge_tgt_in,
                          Py_ssize_t nV, Py_ssize_t nE,
                          Py_ssize_t iterations=400,
                          double repel_strength=3000.0,
                          double attract_ideal=50.0,
                          double attract_strength=0.04,
                          double centering=0.008,
                          double width=700.0, double height=500.0):
    """
    O(nV^2) Fruchterman-Reingold force-directed refinement.

    Parameters
    ----------
    px_in, py_in : array-like, shape (nV,)
        Initial vertex positions (modified in-place).
    edge_src_in, edge_tgt_in : array-like, shape (nE,)
        Edge endpoint arrays.
    nV, nE : int
        Vertex and edge counts.
    iterations : int
        Number of force iterations.
    repel_strength : float
        Coulomb repulsion constant.
    attract_ideal : float
        Ideal edge length (Hooke rest length).
    attract_strength : float
        Hooke spring constant.
    centering : float
        Centering force coefficient.
    width, height : float
        Canvas dimensions.

    Returns
    -------
    px, py : f64[nV]
    """
    if nV <= 1:
        return (np.asarray(px_in, dtype=np.float64),
                np.asarray(py_in, dtype=np.float64))

    px_arr = np.ascontiguousarray(px_in, dtype=np.float64)
    py_arr = np.ascontiguousarray(py_in, dtype=np.float64)
    es_arr = np.ascontiguousarray(edge_src_in, dtype=np.int32)
    et_arr = np.ascontiguousarray(edge_tgt_in, dtype=np.int32)

    cdef f64[::1] pxv = px_arr
    cdef f64[::1] pyv = py_arr
    cdef i32[::1] esv = es_arr
    cdef i32[::1] etv = et_arr

    cdef f64* dx_buf = <f64*>malloc(<size_t>nV * sizeof(f64))
    cdef f64* dy_buf = <f64*>malloc(<size_t>nV * sizeof(f64))
    if dx_buf == NULL or dy_buf == NULL:
        if dx_buf != NULL: free(dx_buf)
        if dy_buf != NULL: free(dy_buf)
        raise MemoryError("force_directed_refine: scratch allocation failed")

    cdef double cx = width * 0.5, cy = height * 0.5
    cdef double pad = 0.02
    cdef double x_lo = pad * width, x_hi = (1.0 - pad) * width
    cdef double y_lo = pad * height, y_hi = (1.0 - pad) * height
    cdef double step, frac
    cdef Py_ssize_t it

    cdef f64* px_ptr = &pxv[0]
    cdef f64* py_ptr = &pyv[0]
    cdef const i32* es_ptr = &esv[0]
    cdef const i32* et_ptr = &etv[0]

    with nogil:
        for it in range(iterations):
            frac = 1.0 - <double>it / <double>iterations
            step = 0.6 * frac
            _fd_naive_iteration(px_ptr, py_ptr,
                                es_ptr, et_ptr, nV, nE,
                                dx_buf, dy_buf,
                                repel_strength, attract_strength,
                                attract_ideal, centering,
                                step, cx, cy,
                                x_lo, x_hi, y_lo, y_hi)

    free(dx_buf)
    free(dy_buf)

    return px_arr, py_arr


def barnes_hut_refine(px_in, py_in,
                      edge_src_in, edge_tgt_in,
                      Py_ssize_t nV, Py_ssize_t nE,
                      Py_ssize_t iterations=400,
                      double repel_strength=3000.0,
                      double attract_ideal=50.0,
                      double attract_strength=0.04,
                      double centering=0.008,
                      double width=700.0, double height=500.0,
                      double theta=0.5):
    """
    Barnes-Hut O(nV log nV) force-directed refinement.

    Uses a quadtree to approximate far-field repulsive forces.

    Parameters
    ----------
    px_in, py_in : array-like, shape (nV,)
        Initial vertex positions (modified in-place).
    edge_src_in, edge_tgt_in : array-like, shape (nE,)
        Edge endpoint arrays.
    nV, nE : int
        Vertex and edge counts.
    iterations : int
        Number of force iterations.
    theta : float
        Barnes-Hut opening angle (default 0.5).

    Returns
    -------
    px, py : f64[nV]
    """
    if nV <= 1:
        return (np.asarray(px_in, dtype=np.float64),
                np.asarray(py_in, dtype=np.float64))

    px_arr = np.ascontiguousarray(px_in, dtype=np.float64)
    py_arr = np.ascontiguousarray(py_in, dtype=np.float64)
    es_arr = np.ascontiguousarray(edge_src_in, dtype=np.int32)
    et_arr = np.ascontiguousarray(edge_tgt_in, dtype=np.int32)

    cdef f64[::1] pxv = px_arr
    cdef f64[::1] pyv = py_arr
    cdef i32[::1] esv = es_arr
    cdef i32[::1] etv = et_arr

    cdef f64* dx_buf = <f64*>malloc(<size_t>nV * sizeof(f64))
    cdef f64* dy_buf = <f64*>malloc(<size_t>nV * sizeof(f64))
    if dx_buf == NULL or dy_buf == NULL:
        if dx_buf != NULL: free(dx_buf)
        if dy_buf != NULL: free(dy_buf)
        raise MemoryError("barnes_hut_refine: scratch allocation failed")

    cdef double cx = width * 0.5, cy = height * 0.5
    cdef double pad = 0.02
    cdef double x_lo = pad * width, x_hi = (1.0 - pad) * width
    cdef double y_lo = pad * height, y_hi = (1.0 - pad) * height
    cdef double step, frac
    cdef Py_ssize_t it

    cdef f64* px_ptr = &pxv[0]
    cdef f64* py_ptr = &pyv[0]
    cdef const i32* es_ptr = &esv[0]
    cdef const i32* et_ptr = &etv[0]

    with nogil:
        for it in range(iterations):
            frac = 1.0 - <double>it / <double>iterations
            step = 2.0 * pow(frac, 1.5)
            _fd_bh_iteration(px_ptr, py_ptr,
                             es_ptr, et_ptr, nV, nE,
                             dx_buf, dy_buf,
                             repel_strength, attract_strength,
                             attract_ideal, centering,
                             step, cx, cy,
                             x_lo, x_hi, y_lo, y_hi,
                             theta)

    free(dx_buf)
    free(dy_buf)

    return px_arr, py_arr


# Combined layout

def compute_layout(evecs_in,
                   Py_ssize_t nV, Py_ssize_t nE,
                   edge_src_in, edge_tgt_in,
                   double width=700.0, double height=500.0,
                   Py_ssize_t iterations=400,
                   evals_in=None):
    """
    Spectral embedding followed by force-directed refinement.

    Selects O(nV^2) or Barnes-Hut O(nV log nV) refinement based
    on vertex count.

    Parameters
    ----------
    evecs_in : array-like, shape (nV, k)
        Eigenvectors of L_0.
    nV, nE : int
        Vertex and edge counts.
    edge_src_in, edge_tgt_in : array-like, shape (nE,)
        Edge endpoint arrays.
    width, height : float
        Canvas dimensions.
    iterations : int
        Force-directed iterations.
    evals_in : array-like, shape (k,), optional
        Eigenvalues of L_0 for sorting eigenvectors.

    Returns
    -------
    px, py : f64[nV]
        Final vertex positions in pixel coordinates.
    """
    px, py = spectral_layout(evecs_in, nV, width, height, evals_in=evals_in)

    if nV <= 2 or nE == 0:
        return px, py

    if nV <= _BARNES_HUT_THRESHOLD:
        return force_directed_refine(px, py, edge_src_in, edge_tgt_in,
                                     nV, nE, iterations=iterations,
                                     width=width, height=height)
    return barnes_hut_refine(px, py, edge_src_in, edge_tgt_in,
                             nV, nE, iterations=iterations,
                             width=width, height=height)
