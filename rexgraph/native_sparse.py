"""Native DualCSR actions and boundary views; SciPy is an explicit export only.

This is a shape/dispatch adapter for ``core._sparse``, not a second sparse
implementation. Numerical vector products run in the existing Cython kernels.
Blocks are dispatched columnwise and complex vectors as two real actions: the
real valued kernels must never silently discard an imaginary component.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rexgraph.core import _sparse


@dataclass(frozen=True)
class NativeSparse:
    dual: object

    def __post_init__(self):
        if not isinstance(self.dual, _sparse.DualCSR):
            raise TypeError("native sparse actions require a core DualCSR")

    @property
    def shape(self):
        return int(self.dual.nrow), int(self.dual.ncol)

    @property
    def nnz(self):
        return int(self.dual.nnz)

    @property
    def data(self):
        return self.dual.vals

    def columns(self):
        """Yield canonical numeric columns, coalescing stored duplicates/zeros."""
        ptr, rows, values = self.dual.col_ptr, self.dual.row_idx, self.dual.vals_csc
        for j in range(self.shape[1]):
            column = {}
            for pos in range(int(ptr[j]), int(ptr[j+1])):
                i = int(rows[pos])
                column[i] = column.get(i, 0) + values[pos]
            yield {i: v for i, v in column.items() if v != 0}

    def apply(self, values, *, transpose=False):
        from rexgraph.linear_operator import _numeric_array
        values = _numeric_array(values, operation="native sparse action")
        m, n = self.shape[::-1] if transpose else self.shape
        if values.ndim not in (1, 2) or values.shape[0] != n:
            raise ValueError("native sparse action requires a matching vector/block axis")
        if not np.all(np.isfinite(self.data)):
            raise FloatingPointError("native sparse coefficients are outside numerical range")
        kernel = _sparse.rmatvec if transpose else _sparse.matvec

        def vector(v):
            if np.iscomplexobj(v):
                return kernel(self.dual, np.ascontiguousarray(v.real)) + 1j * kernel(
                    self.dual, np.ascontiguousarray(v.imag))
            return kernel(self.dual, np.ascontiguousarray(v))

        if values.ndim == 1:
            result = vector(values)
        elif values.shape[1] == 0:
            result = np.zeros((m, 0), dtype=values.dtype)
        else:
            # One pass over the nonzeros for the whole block. The column loop below is
            # the same arithmetic in the same order, and stays for the carriers the
            # block kernel does not take (f32, complex).
            block = None
            if not np.iscomplexobj(values) and self.dual.csr.val_bits != 32:
                kernel = _sparse.rspmm if transpose else _sparse.spmm
                block = kernel(self.dual, np.ascontiguousarray(values, dtype=np.float64))
            result = block if block is not None else np.column_stack(
                [vector(values[:, j]) for j in range(values.shape[1])])
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("native sparse result is outside numerical range")
        return result

    def transpose_apply(self, values):
        return self.apply(values, transpose=True)

    def with_data(self, values):
        """Replace CSR coefficients and let the core rebuild the column view."""
        values = np.ascontiguousarray(values, dtype=np.float64)
        if values.shape != self.data.shape:
            raise ValueError("replacement coefficients must match stored nonzeros")
        return NativeSparse(_sparse.dual_from_csr(_sparse.CSRMatrix(
            self.dual.row_ptr, self.dual.col_idx, values, *self.shape)))

    def column_quadrances(self):
        """Squared column norms for canonical storage (no duplicate addresses)."""
        return np.bincount(self.dual.col_idx, weights=self.data**2, minlength=self.shape[1])

    @property
    def T(self):
        return NativeSparse(_sparse.dual_from_csr(_sparse.CSRMatrix(
            self.dual.col_ptr, self.dual.row_idx, self.dual.vals_csc, *self.shape[::-1])))

    def diagonal(self):
        return _sparse.diag(self.dual)

    def product(self, other):
        return NativeSparse(_sparse.csr_product(self.dual, as_native(other).dual))

    def row_inner(self, other):
        return _sparse.csr_hadamard_rows(self.dual, as_native(other).dual)

    def add(self, other, coefficient=1.0):
        other = as_native(other)
        if self.shape != other.shape:
            raise ValueError("sparse sum shapes do not match")
        rows = np.concatenate([np.repeat(np.arange(a.shape[0]), np.diff(a.dual.row_ptr))
                               for a in (self, other)])
        return native_coo(rows, np.concatenate((self.dual.col_idx, other.dual.col_idx)),
                          np.concatenate((self.data, coefficient * other.data)), self.shape)

    def as_scipy(self):
        """Explicit optional compatibility export, never used by native actions."""
        return _sparse.to_scipy_csr(self.dual)

    def torch_sparse(self, dtype=None, device=None, *, transpose=False):
        import torch

        from rexgraph.compute import sparse_coo_tensor
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.dual.row_ptr))
        indices = np.vstack((rows, self.dual.col_idx))
        if transpose:
            indices = indices[::-1].copy()
        return sparse_coo_tensor(
            torch.as_tensor(indices, dtype=torch.long, device=device),
            torch.as_tensor(self.data, dtype=dtype or torch.get_default_dtype(), device=device),
            self.shape[::-1] if transpose else self.shape, device=device,
        ).coalesce()


def native_coo(rows, columns, values, shape):
    """Build canonical numerical storage through the compiled COO constructor."""
    from rexgraph.linear_operator import _numeric_array
    rows, columns = np.asarray(rows), np.asarray(columns)
    if any(axis.size and axis.dtype.kind not in 'iu' for axis in (rows, columns)):
        raise ValueError("COO addresses must be integers")
    if any(np.any(axis > np.iinfo(np.int64).max) for axis in (rows, columns)):
        raise ValueError("COO addresses exceed the native index range")
    rows, columns = np.ascontiguousarray(rows, dtype=np.int64), np.ascontiguousarray(columns, dtype=np.int64)
    values = _numeric_array(values, operation="COO coefficients")
    if np.iscomplexobj(values):
        raise TypeError("COO coefficients must be real")
    values = np.ascontiguousarray(values)
    if rows.ndim != 1 or rows.shape != columns.shape or rows.shape != values.shape:
        raise ValueError("COO arrays must be matching vectors")
    if len(shape) != 2 or any(isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or n < 0 for n in shape):
        raise ValueError("sparse shape requires two nonnegative integer dimensions")
    if (np.any(rows < 0) or np.any(rows >= shape[0])
            or np.any(columns < 0) or np.any(columns >= shape[1])):
        raise ValueError("COO address is outside the sparse shape")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("sparse coefficients are outside numerical range")
    result = NativeSparse(_sparse.canonical_dual(_sparse.dual_from_coo(rows, columns, values, *shape)))
    if not np.all(np.isfinite(result.data)):
        raise FloatingPointError("coalesced sparse coefficients are outside numerical range")
    return result


def native_diagonal(values):
    values = np.asarray(values, dtype=np.float64)
    index = np.arange(values.size)
    return native_coo(index, index, values, (values.size, values.size))


def as_native(matrix):
    """Read supplied sparse arrays without importing their compatibility package."""
    if isinstance(matrix, NativeSparse):
        return matrix
    if isinstance(matrix, _sparse.DualCSR):
        return NativeSparse(matrix)
    if isinstance(matrix, _sparse.CSRMatrix):
        ptr, indices, values, shape = sparse_arrays(matrix)
        return native_coo(np.repeat(np.arange(shape[0]), np.diff(ptr)), indices, values, shape)
    if hasattr(matrix, 'tocsr'):
        matrix = matrix.tocsr()
        return native_coo(np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr)),
                          matrix.indices, matrix.data, matrix.shape)
    array = np.asarray(matrix)
    if array.ndim != 2 or np.iscomplexobj(array):
        raise TypeError("native sparse storage requires a real matrix")
    rows, cols = np.nonzero(array)
    return native_coo(rows, cols, array[rows, cols], array.shape)


def empty_native(shape):
    rows = np.empty(0, dtype=np.int64)
    return NativeSparse(_sparse.dual_from_coo(rows, rows, np.empty(0, dtype=float), *shape))


def sparse_arrays(matrix):
    """Original CSR arrays and shape, without coefficient casts or arithmetic."""
    if isinstance(matrix, NativeSparse):
        matrix = matrix.dual
    if isinstance(matrix, (_sparse.DualCSR, _sparse.CSRMatrix)):
        return matrix.row_ptr, matrix.col_idx, matrix.vals, (int(matrix.nrow), int(matrix.ncol))
    csr = matrix.tocsr()
    return csr.indptr, csr.indices, csr.data, csr.shape


def csr_carrier(ptr, indices, data, shape):
    """Restore original coefficient storage in the core CSR carrier.

    Validate addresses before exposing them to unchecked compiled kernels. Keep
    coefficient and index dtypes for exact state identity. Numerical actions get
    a separate DualCSR view; integral storage is never a disguised float buffer.
    """
    ptr, indices, data = np.asarray(ptr), np.asarray(indices), np.asarray(data)
    if (len(shape) != 2 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                               or n < 0 for n in shape)):
        raise ValueError("CSR carrier shape requires two nonnegative integer axes")
    m, n = map(int, shape)
    if (ptr.ndim != 1 or indices.ndim != 1 or data.ndim != 1 or
            ptr.dtype not in (np.dtype("int32"), np.dtype("int64")) or
            indices.dtype not in (np.dtype("int32"), np.dtype("int64")) or
            len(ptr) != m + 1 or len(indices) != len(data) or
            ptr[0] != 0 or ptr[-1] != len(data) or np.any(ptr[1:] < ptr[:-1]) or
            np.any(indices < 0) or np.any(indices >= n)):
        raise ValueError("invalid CSR carrier pointers, indices or data axes")
    return _sparse.CSRMatrix(ptr.copy(), indices.copy(), data.copy(), m, n)


def scipy_csr_export(matrix):
    """Explicit compatibility export preserving original coefficient dtypes."""
    import scipy.sparse as sp
    ptr, indices, data, shape = sparse_arrays(matrix)
    return sp.csr_matrix((data, indices, ptr), shape=shape)


def restrict_carrier(matrix, rows, columns):
    """Copy a sparse submatrix without converting its coefficient storage.

    Axes are ordered, unique source indices. Work visits only selected CSR rows;
    no dense matrix or optional sparse backend is involved.
    """
    ptr, indices, data, shape = sparse_arrays(matrix)
    axes = []
    for selected, size in zip((rows, columns), shape, strict=True):
        axis = np.asarray(selected)
        if (axis.ndim != 1 or (axis.size and axis.dtype.kind not in "iu")
                or np.any(axis < 0) or np.any(axis >= size)
                or len(set(map(int, axis))) != axis.size):
            raise ValueError("restriction axes must be unique integer source indices")
        axes.append(axis.astype(np.int64))
    rows, columns = axes
    remap = {int(old): new for new, old in enumerate(columns)}
    out_ptr, out_idx, positions = [0], [], []
    for row in rows:
        for pos in range(int(ptr[row]), int(ptr[row+1])):
            target = remap.get(int(indices[pos]))
            if target is not None:
                out_idx.append(target)
                positions.append(pos)
        out_ptr.append(len(out_idx))
    return csr_carrier(np.asarray(out_ptr, dtype=ptr.dtype),
                       np.asarray(out_idx, dtype=indices.dtype),
                       data[np.asarray(positions, dtype=np.int64)], (len(rows), len(columns)))


def raw_boundary_carriers(rex):
    """Read the complete stored tower without Hodge filtering or coefficient casts.

    Exact certificates need raw B2, even when its composition with B1 is invalid.
    Empty grade two stays in place before higher grades.
    """
    rex._ensure_clean()
    bounds = [NativeSparse(rex._B1_dual)]
    higher = [NativeSparse(m) if isinstance(m, _sparse.DualCSR) else m
              for m in (getattr(rex, "_graded_duals", None) or ())]
    if int(rex.nF) > 0:
        if rex._B2_dual is None:
            raise ValueError("source RexGraph has faces but no stored B2")
        bounds.append(NativeSparse(rex._B2_dual))
    elif higher:
        first = higher[0]
        n = first.nrow if isinstance(first, _sparse.CSRMatrix) else first.shape[0]
        bounds.append(empty_native((int(rex.nE), int(n))))
    bounds.extend(higher)
    return bounds


def boundary_carriers(rex):
    """Read Hodge boundary views without coercing higher coefficients.

    B1 and the validated Hodge B2 retain DualCSR. Higher grades retain CSR arrays,
    including explicit compatibility inputs. ``from_cells`` constructs native
    storage. An explicit empty B2 never shifts B3's grade.
    """
    rex._ensure_clean()
    bounds = [NativeSparse(rex._B1_dual)]
    higher = getattr(rex, "_graded_duals", None)
    if int(rex.nF) > 0 and rex._B2_hodge_dual is not None:
        bounds.append(NativeSparse(rex._B2_hodge_dual))
    elif higher and int(rex.nF) == 0:
        first = higher[0]
        n = first.nrow if isinstance(first, (_sparse.DualCSR, _sparse.CSRMatrix)) else first.shape[0]
        bounds.append(empty_native((int(rex.nE), int(n))))
    if higher:
        bounds.extend(NativeSparse(b) if isinstance(b, _sparse.DualCSR) else b for b in higher)
    return bounds


def native_boundaries(rex):
    """Numerical DualCSR views; exact readers use the original boundary carriers."""
    bounds = []
    for matrix in boundary_carriers(rex):
        if isinstance(matrix, NativeSparse):
            bounds.append(matrix)
        else:
            ptr, indices, data, shape = sparse_arrays(matrix)
            bounds.append(NativeSparse(_sparse.dual_from_csr(_sparse.CSRMatrix(
                np.asarray(ptr, dtype=np.int64), np.asarray(indices, dtype=np.int64),
                np.asarray(data, dtype=np.float64), *shape))))
    return bounds


def sparse_columns(matrix):
    """Canonical columns from native storage or an explicitly supplied CSR/CSC."""
    if isinstance(matrix, NativeSparse):
        yield from matrix.columns()
        return
    csc = matrix.tocsc(copy=True)
    csc.sum_duplicates()
    csc.eliminate_zeros()
    for j in range(csc.shape[1]):
        lo, hi = csc.indptr[j:j+2]
        yield dict(zip(csc.indices[lo:hi], csc.data[lo:hi], strict=True))
