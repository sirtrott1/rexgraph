# rexgraph/core/ — Module Documentation






## `_common` — Shared Infrastructure Layer

**Files:** `_common.pxd` (745 lines), `_common.pyx` (821 lines)

Every Cython module in `rexgraph/core/` imports from `_common`. It defines the
type system, error codes, memory management, parallelization decisions, numeric
utilities, and data structure primitives. No mathematical computation happens
here. This module ensures that all computation elsewhere is type-safe,
memory-bounded, overflow-aware, and correctly parallelized.

---

### Type System

All modules share a unified set of type aliases for consistent precision:

- `idx_t` (Py_ssize_t) — array indices, loop counters
- `idx32_t` (int32_t) — CSR index arrays (boundary_ptr, boundary_idx)
- `idx64_t` (int64_t) — large-graph indices (>2B nodes)
- `i32` (int32_t), `i64` (int64_t) — general integers
- `f32` (float), `f64` (double) — floating point

The mathematical core uses `f64` for all Laplacian, eigenvalue, and character
computation. `f32` exists for future GPU paths and visualization buffers.

---

### Error Codes

All `nogil` C-level functions return integer error codes rather than raising
Python exceptions (which require the GIL):

- `ERR_SUCCESS` (0) — completed successfully
- `ERR_MEMORY` (-1) — malloc/calloc returned NULL
- `ERR_INVALID_ARG` (-2) — parameter outside valid range
- `ERR_OUT_OF_BOUNDS` (-3) — index exceeds array dimension
- `ERR_OVERFLOW` (-4) — integer multiplication overflow
- `ERR_SHAPE_MISMATCH` (-5) — matrix dimensions incompatible
- `ERR_NOT_CONVERGED` (-6) — iterative solver failed
- `ERR_SINGULAR` (-7) — matrix is singular or degenerate
- `ERR_MEMORY_LIMIT` (-8) — allocation exceeds configured limit
- `ERR_CANCELLED` (-9) — operation cancelled

Python-level code converts these to typed exceptions via `raise_on_error()`:

```python
from rexgraph.core._common import raise_on_error, CoreMemoryError
raise_on_error(code, "build_RL")  # raises CoreMemoryError if code == -1
```

Exception hierarchy:

- `CoreError` — base for all core layer errors
- `CoreMemoryError(CoreError, MemoryError)` — allocation failure
- `CoreMemoryLimitError(CoreMemoryError)` — exceeds configured limit
- `CoreValueError(CoreError, ValueError)` — invalid argument
- `CoreOverflowError(CoreError, OverflowError)` — integer overflow

---

### Memory Management

The library enforces three memory ceilings, auto-detected from system RAM at
import time and reconfigurable at runtime:

- `max_parallel_buffer_bytes` (default: 25% of system RAM) — per-operation
  scratch for parallel loops
- `max_total_allocation_bytes` (default: 75% of system RAM) — global allocation
  ceiling
- `max_dense_allocation_bytes` (default: 25% of RAM, clamped to 100 MB - 4 GB)
  — single dense matrix ceiling, controls whether Laplacian construction uses
  dense BLAS (dgemm) or sparse scipy

These limits are checked by inline helpers that every module calls:

- `can_allocate_dense_f64(nrows, ncols)` — True if nrows x ncols x 8 bytes fits
  within the limit
- `should_use_dense_eigen(n)` — True if n <= `eigen_dense_limit` (default 2000)
- `should_use_dense_matmul(n)` — True if both the allocation fits AND the
  eigensolver will use the dense path

System memory is detected via psutil (preferred), `os.sysconf` (Linux fallback),
or 8 GB hardcoded fallback.

Runtime configuration:

```python
from rexgraph.core._common import configure_memory, configure_algorithms

# Allow larger dense matrices (for a machine with 64 GB RAM)
configure_memory(max_dense_allocation=8_000_000_000)

# Use dense eigensolver up to 5000x5000 matrices
configure_algorithms(eigen_dense_limit=5000)
```

Environment variable configuration (applied at import time):

```bash
REXGRAPH_MAX_DENSE_GB=4
REXGRAPH_EIGEN_DENSE_LIMIT=3000
REXGRAPH_DEFAULT_K=20
```

---

### Parallelization

OpenMP is detected at compile time via the `_OPENMP` preprocessor macro. When
unavailable, all OpenMP functions are replaced with stubs returning 1. The
`prange` import compiles without OpenMP and runs serially.

Parallelization decisions are made per-operation by inline helpers:

- `should_parallelize(work_size, threshold)` — True if OpenMP is available AND
  work_size >= max(threshold, 1000). The absolute minimum of 1000 prevents
  parallel overhead from dominating small operations.
- `should_parallelize_with_memory(work_size, threshold, memory_required)` —
  additionally checks that per-thread scratch fits in the parallel buffer limit.
- `get_num_threads(requested)` — returns the effective thread count after
  applying `max_threads_limit` and `reserved_threads` caps.

Default thresholds:

- Simple row-parallel loops: 50,000 elements
- CSR transpose: 500,000 elements
- Reduction operations: 100,000 elements

---

### Numeric Utilities

All helpers are `cdef inline ... noexcept nogil`, callable from any parallel
region with zero overhead:

Constants:

- `get_EPSILON_NORM()` -> 1e-10 (eigenvalue zero threshold)
- `get_EPSILON_DIV()` -> 1e-12 (division safety)
- `get_EPSILON_EQ()` -> 1e-12 (equality comparison)
- `get_PI()` -> 3.141592653589793

Clamping and sanitization:

- `clamp(x, lo, hi)` — clamp f64 to [lo, hi]
- `sanitize_float64(x)` — replace NaN/Inf with 0.0
- `is_near_zero(x, eps)` — True if |x| <= eps
- `log_clamp_min(x, min_x)` — log(max(x, min_x))
- `sqrt_clamp_min(x)` — sqrt(max(x, 0))

---

### Bit Operations

- `popcount_u64(x)` — population count (Hamming weight) of a uint64
- `next_power_of_two(x)` — smallest power of 2 >= x
- `is_power_of_two(x)` — True iff x is a power of 2

---

### Hash Functions

Two hash functions for internal hash tables (used in `_sparse`, `_cycles`,
`_joins`):

- `fnv1a_hash_u64(x)` — FNV-1a bytewise hash of a uint64
- `mix64(x)` — SplitMix64 finalizer (bijective mixing of 64 bits)

---

### CSR Utilities

Helpers for compressed sparse row format used in boundary operator
representation:

- `csr_needs_int64_indptr(nnz)` — True if nnz > 2^31 - 1
- `csr_row_length_i32(indptr, row)` — row length from int32 indptr
- `csr_row_length_i64(indptr, row)` — row length from int64 indptr

---

### Union-Find (Disjoint Set Union)

Two implementations for connected component computation:

`UnionFind` (int32, for graphs with < 2B nodes):

- `uf_init(uf, n)` -> ERR_SUCCESS or ERR_MEMORY
- `uf_find(uf, x)` -> root of x with path compression
- `uf_union(uf, x, y)` -> True if x and y were in different components
- `uf_component_count(uf)` -> number of connected components
- `uf_connected(uf, x, y)` -> True if same component
- `uf_free(uf)` -> release memory

`UnionFind64` (int64, for graphs with > 2B nodes): identical API with `uf64_`
prefix.

Both use union-by-rank with path compression, giving amortized O(alpha(n)) per
operation where alpha is the inverse Ackermann function.

---

### Sorted Array Set Operations

For CSR neighbor lists (sorted int32 arrays), these compute set operations in
O(|a| + |b|) time without allocation:

- `sorted_intersection_count_i32(a, len_a, b, len_b)` -> |a intersect b|
- `sorted_union_count_i32(a, len_a, b, len_b)` -> |a union b|
- `sorted_jaccard_i32(a, len_a, b, len_b)` -> |a intersect b| / |a union b|
- `sorted_intersection_write_i32(a, len_a, b, len_b, out, max_out)` -> writes
  intersection to out buffer

---

### Binary Search

For lookups in sorted CSR index arrays:

- `binary_search_i32(arr, n, target)` -> index of target, or -1
- `binary_search_contains_i32(arr, n, target)` -> True if found
- `lower_bound_i32(arr, n, target)` -> first index where arr[i] >= target
- `upper_bound_i32(arr, n, target)` -> first index where arr[i] > target

---

### Validation (Python-level)

For input validation at the Python/Cython boundary:

- `validate_csr_arrays(indptr, indices, data, name)` — checks shapes, dtypes,
  indptr[0]==0, len(indices)==indptr[-1]
- `validate_array_size(arr, name, min_size, max_size)` — bounds check
- `check_parallel_memory(op_name, threads, elements, element_size)` — raises
  CoreMemoryLimitError if parallel scratch exceeds limit
- `check_dense_allocation(op_name, nrows, ncols)` — raises if dense matrix
  exceeds limit

---

### Memory Estimation

For capacity planning before building large graphs:

```python
from rexgraph.core._common import estimate_memory_usage, estimate_dense_matrix_bytes

# How much memory for a graph with 1M nodes, 10M edges?
est = estimate_memory_usage(1_000_000, 10_000_000)
print(f"Total: {est['total_gb']:.2f} GB")

# Will a 5000x5000 dense Laplacian fit?
dense = estimate_dense_matrix_bytes(5000)
print(f"Dense: {dense['gb']:.2f} GB, fits: {dense['fits_in_limit']}")
```

---

### Diagnostics

```python
from rexgraph.core._common import get_build_info, get_configuration, test_parallel_execution

# Check build features
print(get_build_info())
# {'openmp_enabled': True, 'debug_enabled': False, 'max_threads': 16, 'compiled': True}

# Full runtime configuration
print(get_configuration())

# Verify OpenMP works
result = test_parallel_execution(1_000_000)
print(f"Threads: {result['threads_used']}, correct: {result['result_correct']}")
```

---

### Interop with Other Modules

`_common` is imported by every other module in `rexgraph/core/`. The `.pxd`
header is the C-level interface. `cimport` makes all inline functions, type
aliases, and constants available at zero runtime cost. The `.pyx` implementation
provides the Python-accessible configuration API and system detection logic.




## `_linalg` — LAPACK/BLAS Wrappers and RL Pipeline

**File:** `_linalg.pyx` (324 lines)

Provides Python-callable wrappers around the LAPACK and BLAS routines used
throughout rexgraph. Also contains the `rl_pipeline` function, which runs the
entire trace-normalize -> sum -> eigendecompose -> chi -> phi -> kappa
computation in a single C-level call with zero Python overhead in the hot path.

A static workspace buffer (`WORK_SIZE` doubles) is allocated once at module load
for dsyev_ calls, avoiding per-call allocation.

---

### Eigendecomposition

- `eigh(A)` -> (evals, evecs)

  Symmetric eigendecomposition via LAPACK dsyev_. Input must be square
  symmetric. Returns eigenvalues sorted ascending and eigenvectors as columns of
  a row-major array. Near-zero negative eigenvalues (|val| < 1e-10) are cleaned
  to 0.0.

---

### SVD

- `svd(A)` -> (U, S, Vt)

  General SVD via LAPACK dgesvd_. Returns U (m x m), S (min(m,n),), Vt (n x n)
  such that A = U @ diag(S) @ Vt.

---

### Least Squares

- `lstsq(A, b)` -> (x, rank)

  Least squares min ||A @ x - b||_2 via LAPACK dgelsd_. Returns the solution
  vector x and the numerical rank.

---

### Matrix Rank

- `matrix_rank(A, tol=1e-10)` -> int

  Numerical rank via SVD. Singular values below tol are treated as zero.

---

### Matrix Multiply (BLAS dgemm)

Three variants covering all transpose combinations:

- `gemm_nn(A, B)` -> A @ B
- `gemm_nt(A, B)` -> A @ B^T
- `gemm_tn(A, B)` -> A^T @ B

All call BLAS dgemm directly. No intermediate copies.

---

### Spectral Pseudoinverse

- `pinv_spectral(evals, evecs, tol=1e-10)` -> RL^+

  Computes the pseudoinverse from eigendecomposition: RL^+ = V diag(1/lambda_k)
  V^T, where the sum is over eigenvalues above tol. This avoids forming the full
  matrix and inverting it.

- `pinv_matvec(evals, evecs, x, tol=1e-10)` -> RL^+ @ x

  Applies the pseudoinverse to a vector without forming the full RL^+ matrix.
  Used in the sparse phi computation path where only RL^+ @ (B1^T e_v) is needed
  per vertex.

---

### rl_pipeline — Full RCF Computation

`rl_pipeline(B1, L1, L_O, L_SG)` -> dict

Runs the complete relational complex field computation in one call with all
operations at the C level (BLAS/LAPACK, no Python in the loop):

1. Trace-normalize L1, L_O, L_SG into hat operators h1, hO, hSG
2. Sum: RL = h1 + hO + hSG
3. Eigendecompose RL via dsyev_
4. Compute RL^+ via spectral pseudoinverse
5. Compute chi: chi(e, k) = hat_k[e,e] / RL[e,e]
6. Compute B1 @ RL^+ via dgemm
7. Compute S0 diagonal (vertex self-response)
8. Compute phi: for each hat k, diag(B1_RLp @ hat_k @ B1_RLp^T) / S0
9. Compute chi_star: mean of chi over incident edges
10. Compute kappa: 1 - 0.5 * ||phi - chi_star||_1

This is the 3-hat (RL3) variant used in the test suite and benchmarks. The
production path uses `_relational.build_RL` which handles N hats generically.

Returns a dict with: RL, evals, evecs, RLp, chi, phi, chi_star, kappa,
B1_RLp, S0_diag, hats, nhats.




## `_sparse` — Sparse Matrix Storage and Operations

**File:** `_sparse.pyx` (1407 lines)

DualCSR sparse format (CSR + CSC in one pass) with sorted indices, typed
variants (i32/i64 x f32/f64), and operations: matvec, rmatvec, Gram
products (A^T A, A A^T), diagonal extraction, dense conversion, scipy
interop, and memory reporting. Internal sorting uses iterative quicksort
with median-of-3 pivot and insertion sort fallback. All public functions
auto-dispatch by index and value type.

---

### Classes

- `CSRMatrix(row_ptr, col_idx, vals, nrow, ncol)`

  Compressed Sparse Row matrix backed by contiguous numpy arrays. Stores
  nrow, ncol, nnz, idx_bits (32 or 64), val_bits (32 or 64). Properties:
  row_ptr, col_idx, vals.

- `DualCSR(csr, col_ptr, row_idx, vals_csc)`

  Dual CSR/CSC storage built in a single pass through COO data. Exposes
  both row-major (CSR) and column-major (CSC) views for efficient row and
  column access. Properties: nrow, ncol, nnz, row_ptr, col_idx, vals,
  col_ptr, row_idx, vals_csc, idx_bits, val_bits.

---

### Construction

- `csr_from_coo(rows, cols, vals, nrow, ncol)` -> CSRMatrix

  Builds CSR from COO triplets. Duplicates summed. Rows sorted by column.

- `dual_from_coo(rows, cols, vals, nrow, ncol)` -> DualCSR

  Builds DualCSR (CSR + CSC) in a single pass. Both representations have
  sorted indices. This is the primary construction path used throughout
  rexgraph for B1, B2, and Laplacian storage.

- `dual_from_csr(csr)` -> DualCSR

  Adds CSC storage to an existing CSRMatrix.

---

### Type Selection

- `select_idx_bits(max_dim)` -> int — 32 if max_dim < 2^31, else 64
- `select_val_bits(nnz, force_64=False)` -> int — 64 if small or forced, 32 for large
- `aligned_empty_idx(n, use_64)`, `aligned_empty_val(n, use_64)` — typed allocation
- `aligned_zeros_idx(n, use_64)`, `aligned_zeros_val(n, use_64)`

---

### Matrix-Vector Products

- `matvec(A, x)` -> f64[nrow] — y = A @ x via CSR. Auto-dispatches type.
- `rmatvec(A, x)` -> f64[ncol] — y = A^T @ x via CSC. Auto-dispatches type.

  Both accept DualCSR and handle f32/f64 x i32/i64 combinations.

---

### Gram Products

- `spmm_AtA_dense_f64(A)` -> f64[ncol, ncol] — A^T A as dense matrix.
- `spmm_AAt_dense_f64(A)` -> f64[nrow, nrow] — A A^T as dense matrix.

  Both raise MemoryError if the output exceeds max_dense_allocation.

---

### Access and Extraction

- `diag(A)` -> f64[min(nrow, ncol)] — extract diagonal
- `row_entries(A, row)` -> (col_idx, vals) for a single row
- `col_entries(A, col)` -> (row_idx, vals) for a single column
- `row_nnz(A, row)` -> int
- `col_nnz(A, col)` -> int

---

### Conversion

- `to_dense_f64(A)` -> f64[nrow, ncol]

  Convert CSR or DualCSR to dense. Raises MemoryError if too large. Used by
  `graph.py` properties B1, B2, L0, L1, etc. for small matrices.

- `from_dense_f64(D, tol=-1.0)` -> DualCSR

  Convert dense matrix to DualCSR. Drops entries below tol. Used by
  `graph.py`'s `from_dense_f64` for rebuilding sparse storage.

- `to_scipy_csr(A)` -> scipy.sparse.csr_matrix
- `from_scipy_csr(sp_matrix)` -> DualCSR

---

### Memory

- `memory_bytes(A)` -> int — total bytes used by CSR + CSC arrays
- `memory_report(A)` -> str — human-readable memory usage
- `validate_csr(A, name="CSR")` -> (valid, error_msg) — structural integrity check




## `_rex` — Structural Operations for the Relational Complex

**File:** `_rex.pyx` (1230 lines)

Array-level kernels for constructing, modifying, querying, and projecting
relational complex structures at all dimension levels (vertices, edges, faces).
This module treats edges as the primitive objects: vertices are derived from edge
boundaries, and faces are attached via the B2 boundary operator.

Every function has i32 and i64 typed variants plus a dispatcher that selects by
dtype automatically.

---

### Edge Classification

Edges in a relational complex are classified by the size of their boundary
(the set of vertices they connect):

- `EDGE_STANDARD` (0) — boundary has exactly 2 distinct vertices (ordinary edge)
- `EDGE_SELF_LOOP` (1) — boundary has 1 unique vertex with multiplicity 2
- `EDGE_BRANCHING` (2) — boundary has 3+ distinct vertices (hyperedge)
- `EDGE_WITNESS` (3) — boundary has 1 vertex with multiplicity 1 (pendant)

Functions:

- `classify_edges_standard(nE, sources, targets)` — for simple graphs where
  every edge connects exactly 2 vertices. Returns int32 array of edge types.

- `classify_edges_general(nE, boundary_ptr, boundary_idx)` — for hypergraphs
  where edges can have arbitrary boundary sizes. Sorts boundary indices to count
  unique vertices. Returns (edge_types, boundary_sizes) as int32 arrays.

- `classify_edges(nE, sources, targets, boundary_ptr, boundary_idx)` — top-level
  dispatcher that picks the right variant.

---

### Vertex Derivation

In a relational complex, vertices are not stored explicitly. They are derived
from the edge boundary arrays. This function scans source/target arrays to find
the maximum vertex index, then computes degree statistics:

- `derive_vertex_set(nE, sources, targets)` -> (nV, degree, in_deg, out_deg)

All four return values are int32 arrays. nV is the number of vertices (max
index + 1). Degree counts are per-vertex.

---

### CSR Incidence Structures

These build compressed sparse row (CSR) lookup tables from the boundary
operators. They are used for hyperslice queries, neighbor traversal, and
coboundary computation throughout the library.

- `build_vertex_to_edge_csr(nV, nE, sources, targets)` -> (v2e_ptr, v2e_idx)

  For each vertex v, v2e_idx[v2e_ptr[v]:v2e_ptr[v+1]] gives the indices of all
  edges incident to v.

- `build_edge_to_face_csr(nE, nF, B2_col_ptr, B2_row_idx)` -> (e2f_ptr, e2f_idx)

  Transposes the B2 boundary operator (stored as CSC) into a CSR lookup. For
  each edge e, e2f_idx[e2f_ptr[e]:e2f_ptr[e+1]] gives the indices of all faces
  that have e on their boundary.

---

### Branching Edge Expansion

When a relational complex contains hyperedges (branching edges with 3+
boundary vertices), this function clique-expands them into standard 2-vertex
edges for use with the Hodge Laplacian machinery:

- `clique_expand_branching(nE, boundary_ptr, boundary_idx, edge_types)`
  -> (sources, targets, weights, parent_edge)

A branching edge with k boundary vertices produces C(k,2) standard edges, each
with weight 1/(k-1). The parent_edge array maps each expanded edge back to its
original hyperedge index.

---

### Hyperslice Queries

A hyperslice of a cell sigma returns its immediate neighborhood across
dimensions: what is below it (boundary), above it (coboundary), and lateral to
it (overlap neighbors at the same dimension).

- `hyperslice(0, v, ...)` -> (above_edges, lateral_neighbors)

  For a vertex v: above = incident edges, lateral = adjacent vertices.

- `hyperslice(1, e, ...)` -> (below_vertices, above_faces, lateral_edges)

  For an edge e: below = its two boundary vertices, above = faces containing it,
  lateral = edges sharing at least one vertex with e.

- `hyperslice(2, f, ...)` -> (below_edges, lateral_faces)

  For a face f: below = its boundary edges, lateral = faces sharing at least one
  boundary edge with f.

Individual typed functions are also available directly:

- `hyperslice_vertex_i32(v, v2e_ptr, v2e_idx, sources, targets)`
- `hyperslice_edge_i32(e, sources, targets, e2f_ptr, e2f_idx, v2e_ptr, v2e_idx)`
- `hyperslice_face_i32(f, nF, B2_col_ptr, B2_row_idx, e2f_ptr, e2f_idx)`

---

### Edge Insertion and Deletion

Mutation operations that produce new edge arrays without modifying the originals:

- `insert_edges(nV, nE, sources, targets, new_sources, new_targets)`
  -> (all_sources, all_targets, nV_new)

  Concatenates new edges onto existing arrays and extends the vertex set if any
  new vertex indices exceed the current nV.

- `delete_edges(nV, nE, sources, targets, delete_mask)`
  -> (new_sources, new_targets, nV_new, vertex_map, edge_map)

  Removes edges where delete_mask == 1. Vertices that become isolated (no
  remaining incident edges) are removed and the vertex indices are compacted.
  vertex_map[old_v] gives the new index (-1 if removed). edge_map[old_e] gives
  the new index (-1 if deleted).

---

### Dimensional Projection

Computes Betti numbers and Euler characteristic for projections of the complex
to lower dimensions:

- `project_to_dimension(target_dim, nV, nE, nF, rank_B1, rank_B2)`
  -> (betti[3], euler)

  At dimension 2: full complex. At dimension 1: edges and vertices only (faces
  ignored). At dimension 0: vertices only.

- `betti_deltas(nV, nE, nF, rank_B1, rank_B2)` -> dict with "2to1" and "1to0"

  How Betti numbers change when projecting down one dimension.

---

### Subsumption Embeddings

Constructors that embed standard graph formats into the relational complex
representation:

- `from_graph(nV, src, tgt)` -> (sources, targets, edge_types, nV)

  Simple graph to 1-rex. Self-loops are classified as EDGE_SELF_LOOP.

- `from_hypergraph(nV, hedge_ptr, hedge_idx)` -> (boundary_ptr, boundary_idx, edge_types)

  Hypergraph to branching 1-rex. Edges with 1 boundary vertex are WITNESS,
  2 are STANDARD, 3+ are BRANCHING.

- `from_simplicial_2complex(nV, edge_src, edge_tgt, tri_e0, tri_e1, tri_e2, tri_s0, tri_s1, tri_s2)`
  -> (B2_col_ptr, B2_row_idx, B2_vals)

  Simplicial 2-complex (graph + triangles) to 2-rex. Builds B2 in CSC format
  from triangle edge indices and orientation signs.

---

### Generalized k-Chain

For complexes beyond dimension 2 (tetrahedra, etc.):

- `build_Bk_from_cells(n_lower, cell_ptr, cell_idx, cell_signs)`
  -> (col_ptr, row_idx, vals, n_cells)

  Builds the k-th boundary operator Bk in CSC from a list of k-cells given as
  CSC-style pointers into a shared index array with orientation signs.

- `verify_chain_condition_Bk(n_rows_prev, Bkm1_rp, Bkm1_ci, Bkm1_v, Bk_cp, Bk_ri, Bk_v, tol)`
  -> (ok, max_error)

  Verifies B_{k-1} * B_k = 0 via sparse matrix-vector product on each column
  of B_k. Returns True and the maximum absolute entry in the product.

---

### Coboundary Queries

Direct coboundary lookups from the CSR incidence tables:

- `coboundary_vertex_i32(v, v2e_ptr, v2e_idx)` -> edge indices incident to v
- `coboundary_edge_i32(e, e2f_ptr, e2f_idx)` -> face indices incident to e

---

### Convenience

- `build_1rex(nV, nE, sources, targets)` -> dict with all derived structure
  (vertex set, degree, edge types, v2e CSR)




## `_laplacians` — Hodge Laplacians and Spectral Decomposition

**File:** `_laplacians.pyx` (925 lines)

Builds the Hodge Laplacians (L0, L1, L2) from boundary operators B1 and B2,
eigendecomposes them via LAPACK, and assembles the relational Laplacian. This is
the spectral foundation of the library: `graph.py`'s `spectral_bundle` calls
`build_all_laplacians()` once and caches the result.

All matrix products go through BLAS dgemm. All eigendecompositions go through
LAPACK dsyev_. No numpy.linalg calls.

For large graphs where the dense nE x nE path is infeasible, a sparse path
computes Betti numbers via union-find and the Fiedler value via matrix-free
ARPACK, without materializing any nE x nE or nV x nV matrix.

---

### Laplacian Construction

The chain complex V -> E -> F defines three Hodge Laplacians:

- `build_L0(B1)` -> L0 = B1 @ B1^T, shape (nV, nV)

  The vertex Laplacian. Its nullity is beta_0 (number of connected components).
  Its second-smallest eigenvalue is the algebraic connectivity (Fiedler value).

- `build_L1_down(B1)` -> L1_down = B1^T @ B1, shape (nE, nE)

  The downward edge Laplacian. Captures vertex-mediated coupling between edges.

- `build_L1_up(B2)` -> L1_up = B2 @ B2^T, shape (nE, nE)

  The upward edge Laplacian. Captures face-mediated coupling between edges.
  Returns a zero matrix if there are no faces.

- `build_L1_full(L1_down, L1_up)` -> L1 = L1_down + L1_up, shape (nE, nE)

  The full Hodge Laplacian on edges. Its kernel is the space of harmonic
  1-forms, whose dimension is beta_1 (the first Betti number). The Hodge
  decomposition splits any edge signal into gradient (from L1_down), curl (from
  L1_up), and harmonic components.

- `build_L2(B2)` -> L2 = B2^T @ B2, shape (nF, nF)

  The face Laplacian. Its nullity is beta_2.

---

### Eigendecomposition

- `eigen_symmetric(L)` -> (evals, evecs)

  Full eigendecomposition of a symmetric matrix via LAPACK dsyev_. Eigenvalues
  are returned in ascending order. Near-zero eigenvalues (|val| < 1e-12) are
  cleaned to exactly 0.0. Negative eigenvalues from numerical noise (|val| <
  1e-9) are also zeroed.

- `clean_eigenvalues(evals, tol=1e-10)` -> cleaned copy

- `fiedler_value(evals)` -> second-smallest eigenvalue (first eigenvalue > 1e-10)

- `fiedler_vector(evecs, evals)` -> eigenvector for the Fiedler value

---

### Diagonal Extraction

- `extract_diag_L1(B1, B2)` -> (diag_down, diag_up)

  Extracts the diagonals of L1_down and L1_up without forming the full matrices.
  diag_down[e] = sum_v B1[v,e]^2 (column sum of squares). diag_up[e] = sum_f
  B2[e,f]^2 (row sum of squares). Used for fast per-edge diagnostics without
  O(nE^2) memory.

---

### Composite Operators

- `build_L1_alpha(L1, L_O, alpha)` -> L1 + alpha * L_O

  The alpha-coupled Laplacian, mixing Hodge structure with overlap geometry.

- `build_Lambda(B1, L_O)` -> B1 @ L_O @ B1^T, shape (nV, nV)

  The vertex-level projection of the overlap Laplacian. Used in vertex character
  computation.

- `compute_coupling_constants(evals_L1, evals_L_O, beta1, nE)` -> (alpha_G, alpha_T)

  alpha_G = fiedler(L1) / fiedler(L_O) measures the geometric coupling between
  Hodge and overlap structure. alpha_T = beta_1 / nE measures the topological
  coupling (fraction of edges in harmonic cycles).

---

### Trace Normalization

- `trace_normalize(L)` -> (L_hat, trace_value)

  Returns L / tr(L) and the trace. If tr(L) < 1e-15, returns the zero matrix.
  This normalization is applied to each typed Laplacian before summing into the
  relational Laplacian: each hat operator contributes equally regardless of its
  absolute scale.

---

### build_all_laplacians

`build_all_laplacians(B1, B2, L_O, L_SG=None, L_C=None, auto_alpha=True, k=-1)`

This is the single function that computes everything. `graph.py` calls it once
per graph and caches the result dict. It:

1. Builds L0, L1_down, L1_up, L1_full, L2 via BLAS
2. Eigendecomposes each via LAPACK dsyev_
3. Computes Betti numbers from eigenvalue nullities
4. Eigenanalyzes the overlap Laplacian L_O
5. Computes coupling constants alpha_G and alpha_T
6. Builds K1 = |B1|^T @ |B1| (the overlap Gramian)
7. Builds L_C from the line graph of K1 (copath complex Laplacian)
8. Calls `_relational.build_RL` with all available typed Laplacians
9. Calls `_character.compute_chi` on the resulting RL

Parameters:

- `B1_in` — vertex-edge boundary operator (nV, nE)
- `B2_in` — edge-face boundary operator (nE, nF), or None
- `L_O_in` — overlap Laplacian (nE, nE), or None
- `L_SG_in` — frustration Laplacian (nE, nE), or None
- `L_C_in` — copath complex Laplacian (nE, nE), or None

Returns a dict with all Laplacians, eigenvalues, eigenvectors, Betti numbers,
coupling constants, the relational Laplacian RL, its hat operators, trace
values, hat names, and the structural character chi.

---

### Sparse Spectral Bundle

For large graphs where nE x nE dense matrices are infeasible, three functions
provide Betti numbers and the L0 Fiedler value without any dense edge-space
computation.

- `_sparse_betti(B1_in, B2_in, nV, nE, nF)` -> (beta0, beta1, beta2, rank_B1, rank_B2)

  Computes Betti numbers without eigendecomposition. beta_0 comes from
  `_cycles.cycle_space_dimension`, which uses the nogil union-find from
  `_common.UnionFind` in O(nE * alpha(nV)) time with zero scipy calls.
  beta_1 follows from the Euler relation. beta_2 uses sparse SVD on B2
  only when nF > 0. For faceless graphs, this is pure C.

  Accepts DualCSR, scipy sparse, or dense B1/B2.

- `_sparse_fiedler_L0(B1_in, nV, nE)` -> (fiedler_val, fiedler_vec, evals, evecs)

  Computes the Fiedler value and vector of L0 = B1 @ B1^T without
  materializing L0. For nV <= 2000, uses `spmm_AAt_dense_f64` from
  `_sparse` (BLAS kernel) plus LAPACK eigh. For nV > 2000, wraps
  the DualCSR in a `scipy.sparse.linalg.LinearOperator` whose matvec
  applies L0 via two Cython calls: `rmatvec(B1_dual, x)` for B1^T @ x
  through the CSC path, then `matvec(B1_dual, tmp)` for B1 @ tmp
  through the CSR path. ARPACK Lanczos finds the 6 smallest eigenvalues
  in `which='SM'` mode (no shift-invert, no matrix factorization).
  Memory: O(nE) for the existing DualCSR arrays, not O(nV^2) for dense L0.

  Accepts DualCSR (preferred, uses Cython matvec) or scipy sparse (fallback).

- `build_all_laplacians_sparse(B1_in, B2_in, nV, nE, nF)` -> dict

  Sparse spectral bundle entry point. Returns a dict with the same key
  set as `build_all_laplacians` so graph.py can dispatch transparently.
  Populates: beta0, beta1, beta2, fiedler_val_L0, fiedler_vec_L0,
  evals_L0, alpha_T. Sets edge-space operators (L1, L_O, RL, hats,
  chi, K1, L_C) to None. The `_sparse_mode` key is set to True.

  Use `subgraph()` to extract a manageable subset for full dense spectral
  analysis, or `build_quotient_from_sparse()` in `_quotient` to build
  quotient boundary operators directly from sparse parents.




## `_overlap` — Overlap Laplacian L_O

**File:** `_overlap.pyx` (530 lines)

Builds the overlap Laplacian L_O on the edge set, capturing geometric similarity
between edges based on shared boundary vertices. Two edges are similar when they
share endpoints. The similarity is normalized by overlap degree, which guarantees
L_O is PSD with eigenvalues in [0, 1].

The construction is:

    K = |B1|^T W |B1|          unsigned Gramian (shared vertex count)
    d_ov = rowsums(K)           overlap degree per edge
    S = D_ov^{-1/2} K D_ov^{-1/2}   normalized similarity
    L_O = I - S                 overlap Laplacian

K_ij counts the weighted number of vertices shared by edges i and j. W is an
optional diagonal matrix of per-vertex weights (default: uniform). The
normalization by D_ov^{-1/2} makes S a doubly stochastic-like matrix whose
entries are in [0, 1], so L_O has eigenvalues in [0, 1].

L_O enters the relational Laplacian as one of the typed hat operators. Its
Fiedler value also determines the geometric coupling constant alpha_G =
fiedler(L1) / fiedler(L_O).

---

### Algorithm

The Gramian K is built by vertex-driven pair enumeration. For each vertex v
with degree d, the d^2 pairs of incident edges all share v and each contributes
w_v to K. This runs in O(sum_v deg(v)^2) time.

Two paths:

- Dense (nE^2 fits in the dense allocation budget): fills K as an nE x nE
  array directly, then normalizes in-place. No COO intermediate.
- Sparse (large nE): accumulates COO triples, converts to scipy CSR, then
  normalizes via sparse diagonal multiplication.

The `method` parameter controls selection: "auto" (default) picks based on the
memory budget, "dense" forces dense, "sparse" forces sparse.

---

### Functions

- `build_L_O(nV, nE, sources, targets, method="auto", vertex_weights=None)` -> L_O

  The main entry point. Returns a dense ndarray (nE, nE) or scipy CSR depending
  on the method. L_O is symmetric PSD with eigenvalues in [0, 1].

  Parameters:
  - `nV, nE` — vertex and edge counts
  - `sources, targets` — int32 edge endpoint arrays (length nE)
  - `method` — "auto", "dense", or "sparse"
  - `vertex_weights` — optional float64 array (nV,) for weighted Gramian

- `build_overlap_adjacency(nV, nE, sources, targets, vertex_weights=None)` -> (S, d_ov)

  Returns the normalized similarity matrix S (nE, nE) and overlap degree vector
  d_ov (nE,). Always dense. Used by graph.py's `_overlap_bundle`.

- `build_overlap_pairs(nV, nE, sources, targets, topk=30, vertex_weights=None)` -> list of dict

  Returns the top-k most similar edge pairs by overlap similarity. Each entry
  has keys: edge_i, edge_j, similarity, shared (number of shared vertices).
  Sorted by descending similarity. Uses the sparse path internally to avoid
  materializing the full nE x nE matrix for large graphs.




## `_spectral` — Spectral Layout and Force-Directed Refinement

**File:** `_spectral.pyx` (794 lines)

Computes 2D vertex positions for visualization in two phases:

1. Spectral embedding from L0 eigenvectors (Fiedler vector and third
   eigenvector as x/y coordinates).
2. Force-directed refinement via Fruchterman-Reingold with Coulomb repulsion,
   Hooke attraction, and centering forces.

For graphs with more than 200 vertices, O(nV^2) all-pairs repulsion is replaced
by Barnes-Hut quadtree approximation at O(nV log nV) with opening angle theta =
0.5.

---

### spectral_layout

`spectral_layout(evecs, nV, width=700.0, height=500.0, pad=0.10, evals=None)`

Uses eigenvectors of the vertex Laplacian L0 to place vertices in 2D. Column 1
(Fiedler vector) becomes the x-coordinate, column 2 (third eigenvector) becomes
y. Coordinates are linearly rescaled to fit the canvas with padding on each
side.

Falls back to deterministic placement (golden-ratio-based grid) when fewer than
3 eigenvectors are available.

Parameters:

- `evecs` — eigenvectors of L0, shape (nV, k)
- `nV` — number of vertices
- `width, height` — canvas size in pixels
- `pad` — fractional padding on each side (default 10%)
- `evals` — eigenvalues for sorting eigenvectors by ascending eigenvalue

Returns (px, py) as float64 arrays of length nV.

---

### force_directed_refine

`force_directed_refine(px, py, edge_src, edge_tgt, nV, nE, iterations=400, ...)`

O(nV^2) Fruchterman-Reingold refinement. Each iteration computes:

- Coulomb repulsion between all vertex pairs: force proportional to
  1/distance^2, pushing apart
- Hooke attraction along edges: spring force proportional to
  (distance - ideal_length), pulling connected vertices together
- Centering force: pulls all vertices toward the canvas center

The step size decays linearly from 0.6 to 0 over the iteration count, providing
simulated annealing behavior.

Parameters:

- `px, py` — initial positions (modified in-place)
- `edge_src, edge_tgt` — int32 edge endpoint arrays
- `iterations` — number of force iterations (default 400)
- `repel_strength` — Coulomb constant (default 3000)
- `attract_ideal` — ideal edge length in pixels (default 50)
- `attract_strength` — spring constant (default 0.04)
- `centering` — centering force coefficient (default 0.008)
- `width, height` — canvas dimensions for boundary clamping

---

### barnes_hut_refine

`barnes_hut_refine(px, py, edge_src, edge_tgt, nV, nE, iterations=400, theta=0.5, ...)`

O(nV log nV) force-directed refinement using a quadtree for far-field repulsion
approximation. Same force model as `force_directed_refine`, but repulsion is
computed via Barnes-Hut tree traversal instead of all-pairs.

The quadtree is rebuilt each iteration from the current vertex positions. For
each vertex, the tree is traversed: if a cell's size divided by its distance to
the vertex is less than theta, the entire cell is treated as a single point mass
at its center of mass. Otherwise, the traversal recurses into children.

The step size decays as 2 * (1 - t/T)^1.5, which gives faster initial settling
than the linear decay in the naive method.

Additional parameter:

- `theta` — Barnes-Hut opening angle (default 0.5). Smaller values give more
  accurate but slower computation.

---

### compute_layout

`compute_layout(evecs, nV, nE, edge_src, edge_tgt, width=700.0, height=500.0, iterations=400, evals=None)`

Runs spectral embedding followed by force-directed refinement, automatically
selecting the naive or Barnes-Hut method based on vertex count:

- nV <= 200: O(nV^2) naive refinement
- nV > 200: O(nV log nV) Barnes-Hut refinement

This is the function called by `graph.py`'s `layout` property.

---

### Internal Details

The quadtree implementation (`QuadTree` struct) uses flat arrays for all node
data (center of mass, mass, cell bounds, child pointers, body index) allocated
in a single block. Maximum tree size is 4 * nV + 64 nodes. Depth is capped at
50 to prevent infinite recursion from coincident points.

Boundary clamping keeps all vertices within a 2% padding from the canvas edges
after each iteration.




## `_boundary` — Chain Complex Construction

**File:** `_boundary.pyx` (607 lines)

Assembles the boundary operators B1 (nV x nE) and B2 (nE x nF) that define the
chain complex V -> E -> F, verifies the chain condition B1 @ B2 = 0, and
computes Betti numbers from Laplacian eigenvalues.

B1 and B2 are stored as DualCSR matrices (from `_sparse`), which provide both
CSR and CSC access without duplication.

---

### B1 Construction

- `build_B1(nV, nE, sources, targets)` -> DualCSR (nV x nE)

  The signed vertex-edge incidence matrix. For each edge j with source s and
  target t: B1[s, j] = -1, B1[t, j] = +1. This convention orients edges from
  source to target.

  Column sums of B1 are zero (each column has exactly one -1 and one +1). Row
  sums give signed degree.

  Parameters:
  - `nV, nE` — vertex and edge counts
  - `sources, targets` — int32 or int64 edge endpoint arrays

  Dispatches to i32 or i64 variant based on dtype. Uses i64 when nV or nE
  exceeds 2^31 - 1.

---

### B2 Construction

- `build_B2_from_cycles(nE, cycle_edges, cycle_signs, cycle_lengths)` -> DualCSR (nE x nF)

  Builds the edge-face boundary operator from flat cycle data. Each face is
  defined by a list of boundary edge indices with orientation signs (+/-1).

  The input arrays are concatenated across faces: cycle_edges and cycle_signs
  hold all boundary edges for all faces sequentially, and cycle_lengths[f] gives
  how many edges face f has on its boundary.

  For a triangle face with edges [0, 1, 2] and signs [+1, +1, -1]:
  cycle_edges = [0, 1, 2], cycle_signs = [1.0, 1.0, -1.0], cycle_lengths = [3].

  Parameters:
  - `nE` — number of edges
  - `cycle_edges` — int32/int64 array of boundary edge indices, concatenated
  - `cycle_signs` — float64 array of orientation signs, concatenated
  - `cycle_lengths` — int32/int64 array of boundary sizes per face

- `build_B2_from_dense(nE, nF, matrix)` -> DualCSR (nE x nF)

  Builds B2 from a dense matrix. Entries are rounded to the nearest integer
  (-1, 0, or +1). Accepts (nE, nF) or (nF, nE) orientation; if the shape
  suggests transposition is needed, it is applied automatically.

---

### Chain Complex Verification

- `verify_chain_complex(B1, B2, tol=1e-10)` -> (ok, max_error)

  Checks B1 @ B2 = 0 by iterating over columns of B2, applying B1 via sparse
  matrix-vector product, and checking each result is zero. This avoids forming
  the dense nV x nF product.

  Returns True if max|B1 @ B2| < tol, along with the actual maximum absolute
  entry. This is the computational proof of the chain condition: the boundary of
  a boundary is zero.

---

### Betti Numbers from Eigenvalues

The spectral path computes Betti numbers in O(k) time from pre-computed
Laplacian eigenvalues, avoiding any matrix factorization:

- `count_zero_eigenvalues(evals, tol=1e-10)` -> int

  Counts eigenvalues at or below the tolerance. Expects sorted ascending input.
  beta_k = dim ker(L_k) = number of zero eigenvalues of L_k.

- `rank_from_eigenvalues(evals, full_dim, tol=1e-10)` -> int

  rank(A) = full_dim - count_zero(evals) where L = A^T A or A A^T.

- `betti_from_eigenvalues(evals_L0, evals_L1, evals_L2, nV, nE, nF, tol=1e-10)` -> dict

  Computes all Betti numbers from Laplacian spectra in one call. Returns a dict
  with beta0, beta1, beta2, rank_B1, rank_B2, euler_char, and euler_check (True
  if beta_0 - beta_1 + beta_2 = nV - nE + nF).

  Also computes beta1_rank_check: whether beta_1 from eigenvalue nullity matches
  beta_1 = nE - rank(B1) - rank(B2) from operator ranks.

---

### Unified Betti Interface

- `betti_numbers(B1, B2=None, evals_L0=None, evals_L1=None, evals_L2=None)` -> tuple

  Uses the spectral path when eigenvalue arrays are available, otherwise falls
  back to SVD-based rank computation (with a warning). Returns (beta_0, beta_1)
  for a 1-rex or (beta_0, beta_1, beta_2) for a 2-rex.

---

### SVD Rank (Legacy)

- `compute_rank(M, method="auto", tol=1e-10)` -> int

  Numerical rank via SVD. Slow for large matrices. Prefer
  `betti_from_eigenvalues` when Laplacian eigenvalues are available. The "auto"
  method selects dense SVD (via `_linalg`) when the matrix is small enough, or
  scipy sparse SVDs otherwise.




## `_hodge` — Hodge Decomposition of Edge Signals

**File:** `_hodge.pyx` (535 lines)

Decomposes an edge signal g into three mutually orthogonal components:

    g = B1^T phi + B2 psi + eta
        (gradient)  (curl)  (harmonic)

The gradient component lies in im(B1^T), the curl component lies in im(B2),
and the harmonic residual lies in ker(L1). Orthogonality holds when the chain
condition B1 @ B2 = 0 is satisfied. If the complex has self-loop faces, they
must be filtered from B2 before calling this module (graph.py handles this via
B2_hodge).

Potentials are recovered via pseudoinverse: phi = L0^+ (B1 g) and psi = L2^+
(B2^T g). The dense path uses LAPACK dgelsd (lstsq), the sparse path uses
scipy lsqr (iterative).

---

### Signal Construction

- `build_flow_signal(weights, edge_type_indices=None, negative_type_mask=None)` -> f64[nE]

  Builds an oriented edge signal from weights and type information. For each
  edge, the flow is weights[e] * sign, where sign is -1 if the edge's type is
  marked negative in the mask, +1 otherwise. This connects the CSV/JSON polarity
  classification to the Hodge decomposition: inhibition edges get negative flow.

- `normalize_signal(x)` -> f64[n]

  Scales a signal to [-1, 1] by dividing by the maximum absolute value. Returns
  zeros if the input is all-zero.

---

### Vertex Divergence and Face Curl

- `compute_divergence(B1, flow)` -> f64[nV]

  B1 @ g. Measures net outflow at each vertex. Zero for harmonic signals.

- `compute_face_curl(B2, flow)` -> f64[nF]

  B2^T @ g. Measures circulation around each face. Zero for gradient signals.

---

### Per-Edge Resistance Ratio

- `compute_rho(harm, flow)` -> f64[nE]

  rho(e) = |eta_e| / |g_e|, the fraction of each edge's signal that is
  harmonic. Values in [0, 1]. Zero where the original flow is zero. Edges with
  high rho carry signal that is neither gradient nor curl: they represent
  topological flow through independent cycles.

---

### Energy Decomposition

- `compute_energy_percentages(grad, curl, harm)` -> (pct_grad, pct_curl, pct_harm)

  ||g||^2 = ||grad||^2 + ||curl||^2 + ||harm||^2. Returns the three energy
  fractions summing to 1.0. A gradient-dominated signal flows along potential
  differences. A curl-dominated signal circulates around faces. A
  harmonic-dominated signal flows through cycles that are not face boundaries.

---

### Orthogonality Verification

- `check_orthogonality(grad, curl, harm)` -> dict

  Computes the absolute inner products between all three component pairs:
  grad_curl, grad_harm, curl_harm. Returns max_inner (largest of the three) and
  orthogonal (True if max_inner < 1e-6). Large values indicate the chain
  condition is violated, usually because self-loop faces were not filtered.

---

### Hodge Decomposition

- `hodge_decomposition(B1, B2, flow, L0=None, L2=None)` -> (grad, curl, harm)

  The core decomposition. Automatically selects the dense or sparse path based
  on matrix dimensions and the memory budget. Builds L0 and L2 internally if
  not provided.

  Parameters:
  - `B1` — DualCSR (nV, nE)
  - `B2` — DualCSR (nE, nF) or None. Should have self-loop faces filtered.
  - `flow` — f64[nE] edge signal
  - `L0` — precomputed vertex Laplacian (dense or sparse), or None
  - `L2` — precomputed face Laplacian (dense or sparse), or None

---

### build_hodge — Full Analysis

- `build_hodge(B1, B2, flow, L0=None, L2=None)` -> dict

  Runs the decomposition and computes all derived quantities. Returns a dict
  with:

  - `grad, curl, harm` — raw f64[nE] components
  - `grad_norm, curl_norm, harm_norm, flow_norm` — normalized to [-1, 1]
  - `rho` — per-edge harmonic resistance ratio
  - `pct_grad, pct_curl, pct_harm` — energy fractions
  - `divergence, div_norm` — vertex divergence and its normalization
  - `face_curl` — face curl B2^T g
  - `orthogonality` — inner product dict from check_orthogonality




## `_faces` — Face Classification, Extraction, and Metrics

**File:** `_faces.pyx` (1201 lines)

Classifies faces into proper and self-loop types, filters B2 for exact Hodge
decomposition, extracts per-face descriptors, computes structural metrics
relating faces to vertices and edges, and provides typed and context-based
face selection for building new chain complexes from structural criteria.

Self-loop faces arise when edges connect a vertex to itself (v -> v). Their B2
column has nonzero entries but B1 @ B2 != 0 for those columns, because the
boundary of a self-loop is v - v = 0 in the vertex chain group yet nonzero in
the edge chain. Filtering them out gives B2_hodge where B1 @ B2 = 0 holds
exactly. This filtering is what makes the Hodge decomposition produce orthogonal
components.

---

### Face Classification

- `classify_faces(B2, edge_src, edge_tgt)` -> dict

  Classifies each face as proper (2+ unique boundary vertices) or self-loop
  (single vertex). Returns:

  - `proper_mask` — bool[nF], True for proper faces
  - `self_loop_mask` — bool[nF], True for self-loop faces
  - `n_proper, n_self_loop` — counts
  - `proper_indices, self_loop_indices` — index arrays

- `filter_b2_hodge(B2_dense, proper_mask)` -> ndarray[nE, nF_hodge]

  Extracts only the proper-face columns from B2. The returned matrix satisfies
  B1 @ B2_hodge = 0.

---

### Vertex Face Count

- `vertex_face_count(B2, edge_src, edge_tgt, nV)` -> int32[nV]

  Counts the number of distinct faces incident to each vertex. Uses a
  generation-counter technique (last_seen[v] = face_index) to avoid per-face
  set allocation. O(nnz(B2)) time.

---

### Face Extraction

- `extract_faces(B2, edge_src, edge_tgt, vertex_names, edge_names, face_class=None)` -> list of dict

  Produces a descriptor for each face with:
  - `id` — string label (f1, f2, ...)
  - `boundary` — dict mapping edge names to orientation signs (+/-1)
  - `vertices` — sorted list of vertex names on the face boundary
  - `size` — number of boundary edges
  - `is_self_loop` — True if the face is a self-loop

---

### Face Metrics

- `compute_face_metrics_i32(B2, edge_src, edge_tgt, nV, nE, nF, vfc, rho)` -> dict

  Computes structural metrics in six phases, all in O(nnz(B2) + nE) total:

  Phase 1 — Face sizes (boundary edge count) and face vertex counts.

  Phase 2 — Per-edge contribution: for each edge, the average reciprocal face
  size (1/|boundary|) across its incident faces, and the average face size.

  Phase 3 — Boundary asymmetry: |fc(src) - fc(tgt)| / max(fc(src), fc(tgt)),
  where fc is the vertex face count. Measures how asymmetric a face
  distribution is across an edge's endpoints.

  Phase 4 — Per-vertex contribution: average reciprocal face-vertex-count and
  average face size across a vertex's incident faces.

  Phase 5 — Face concentration: coefficient of variation (CV = std/mean) of
  vertex face counts within each face. Uses Welford's single-pass algorithm to
  compute mean and variance in one traversal of B2 per face.

  Phase 6 — Pearson correlation between boundary asymmetry and harmonic
  resistance ratio (rho). Measures whether topologically asymmetric edges tend
  to carry more harmonic flow.

  Returns dict with:
  - `v_avg_contrib, v_total_contrib, v_avg_face_size` — f64[nV]
  - `e_avg_contrib, e_total_contrib, e_avg_face_size` — f64[nE]
  - `e_bnd_asym` — f64[nE], boundary asymmetry
  - `f_concentration` — f64[nF], face concentration (CV)
  - `v_tc_sum, e_tc_sum` — scalar sums
  - `asym_rho_corr` — Pearson correlation

---

### build_face_data — Combined Builder

- `build_face_data(B2, edge_src, edge_tgt, nV, vertex_names, edge_names, rho)` -> dict

  Runs classification, extraction, vertex face count, and metrics in one call.
  Returns:
  - `faces` — list of face descriptors
  - `face_class` — classification dict
  - `vertex_face_count` — int32[nV]
  - `metrics` — metrics dict

  Returns zero-filled arrays when nF = 0.

---

### Typed Face Selection

- `typed_face_selection(edge_type_labels, adj_ptr, adj_idx, adj_edge, nV, nE, n_types)` -> dict

  Enumerates all triangles in the 1-skeleton via sorted adjacency
  merge-intersection. A triangle is a realized face iff all three boundary
  edges share the same type label. Cross-type triangles become voids.

  Returns: nF_realized, nF_void, n_triangles, realized_edges (i32[nF*3]),
  realized_signs (f64[nF*3]), void_edges (i32[nF_void*3]), face_types
  (i32[nF_realized]).

  Called by `graph.py`'s `typed_face_selection` method, which builds a new
  RexGraph from the realized faces.

---

### Context Face Selection

- `context_face_selection(B1, context_matrix, adj_ptr, adj_idx, adj_edge, nV, nE)` -> dict

  Selects faces based on a context matrix (uint8[n_contexts, nV]). A
  triangle is realized iff at least one context covers all three boundary
  vertices (E = C^T |B1| > 0 per context row). Also computes per-context
  face counts and void fractions.

  Returns: nF, n_triangles, cycle_edges, cycle_signs, cycle_lengths,
  per_context_face_count (i32[n_contexts]), per_context_void_fraction
  (f64[n_contexts]).

  Called by `graph.py`'s `context_face_selection` method.

---

### Void Type Composition

- `void_type_composition(void_edges, edge_type_labels, nF_void, n_types)` -> dict

  Analyzes the edge-type composition of void triangles. For each void,
  identifies the set of distinct edge types present. Returns pair_counts
  (how many voids have each type combination), pair_fractions (normalized),
  and type_pairs (the distinct type sets).




## `_cycles` — Deterministic Fundamental Cycle Basis

**File:** `_cycles.pyx` (1033 lines)

Computes a fundamental cycle basis for the 1-skeleton of a relational complex
via tree-cotree decomposition. The output defines the face set and provides the
data needed to build B2 (the edge-face boundary operator).

The algorithm is deterministic: neighbor lists are sorted by vertex index before
BFS traversal, so the same graph always produces the same spanning tree and the
same cycle basis. This matters for reproducibility of the entire RCF pipeline,
since faces determine B2, which determines L1_up, which enters the Hodge
Laplacian and the relational Laplacian.

---

### Algorithm

1. Build symmetric undirected adjacency in CSR form from directed edge arrays.
   Each directed edge produces two adjacency entries. Neighbors within each row
   are sorted by vertex index.

2. BFS spanning forest with sorted neighbor traversal. Visits all connected
   components. Produces parent, parent_edge, depth, and is_tree arrays.

3. Identify cotree edges: the nE - nV + beta_0 edges not in the spanning tree.
   Each cotree edge closes exactly one fundamental cycle.

4. For each cotree edge (u, v), trace the cycle through the tree: u -> LCA(u,v)
   -> v -> u. The LCA (lowest common ancestor) is found by walking both
   endpoints up toward the root until they meet.

5. Assign orientation signs: +1 if the cycle traverses the edge from source to
   target, -1 if reversed.

The output format matches `build_B2_from_cycles` in `_boundary.pyx`:
concatenated edge indices, orientation signs, and per-face boundary lengths.

---

### Symmetric Adjacency

- `build_symmetric_adjacency(nV, nE, sources, targets)` -> (adj_ptr, adj_idx, adj_edge)

  CSR adjacency where each directed edge appears in both directions. Rows are
  sorted by neighbor vertex index. adj_edge[k] maps each adjacency entry back to
  the original directed edge index.

---

### BFS Spanning Forest

- `bfs_spanning_forest(adj_ptr, adj_idx, adj_edge, nV, nE)` -> (parent, parent_edge, depth, is_tree, n_components)

  Deterministic BFS from vertex 0. Produces one spanning tree per connected
  component. Returns:
  - `parent[v]` — parent vertex in BFS tree (root has parent[v] = v)
  - `parent_edge[v]` — edge connecting v to its parent (-1 for roots)
  - `depth[v]` — BFS distance from component root
  - `is_tree[e]` — 1 for spanning tree edges, 0 for cotree edges
  - `n_components` — number of connected components (beta_0)

---

### Fundamental Cycle Basis

- `find_fundamental_cycles(nV, nE, sources, targets)` -> (cycle_edges, cycle_signs, cycle_lengths, nF, n_components)

  The main entry point. Returns:
  - `cycle_edges` — int32 array of concatenated boundary edge indices
  - `cycle_signs` — float64 array of orientation signs (+/-1.0)
  - `cycle_lengths` — int32 array of boundary lengths per face
  - `nF` — number of fundamental cycles (= beta_1)
  - `n_components` — number of connected components (= beta_0)

  These arrays are passed directly to `_boundary.build_B2_from_cycles` to
  construct B2.

---

### Cycle Space Dimension

- `cycle_space_dimension(nV, nE, sources, targets)` -> int

  Computes beta_1 = nE - nV + beta_0 without tracing any cycles. Uses
  union-find for fast component counting. O(nE * alpha(nV)) time.

---

### Verification

- `verify_cycles_in_kernel(nV, nE, sources, targets, cycle_edges, cycle_signs, cycle_lengths, tol=1e-10)` -> (ok, max_error)

  Checks that every cycle lies in ker(B1): for each cycle, constructs the signed
  edge vector and verifies that B1 @ vector = 0 within tolerance. This is the
  computational proof that each cycle is a valid 1-boundary.

---

### Combined Builder

- `build_adjacency_and_forest(nV, nE, sources, targets)` -> (adj_ptr, adj_idx, adj_edge, parent, parent_edge, depth, is_tree, n_components)

  Exposes the symmetric adjacency and BFS forest together for inspection.




## `_character` — Structural Character Decomposition

**File:** `_character.pyx` (779 lines)

Computes the structural character of every edge and vertex in the relational
complex. The character decomposes each cell's identity into a probability
distribution over the typed Laplacian channels (Hodge, overlap, frustration,
copath). All hot paths use BLAS/LAPACK with zero Python overhead.

Compiled with `-fno-finite-math-only` to restore IEEE inf/nan semantics
(required for mixing time and anisotropy computations that return inf for
degenerate channels).

---

### chi — Edge Structural Character

- `compute_chi(RL, hats, nhats, nE)` -> f64[nE, nhats]

  chi(e, k) = hat_k[e,e] / RL[e,e]. For each edge, the diagonal of each hat
  operator divided by the RL diagonal gives the fraction of that edge's
  relational weight coming from each channel. The result is a probability vector
  on the nhats-simplex: chi(e) sums to 1 for every edge.

  When RL[e,e] is near zero (degenerate edge), chi defaults to uniform
  (1/nhats per channel).

---

### phi — Vertex Structural Character

- `compute_phi(B1, RL, hats, nhats, nV, nE, green_cache=None)` -> f64[nV, nhats]

  phi(v, k) = diag(B1 @ RL^+ @ hat_k @ RL^+ @ B1^T)[v] / diag(B1 @ RL^+ @ B1^T)[v].
  Projects the edge-level character to vertices through the Green's function
  RL^+ (pseudoinverse of RL). The denominator S0[v,v] = (B1 @ RL^+ @ B1^T)[v,v]
  is the vertex self-response.

  Dense path (via `compute_phi_dense`): computes B1 @ RL^+ once, then reuses it
  for all hat operators. All matrix products via BLAS dgemm.

  Sparse path (via `compute_phi_sparse_single`): solves RL x = B1^T e_v per
  vertex via LAPACK lstsq. Used when RL is too large for the dense path.

  When S0[v,v] is near zero (isolated vertex), phi defaults to uniform.

---

### chi_star — Star-Averaged Edge Character

- `compute_chi_star(chi, v2e_ptr, v2e_idx, nV, nhats)` -> f64[nV, nhats]

  chi_star(v) = mean of chi(e) over all edges incident to v. This lifts edge
  character to vertices by simple averaging over the star neighborhood.

---

### kappa — Cross-Dimensional Coherence

- `compute_kappa(phi, chi_star, nV, nhats)` -> f64[nV]

  kappa(v) = 1 - 0.5 * ||phi(v) - chi_star(v)||_1. Measures agreement between
  the vertex's intrinsic character (phi, from the Green's function) and the
  character it would have if it simply inherited from its incident edges
  (chi_star). Values in [0, 1]: kappa = 1 means perfect agreement, kappa near
  0 means the vertex sees its structural role very differently from its edges.

---

### build_character_bundle

- `build_character_bundle(B1, RL, hats, nhats, nV, nE, v2e_ptr, v2e_idx, green_cache=None)` -> dict

  Computes chi, phi, chi_star, kappa in one call. Returns a dict with all four
  arrays. This is called by graph.py's character property.

---

### Hat Eigendecomposition

- `hat_eigen(hat, nE)` -> (evals f64[nE], evecs f64[nE, nE])

  Eigendecompose a single trace-normalized hat operator via LAPACK dsyev_.
  Eigenvalues are ascending, near-zero values cleaned to exactly 0.0.
  Eigenvectors are returned as columns of a contiguous array.

- `hat_eigen_all(hats, nhats, nE)` -> list of (evals, evecs)

  Eigendecompose all hat operators. Results are cached in graph.py's
  `_hat_eigen_bundle` property and reused by `per_channel_mixing_times`
  and `primal_signal_character`.

---

### Per-Channel Mixing Time

- `per_channel_mixing_time(hat_evals, nE)` -> float

  mu_X = ln(nE) / lambda_2(hat_L_X). Takes pre-computed eigenvalues from
  `hat_eigen`. Returns inf if no spectral gap (lambda_2 < 1e-10) or nE <= 1.

- `per_channel_mixing_times_from_evals(hat_evals_list, nhats, nE)` -> f64[nhats]

  Per-channel mixing times from pre-computed eigenvalue lists. Avoids
  redundant eigendecomposition when hat eigendata is already cached.

- `per_channel_mixing_times(hats, nhats, nE)` -> f64[nhats]

  Convenience wrapper that eigendecomposes each hat internally. Used when
  hat eigendata is not cached.

---

### Mixing Time Anisotropy

- `mixing_time_anisotropy(times, nhats)` -> dict

  Computes pairwise ratios of per-channel mixing times. Returns:
  - `ratios` — f64[nhats, nhats], ratios[i,j] = times[i] / times[j]
  - `dominant_channel` — channel with smallest finite mixing time
  - `slowest_channel` — channel with largest finite mixing time
  - `anisotropy` — ratio of slowest to fastest (inf if any channel has no gap)

---

### Face-Void Dipole

- `face_void_dipole(psi, B2, Bvoid, nE, nF)` -> dict

  Projects an edge signal onto the realized face basis (B2 columns) and
  the void basis (Bvoid columns). face_affinity = sum of squared
  projections onto B2 columns, normalized by ||psi||^2. void_affinity
  is the same for Bvoid. dipole_ratio = (face - void) / (face + void),
  in [-1, 1]. Returns dict with face_affinity (>= 0), void_affinity
  (>= 0), dipole_ratio, total_projection.

  Called by `graph.py`'s `face_void_dipole` method.

---

### Per-Vertex Curvature

- `per_vertex_curvature(phi, chi_star, kappa, nV, nhats)` -> f64[nV]

  ||phi(v) - chi_star(v)||_2^2 / kappa(v). Squared L2 deviation between vertex
  and star character, scaled by coherence. High curvature means the vertex is a
  boundary between structurally different regions.

- `per_vertex_weighted_curvature(chi, edge_weights, sources, targets, nV, nE, nhats)` -> f64[nV]

  (1/deg(v)) * sum_{e in star(v)} w(e) * ||chi(e) - 1/nhats||_1. Measures how
  far a vertex's incident edges deviate from uniform character, weighted by edge
  importance.

---

### Structural Summary

- `structural_summary(chi, phi, kappa, nE, nV, nhats)` -> dict

  Mean/std of chi per channel, mean/std of kappa, count of low-kappa vertices,
  dominant channel counts.

---

### Topological Integrity

- `topological_integrity(chi, edge_weights, nE, nhats)` -> dict

  IT = sum_e w(e) * chi(e, 0) (topological channel weight), IF = sum_e w(e) *
  chi(e, 2) (frustration channel weight). Regime is "NORMAL" if IT > IF,
  "INVERTED" otherwise. Measures whether the graph is topologically or
  frustration-dominated.

---

### Structural Entropy

- `structural_entropy(chi, nE, nhats)` -> float

  Shannon entropy of the mean character across edges: H = -sum_k chi_bar_k *
  ln(chi_bar_k). Maximum entropy = ln(nhats) when the mean character is uniform.
  Low entropy means one channel dominates the graph's structure.

---

### RL-Derived Operators

- `self_response(RLp, nE)` -> f64[nE]

  R_self(e) = RL^+[e,e]. Per-edge effective resistance. Higher values mean the
  edge is more structurally isolated.

- `signed_cosine_matrix(RL, nE)` -> f64[nE, nE]

  cos_s(i,j) = RL[i,j] / sqrt(RL[i,i] * RL[j,j]). Measures structural coupling
  between edge pairs through the relational Laplacian.

- `mixing_time(rl_evals, nE)` -> float

  tau_mix = ln(nE) / lambda_2 where lambda_2 is the smallest positive eigenvalue
  of RL. Smaller mixing time means faster information propagation across the
  graph's relational structure.

---

### Derived Constants

- `derived_constants(nV)` -> dict

  H0 = 1.0, amp_coeff = (nV-1)/nV, scaffold_floor = 1/nV^2,
  probe_floor = 1/nV^3. Used in signal propagation and perturbation analysis.




## `_relational` — Relational Laplacian and Green Function

**File:** `_relational.pyx` (325 lines)

Builds the relational Laplacian RL from N typed Laplacians (Hodge, overlap,
frustration, copath), computes its eigendecomposition and pseudoinverse, and
assembles the Green function cache used by `_character` for vertex character
computation. Also constructs the line graph of the overlap Gramian and the
copath complex Laplacian L_C. All hot paths use BLAS/LAPACK via `_linalg`
with zero Python overhead.

---

### Trace Normalization

- `trace_normalize(L)` -> (L_hat, trace_value)

  Returns L / tr(L) and the trace. If tr(L) < 1e-15, returns the zero matrix
  and trace 0. This normalization makes each typed Laplacian contribute equally
  to RL regardless of absolute scale.

  The C-level helper `_trace_normalize_inplace` operates directly on a memory
  buffer without allocation.

---

### Building RL

- `build_RL(laplacians, names)` -> dict

  The main entry point for constructing the relational Laplacian from an
  arbitrary number of typed Laplacians. Each input is trace-normalized. Those
  with trace below 1e-15 are dropped. RL is the sum of the surviving hat
  operators, so tr(RL) = nhats (the number of active hats).

  For N=3 and N=4, C-level fast paths (`_build_RL_3`, `_build_RL_4`) copy all
  inputs, normalize, and sum in a single nogil block with no Python calls.
  The general path handles any N through a Python loop with per-operator
  C-level normalization.

  Returns a dict with:
  - `RL` -- f64[nE, nE], the relational Laplacian
  - `hats` -- list of f64[nE, nE], the active hat operators
  - `nhats` -- int, number of active hats (= tr(RL))
  - `trace_values` -- f64 array of original traces before normalization
  - `hat_names` -- list of str labels for each active hat

  Called by `_laplacians.build_all_laplacians`, which collects L1, L_O, L_SG,
  and optionally L_C before passing them here.

- `build_RL_from_laplacians(L1, L_O, L_SG)` -> dict

  Convenience wrapper that calls `build_RL` with the three standard Laplacians
  and names `['L1_down', 'L_O', 'L_SG']`.

---

### Eigendecomposition

- `rl_eigen(RL)` -> (evals, evecs)

  Symmetric eigendecomposition of RL via LAPACK dsyev_. Eigenvalues are sorted
  ascending. Small negative eigenvalues from numerical noise (|val| < 1e-10)
  are cleaned to 0.0. Values below 1e-12 in absolute value are also zeroed.

  Called by `graph.py`'s `_rl_eigen` property.

---

### Pseudoinverse

- `rl_pinv_dense(evals, evecs)` -> f64[nE, nE]

  Spectral pseudoinverse: RL^+ = V diag(1/lambda_k) V^T summed over
  eigenvalues above 1e-10. Builds the full nE x nE matrix.

- `rl_pinv_matvec(evals, evecs, x)` -> f64[nE]

  Applies RL^+ to a vector without forming the full matrix:
  RL^+ x = sum_k (1/lambda_k) (v_k^T x) v_k. Used in the sparse phi
  computation path where only RL^+ @ (B1^T e_v) is needed per vertex.

---

### Green Function Cache

- `build_green_cache(RL, B1, evals, evecs)` -> dict

  Precomputes the matrices needed for vertex character (phi) and related
  quantities in a single call:

  1. RL^+ via spectral pseudoinverse
  2. B1_RLp = B1 @ RL^+ via BLAS dgemm
  3. S0 = B1_RLp @ B1^T via BLAS dgemm (vertex self-response matrix)

  Returns a dict with: RL_pinv, B1_RLp, S0, evals, evecs, nV, nE, dense.
  Called by `graph.py`'s `_green_cache` property, which passes the result to
  `_character.build_character_bundle`.

---

### Linear Solve

- `rl_cg_solve(RL, b)` -> f64[nE]

  Solves RL x = b via LAPACK dgelsd_ (least squares). Despite the name, this
  uses a direct factorization, not conjugate gradient.

- `rl_solve_column(RL, B1, vertex_idx)` -> f64[nE]

  Solves RL x = B1[vertex_idx, :] for a single vertex. Extracts the row of
  B1 as the right-hand side and calls `rl_cg_solve`. Used in the sparse path
  for per-vertex phi computation.

---

### Line Graph and Copath Laplacian

- `build_line_graph(K1, nE)` -> dict

  Constructs the line graph of the overlap Gramian K1 = |B1|^T |B1|. Two
  edges i and j in the original graph are connected in the line graph when
  K1[i,j] > 0 (they share at least one vertex). Edge weights in the line
  graph are K1[i,j].

  Returns a dict with: src, tgt (int32 edge endpoints), weights (f64),
  nV_L (= nE, the vertex count of the line graph), nE_L (edge count).

- `build_L_coPC(line_graph_info)` -> f64[nE, nE]

  Builds the copath complex Laplacian L_C = B1_L^T @ B1_L where B1_L is
  the signed incidence matrix of the line graph. Returns the zero matrix if
  the line graph has no edges (happens when no two edges share a vertex, for
  example a matching).

  L_C is the fourth hat operator in RL4 (when available). It captures
  higher-order path structure beyond what L1, L_O, and L_SG measure.




## `_frustration` — Frustration Laplacian L_SG

**File:** `_frustration.pyx` (278 lines)

Builds the frustration Laplacian L_SG from edge signs and inverse-log-degree
vertex weights. L_SG captures signed coupling between edges: two edges sharing
a vertex contribute positively or negatively to the Gramian depending on
their signs. This is the third typed Laplacian in the standard RL3
construction.

The construction is:

    w(v) = 1 / log(deg(v) + e)           inverse-log-degree vertex weight
    K_s = B1^T diag(w) B1                signed weighted Gramian
    K_off = K_s with diagonal zeroed
    L_SG = D_{|K_off|} - K_off           frustration Laplacian

K_s[i,j] accumulates sign(i) * sign(j) * w(v) over all vertices v shared by
edges i and j. The diagonal K_s[i,i] = sum of w(v) over boundary vertices of
edge i (signs cancel on the diagonal). The frustration Laplacian L_SG is then
the standard graph Laplacian of the absolute off-diagonal coupling matrix.

L_SG is symmetric and PSD. When all edge signs are +1 (no frustration), L_SG
reduces to a degree-weighted graph Laplacian on the edge adjacency graph. When
some edges carry -1 signs (inhibition, repression), L_SG captures how sign
conflicts distribute across the graph.

---

### Vertex Weights

- `build_vertex_weights(nV, nE, sources, targets)` -> f64[nV]

  w(v) = 1 / log(deg(v) + e), where deg(v) is the undirected degree and e is
  Euler's number. High-degree vertices contribute less per-edge, which prevents
  hub vertices from dominating the Gramian. The log-based weighting is smoother
  than inverse degree.

- `build_vertex_weights_i64(nV, nE, sources, targets)` -> f64[nV]

  Same computation with int64 edge arrays.

---

### Signed Gramian

- `build_signed_gramian_dense(nV, nE, sources, targets, signs, vertex_weights)` -> f64[nE, nE]

  Builds K_s via vertex-driven pair enumeration. For each vertex v with
  incident edges {e_1, ..., e_d}, adds w(v) to the diagonal for each incident
  edge, and w(v) * sign(e_i) * sign(e_j) to each off-diagonal pair. Runs in
  O(sum_v deg(v)^2) time.

  Internally builds a vertex-to-edge CSR index for traversal.

---

### Frustration Laplacian

- `build_L_SG_dense(nV, nE, sources, targets, signs, vertex_weights)` -> f64[nE, nE]

  Builds K_s, zeros the diagonal to get K_off, then computes
  L_SG[i,i] = sum_j |K_off[i,j]| and L_SG[i,j] = -K_off[i,j] for i != j.

- `build_L_SG_sparse(nV, nE, sources, targets, signs, vertex_weights)` -> f64[nE, nE]

  Currently calls the dense path. The sparse path exists for API symmetry
  with `_overlap`.

- `build_L_SG(nV, nE, sources, targets, signs=None, method="auto")` -> f64[nE, nE]

  The main entry point. Computes vertex weights internally. If signs is None,
  defaults to all +1.

  Parameters:
  - `nV, nE` -- vertex and edge counts
  - `sources, targets` -- int32 edge endpoint arrays
  - `signs` -- optional f64[nE] of +1/-1 per edge (default: all +1)
  - `method` -- "auto", "dense", or "sparse"

  Called by `graph.py`'s `L_frustration` property and by
  `_laplacians.build_all_laplacians` during spectral bundle construction.

---

### Frustration Rate

- `frustration_rate(signs, edge_types, nE, n_types)` -> f64[n_types]

  Fraction of negative-signed edges per edge type. For each type t,
  rate[t] = (number of edges with type t and sign -1) / (total edges with
  type t). Returns 0.0 for types with no edges.




## `_signal` — Perturbation Analysis Pipeline

**File:** `_signal.pyx` (936 lines)

End-to-end pipeline for perturbation analysis on the rex chain complex.
Orchestrates calls to `_state`, `_field`, `_hodge`, `_temporal`, and
`_wave` to produce a complete signal analysis from a single perturbation
input. Field states live on (E, F) only; vertex observables are derived
via f_V = B1 f_E.

The pipeline runs in six stages: perturbation construction, propagation
(spectral diffusion), energy decomposition (E_kin/E_pot per timestep and
per edge), cascade analysis (activation order and face emergence), temporal
tagging (BIOES from energy ratio), and Hodge decomposition of initial and
final states.

---

### Perturbation Construction

- `build_edge_perturbation(nE, nF, edge_idx, amplitude=1.0)` -> (f_E, f_F)

  Dirac delta on a single edge: f_E[edge_idx] = amplitude, all other entries
  zero. f_F is always zero. This is the simplest perturbation type.

- `build_vertex_perturbation(vertex_idx, B1_T, nE, nF)` -> (f_E, f_F)

  Vertex perturbation converted to edge signal via B1^T. Perturbing vertex v
  means activating all edges incident to v: f_E = B1^T delta_v. The B1_T
  argument is the transpose of the boundary operator (nE, nV).

- `build_multi_edge_perturbation(nE, nF, edge_indices, amplitudes)` -> (f_E, f_F)

  Superposition of Dirac deltas on multiple edges with independent amplitudes.
  f_E[edge_indices[k]] += amplitudes[k] for each k. Models simultaneous
  perturbations at multiple sites.

- `build_spectral_perturbation(nE, nF, evecs_RL1, mode_idx, amplitude=1.0)` -> (f_E, f_F)

  Perturbation along a single eigenmode of RL. f_E = amplitude * evecs[:, mode_idx].
  Low modes are smooth (topological), high modes are rough (geometric).

---

### Propagation

- `propagate_diffusion(f_E, L_operator, evals, evecs, times)` -> f64[T, nE]

  Heat equation on edges: f(t) = exp(-L t) f(0) via spectral decomposition.
  Computes spectral coefficients c_k = v_k^T f(0) once, then for each
  timestep t applies f(t) = sum_k c_k exp(-lambda_k t) v_k. Works with any
  edge-space Laplacian (L1, L_O, or RL).

- `propagate_diffusion_comparative(f_E, L1, LO, RL1, evals_L1, evecs_L1, evals_LO, evecs_LO, evals_RL1, evecs_RL1, times)` -> (traj_L1, traj_LO, traj_RL1)

  Runs diffusion under L1, L_O, and RL simultaneously for comparison. Returns
  three trajectory arrays showing how the same perturbation behaves under
  topological-only, geometric-only, and combined dynamics.

---

### Energy Decomposition

- `energy_trajectory(trajectory, L1, LO)` -> (E_kin, E_pot, ratio, norms)

  Computes E_kin(t) = f(t)^T L1 f(t) (topological energy) and
  E_pot(t) = f(t)^T L_O f(t) (geometric energy) at each timestep. The ratio
  E_kin/E_pot is capped at 1e15 when E_pot is near zero. norms[t] = ||f(t)||^2.

- `hodge_energy_decomposition(f_E, B1, B2, L0, L2, L1)` -> (grad, curl, harm, E_grad, E_curl, E_harm, pct_grad, pct_curl, pct_harm)

  Hodge-decomposes an edge signal and computes energy per component:
  E_grad = grad^T L1 grad, E_curl = curl^T L1 curl, E_harm = harm^T L1 harm
  (E_harm should be near 0 since L1 harm = 0). Also returns norm-squared
  percentages.

- `per_edge_energy_trajectory(trajectory, L1, LO)` -> (Ekin_per_edge, Epot_per_edge)

  Per-edge energy at each timestep: E_kin_e(t) = f_e(t) * (L1 f(t))_e.
  These per-edge values sum to the total E_kin and E_pot.

---

### Cascade Analysis

- `cascade_from_edge(trajectory, threshold=-1.0)` -> (activation_time, activation_order, activation_rank, threshold_used)

  Computes cascade activation order from an edge trajectory. An edge is
  activated at the first timestep its absolute signal exceeds the threshold.
  If threshold is negative, auto-computed as 0.5% of peak signal magnitude.

  Returns per-edge activation times (-1 for never-activated), the sorted
  activation order, per-edge rank in the order, and the threshold used.

- `face_emergence(trajectory, B2, threshold=-1.0)` -> (face_activation_time, face_order)

  Tracks when faces activate during propagation. A face is active when the
  minimum |signal| across all its boundary edges exceeds the threshold. This
  models higher-order emergence: a face becomes active only when all its
  edges are active.

- `cascade_depth(activation_order, edge_src, edge_tgt, nE)` -> i32[nE]

  Topological distance from the perturbation source via BFS on the edge
  adjacency graph (edges sharing a vertex). Depth 0 = perturbed edge,
  depth 1 = star neighborhood, depth 2 = star of star, etc. Returns -1 for
  unreached edges.

---

### Temporal Tagging

- `tag_energy_phases(E_kin, E_pot, ratio_tol=0.2, min_phase_len=2, floor=1e-12)` -> (tags, phase_start, phase_end, phase_regime, log_ratios, crossover_times)

  BIOES tagging from energy ratio timeseries. Tags each timestep as B(egin),
  I(nside), O(utside), E(nd), or S(ingle). Phase regime is 0=kinetic,
  1=crossover, 2=potential based on log(E_kin/E_pot) relative to ratio_tol.
  Wraps `_temporal.compute_bioes_energy`.

- `tag_cascade_phases(activation_time, T)` -> (step_tags, new_per_step)

  Tags each timestep by cascade activity: 0 = quiet (no new activations),
  1 = wavefront advancing, 2 = peak activation step. new_per_step counts
  newly activated edges per timestep.

---

### Full Pipeline

- `analyze_perturbation(f_E, f_F, L1, LO, evals_RL1, evecs_RL1, B1, B2, times, ...)` -> dict

  One-call entry point that runs all six stages. Propagates f_E under RL
  diffusion, computes energy trajectory, per-edge energy, cascade activation,
  face emergence, BIOES tags, and Hodge decomposition of initial and final
  states. Also derives vertex observables via B1.

  Called by `graph.py`'s `analyze_perturbation` method.

  Returns a dict with: trajectory, E_kin, E_pot, ratio, norms,
  Ekin_per_edge, Epot_per_edge, activation_time, activation_order,
  activation_rank, face_activation_time, face_order, bioes_tags,
  phase_start, phase_end, phase_regime, log_ratios, crossover_times,
  hodge_initial, hodge_final, cascade_depth, f_V_initial, f_V_final.

- `analyze_perturbation_field(f_E, f_F, M_field, evals_M, evecs_M, freqs_M, L1, LO, B1, times, nE, nF, mode="diffusion")` -> dict

  Perturbation analysis using the full (E, F) field operator M from
  `_field.pyx`. Propagates the packed field state F = [f_E, f_F] under M,
  then extracts per-dimension trajectories and energy decomposition.

  In wave mode, also computes wave kinetic/potential energy and verifies
  energy conservation.

  Called by `graph.py`'s `analyze_perturbation_field` method.




## `_dirac` — Dirac Operator and Graded State Evolution

**File:** `_dirac.pyx` (321 lines)

Builds the Dirac operator D = d + d* on the graded cell space
R^nV + R^nE + R^nF. D encodes the full chain complex in a single real
symmetric matrix. Its square D^2 = blkdiag(L0, L1, L2) when B1 @ B2 = 0.
Schrodinger evolution exp(-iDt) preserves ||Psi||^2 exactly and couples
all three dimensional sectors (vertices, edges, faces) simultaneously.

---

### Dirac Operator Construction

- `build_dirac_operator(B1, B2)` -> (D, sizes)

  Assembles the (nV + nE + nF) x (nV + nE + nF) block matrix:

      D = [[ 0,     B1,    0    ],
           [ B1^T,  0,     B2   ],
           [ 0,     B2^T,  0    ]]

  D is real symmetric by construction. The off-diagonal blocks place B1 and
  B2 (and their transposes) in the appropriate positions. Returns D and the
  tuple (nV, nE, nF).

  Called by `graph.py`'s `dirac_operator` property, which passes B1 and
  B2_hodge (self-loop faces filtered).

---

### Eigendecomposition

- `dirac_eigen(D)` -> (evals, evecs)

  Full eigendecomposition via LAPACK dsyev_. Eigenvalues are sorted ascending
  and can be positive or negative (D is not PSD). Eigenvectors are returned
  as columns of a row-major array.

  Called by `graph.py`'s `_dirac_eigen` cached property.

---

### D^2 Verification

- `verify_d_squared(D, L0, L1, L2, nV, nE, nF, tol=1e-10)` -> (is_valid, max_error)

  Checks that D^2 = blkdiag(L0, L1, L2). The off-diagonal blocks of D^2
  vanish because B1 @ B2 = 0. Returns True if the maximum absolute entry
  in (D^2 - expected) is below tol.

---

### Schrodinger Evolution

- `schrodinger_evolve(evals, evecs, psi0, t)` -> (psi_re, psi_im)

  Evolves a graded state under exp(-iDt). Since D is real symmetric and psi0
  is real, the result separates into real and imaginary parts:

      psi_re(t) = sum_j cos(lambda_j t) c_j phi_j
      psi_im(t) = -sum_j sin(lambda_j t) c_j phi_j

  where c_j = phi_j^T psi0 are the spectral coefficients. ||psi_re||^2 +
  ||psi_im||^2 = ||psi0||^2 is preserved exactly.

  Called by `graph.py`'s `graded_state` method.

- `schrodinger_trajectory(evals, evecs, psi0, times)` -> (traj_re, traj_im, born)

  Evolves at multiple timepoints. traj_re and traj_im are f64[T, N] arrays.
  born[t, k] = |psi_re(t,k)|^2 + |psi_im(t,k)|^2 is the Born probability
  per cell at each timestep.

  Called by `graph.py`'s `graded_trajectory` method.

---

### Canonical Collapse

- `canonical_collapse(B1, nV, nE, nF, vertex_idx)` -> f64[N]

  The canonical graded state from observing vertex v:

      psi = (delta_v, B1^T delta_v, 0) / ||...||

  The vertex component is a Dirac delta at v. The edge component is B1^T
  applied to that delta (activating all incident edges with orientation
  signs). The face component is exactly zero because B2^T B1^T delta_v =
  (B1 B2)^T delta_v = 0 by the chain condition. The result is L2-normalized.

  Called by `graph.py`'s `canonical_collapse` method and as the default
  initial state for `graded_state`.

---

### Born Probabilities

- `born_graded(psi_re, psi_im, nV, nE, nF)` -> (per_cell, per_dim)

  Born probability |psi_k|^2 = psi_re[k]^2 + psi_im[k]^2 per cell.
  per_dim sums probabilities within each dimensional sector:
  per_dim = [P_V, P_E, P_F].

  Called by `graph.py`'s `born_graded` method.

- `energy_partition(psi_re, psi_im, nV, nE, nF)` -> f64[3]

  Fraction of total Born probability in each sector. Returns [P_V, P_E, P_F]
  summing to 1.0. Measures how much of the quantum state's probability mass
  lives at each dimensional level.

  Called by `graph.py`'s `energy_partition` method.




## `_rcfe` — RCFE Curvature, Strain, and Conservation Laws

**File:** `_rcfe.pyx` (494 lines)

Computes RCFE curvature, strain, Bianchi conservation, coupling tensors, and
dynamic strain equilibrium. Curvature C(e) measures how concentrated face
structure is at each edge. Strain S = sum C(e) * RL[e,e] is the total
structural stress. The Bianchi identity B1 @ diag(C) @ B2 = 0 guarantees that
curvature is a cocycle: structural stress is conserved across the chain complex.

---

### Curvature

- `compute_curvature(B2, nE, nF)` -> f64[nE]

  C(e) = sum_f B2[e,f]^2 / ||B2[:,f]||^2. Each face f distributes one unit of
  curvature across its boundary edges proportional to B2[e,f]^2. The total
  curvature sums to nF: sum_e C(e) = nF. Returns zeros when nF = 0.

  Called by `graph.py`'s `rcfe_curvature` property.

---

### Strain

- `compute_strain(curvature, rl_diag, nE)` -> float

  S = sum_e C(e) * RL[e,e]. Weighs curvature by the relational self-weight of
  each edge. Higher strain means more structural stress is concentrated on
  relationally important edges.

  Called by `graph.py`'s `rcfe_strain` property.

- `compute_strain_per_face(B2, curvature, nE, nF)` -> f64[nF]

  Per-face strain contribution: strain_f = sum of C(e) over boundary edges of
  face f. Identifies which faces carry the most structural stress.

---

### Bianchi Identity

- `verify_bianchi(B1, B2, curvature, nE, nF, tol=1e-10)` -> (is_valid, max_error)

  Checks B1 @ diag(C) @ B2 = 0. This holds because C(e) is a function of
  B2 column norms, and B1 @ B2 = 0 by the chain condition. Returns True if
  the maximum absolute entry in the product is below tol.

- `bianchi_residual(B1, B2, curvature, nE, nF)` -> f64[nF]

  Per-face Bianchi residual: ||B1 diag(C) B2[:,f]||_2 for each face. Should
  be zero (or near-zero numerically) for every face.

---

### Face Realization Rates

- `face_realization_rates(B2, tri_edges, nT, nV, nE, sources, targets)` -> (per_vertex, per_edge)

  Fraction of potential triangles that are realized as faces, computed per
  vertex and per edge. Uses `_void.classify_triangles` to distinguish realized
  from void triangles.

---

### Coupling Tensor

- `coupling_tensor(B2, RL, hats, nhats, nE, nF)` -> f64[nF, nhats]

  Per-face energy decomposition by operator channel. For each face f and
  channel k: tensor[f,k] = sum of hat_k[e,e] / RL[e,e] over boundary edges
  of f. Shows how each face's structural role distributes across the typed
  Laplacian channels (Hodge, overlap, frustration, copath).

---

### Derived Quantities

- `relational_integrity(curvature, rl_diag, nE, B2=None, nF=0)` -> dict

  RI = 1 / (1 + kappa_total) where kappa_total = strain. Values near 1 mean
  low stress, values near 0 mean high stress. If B2 is provided, also returns
  per-face RI.

- `face_overlap_K2(B2, nE, nF)` -> f64[nF, nF]

  Face overlap matrix K2 = |B2|^T |B2|. K2[f,f'] counts the number of
  boundary edges shared between faces f and f'. Symmetric with K2[f,f] equal
  to the boundary size of face f.

- `edge_weight_conjugation(L, sqw, nE)` -> f64[nE, nE]

  Weighted Laplacian L_w[i,j] = sqrt(w_i) * L[i,j] * sqrt(w_j). For dynamic
  edge weighting where w(e) depends on vertex amplitudes.

---

### Dynamic Strain Equilibrium

The dynamic strain framework couples curvature to quantum state probabilities.

- `attributed_curvature(B1, B2, w_e, a_v, nV, nE, nF)` -> dict

  Builds attributed boundary operators B1^w and B2^w with edge weights and
  vertex amplitudes applied, then computes the curvature residual R = B1^w @ B2^w
  and per-face attributed curvature kappa_f = ||R[:,f]||_2.

  Called by `graph.py`'s `attributed_curvature` method.

- `face_deficit(kappa_f, alpha, born_face, nF)` -> f64[nF]

  delta_f = kappa_f - alpha * |Psi_f|^2. The deficit between attributed
  curvature and quantum occupation probability scaled by the coupling constant.

- `relational_strain_dynamic(B2, delta, nE, nF)` -> f64[nE]

  sigma = B2 @ delta. The relational strain at each edge is the boundary
  operator applied to the face deficit. B1 @ sigma = 0 by the chain condition.

- `optimal_alpha(B2, kappa_f, born_face, nE, nF)` -> float

  alpha = (B2 kappa)^T (B2 pF) / ||B2 pF||^2. Minimizes ||sigma||^2 over
  the coupling constant. Returns 0 if the denominator is zero.

- `verify_bianchi_strain(B1, sigma, nV, nE, tol=1e-10)` -> (is_valid, max_residual)

  Checks B1 @ sigma = 0. Since sigma = B2 @ delta and B1 @ B2 = 0, this holds
  exactly. The Bianchi identity guarantees conservation of relational strain.

- `strain_equilibrium(B1, B2, kappa_f, born_face, nV, nE, nF)` -> dict

  Full strain equilibrium analysis in one call. Computes optimal alpha, face
  deficit, relational strain, Bianchi check, and strain norm.

  Called by `graph.py`'s `strain_equilibrium` method.

  Returns dict with: alpha, delta, sigma, bianchi_ok, bianchi_residual,
  strain_norm.




## `_fiber` — Fiber Character and Similarity Complex

**File:** `_fiber.pyx` (455 lines)

Computes pairwise similarity between edges (via chi cosine) and between
vertices (via phi similarity and fiber bundle similarity). Builds
threshold graphs from similarity matrices, constructs linkage complexes
from fiber bundle similarity, and projects simplex coordinates to 3D for
visualization.

---

### Cosine Similarity

- `chi_cosine(chi, nE, nhats)` -> f64[nE, nE]

  Pairwise cosine similarity of edge structural character vectors.
  sim[i,j] = chi_i . chi_j / (||chi_i|| ||chi_j||). Symmetric with diagonal
  1.0. Values in [-1, 1], though chi entries are nonneg so values are
  typically in [0, 1].

- `phi_cosine(phi, nV, nhats)` -> f64[nV, nV]

  Pairwise cosine similarity of vertex character vectors. Same formula as
  chi_cosine but on phi vectors.

---

### Phi Similarity

- `phi_similarity_score(phi_a, phi_b, nhats)` -> float

  S_phi = 1 - 0.5 * ||phi_a - phi_b||_1. Same metric as cross-dimensional
  coherence (kappa) but between two vertices instead of between a vertex and
  its star. Values in [0, 1]: 1 means identical character, 0 means maximally
  different.

- `phi_similarity_matrix(phi, nV, nhats)` -> f64[nV, nV]

  Full pairwise phi similarity matrix. S_phi[i,j] = 1 - 0.5 * ||phi_i - phi_j||_1.
  Symmetric with diagonal 1.0. Values in [0, 1].

  Called by `graph.py`'s `phi_similarity` property.

---

### Fiber Bundle Similarity

- `sfb_similarity_matrix(fchi, phi, n, nhats)` -> f64[n, n]

  S_fb[i,j] = max(cos(fchi_i, fchi_j), 0) * phi_similarity(phi_i, phi_j).
  Combines star character cosine (fiber alignment) with vertex character
  agreement. The cosine term is clamped to zero (negative cosines mean
  opposing fiber orientations). The product is zero when either factor is
  zero: vertices must agree on both their fiber structure and their
  cross-dimensional character to score high.

  fchi is the star character chi* (nV, nhats), phi is the vertex character
  (nV, nhats).

  Called by `graph.py`'s `fiber_similarity` property.

---

### Threshold Graph

- `threshold_graph(similarity, n, threshold)` -> (src, tgt, weights, n_edges)

  Extracts edges where similarity[i,j] > threshold for i < j. Returns int32
  edge arrays and float64 weights.

- `similarity_complex(similarity, n, threshold)` -> dict

  Builds a full chain complex from a thresholded similarity matrix. Calls
  `threshold_graph` to get edges, then uses `_cycles.find_fundamental_cycles`
  and `_boundary.build_B1`/`build_B2_from_cycles` to construct B1, B2, and
  compute Betti numbers. Returns a dict with src, tgt, weights, n_edges, nV,
  nF, B1, B2, and beta tuple.

---

### Simplex Projection

- `signal_sphere_proj(chi, nE, nhats)` -> f64[nE, 3]

  Projects chi vectors from the probability simplex to 3D Cartesian
  coordinates for visualization.

  For nhats=3: barycentric coordinates on an equilateral triangle.
  Simplex vertices map to (0,0), (1,0), (0.5, sqrt(3)/2) in the xy-plane.

  For nhats=4: barycentric coordinates on a regular tetrahedron. The fourth
  component lifts into the z-axis via sqrt(2/3).

  For nhats > 4: uses the first 3 components directly.

---

### Linkage Complex

- `linkage_complex(sfb, threshold, n_entities)` -> dict

  Builds a full chain complex from thresholded fiber bundle similarity.
  Edges are created where S_fb[i,j] > threshold. All triangles in the
  resulting 1-skeleton are enumerated as faces via sorted adjacency
  merge-intersection. Boundary operators B1 and B2 are constructed and
  converted to dense arrays. Betti numbers are computed via SVD rank of B2
  and the Euler relation.

  Returns: src, tgt, weights, n_edges, nV, nF, B1 (f64[nV, nE]),
  B2 (f64[nE, nF]), beta (beta_0, beta_1, beta_2), triangles (i32[nF, 3]).

  Called by `graph.py`'s `linkage_complex` method, which wraps the result
  in a new RexGraph.




## `_hypermanifold` — Filtered Manifold Sequence and Dimensional Analysis

**File:** `_hypermanifold.pyx` (185 lines)

Builds the filtered manifold sequence M1 < M2 where each inclusion adds
cells, degrees of freedom, and Bianchi identities. Computes the harmonic
shadow (cycles at dimension d that become boundaries at dimension d+1) and
verifies the dimensional subsumption property: Betti numbers cannot increase
when higher-dimensional cells are added.

---

### Manifold Sequence

- `build_manifold_sequence(evals_L0, evals_L1, evals_L2, nV, nE, nF, tol=1e-8)` -> dict

  Constructs the filtered family of truncated complexes:

  M1 (1-rex): vertices + edges only. Betti numbers are beta_0 from L0 and
  beta_1(1) = nE - rank(B1) = nE - (nV - beta_0). No Bianchi identity at
  this level.

  M2 (2-rex): vertices + edges + faces. Betti numbers are beta_0, beta_1
  (from full L1), and beta_2 (from L2). One Bianchi identity (B1 @ B2 = 0).
  Only included when nF > 0.

  Each manifold entry contains: dimension, cell counts, total DOF (N),
  Betti numbers, and number of Bianchi identities.

  Returns a dict with: manifolds (list of per-level dicts), max_dimension,
  total_N.

  Called by `graph.py`'s `hypermanifold` property.

---

### Harmonic Shadow

- `harmonic_shadow(evals_Ld_at_d, evals_Ld_at_d1, tol=1e-8)` -> (shadow_dim, beta_d, beta_d1)

  The harmonic shadow at dimension d is the set of harmonic d-forms that
  become exact when (d+1)-cells are added:
  ker(L_d(d)) minus ker(L_d(d+1)).

  shadow_dim = beta_d(d) - beta_d(d+1), clamped to zero. This equals
  rank(B_{d+1}): the number of (d+1)-cells needed to fill all d-cycles.

  For d=1: evals_Ld_at_d are eigenvalues of L1_down (no face contribution),
  evals_Ld_at_d1 are eigenvalues of the full L1 (with L1_up from faces).
  The shadow dimension is the number of 1-cycles that become boundaries of
  2-cells.

  Called by `graph.py`'s `harmonic_shadow` property.

---

### Dimensional Subsumption

- `dimensional_subsumption(betti_sequence)` -> (is_valid, violations)

  Verifies beta_k(d+1) <= beta_k(d) for all k and d. Adding cells can only
  kill cycles, not create new ones in lower dimensions. The betti_sequence
  is a list of lists: betti_sequence[d] = [beta_0, ..., beta_d] at
  truncation level d.

  Returns True if the property holds, along with a list of violation tuples
  (d, k, beta_k_d, beta_k_d1) if any.

  Called by `graph.py`'s `dimensional_subsumption` property.

---

### Betti from Eigenvalues

- `compute_betti_from_evals(evals, tol=1e-8)` -> int

  Counts eigenvalues with |val| < tol. A standalone helper for computing a
  single Betti number from Hodge theory: beta_k = dim ker(L_k) = number of
  zero eigenvalues of L_k.



## `_field` — Cross-Dimensional Field Dynamics on (E, F)

**File:** `_field.pyx` (850 lines)

Implements the coupled field operator and dynamics on the (E, F) field space,
where edges and faces are the independent degrees of freedom. Vertices are
derived via f_V = B1 f_E. Supports diffusion (heat equation), wave equation,
mode classification, and energy conservation analysis.

The field operator M couples edges and faces through B2:

    M = [[ RL,       -g * B2    ],
         [-g * B2^T,     L2     ]]

M is PSD when the coupling g is small enough. The default auto-coupling
g = 1 / max(||B2||_F, 1) stays in the PSD regime for typical complexes.

---

### Field Operator Construction

- `build_field_operator(RL1, L2, B2, g=-1.0)` -> (M, g_used, is_psd)

  Assembles the (nE+nF) x (nE+nF) block matrix. RL1 is the relational
  Laplacian on edges, L2 is the face Laplacian, B2 is the edge-face boundary.
  If g < 0, auto-computes coupling from the Frobenius norm of B2. Runs a PSD
  check via eigendecomposition.

  Called by `graph.py`'s `field_operator` property.

- `field_operator_matvec(F, RL1, L2, B2, g, nE, nF)` -> f64[nE+nF]

  Applies M @ F without building the dense matrix. Uses operator-vector
  products directly. For large complexes where the dense (nE+nF)^2 matrix is
  too expensive.

---

### Eigendecomposition

- `field_eigendecomposition(M)` -> (evals, evecs, freqs)

  Eigendecomposition of the field operator via LAPACK dsyev_. Near-zero
  eigenvalues are cleaned. Frequencies are freqs[k] = sqrt(max(evals[k], 0)).

  Called by `graph.py`'s `field_eigen` property.

- `field_spectral_coefficients(F, evecs)` -> f64[n]

  Projects a field state onto the eigenbasis: c_k = v_k^T F.

---

### Wave Evolution

- `wave_evolve(F0, evals, evecs, freqs, t)` -> (Ft, dFdt)

  Exact spectral wave evolution: d^2F/dt^2 = -M F with F(0) = F0,
  dF/dt(0) = 0. Solution: F(t) = sum_k c_k cos(omega_k t) v_k. Returns
  both position and velocity at time t.

- `wave_evolve_trajectory(F0, evals, evecs, freqs, times)` -> (traj, vel)

  Wave evolution at multiple timepoints. Reuses spectral coefficients for
  O(n * T) work after the initial O(n^2) projection. Returns f64[T, n]
  arrays for position and velocity.

  Called by `graph.py`'s `field_wave_evolve` method.

---

### Diffusion

- `field_diffusion_spectral(F0, evals, evecs, t)` -> f64[n]

  Heat equation F(t) = exp(-M t) F(0) = sum_k c_k exp(-lambda_k t) v_k.

- `field_diffusion_trajectory(F0, evals, evecs, times)` -> f64[T, n]

  Diffusion at multiple timepoints.

  Called by `graph.py`'s `field_diffuse` method.

---

### Energy and Conservation

- `wave_energy(F, dFdt, M)` -> (KE, PE, total)

  KE = 0.5 ||dF/dt||^2, PE = 0.5 F^T M F, total = KE + PE. Total energy
  is conserved under wave evolution.

- `field_energy_kin_pot(F, L1, LO, nE)` -> (E_kin, E_pot, ratio)

  Topological/geometric energy decomposition of the edge block:
  E_kin = f_E^T L1 f_E, E_pot = f_E^T L_O f_E. Extracts f_E from the packed
  field state F = [f_E, f_F].

- `wave_dimensional_energy(F, dFdt, nE, nF)` -> dict

  Splits field energy into edge and face components: norm_E, norm_F,
  ke_E, ke_F.

---

### Mode Classification

- `classify_modes(evals, evecs, nE, nF, threshold=0.1)` -> (labels, weights_E, weights_F, n_resonant)

  Classifies each eigenmode by its dimensional weight. For mode k,
  w_E = ||v_k[:nE]||^2 / ||v_k||^2 measures edge content, w_F measures face
  content. Labels: 0 = edge-dominated, 1 = face-dominated, 2 = EF-resonant.
  Resonant modes transfer energy between edges and faces.

  Called by `graph.py`'s `classify_modes` method.

- `resonance_frequencies(freqs, labels)` -> (res_freqs, res_indices)

  Extracts frequencies of EF-resonant modes.

---

### Vertex Observables

- `derive_vertex_trajectory(traj_EF, B1, nE)` -> f64[T, nV]

  Derives vertex observables from a field trajectory: f_V(t) = B1 f_E(t)
  at each timestep.

- `derive_vertex_state(F, B1, nE)` -> f64[nV]

  Derives vertex observable from a single field state: f_V = B1 F[:nE].

  Called by `graph.py`'s `derive_vertex_state` method.

---

### RK4 Integration

- `field_rk4_step(F, dFdt, RL1, L2, B2, g, nE, nF, dt)` -> (F_new, dFdt_new)

  Single RK4 step for the second-order wave equation, rewritten as a
  first-order system on (position, velocity). For large systems where
  spectral decomposition is too expensive.

- `field_diffusion_rk4_step(F, RL1, L2, B2, g, nE, nF, dt)` -> f64[n]

  Single RK4 step for the heat equation dF/dt = -M F.



## `_wave` — Complex-Amplitude Wave Mechanics

**File:** `_wave.pyx` (1130 lines)

Complex-valued wave mechanics on the rex chain complex. Operates on complex
amplitudes psi in C^n under Schrodinger evolution exp(-i L t), where L is any
Laplacian. Covers state operations, information theory, wave evolution
(spectral, RK4, Trotter-Suzuki), interference, entanglement, decoherence
channels, measurement, and density matrix operations.

---

### Complex State Operations

- `normalize_c128(psi)` -> float

  L2-normalizes a complex vector in place. Returns the original norm.

- `norm_c128(psi)` -> float

  L2 norm of a complex vector.

- `inner_product(psi, phi)` -> complex128

  Quantum inner product: sum_i conj(psi_i) * phi_i.

- `born_probabilities(psi)` -> f64[n]

  Born rule: P(i) = |psi_i|^2. Called by `graph.py`'s `born_probabilities`.

- `fidelity_pure(psi, phi)` -> float

  Fidelity |<psi|phi>|^2 between pure states. Values in [0, 1].

- `trace_distance_pure(psi, phi)` -> float

  Trace distance sqrt(1 - F) for pure states.

- `extract_phases(psi, ref=0)` -> f64[n]

  Relative phases with respect to a reference component.

- `apply_phase_gate(psi, phases)` -> complex128[n]

  Diagonal phase rotation: out_i = exp(i * phases_i) * psi_i.

---

### Information Theory

- `shannon_entropy(psi)` -> float

  H = -sum p_i log2(p_i) from Born probabilities. In bits.

- `renyi_entropy(probs, alpha)` -> float

  H_alpha = log2(sum p_i^alpha) / (1 - alpha). Reduces to Shannon at alpha=1.

- `participation_ratio(psi)` -> float

  PR = 1 / sum |psi_i|^4. Measures signal delocalization. PR = 1 for a
  Dirac delta, PR = n for uniform.

- `signal_purity(psi)` -> float

  sum |psi_i|^4. Inverse of participation ratio. 1 for Dirac delta, 1/n for
  uniform.

- `kl_divergence(p, q)` -> float

  KL divergence D(p||q) = sum p_i log2(p_i / q_i) in bits. Returns inf if
  any q_i = 0 where p_i > 0.

- `linear_entropy(psi)` -> float

  S_L = 1 - sum |psi_i|^4 = 1 - purity.

---

### Wave Evolution

- `schrodinger_spectral(psi, evals, evecs, t)` -> complex128[n]

  psi(t) = exp(-i L t) psi(0) via spectral decomposition. Works with any
  real symmetric Laplacian.

- `schrodinger_spectral_trajectory(psi, evals, evecs, times)` -> complex128[nT, n]

  Evolution at multiple timepoints.

- `rk4_step_complex(psi, L, dt)` -> complex128[n]

  Single RK4 step for d|psi>/dt = -i L |psi>. Dense Laplacian, complex state.

- `rk4_integrate_complex(psi0, L, t0, t1, n_steps)` -> (psi_final, trajectory)

  Full RK4 integration from t0 to t1.

- `field_schrodinger_evolve(psi_E, psi_F, evals_RL1, evecs_RL1, evals_L2, evecs_L2, t, B1=None)` -> (psi_E_t, psi_F_t, psi_V_t)

  Schrodinger evolution on the rex field (E, F). Edge tier uses RL, face
  tier uses L2, evolved independently. Vertex observables are derived via
  B1 when provided. Called by `graph.py`'s `evolve_field_wave`.

- `field_schrodinger_trajectory(psi_E, psi_F, ..., times, B1=None)` -> (traj_E, traj_F, traj_V)

  Field evolution at multiple timepoints.

- `trotter_step(psi, diag, L_off, dt)` -> complex128[n]

  Trotter-Suzuki split-operator step for L = L_diag + L_off.

---

### Interference

- `superpose(psi_list, weights)` -> complex128[n]

  Weighted superposition: Psi = sum_k w_k * psi_k. Unnormalized.

- `interference_pattern(psi1, psi2, w1, w2)` -> f64[n]

  P(i) = |w1 psi1_i + w2 psi2_i|^2. Includes the interference term.

- `classical_mixture(psi1, psi2, w1, w2)` -> f64[n]

  P(i) = w1 |psi1_i|^2 + w2 |psi2_i|^2. No interference.

- `interference_term(psi1, psi2)` -> f64[n]

  I(i) = 2 Re(conj(psi1_i) psi2_i). P_quantum = P_classical + I.

- `visibility(psi1, psi2)` -> float

  Fringe visibility V = 2|<psi1|psi2>| / (1 + |<psi1|psi2>|^2). 1 for
  identical states, 0 for orthogonal.

- `coherence_measure(rho)` -> float

  l1-norm coherence C = sum_{i!=j} |rho_ij|.

---

### Entanglement

- `tensor_product(psi_A, psi_B)` -> complex128[dA * dB]

- `partial_trace_A(rho, dim_A, dim_B)` -> complex128[dim_B, dim_B]

  Trace out subsystem A.

- `partial_trace_B(rho, dim_A, dim_B)` -> complex128[dim_A, dim_A]

  Trace out subsystem B.

- `partial_transpose_A(rho, dim_A, dim_B)` -> complex128[dA*dB, dA*dB]

  Negative eigenvalues certify entanglement (PPT criterion).

- `entanglement_entropy(psi, dim_A, dim_B)` -> float

  Via Schmidt decomposition: S = -sum lambda_i^2 log2(lambda_i^2).
  Called by `graph.py`'s `entanglement_entropy`.

- `schmidt_decomposition(psi, dim_A, dim_B)` -> (values, vectors_A, vectors_B)

---

### Decoherence Channels

- `dephasing_channel(rho, gamma, dt)` — off-diagonals decay exponentially
- `amplitude_damping(rho, gamma, dt)` — irreversible decay toward ground state
- `depolarizing_channel(rho, p)` — rho -> (1-p) rho + (p/d) I
- `lindblad_step(rho, H, lindblad_ops, dt)` — Euler step of Lindblad master equation

---

### Measurement

- `born_sample(psi, n_samples)` -> int64[n_samples]

  Sample cell indices from Born distribution.

- `projective_collapse(psi, projector)` -> (collapsed_state, probability)

  P|psi> / ||P|psi>||.

- `measure_in_eigenbasis(psi, evecs)` -> (outcome, probability, collapsed)

  Born-sample from eigenbasis coefficients and collapse. Called by
  `graph.py`'s `measure_in_eigenbasis`.

---

### Density Matrix Operations

- `pure_to_density(psi)` -> complex128[n, n] — rho = |psi><psi|
- `density_from_ensemble(states, weights)` -> complex128[n, n] — mixed state
- `density_trace(rho)` -> complex128
- `density_purity(rho)` -> float — Tr(rho^2)
- `von_neumann_entropy(rho)` -> float — S = -Tr(rho log2 rho)
- `fidelity_mixed(rho, sigma)` -> float — (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2

---

### RCF Dynamics Operators

- `amplitude_graded_projection(B1, B2, amplitudes, nV, nE, nF)` -> f64[N]

  Graded projection using continuous vertex amplitudes with geometric-mean
  edge coupling. Unlike canonical collapse, uses sqrt(a_i * a_j) for edge
  amplitudes.

- `lagrangian_step(amplitudes, prev_amplitudes, sources, targets, edge_weights, nV, nE, dt, H0=1.0)` -> dict

  One step of Lagrangian dynamics. Returns kinetic energy T, coupling
  potential V, and Lagrangian L = T - V.

- `harmonic_basis_extract(evals, evecs, n, tol=1e-10)` -> f64[n_harm, n]

  Extracts harmonic eigenvectors as a basis.

- `harmonic_project(signal, basis)` -> f64[n]

  Projects signal onto harmonic basis.

- `face_partition(B1, B2, sources, targets, probe_vertex, nV, nE, nF)` -> dict

  Partitions faces into probe-incident and scaffold sets.

- `action_integral(lagrangian_values, dt_values)` -> f64[nT]

  Cumulative action: A[t] = sum_{i<=t} L_i * dt_i.




## `_state` — Rex State Representation and Signal Operations

**File:** `_state.pyx` (555 lines)

Provides signal norms, normalization, packing/unpacking of per-dimension
signals into flat vectors, state differencing, energy decomposition, and
state construction helpers. Also contains the RexState class for managing
signal evolution with cached energy. In the rex framework, edges are
primitive and vertices are derived via f_V = B1 f_E.

---

### Signal Norms

- `signal_norm_l1(signal)` -> float — sum |f_i|
- `signal_norm_l2(signal)` -> float — sqrt(sum f_i^2)
- `signal_norm_linf(signal)` -> float — max |f_i|
- `signal_norm(signal, norm_type=NORM_L2)` -> float — dispatches by type (0=L1, 1=L2, 2=Linf)

---

### Normalization

- `normalize_l1(signal)` -> f64[n] — normalize to L1 = 1 (probability distribution). Returns copy.
- `normalize_l2(signal)` -> f64[n] — normalize to L2 = 1 (unit amplitude). Returns copy.
- `normalize_signal(signal, norm_type=NORM_L2)` -> f64[n] — dispatches by type.

  Called by `graph.py`'s `normalize` method.

---

### State Packing (V + E + F)

- `pack_state(f0, f1, f2)` -> (flat, sizes)

  Concatenates vertex, edge, and face signals into a single f64[nV+nE+nF]
  vector. Returns the flat vector and an int32[3] sizes array [nV, nE, nF].

- `unpack_state(flat, sizes)` -> (f0, f1, f2)

  Splits a flat vector back into per-dimension signals.

---

### Field State Packing (E + F only)

- `field_state_pack(f_E, f_F)` -> (flat, sizes)

  Packs edge and face signals into f64[nE+nF]. Vertices are not part of the
  field state; they are derived via B1. Returns the flat vector and int32[2]
  sizes [nE, nF].

- `field_state_unpack(flat, sizes)` -> (f_E, f_F)

  Splits a field state vector back to edge and face signals.

- `field_state_vertex_observable(f_E, B1)` -> f64[nV]

  Derives vertex observable from edge signal: f_V = B1 f_E.

---

### State Differencing

- `state_diff(state_a, state_b)` -> f64[n] — diff = state_b - state_a
- `state_apply_diff(state, diff)` -> f64[n] — result = state + diff

---

### Energy Computation

- `energy_kin_pot(f_E, L1, LO)` -> (E_kin, E_pot, ratio)

  E_kin = f_E^T L1 f_E (topological energy from Hodge Laplacian),
  E_pot = f_E^T L_O f_E (geometric energy from overlap Laplacian).
  Ratio = E_kin / E_pot, capped at 1e15 when E_pot is near zero.

  Called by `graph.py`'s `energy_kin_pot` method.

---

### State Construction

- `uniform_state(nV, nE, nF, norm_type=NORM_L1)` -> (f0, f1, f2)

  Uniform signal at each dimension. L1-normalized: each entry is 1/n.
  L2-normalized: each entry is 1/sqrt(n).

  Called by `graph.py`'s `uniform_state` method.

- `dirac_state(nV, nE, nF, dim, idx)` -> (f0, f1, f2)

  All zeros except 1.0 at position idx in dimension dim.
  Called by `graph.py`'s `dirac_state` method.

- `dirac_edge(nE, nF, edge_idx)` -> (f_E, f_F)

  Dirac delta on a single edge in the (E, F) field state. f_F is zero.
  Called by `graph.py`'s `dirac_edge` method.

- `vertex_perturbation_to_edges(vertex_idx, B1_T, nE, nF)` -> (f_E, f_F)

  Edge signal from vertex perturbation: f_E = B1^T delta_v.

- `random_state(nV, nE, nF, norm_type=NORM_L1)` -> (f0, f1, f2)

  Random nonnegative signal at each dimension, normalized by norm_type.

---

### RexState Class

`RexState(nV, nE, nF, t=0.0)`

Container for signals on the 2-rex chain complex. Holds f0 (vertices),
f1 (edges), f2 (faces), and current time t. Provides cached energy
decomposition and evolution methods.

- `shapes` -> (nV, nE, nF)
- `set_f0(data)`, `set_f1(data)`, `set_f2(data)` — set signal, marks energy dirty
- `update_energy(L1, LO, alpha)` — recomputes E_kin, E_pot, E_tot under RL
- `energy`, `E_kin`, `E_pot` — cached energy properties (NaN if dirty)
- `derive_vertex_signal(B1)` — sets f0 = B1 @ f1
- `evolve_coupled(system, dt, n_steps=1)` — RK4 cross-dimensional evolution
- `evolve_schrodinger(system, dt)` — unitary evolution of f1 via RL
- `evolve_diffusion(system, dt, dim=0)` — simple diffusion on one dimension




## `_transition` — Transition Operators on the Rex Chain Complex

**File:** `_transition.pyx` (623 lines)

Stateless transition operators for evolving signals on the chain complex.
Covers Markov diffusion (discrete and continuous), Schrodinger unitary
evolution (spectral and matrix exponential), RK4 ODE integration with
coupled cross-dimensional dynamics, and signal resizing after structural
mutation.

---

### Markov Diffusion

- `markov_vertex_step(p, W)` -> f64[nV] — one discrete step: p_new = W @ p
- `markov_edge_step(p, W_O)` -> f64[nE] — discrete step via overlap adjacency
- `markov_face_step(p, W_F)` -> f64[nF] — discrete step via face adjacency

- `markov_multistep(p, W, n_steps)` -> (final, trajectory)

  Applies n discrete Markov steps. Returns the final state and the full
  f64[n_steps+1, n] trajectory.

- `markov_continuous_expm(p, L, t)` -> f64[n]

  p(t) = exp(-L t) p(0) via scipy matrix exponential. Called by
  `graph.py`'s `evolve_markov` method.

- `markov_continuous_spectral(p, evals, evecs, t)` -> f64[n]

  Spectral decomposition path: p(t) = sum_k exp(-lambda_k t) c_k v_k where
  c_k = v_k^T p(0). Avoids forming the matrix exponential.

- `build_vertex_transition_matrix(L0)` -> f64[nV, nV]

  Column-stochastic transition matrix W = I - D^{-1} L0 from the vertex
  Laplacian. Isolated vertices get self-loops (W[i,i] = 1).

- `build_lazy_transition_matrix(W, lazy=0.5)` -> f64[nV, nV]

  Lazy random walk: W_lazy = lazy * I + (1 - lazy) * W. Ensures
  aperiodicity.

---

### Schrodinger (Unitary) Evolution

- `schrodinger_evolve_spectral(f, evals, evecs, t)` -> (f_real, f_imag)

  Unitary evolution exp(-i L t) f(0) via spectral decomposition. Since L is
  real symmetric, the result splits into cos and sin components:
  f_re = sum_k cos(lambda_k t) c_k v_k,
  f_im = -sum_k sin(lambda_k t) c_k v_k.
  Called by `graph.py`'s `evolve_schrodinger` method.

- `schrodinger_evolve_expm(f, L, t)` -> (f_real, f_imag)

  Unitary evolution via complex matrix exponential (scipy expm). Returns
  real and imaginary parts.

- `schrodinger_multistep(f, evals, evecs, times)` -> (traj_re, traj_im)

  Evolves through multiple timepoints. Returns f64[nT, n] arrays for real
  and imaginary parts.

---

### Energy Decomposition

- `quadratic_form(f_re, f_im, L)` -> float

  Computes f_re^T L f_re + f_im^T L f_im. Works with dense or sparse L.

- `kinetic_energy(f_re, f_im, L1)` -> float — E_kin = <f|L1|f>
- `potential_energy(f_re, f_im, LO)` -> float — E_pot = <f|L_O|f>

- `energy_decomposition(f_re, f_im, L1, LO, alpha_G)` -> (E_kin, E_pot, E_RL)

  E_RL = E_kin + alpha_G * E_pot. Used by RexState.update_energy.

---

### RK4 Integration

- `rk4_step(y, t, dt, derivative_func)` -> f64[n]

  Single RK4 step. derivative_func(y, t) returns dy/dt.

- `rk4_integrate(y0, t0, t1, n_steps, derivative_func)` -> (y_final, trajectory, times)

  Full RK4 integration from t0 to t1. Returns final state, the full
  f64[n_steps+1, n] trajectory, and the f64[n_steps+1] time array.
  Called by `graph.py`'s `evolve_coupled` method.

- `diffusion_derivative(f, L, diffusion_rate=1.0)` -> f64[n]

  Heat equation derivative: df/dt = -rate * L @ f.

---

### Coupled Cross-Dimensional Dynamics

- `coupled_derivative(flat_state, sizes, L0, L1, L2, L_O, B1_dense, B2_dense, alpha0, alpha1, alpha2, alpha_G)` -> f64[nV+nE+nF]

  Coupled ODE right-hand side for cross-dimensional diffusion:

      df0/dt = -alpha0 * L0 @ f0
      df1/dt = -(alpha1 * L1 + alpha_G * L_O) @ f1 + B1^T @ f0
      df2/dt = -alpha2 * L2 @ f2 + B2^T @ f1

  The edge tier uses RL = alpha1 * L1 + alpha_G * L_O. Cross-dimensional
  coupling flows downward via B1^T (vertex to edge) and B2^T (edge to face).
  B2 should be B2_hodge (self-loop faces filtered).

  Used as the derivative function for `rk4_integrate` in
  `graph.py`'s `evolve_coupled` method.

---

### Rewrite (Signal Resizing)

For structural mutation (edge insertion/deletion, face changes):

- `rewrite_insert_edges_i32(f0, f1, f2, nV_old, nE_old, nV_new, nE_new, ...)` -> (f0, f1, f2)

  Resizes signals after edge insertion. New vertices/edges get default values.

- `rewrite_delete_edges_i32(f0, f1, f2, vertex_map, edge_map, nV_new, nE_new)` -> (f0, f1, f2)

  Resizes signals after edge deletion. Uses vertex_map and edge_map to
  reindex surviving cells.

- `rewrite_add_faces(f2, n_new_faces, default_face_val=0.0)` -> f64[nF+n_new]
- `rewrite_remove_faces(f2, keep_mask)` -> f64[n_keep]

---

### Transition Dispatch

- `apply_transition(trans_type, f0, f1, f2, target_dim, operator_data, dt, t, n_steps)` -> (f0, f1, f2)

  Unified dispatch for all transition types. trans_type selects the operator:
  TRANS_MARKOV (0), TRANS_SCHRODINGER (1), TRANS_DIFFERENTIAL (2),
  TRANS_REWRITE (3). operator_data is a dict with operator-specific arrays.
  Only the targeted dimension is modified; others are returned as-is.




## `_void` — Void Spectral Theory

**File:** `_void.pyx` (443 lines)

The void complex records potential faces (triangles in the 1-skeleton) that
could exist but don't. Each void v has a boundary cycle bv in ker(B1) with
harmonic content eta(v) in [0, 1]. If eta > 0, filling v decreases beta_1
by 1. The void Laplacian Lvoid = Bvoid @ Bvoid^T measures structural stress
from unrealized faces.

Key identities: B1 @ Bvoid = 0 (void boundaries lie in ker(B1)),
L_up + Lvoid = Bfull @ Bfull^T where Bfull = [B2 | Bvoid].

---

### Triangle Enumeration

- `find_potential_triangles(adj_ptr, adj_idx, adj_edge, nV, nE)` -> (tri_edges, nT)

  Finds all triangles in the 1-skeleton via the symmetric adjacency CSR.
  For each vertex v, checks all neighbor pairs (u, w) with u < w < v for a
  closing edge. Returns tri_edges as an int32[nT, 3] array of edge indices
  per triangle, and the count nT.

---

### Triangle Classification

- `classify_triangles(B2, tri_edges, nT, nE)` -> (realized, void_indices, n_voids)

  For each potential triangle, checks whether its edge triple matches a column
  of B2 (a realized face). realized[k] = 1 if triangle k is a face, 0 if
  void. void_indices lists the indices of void triangles.

---

### Void Boundary Operator

- `build_void_boundary(B1, B2, tri_edges, nT, nV, nE)` -> (Bvoid, void_indices, n_voids)

  Builds the void boundary operator Bvoid (nE x n_voids). Each column is a
  signed edge vector (+/-1 on 3 edges) chosen so that B1 @ Bvoid[:,k] = 0.
  Tries all 8 sign patterns and picks the one in ker(B1). Returns None if
  there are no voids.

---

### Harmonic Content

- `harmonic_content_single(bv, evals_L1, evecs_L1, nE)` -> float

  eta = ||proj_harm(bv)||^2 / ||bv||^2. Projects a void boundary cycle onto
  the harmonic space ker(L1). eta > 0 means the void carries harmonic content
  and filling it would reduce beta_1.

- `harmonic_content_all(Bvoid, evals_L1, evecs_L1, n_voids, nE)` -> f64[n_voids]

  Harmonic content for all voids.

---

### Void Character

- `void_character_single(bv, RL, hats, nhats, nE)` -> f64[nhats]

  chi^void(k) = bv^T hat_k bv / (bv^T RL bv). Decomposes each void's
  relational energy across the typed Laplacian channels. Same formula as
  edge character chi but applied to void boundary cycles.

- `void_character_all(Bvoid, RL, hats, nhats, n_voids, nE)` -> f64[n_voids, nhats]

  Void character for all voids.

---

### Void Strain

- `void_strain(Bvoid, n_voids, nE)` -> float

  S^void = sum ||bv||^2 = tr(Lvoid). Total structural stress from
  unrealized faces. For triangles, each ||bv||^2 = 3 (three edges with
  +/-1 entries), so S^void = 3 * n_voids.

---

### Filling Prediction

- `fills_beta(eta, n_voids)` -> int32[n_voids]

  fills_beta[k] = 1 if eta[k] > epsilon, meaning that filling void k would
  decrease beta_1 by 1.

---

### Void Type Decomposition

- `void_type_decomposition(void_indices, tri_edges, edge_types, n_voids, n_types)` -> int32[2^n_types]

  Counts voids by bitmask of edge types present. For example, if a void has
  edges of types 0 and 2, its bitmask is 0b101 = 5.

---

### Void Identity Verification

- `verify_void_identity(B2, Bvoid, nE, tol=1e-10)` -> (is_valid, residual)

  Checks L_up + Lvoid = Bfull @ Bfull^T where Bfull = [B2 | Bvoid]. This
  identity decomposes the full upward Laplacian into realized and void
  contributions.

---

### Combined Builder

- `build_void_complex(B1, B2, adj_ptr, adj_idx, adj_edge, nV, nE, RL=None, hats=None, nhats=0, evals_L1=None, evecs_L1=None)` -> dict

  Builds the complete void complex in one call: enumerates triangles,
  classifies realized vs void, builds Bvoid, computes harmonic content,
  void character, fills_beta, and void strain.

  Called by `graph.py`'s `void_complex` property.

  Returns dict with: Bvoid, Lvoid, n_voids, n_potential, tri_edges,
  void_indices, eta, chi_void, fills_beta, void_strain.




## `_quotient` — Quotient Complexes and Relative Homology

**File:** `_quotient.pyx` (1931 lines)

Given a 2-rex R and a subcomplex I specified by cell masks, builds the
quotient complex R/I and computes relative homological invariants. Supports
subcomplex selection (by edge type, signal threshold, energy regime, star
neighborhood, hyperslice), quotient construction (reindexing, B1_quot,
B2_quot), relative Betti numbers, congruence testing, signal
restriction/lifting, per-edge energy decomposition, sparse quotient
construction from large parent complexes, and character-based quotient
filtration.

---

### Subcomplex Selection

- `validate_subcomplex(v_mask, e_mask, f_mask, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)` -> (valid, violations)

  Checks closure conditions: boundary vertices of selected edges must be
  selected, boundary edges of selected faces must be selected. Returns True
  if valid, along with a list of violation tuples.

- `closure_of_edges(e_mask, nV, boundary_ptr, boundary_idx)` -> (v_mask, e_mask, f_mask)

  Computes the downward closure of an edge set by adding boundary vertices.

- `closure_of_faces(f_mask, nV, nE, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)` -> (v_mask, e_mask, f_mask)

  Computes the downward closure of a face set by adding boundary edges and
  vertices.

- `closure_of_faces_and_edges(v_mask, e_mask, f_mask, nV, nE, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)` -> (v_mask, e_mask, f_mask)

  Closes masks downward: faces add boundary edges, edges add boundary
  vertices.

- `subcomplex_by_edge_type(edge_types, select_type, nV, boundary_ptr, boundary_idx)` -> (v_mask, e_mask, f_mask)

- `subcomplex_by_threshold(signal, threshold, select_below, nV, boundary_ptr, boundary_idx)` -> (v_mask, e_mask, f_mask)

- `star_of_vertex(v, nV, nE, nF, ...)` -> (v_mask, e_mask, f_mask)

  Vertex, incident edges, incident faces, closed downward.

- `star_of_edge(edge_idx, nV, nE, nF, ...)` -> (v_mask, e_mask, f_mask)

  Edge, overlap neighbors, incident faces, closed downward.

- `subcomplex_by_energy_regime(E_kin_per_edge, E_pot_per_edge, regime, ratio_tol, ...)` -> (v_mask, e_mask, f_mask)

  Edges classified by log(E_kin/E_pot): regime 0 = kinetic, 1 = crossover,
  2 = potential.

---

### Quotient Construction

- `quotient_maps(v_mask, e_mask, f_mask)` -> (v_reindex, v_star, e_reindex, f_reindex, nV_quot, nE_quot, nF_quot)

  Builds reindexing arrays. Subcomplex vertices collapse to a single
  basepoint v_star. Subcomplex edges and faces are removed.

- `quotient_B1(B1, v_mask, e_mask, v_reindex, v_star, e_reindex, nV_quot, nE_quot)` -> f64[nV_quot, nE_quot]

  Quotient boundary operator. Surviving edges are remapped; subcomplex
  vertices map to the basepoint.

- `quotient_B2(B2_col_ptr, B2_row_idx, B2_vals, e_mask, f_mask, e_reindex, f_reindex, nE_quot, nF_quot)` -> f64[nE_quot, nF_quot]

  Quotient face boundary operator from CSC representation.

- `quotient_verify_chain(B1_quot, B2_quot, tol=1e-10)` -> (valid, max_error)

  Checks B1_quot @ B2_quot = 0.

- `build_quotient(B1, v_mask, e_mask, f_mask, B2_col_ptr, B2_row_idx, B2_vals, LO=None, alpha_G=0.0)` -> dict

  Full quotient pipeline: reindexing, B1_quot, B2_quot, chain check, relative
  Betti numbers, L1_quot. Optionally builds RL1_quot if L_O is provided.
  Called by `graph.py`'s `quotient` method.

---

### Relative Homology

- `relative_betti(B1_quot, B2_quot, tol=1e-10)` -> (beta0_rel, beta1_rel, beta2_rel)

  Relative Betti numbers from SVD ranks of quotient boundary operators.

- `relative_cycle_basis(B1_quot, B2_quot, tol=1e-10)` -> f64[nE_quot, beta1_rel]

  Orthonormal harmonic edge signals spanning H_1(R, I).

- `connecting_homomorphism(B1_full, v_mask, e_mask, relative_cycle, e_reindex)` -> f64[nV_I]

  Lifts a relative 1-cycle to the full edge space, applies B1, and
  restricts to subcomplex vertices.

---

### Congruence

- `congruent_edges(a, b, B1, e_mask, tol=1e-10)` -> (is_congruent, residual)

  Tests whether edges a and b are congruent modulo the subcomplex.

- `congruent_faces(a, b, B2, f_mask, tol=1e-10)` -> (is_congruent, residual)

- `congruence_classes_edges(B1, e_mask, tol=1e-10)` -> (labels, n_classes)

  Partitions surviving edges into congruence equivalence classes.

- `congruence_classes_faces(B2, f_mask, tol=1e-10)` -> (labels, n_classes)

---

### Signal Operations

- `restrict_signal(signal, mask)` -> f64[n_quot] — drop subcomplex cells
- `restrict_signal_complex(signal, mask)` -> complex128[n_quot]
- `lift_signal(signal_quot, mask, fill_value=0.0)` -> f64[n] — expand to full
- `lift_signal_complex(signal_quot, mask)` -> complex128[n]
- `restrict_field_state(f_E, f_F, e_mask, f_mask)` -> (f_E_quot, f_F_quot)
- `lift_field_state(f_E_quot, f_F_quot, e_mask, f_mask, fill_value=0.0)` -> (f_E, f_F)
- `quotient_energy(signal_quot, L_quot)` -> float — Rayleigh quotient
- `quotient_RL1(B1_quot, B2_quot, LO_quot, alpha_G)` -> (RL1_quot, L1_quot)
- `quotient_energy_kin_pot(signal_quot, L1_quot, LO_quot)` -> (E_kin, E_pot, ratio)

---

### Per-Edge Energy

- `per_edge_energy(f_E, L1, LO)` -> (E_kin_per_edge, E_pot_per_edge)

  E_kin_e = f_E[e] * (L1 f_E)[e], E_pot_e = f_E[e] * (L_O f_E)[e]. These
  per-edge values sum to the total energies. Called by `graph.py`'s
  `per_edge_energy` method.

---

### Hyperslice, Edge-Type, and Temporal Integration

- `hyperslice_quotient(dim, cell_idx, nV, nE, nF, ...)` -> (v_mask, e_mask, f_mask)

  Subcomplex from the hyperslice around a cell, closed downward.

- `edge_type_quotient(edge_types, type_codes, nV, boundary_ptr, boundary_idx)` -> (v_mask, e_mask, f_mask)

- `temporal_quotient(n_snapshots, time_mask, snapshot_sources, snapshot_targets, nV)` -> (v_mask, e_mask_union)

---

### Sparse Quotient Construction

- `build_quotient_from_sparse(B1_scipy, B2_scipy, v_mask, e_mask, f_mask, nV, nE, nF)` -> dict

  Builds a quotient complex directly from sparse boundary operators without
  densifying the parent at the nE x nE scale. Accepts DualCSR (auto-detected
  via `hasattr(obj, 'row_ptr')` and converted via `to_scipy_csr`), scipy
  sparse, or dense input.

  The function extracts surviving cell indices, reindexes vertices (subcomplex
  vertices collapse to a basepoint), builds dense B1_quot and B2_quot (small
  since the quotient is a subset), checks the chain condition B1_quot @ B2_quot
  = 0, computes L1_quot, and calls `relative_betti` for the quotient Betti
  numbers.

  If the quotient has nEq <= 5000 edges, automatically runs the full dense
  `build_all_laplacians` on it to produce a `spectral_bundle_quot` dict with
  RL, hats, chi, coupling constants, and the complete RCF analysis on the
  tractable subcomplex.

  Returns dict with: B1_quot, B2_quot, L1_quot, betti_rel, chain_valid,
  chain_error, dims (nVq, nEq, nFq), v_surv, e_surv, f_surv, and optionally
  spectral_bundle_quot.

  The intended workflow for large graphs: compute sparse Betti on the full
  graph via `build_all_laplacians_sparse`, then use `build_quotient_from_sparse`
  to analyze structurally interesting subsets with full dense spectral.

---

### Quotient Filtration by Character

- `quotient_filtration_by_character(chi, channel, n_steps, B1, B2, nV, nE, nF)` -> dict

  Removes edges in order of decreasing chi[:, channel] and tracks Betti
  numbers at each step. At each threshold, edges with chi above the
  threshold are removed and Betti numbers are recomputed on the remaining
  subcomplex. The transition point is the step with the largest drop in
  beta_1.

  Returns: thresholds (f64[n_steps]), beta0/beta1/beta2 (i32[n_steps]),
  n_edges_remaining (i32[n_steps]), edges_removed_order (i32[nE] sorted
  by decreasing chi), transition_index (-1 if constant),
  transition_threshold.

  Called by `graph.py`'s `quotient_filtration` method.




## `_temporal` — Temporal Bundle, BIOES Phase Detection, and Lifecycle Tracking

**File:** `_temporal.pyx` (1958 lines)

Delta-encoded snapshot storage, BIOES phase detection on Betti and energy
timeseries, edge and face lifecycle tracking, and cascade event analysis for
temporal rexgraphs. All functions have i32 and i64 typed variants plus
auto-dispatchers. General boundary variants handle branching, self-loop, and
witness edges alongside standard 2-endpoint edges.

---

### Delta Encoding

- `encode_snapshot_delta(prev_src, prev_tgt, curr_src, curr_tgt, directed=False)` -> (born_src, born_tgt, died_src, died_tgt)

  Computes the edge delta between consecutive snapshots via sorted merge-diff
  on canonical edge encodings. Born edges are in curr but not prev; died
  edges are in prev but not curr.

- `encode_snapshot_delta_general(prev_bp, prev_bi, curr_bp, curr_bi)` -> (born_keys, died_keys)

  General boundary variant using sorted boundary vertex tuples as edge keys.

---

### Temporal Index

- `build_temporal_index(snapshots, directed=False, checkpoint_threshold=0.5)` -> (checkpoints, deltas, checkpoint_times)

  Builds an adaptive checkpoint index from a snapshot list. Checkpoints are
  stored when cumulative delta exceeds checkpoint_threshold * current edge
  count. Returns checkpoint snapshots, per-step deltas, and checkpoint times.

  Called by `TemporalRex.temporal_index`.

- `build_temporal_index_general(snapshots, checkpoint_threshold=0.5)` -> same

  General boundary variant.

---

### Edge Lifecycle

- `edge_lifecycle(snapshots, directed=False)` -> (birth_time, death_time, n_unique)

  For each unique edge across all snapshots, records first appearance
  (birth_time) and last appearance (death_time). n_unique is the total number
  of distinct edges seen.

  Called by `TemporalRex.edge_lifecycle`.

- `edge_lifecycle_general(snapshots)` -> same

---

### Edge Metrics

- `compute_edge_metrics(snapshots, directed=False)` -> (edge_counts, edge_born, edge_died)

  Per-timestep edge counts, births, and deaths. Each is an int32[T] array.

  Called by `TemporalRex.edge_metrics`.

- `compute_edge_metrics_general(snapshots)` -> same

---

### Phase Detection (Betti-based)

- `detect_phases(beta0, beta1, phase_tol=0.0)` -> (phase_start, phase_end, phase_b0, phase_b1)

  Detects topological phases: consecutive timesteps where beta_0 and beta_1
  are constant within tolerance. A phase break occurs when either Betti
  number changes.

- `detect_phases_kd(betti_matrix, phase_tol=0.0)` -> (phase_start, phase_end, phase_betti)

  K-dimensional Betti phase detection for betti_matrix[T, K]. A phase breaks
  when any Betti number shifts.

- `detect_phases_with_events(betti_matrix, f_born, f_died, f_split, f_merge, phase_tol, face_event_threshold)` -> (phase_start, phase_end, phase_betti, break_reasons)

  Phase detection with face event triggers. A phase also breaks when
  face births, deaths, splits, or merges exceed the threshold.

---

### BIOES Tagging

- `assign_bioes_tags(T, phase_start, phase_end, min_phase_len=2)` -> int32[T]

  Tags each timestep: B=0 (begin), I=1 (inside), O=2 (outside),
  E=3 (end), S=4 (single-step phase).

- `assign_bioes_per_dimension(T, betti_matrix, phase_tol, min_phase_len)` -> int32[T, K]

  Per-dimension BIOES tags.

---

### Full BIOES Pipelines

- `compute_bioes_full(snapshots, beta0, beta1, directed=False, phase_tol=0.0, min_phase_len=2)` -> tuple

  Full pipeline: edge metrics, phase detection, BIOES tags.

- `compute_bioes_unified(edge_snapshots, face_snapshots, betti_matrix, directed, phase_tol, min_phase_len, face_event_threshold, jaccard_threshold)` -> tuple

  Unified pipeline with face tracking. Returns tags (unified + per-dim),
  edge metrics, face metrics (counts, born, died, split, merge), phases.

  Called by `TemporalRex.bioes`.

- `compute_bioes_general(...)`, `compute_bioes_unified_general(...)` — general boundary variants.

---

### Energy-Ratio BIOES

- `detect_phases_energy_ratio(E_kin, E_pot, ratio_tol=0.2, floor=1e-12)` -> (phase_start, phase_end, phase_regime, log_ratios, crossover_times)

  Detects energy regime phases from the log(E_kin/E_pot) timeseries.
  Regime 0 = kinetic (topological), 1 = crossover, 2 = potential (geometric).

- `compute_bioes_energy(E_kin, E_pot, ratio_tol=0.2, min_phase_len=2, floor=1e-12)` -> (tags, phase_start, phase_end, phase_regime, log_ratios, crossover_times)

  Full energy-ratio BIOES pipeline. Called by `TemporalRex.bioes_energy`
  and by `_signal.tag_energy_phases`.

- `detect_phases_joint(betti_matrix, E_kin, E_pot, betti_tol, ratio_tol)` -> (phase_start, phase_end, phase_betti, phase_regime, break_reasons, log_ratios)

  Joint Betti + energy phase detection. A phase breaks when any Betti
  number shifts or the energy regime changes.

  Called by `TemporalRex.bioes_joint`.

---

### Cascade Tracking

- `cascade_edge_activation(edge_signals, threshold)` -> (activation_time, activation_order, activation_rank)

  Tracks when each edge first exceeds a signal threshold. activation_time[e]
  is the first timestep (-1 if never). activation_order lists edges sorted by
  activation time. activation_rank[e] is the position in the order.

  Called by `TemporalRex.cascade_activation` and `_signal.cascade_from_edge`.

- `cascade_wavefront(edge_signals, src, tgt, threshold)` -> dict

  Wavefront tracking with spatial propagation analysis. Records per-timestep
  wavefront edges, their topological depth from the source, and propagation
  velocity.

  Called by `TemporalRex.cascade_wavefront`.

---

### Face Tracking

- `track_faces(B2_cp_prev, B2_ri_prev, src_prev, tgt_prev, B2_cp_curr, B2_ri_curr, src_curr, tgt_curr, jaccard_threshold=0.5)` -> (events, n_persist, n_born, n_died, n_split, n_merge)

  Tracks faces across two snapshots. Exact boundary match detects persists;
  Jaccard overlap above threshold detects splits and merges.

- `face_lifecycle(face_snapshots, edge_snapshots, directed=False)` -> tuple

  Full face lifecycle across all timesteps.

  Called by `TemporalRex.face_lifecycle_data`.

- `track_faces_general(...)` — general boundary variant.




## `_standard` — Classical Graph Algorithms on the 1-Skeleton

**File:** `_standard.pyx` (1005 lines)

Classical graph algorithms operating on the undirected 1-skeleton via
symmetric CSR adjacency. All functions have i32 and i64 typed variants
plus auto-dispatchers.

---

### PageRank

- `pagerank(adj_ptr, adj_idx, adj_wt, nV, nE, damping=0.85, max_iter=100, tol=1e-8)` -> f64[nV]

  Weighted damped random walk via power iteration. Converges when the L1
  change between iterations drops below tol. Scores sum to 1.0.

  Called by `graph.py`'s `pagerank` property via `build_standard_metrics`.

---

### Betweenness Centrality

- `betweenness(adj_ptr, adj_idx, adj_edge, nV, nE, max_sources=0)` -> (bc_v, bc_e)

  Vertex and edge betweenness centrality via BFS dependency accumulation
  (Brandes algorithm). O(nV * nE). Vertex betweenness normalized by
  (nV-1)(nV-2)/2, edge betweenness by nV(nV-1)/2. Edge counts are halved
  since each edge appears in both directions of the symmetric adjacency.

  max_sources > 0 samples that many source vertices and rescales (approximate
  betweenness for large graphs).

  Called by `graph.py`'s `betweenness` property via `build_standard_metrics`.

---

### Clustering Coefficient

- `clustering(adj_ptr, adj_idx, nV)` -> f64[nV]

  Local clustering coefficient via sorted neighbor intersection (two-pointer
  merge). C(v) = 2 T(v) / (deg(v) (deg(v) - 1)) for deg >= 2, where T(v)
  is the triangle count at v. Zero for vertices with degree < 2.

  Called by `graph.py`'s `clustering` property via `build_standard_metrics`.

---

### Louvain Community Detection

- `louvain(adj_ptr, adj_idx, adj_wt, nV, nE, max_passes=20)` -> (labels, n_communities, modularity)

  Modularity-based community detection. For each vertex, evaluates the
  modularity gain of moving to each neighbor's community and picks the best.
  Repeats until no improvement or max_passes reached. Returns community
  labels (int32[nV]), number of communities, and final modularity Q.

  Called by `graph.py`'s `partition_communities` property via
  `build_standard_metrics`.

---

### Pearson Correlation

- `safe_correlation(a, b)` -> float

  Pearson correlation with zero-variance guard. Returns 0.0 if either
  signal has zero variance or n < 2.

---

### Adjacency Weights

- `build_adj_weights(adj_edge, edge_weights)` -> f64[nnz]

  Maps per-edge weights to per-adjacency-entry weights:
  adj_wt[k] = edge_weights[adj_edge[k]].

---

### Combined Builder

- `build_standard_metrics(adj_ptr, adj_idx, adj_edge, adj_wt, nV, nE, damping=0.85, pagerank_iter=100, btw_max_sources=0, louvain_max_passes=20)` -> dict

  Computes all standard graph metrics in one call. Returns dict with:
  pagerank, betweenness_v, betweenness_e, btw_norm_v, btw_norm_e,
  clustering, community_labels, n_communities, modularity.

  Called by `graph.py`'s `standard_metrics` cached property.




## `_persistence` — Persistent Homology on the Rex Chain Complex

**File:** `_persistence.pyx` (1237 lines)

Given a 2-rex and a filtration function on cells, computes persistence
pairs tracking birth and death of homological features. Supports multiple
filtration sources, column reduction over Z/2 coefficients, barcode
extraction, diagram distances, landscape functions, entropy, and
enrichment with edge type and Hodge component data.

---

### Filtration Construction

- `filtration_sublevel_vertex(f0, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)` -> (filt_v, filt_e, filt_f)

  Sublevel filtration from a vertex signal. f(e) = max over boundary
  vertices, f(f) = max over boundary edges.

- `filtration_sublevel_edge(f1, nV, v2e_ptr, v2e_idx, B2_col_ptr, B2_row_idx)` -> (filt_v, filt_e, filt_f)

  Sublevel filtration from an edge signal. f(v) = min over incident edges,
  f(f) = max over boundary edges.

- `filtration_sublevel_face(f2, nV, nE, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)` -> (filt_v, filt_e, filt_f)

  Sublevel filtration from a face signal. f(e) = min over incident faces,
  f(v) = min over incident edges.

- `filtration_hodge_component(grad, curl, harmonic, nV, v2e_ptr, v2e_idx, B2_col_ptr, B2_row_idx, component=2)` -> (filt_v, filt_e, filt_f)

  Filtration by Hodge component magnitude. component: 0=gradient, 1=curl,
  2=harmonic.

- `filtration_spectral(eigenvector, nV, ...)` -> (filt_v, filt_e, filt_f)

  Filtration from a Laplacian eigenvector (absolute value).

- `filtration_rips(positions, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)` -> (filt_v, filt_e, filt_f)

  Vietoris-Rips filtration from vertex positions. f(v)=0,
  f(e)=max pairwise distance among boundary vertices.

- `filtration_temporal(snapshot_sources, snapshot_targets, nV, nE, directed)` -> (filt_v, filt_e, filt_f)

  Temporal appearance order as filtration.

- `filtration_temporal_general(snapshots, nV, nE)` -> same

- `filtration_dimension(nV, nE, nF)` -> (filt_v, filt_e, filt_f)

  Trivial filtration by cell dimension (0, 1, 2).

---

### Filtration Ordering and Boundary Matrix

- `build_filtration_order(filt_v, filt_e, filt_f)` -> (order, cell_dim, cell_idx, filt_vals)

  Sorts all cells by filtration value with ties broken by (dim, index).
  Returns the permutation order, per-position dimension and cell index,
  and the sorted filtration values.

- `build_boundary_matrix(order, cell_dim, cell_idx, nV, nE, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)` -> list

  Constructs the combined boundary matrix D as a list of column sets,
  indexed by filtration order. Edges reference their boundary vertices,
  faces reference their boundary edges.

---

### Column Reduction

- `reduce_boundary_matrix_mod2(boundary_cols)` -> (reduced, pivot_to_col)

  Left-to-right column reduction over Z/2. Returns the reduced column
  lists and a pivot-to-column mapping for pair extraction.

- `reduce_boundary_matrix(boundary_cols, coefficients="Z2")` -> same

  Wrapper supporting Z/2 coefficients (Z coefficients planned).

---

### Persistence Diagram

- `persistence_diagram(filt_v, filt_e, filt_f, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)` -> dict

  Full persistence computation. Returns dict with: pairs (f64[n, 5] with
  birth, death, dim, birth_cell, death_cell), essential (f64[n, 3] for
  features that never die), betti (beta0, beta1, beta2 at final step),
  order, cell_dim, cell_idx, filt_vals.

  Called by `graph.py`'s `persistence` method.

---

### Barcodes

- `persistence_barcodes(pairs, essential, target_dim=-1)` -> f64[k, 2]

  Extracts (birth, death) pairs for a specific dimension, sorted by
  persistence. Called by `graph.py`'s `persistence_barcodes` method.

---

### Diagram Distances

- `bottleneck_distance(dgm1, dgm2)` -> float

  Approximate bottleneck distance via greedy matching heuristic. O(n^2).

- `wasserstein_distance(dgm1, dgm2, p=2.0)` -> float

  Approximate p-Wasserstein distance via greedy matching.

  Called by `graph.py`'s `persistence_distance` method.

---

### Enrichment

- `enrich_pairs_edge_type(pairs, edge_types)` -> (type_labels, type_counts)

  Annotates dim-1 pairs with the edge type of their birth cell.

- `enrich_pairs_hodge(pairs, grad_energy, curl_energy, harm_energy)` -> (dominant, fractions)

  Annotates dim-1 pairs with dominant Hodge component (0=grad, 1=curl,
  2=harmonic) and per-component energy fractions.

  Called by `graph.py`'s `enrich_persistence` method.

---

### Persistence Entropy

- `persistence_entropy(barcodes)` -> float

  Shannon entropy of normalized barcode lengths: H = -sum (l_i/L) log2(l_i/L).

  Called by `graph.py`'s `persistence_entropy` method.

---

### Persistence Landscape

- `persistence_landscape(barcodes, grid, k_max=5)` -> f64[k_max, G]

  Lambda_k(t) = k-th largest min(t - birth_i, death_i - t) at each grid
  point. Called by `graph.py`'s `persistence_landscape` method.

---

### Relative Persistence

- `relative_persistence(filt_v, filt_e, filt_f, boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx, v_mask, e_mask, f_mask)` -> dict

  Persistent relative homology H_*(R, I) across a filtration. Restricts
  to surviving cells and computes persistence on the quotient complex.




## `_query` — Relational Complex Query Engine

**File:** `_query.pyx` (323 lines)

Predicate masking (SELECT WHERE), signal imputation (INSERT missing),
spectral propagation (AGGREGATE), and cell explanation (EXPLAIN) on the
relational complex.

---

### Predicate Masking

- `predicate_mask(values, n, op, threshold, threshold_high=0.0)` -> (mask, count)

  Applies a predicate to a signal array. Returns a uint8 mask and match
  count. Ops: GT(0), GE(1), LT(2), LE(3), EQ(4), NE(5), BETWEEN(6).

- `chi_mask(chi, nE, nhats, channel, op, threshold)` -> (mask, count)

  Predicate on edge character chi[:, channel].

- `phi_mask(phi, nV, nhats, channel, op, threshold)` -> (mask, count)

  Predicate on vertex character phi[:, channel].

- `kappa_mask(kappa, nV, op, threshold)` -> (mask, count)

  Predicate on vertex coherence kappa.

- `mask_and(a, b, n)`, `mask_or(a, b, n)`, `mask_not(a, n)` — boolean ops on masks.

---

### Signal Imputation

- `signal_impute(RL, observed_signal, observed_mask, nE)` -> dict

  Imputes missing edge signal values via harmonic interpolation through RL.
  Partitions RL into observed/missing blocks: g_missing = -RL_mm^+ RL_mo g_obs.
  Uses dense pseudoinverse for small systems, CG for large.

  Returns dict with: imputed (f64[nE]), confidence (f64[nE] from RL diagonal),
  residual, n_observed, n_imputed.

  Called by `graph.py`'s `impute` method.

---

### Spectral Propagation

- `spectral_propagate(RL, hats, nhats, source, target, nE)` -> dict

  Computes propagation score = source^T RL^+ target / (||source|| ||target||).
  Also computes per-channel typed scores, total energy, and spectral
  coverage (fraction of RL eigenmodes activated by the source).

  Returns dict with: score, typed_scores (f64[nhats]), energy, coverage.

  Called by `graph.py`'s `propagate` method.

---

### Explain Edge

- `explain_edge(B1, B2, K1, RL, hats, nhats, edge_idx, nV, nE, nF)` -> dict

  Full diagnostic for a single edge. Returns: below (boundary vertices),
  above (coboundary faces), lateral (K1 neighbors), chi (character vector),
  dominant_channel, effective_resistance (RL^+[e,e]), n_incident_faces,
  degree.

  Called by `graph.py`'s `explain(dim=1, idx)`.

---

### Explain Vertex

- `explain_vertex(B1, RL, hats, nhats, phi, kappa, chi_edges, v2e_ptr, v2e_idx, vertex_idx, nV, nE)` -> dict

  Full diagnostic for a single vertex. Returns: phi (vertex character),
  chi_star (averaged chi over incident edges), kappa (coherence),
  discrepant_channel (largest phi-chi_star gap), channel_gap, dominant_channel,
  degree, incident_edges, neighbor_vertices.

  Called by `graph.py`'s `explain(dim=0, idx)`.




## `_joins` — Chain Complex Join Operations

**File:** `_joins.pyx` (325 lines)

Two complexes share structure through a vertex identification map. All
joins produce valid chain complexes (B1j @ B2j = 0 guaranteed because
restriction/extension of chain complexes preserves the chain condition).

All three join functions guard `matrix_rank` and `chain_residual` against
empty matrices (zero matched edges, zero faces). When any output dimension
is zero, rank defaults to 0 and chain residual defaults to 0.0.

---

### Shared Maps

- `build_shared_vertex_map(labels_R, labels_S)` -> int32[nV_R]

  Maps R-vertices to S-vertices by matching labels. Returns -1 for
  unmatched vertices.

- `build_shared_edge_map(src_R, tgt_R, src_S, tgt_S, shared_vertices)` -> int32[nE_R]

  Maps R-edges to S-edges by matching vertex pairs through the shared
  vertex map. Returns -1 for unmatched edges.

---

### Inner Join (Intersection)

- `inner_join(B1_R, B2_R, nV_R, nE_R, nF_R, B1_S, B2_S, nV_S, nE_S, nF_S, shared_vertices)` -> dict

  Keeps only cells present in both complexes. An edge qualifies if both
  endpoints are shared and the same vertex pair is connected in both R
  and S. Faces qualify if all their boundary edges are matched.

  When no edges match (e.g., shared vertices exist but no shared edge
  pairs), the result has nEj = 0 and Betti numbers reflect the empty
  complex.

  Returns dict with: B1j, B2j, nVj, nEj, nFj, beta (Betti numbers),
  chain_residual, matched_edges_R.

  Called by `graph.py`'s `inner_join` method.

---

### Outer Join (Pushout)

- `outer_join(B1_R, B2_R, nV_R, nE_R, nF_R, B1_S, B2_S, nV_S, nE_S, nF_S, shared_vertices)` -> dict

  Union of both complexes. R vertices keep their indices; unshared S
  vertices get new indices. Shared vertices are identified (merged).
  All edges and faces from both complexes are included. nEj = nE_R + nE_S
  always (no edge deduplication — shared edges appear twice with different
  boundary representations in the joined complex).

  Returns dict with: B1j, B2j, nVj, nEj, nFj, beta, chain_residual.

  Called by `graph.py`'s `outer_join` method.

---

### Left Join

- `left_join(B1_R, B2_R, nV_R, nE_R, nF_R, B1_S, B2_S, nV_S, nE_S, nF_S, shared_vertices)` -> dict

  Keeps all of R, adds S-edges between shared vertices that are not already
  in R (no duplicates). Only R faces are kept (S faces would need
  cross-complex B2 construction). New edges get sign convention -1/+1 on
  their two shared endpoints.

  Returns dict with: B1j, B2j, nVj, nEj, nFj, beta, chain_residual,
  n_new_edges.

  Called by `graph.py`'s `left_join` method.

---

### Attribute Merge

- `attribute_merge(nV_R, nE_R, ew_R, amps_R, ew_S, amps_S, shared_vertices, alpha=0.5)` -> dict

  Blends vertex amplitudes at shared vertices:
  merged = (1 - alpha) * R + alpha * S. Returns dict with: merged_ew,
  merged_amps, n_enriched.




## `_interfacing` — Interfacing Vector and Channel Scoring

**File:** `_interfacing.pyx` (623 lines)

Maps a set of source vertices through typed response operators and projects
onto a target edge vector to produce per-channel scores. The interfacing
vector I lives on S^{n-1} after normalization and classifies entities by
their structural mechanism. All hot paths are `cdef nogil` with BLAS/LAPACK
calls.

---

### Vertex Source

- `build_vertex_source(target_indices, target_weights, vertex_weights, nV)` -> f64[nV]

  Weighted vertex source vector. rho[v] = sum of target_weight * vertex_weight
  at each target vertex. Non-target vertices are zero.

---

### Edge Signal

- `build_edge_signal(rho, B1, evals_L0, evecs_L0, nV, nE)` -> f64[nE]

  Edge gradient of the vertex Poisson solution: psi = B1^T @ L0^+ @ rho.
  Uses spectral_pinv_matvec to apply L0^+ without materializing the full
  pseudoinverse. L0 eigendata is passed in from graph.py's spectral_bundle
  (no redundant eigensolves).

---

### Response Operators

- `build_response_operators(B1, evals_L0, evecs_L0, L_O, L_SG, nV, nE)` -> dict

  Builds typed response operators for the three structural channels:
  S_T = B1^T @ L0^+ @ B1 (gradient flow through vertex space, nE x nE),
  S_G = L_O (overlap co-membership), S_F = L_SG (frustration sign coherence).
  Also returns L0_pinv.

---

### Channel Scores

- `channel_scores(psi, S_T, S_G, S_F, target, nE)` -> f64[3]

  Per-channel interfacing scores: I_X = target^T @ S_X @ psi for
  X in {topological, geometric, frustration}. Uses bl_gemv_n + bl_dot
  per channel in a cdef nogil inner function.

- `schrodinger_score(psi, evals_RL, evecs_RL, target, nE)` -> float

  Time-averaged Born probability: I_Sch = sum_j |c_j|^2 * |t_j|^2 where
  c_j = <v_j, psi> and t_j = <v_j, target>. Only eigenmodes with
  evals > 0 contribute.

---

### Quality Gate

- `quality_gate(scores)` -> f64[n_entities, n_channels]

  Bayesian quality gate: q(x) = x / (x + median(|x|)) per channel.
  Applied column-wise across entities.

---

### Interfacing Vector Assembly

- `interfacing_vector(scores, quality)` -> f64[n_channels]

  I = scores * quality elementwise.

- `sphere_position(iv)` -> f64[n_channels]

  Project to unit sphere: iv / ||iv||. Returns zero vector for zero input.

---

### Spectral Coverage

- `coverage(psi, evals_RL, evecs_RL, nE, probe_floor)` -> float

  Fraction of RL eigenmodes activated by psi: count(|c_j| > probe_floor) /
  count(lambda_j > 0). Values in [0, 1].

- `poisson_floor()` -> float

  Minimum acceptable coverage: 1 - 1/e = 0.6321.

---

### Source Efficiency

- `source_efficiency(target_indices, B1, nV, nE)` -> float

  Fraction of boundary entries that are activating (positive) across all
  edges incident to target vertices. Values in [0, 1].

---

### Confidence Flags

- `confidence_flags(coverage_val, efficiency, phi_T)` -> dict

  Confidence diagnostics: CONFIDENT (coverage >= Poisson floor, no conflict),
  LOW_SIGNAL (coverage < 1 - 1/e), CHANNEL_CONFLICT (efficiency < 0.5 and
  topological fraction < 2/3).

---

### Full Pipeline

- `build_interfacing_bundle(target_indices, target_weights, vertex_weights, B1, evals_L0, evecs_L0, L_O, L_SG, evals_RL, evecs_RL, target, nV, nE)` -> dict

  Chains the full pipeline: vertex source -> edge signal -> response operators
  -> channel scores -> Schrodinger score -> sphere position -> coverage ->
  efficiency -> confidence. Returns dict with rho, psi, scores, schrodinger,
  iv, sphere_pos, signal_magnitude, coverage, efficiency, confidence.

  Called by `graph.py`'s `interfacing_vector` method.




## `_channels` — Per-Channel Signal Decomposition and Group Scoring

**File:** `_channels.pyx` (281 lines)

Decomposes an edge signal's energy across typed Laplacian channels via hat^+
quadratic forms. Spectral channel scores propagate a source signal through RL
eigenmodes and project onto a target. Group scores aggregate spectral scores
across entity groups defined by vertex masks. All eigendata is pre-computed
and passed in; no eigensolves here.

---

### Primal Signal Character

- `primal_signal_character(psi, hat_evals_list, hat_evecs_list, nhats, nE)` -> f64[nhats]

  E_X = psi^T @ hat_X^+ @ psi for each channel X, normalized so fractions
  sum to 1. Uses a cdef nogil `_quadratic_pinv` inner function that computes
  the pseudoinverse quadratic form via spectral decomposition:
  sum_{lam>tol} (v_j . psi)^2 / lam_j. Zero signal returns uniform 1/nhats.

  Called by `graph.py`'s `primal_signal_character` method. Hat eigendata comes
  from `_character.hat_eigen_all` cached in `_hat_eigen_bundle`.

---

### Spectral Channel Score

- `spectral_channel_score(source, target, evals_RL, evecs_RL, nE)` -> float

  s = sum_j (c_j^src * c_j^tgt) / lambda_j for lambda_j > 0. Propagates
  source through RL eigenmodes and projects onto target. Symmetric:
  score(a, b) = score(b, a). Uses a cdef nogil inner function with inline
  dot products per eigenmode.

  Called by `graph.py`'s `spectral_channel_score` method.

---

### Group Channel Scores

- `group_channel_scores(group_masks, target, evals_RL, evecs_RL, B1, nV, nE, n_groups)` -> f64[n_groups]

  Per-group spectral channel scores. For each group, builds an edge source
  from vertex membership via bl_gemv_t (B1^T @ mask), normalizes via bl_nrm2,
  then computes the spectral channel score against the target vector.

---

### Multi-Channel Profile

- `multi_channel_profile(iv, primal_char, coverage_val, kappa_mean, efficiency)` -> dict

  Assembles a multi-dimensional profile for visualization. Combines
  interfacing vector components (iv_T, iv_G, iv_F, iv_Sch), primal
  character fractions (pc_T, pc_G, pc_F), coverage, mean coherence,
  and efficiency into a single dict.




## `_cross_complex` — Cross-Complex Structural Comparison

**File:** `_cross_complex.pyx` (303 lines)

Aligns two relational complexes by shared vertex labels and compares
structural invariants (coherence kappa, void fraction, spectral channel
scores) across them. All data is passed as arrays; this module does not
import or depend on RexGraph.

---

### Label Alignment

- `align_by_labels(labels_A, labels_B)` -> (shared_labels, idx_A, idx_B)

  Finds shared vertices between two complexes by label matching.
  Returns the shared labels and corresponding index arrays into each
  complex. Uses a dict-based lookup for O(n) matching.

---

### Kappa Correlation

- `cross_complex_kappa(kappa_A, kappa_B, idx_A, idx_B)` -> dict

  Correlates coherence kappa across two complexes at shared vertices.
  Uses a single-pass Pearson correlation (cdef nogil `_pearson`).
  Returns correlation, n_shared, kappa_A_shared, kappa_B_shared,
  mean_A, mean_B. Returns 0.0 for n_shared < 2.

---

### Void Fraction Comparison

- `cross_complex_void_fraction(n_voids_A, n_potential_A, n_voids_B, n_potential_B)` -> dict

  Compares void fractions (n_voids / n_potential) between two complexes.
  Returns void_fraction_A, void_fraction_B, and their difference.

---

### Channel Score Correlation

- `cross_complex_channel_scores(scores_A, scores_B)` -> dict

  Pearson correlation of per-group spectral channel scores between two
  complexes. Measures whether the two complexes rank groups similarly.

---

### Full Bridge Analysis

- `cross_complex_bridge(kappa_A, kappa_B, idx_A, idx_B, n_voids_A, n_potential_A, n_voids_B, n_potential_B, channel_scores_A=None, channel_scores_B=None)` -> dict

  Chains kappa correlation, void fraction comparison, and optionally
  channel score correlation into a single result. Returns dict with kappa,
  void, n_shared sub-dicts, and optional channel sub-dict.

  Called by the standalone `cross_complex_bridge()` function in `graph.py`,
  which extracts coherence and void_complex data from two RexGraph objects.
