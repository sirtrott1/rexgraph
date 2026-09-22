"""Sheaves over a relational complex.

A fiber assigns the same space to every cell and carries its content in the
connection. A sheaf assigns possibly different data per cell, and gluing asks
whether cells that meet agree where they meet.

Two cells meet through a MEDIATOR, a cell one grade away incident to both: at
grade 1 a shared vertex, at grade 0 an edge containing both, at grade 2 a shared
edge. A pair sharing several mediators is one structural edge, and it must agree
at all of them.

Restrictions live on INCIDENCES rather than cells, which is what lets a connection
carry holonomy: stated per cell the two ends carry the same map and every cycle
closes.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

__all__ = [
    "ExactGlueResult", "ExactGluingObstruction", "ExactSectionCheck", "ExactSheaf", "Sheaf",
    "UndeclaredRestrictionError",
]


def _cell_count(rex, grade: int) -> int:
    # Grade two is the carried, chain consistent face basis, not all decorative
    # face records. Its population must match the B2 used by _incidences.
    return {0: int(rex.nV), 1: int(rex.nE), 2: int(rex.nF_hodge)}[grade]


def _dimension(value, *, context, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{context} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return value


def _grade(value):
    grade = _dimension(value, context="grade")
    if grade not in (0, 1, 2):
        raise ValueError(f"grade must be 0, 1 or 2; got {grade!r}")
    return grade


def _incidences(rex, grade: int, cell_count: int) -> list[list[int]]:
    """Return the declared mediators for each cell at one grade.

    This is structural data shared by the approximate and exact sheaf readers.
    Keeping it outside either carrier prevents the exact path from allocating an
    unused float64 stalk matrix merely to discover the incidence lattice.
    """
    if grade == 1:
        ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
        idx = np.asarray(rex._boundary_idx, dtype=np.int64)
        return [sorted(set(idx[ptr[edge]:ptr[edge + 1]].tolist()))
                for edge in range(cell_count)]
    if grade == 0:
        ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
        idx = np.asarray(rex._boundary_idx, dtype=np.int64)
        out: list[list[int]] = [[] for _ in range(cell_count)]
        for edge in range(int(rex.nE)):
            for vertex in idx[ptr[edge]:ptr[edge + 1]]:
                out[int(vertex)].append(edge)      # the mediator IS the relation
        return [sorted(set(items)) for items in out]
    b2 = getattr(rex, "_B2_hodge_dual", None)
    if b2 is None or cell_count == 0:
        return [[] for _ in range(cell_count)]
    from rexgraph.native_sparse import NativeSparse
    return [sorted(column) for column in NativeSparse(b2).columns()]


class Sheaf:
    """Approximate numerical stalks and restrictions on the cells of one grade.

    This legacy reader stores ``float64`` sections and uses a relative tolerance when
    comparing transported values.  It is appropriate for real valued connection and
    holonomy calculations, not for certifying equality of integer/rational sections.
    Use :class:`ExactSheaf` for the latter; its separate carrier prevents a rational
    share such as ``1/3`` from being rounded before gluing is even evaluated.

    `grade` selects what a cell is: 1 edges (the default, because edges are primary),
    0 vertices, 2 chain consistent carried faces. Legacy ``H0``/``H1`` result keys
    mean agreement-component/failed-pair counts, not sheaf cohomology dimensions.
    """

    def __init__(self, rex, stalk_dim: int = 1, grade: int = 1):
        grade = _grade(grade)
        stalk_dim = _dimension(stalk_dim, context="stalk_dim", minimum=1)
        rex._ensure_clean()
        self.rex = rex
        self.grade = int(grade)
        self.d = int(stalk_dim)
        self._ncells = self._count_cells()
        self.stalks = np.zeros((self._ncells, self.d), dtype=np.float64)
        #: (cell, mediator) -> d x d. Absent means identity, which is "inherit unchanged"
        #: and is the right default: a restriction is a statement, and no statement is
        #: not the zero map.
        self._R: dict[tuple[int, int], np.ndarray] = {}
        self._inc = self._incidences()
        self._B1c = None

    # the cells, and what mediates between them
    def _count_cells(self) -> int:
        return _cell_count(self.rex, self.grade)

    @property
    def n_cells(self) -> int:
        return self._ncells

    def _incidences(self) -> list[list[int]]:
        """cell -> the mediators incident to it, at this grade."""
        return _incidences(self.rex, self.grade, self._ncells)

    def mediators(self, a: int, b: int) -> list[int]:
        """Every cell one grade away incident to BOTH, which is what makes them meet."""
        return sorted(set(self._inc[self._check_cell(a)]) & set(self._inc[self._check_cell(b)]))

    def _check_cell(self, cell):
        index = _dimension(cell, context="cell")
        if index >= self._ncells:
            raise ValueError(f"cell {index} is not present at grade {self.grade}")
        return index

    def _check_mediator(self, cell, mediator):
        index = _dimension(mediator, context="mediator")
        if index not in self._inc[cell]:
            raise ValueError(f"mediator {index} is not incident to cell {cell}")
        return index

    def meets(self) -> list[tuple[int, int, list[int]]]:
        """Every unordered meeting pair ONCE, with all the mediators it shares.

        These are the edges the fully connected sheaf (the LATTICE) would have over
        this complex, so their count is the denominator of the glue ratio.
        """
        by_med: dict[int, list[int]] = {}
        for c, meds in enumerate(self._inc):
            for m in meds:
                by_med.setdefault(m, []).append(c)
        pairs: dict[tuple[int, int], set] = {}
        for m, cells in by_med.items():
            for i, a in enumerate(cells):
                for b in cells[i + 1:]:
                    pairs.setdefault((a, b) if a < b else (b, a), set()).add(m)
        return [(a, b, sorted(ms)) for (a, b), ms in sorted(pairs.items())]

    # the data and the inheritance rules
    def assign(self, cell: int, vec) -> None:
        cell = self._check_cell(cell)
        v = np.asarray(vec, dtype=np.float64).ravel()
        if not np.all(np.isfinite(v)):
            raise ValueError("numerical stalk values must be finite")
        self.stalks[int(cell), :min(self.d, v.size)] = v[:self.d]

    def restrict(self, cell: int, mat, *, mediator: int | None = None) -> None:
        """The inheritance rule. Identity means inherit unchanged, a scale means inherit
        at a factor, zero means do not inherit that component at all.

        `mediator=None` states the rule about the CELL, which applies it at every
        incidence of that cell, which is convenient and exactly the degenerate case that cannot
        express holonomy. Name a mediator to distinguish the ends.
        """
        M = np.asarray(mat, dtype=np.float64).reshape(self.d, self.d)
        if not np.all(np.isfinite(M)):
            raise ValueError("numerical restrictions must be finite")
        cell = self._check_cell(cell)
        meds = self._inc[cell] if mediator is None else [self._check_mediator(cell, mediator)]
        for m in meds:
            self._R[(int(cell), int(m))] = M

    def _transport(self, cell: int, med: int) -> np.ndarray:
        R = self._R.get((int(cell), int(med)))
        s = self.stalks[int(cell)]
        return s if R is None else R @ s

    # the reading
    def glue(self, *, tol: float = 1e-9) -> dict:
        """Approximate relative agreement across every meeting, in [0, 1].

        Read relatively: the scale of a restriction belongs to the caller, so the reading
        is normalised by the transported magnitudes rather than by a fixed tolerance.
        This is not an exact obstruction calculation; :meth:`ExactSheaf.glue` is.
        """
        tol = self._check_numeric(tol)
        pairs = self.meets()
        glued = 0
        parent = list(range(self._ncells))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        failed = []
        for a, b, meds in pairs:
            ok = True
            for m in meds:
                ta, tb = self._transport(a, m), self._transport(b, m)
                if not self._agree(ta, tb, tol):
                    ok = False
                    break
            if ok:
                glued += 1
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            else:
                failed.append((a, b))
        gluable = len(pairs)
        comps = len({find(c) for c in range(self._ncells)})
        return {"ratio": (glued / gluable) if gluable else 1.0,
                "gluable": gluable, "glued": glued,
                "H0": comps, "H1": len(failed), "failed": failed}

    # holonomy: the curl of a connection
    def holonomy(self, theta) -> np.ndarray:
        """`B2^T theta`: the transport accumulated around each face."""
        b2 = getattr(self.rex, "_B2_hodge_dual", None)
        if b2 is None or int(self.rex.nF_hodge) == 0:
            return np.zeros(0, dtype=np.float64)
        from rexgraph.native_sparse import NativeSparse
        return NativeSparse(b2).transpose_apply(np.asarray(theta, dtype=np.float64).ravel())

    def is_flat(self, theta) -> bool:
        """Is every face's holonomy zero, i.e. is the section parallel."""
        h = self.holonomy(theta)
        return bool(h.size == 0 or np.abs(h).max() == 0.0)

    def gradient_angles(self, potential) -> np.ndarray:
        """The per incidence angle a bound connection carries."""
        from rexgraph.native_sparse import NativeSparse
        return NativeSparse(self.rex._B1_dual).transpose_apply(np.asarray(potential, dtype=np.float64).ravel())

    def bind_connection(self, theta) -> None:
        """Bind a U(1) connection to the incidences, signed by the boundary entry.

        Per incidence rather than per cell, so the two ends of a cell carry opposite
        rotations and a cycle can fail to close. A 1 dimensional stalk cannot see this;
        use `stalk_dim >= 2`.
        """
        th = np.asarray(theta, dtype=np.float64).ravel()
        for cell, meds in enumerate(self._inc):
            for m in meds:
                if self.grade == 1:
                    e, sigma = cell, self._incidence_sign(cell, m)
                elif self.grade == 0:
                    e, sigma = m, self._incidence_sign(m, cell)
                else:
                    e, sigma = m, 1.0
                if 0 <= e < th.size:
                    self._R[(cell, m)] = _rotation(self.d, float(sigma) * float(th[e]))

    def _incidence_sign(self, edge: int, vertex: int) -> float:
        """The SIGN of vertex `v` in edge `e`'s boundary column. Sign only: the share
        carries the arity and must not enter the angle."""
        if self._B1c is None:
            from rexgraph.native_sparse import NativeSparse
            self._B1c = list(NativeSparse(self.rex._B1_dual).columns())
        v = float(self._B1c[int(edge)].get(int(vertex), 0))
        return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


    # global sections
    def sections(self, *, tol: float = 1e-9) -> list[list[int]]:
        """Components of the agreement graph for the CURRENT assignment.

        These are not a basis of the space of all global sections. Even for
        identity restrictions that space depends on structural connectivity and
        stalk dimension, not on whether this particular assignment agrees.
        """
        tol = self._check_numeric(tol)
        parent = list(range(self._ncells))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b, meds in self.meets():
            ok = True
            for m in meds:
                ta, tb = self._transport(a, m), self._transport(b, m)
                if not self._agree(ta, tb, tol):
                    ok = False
                    break
            if ok:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        comp: dict[int, list[int]] = {}
        for c in range(self._ncells):
            comp.setdefault(find(c), []).append(c)
        return [cells for _root, cells in sorted(comp.items())]

    def _check_numeric(self, tol):
        if isinstance(tol, (bool, np.bool_)) or not np.isscalar(tol):
            raise TypeError("gluing tolerance must be a finite nonnegative real")
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("gluing tolerance must be finite and nonnegative")
        if not np.all(np.isfinite(self.stalks)) or any(
                not np.all(np.isfinite(matrix)) for matrix in self._R.values()):
            raise ValueError("numerical sections and restrictions must be finite")
        return tol

    @staticmethod
    def _agree(left, right, tol):
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("numerical restriction transport overflowed or is nonfinite")
        # Scale before norms/subtraction so finite, very large values cannot turn
        # inf > inf into a false success. This is the original relative test.
        scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
        a, b = left / scale, right / scale
        denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0 / scale)
        return float(np.linalg.norm(a - b)) / denominator <= tol

    def bind_boundary(self) -> None:
        """The restriction READ OFF B1, which is where it already was.

        At the incidence (cell e, mediator v) the boundary entry `B1[v,e]` is one number
        carrying all three pillars at once:

            existence     it is nonzero at all
            orientation   its SIGN, where -1 marks the distinguished vertex
            share         its MAGNITUDE 1/(k-1), derived from the span width

        So the restriction is that number, and nothing has to be invented, stored or
        encoded. `select()` below takes a caller's own labels instead, which is a
        different and much weaker thing: an indicator is EXISTENCE ONLY, with orientation
        and share thrown away.
        """
        if self.grade not in (0, 1):
            raise ValueError("boundary restrictions are currently declared for grades 0 and 1")
        if self._B1c is None:
            from rexgraph.native_sparse import NativeSparse
            self._B1c = list(NativeSparse(self.rex._B1_dual).columns())
        B = self._B1c
        for cell, meds in enumerate(self._inc):
            for m in meds:
                edge, vert = (cell, m) if self.grade == 1 else (m, cell)
                if not (0 <= edge < len(B)):
                    continue
                if vert in B[edge]:
                    self._R[(cell, m)] = float(B[edge][vert]) * np.eye(self.d)

    def select(self, cell: int, mediator: int, mask) -> None:
        """An INDICATOR restriction over labels the CALLER brings.

        Weaker than `bind_boundary` and worth saying why: an indicator is EXISTENCE
        alone (present or absent) with orientation and share discarded. The pillars
        are already at every incidence in `B1`, so a mask is an encoding invented beside
        a tensor that already carried the answer. Kept because a caller may genuinely have
        exogenous labels to carry; not the way to read structure.
        """
        m = np.asarray(mask, dtype=np.float64).ravel()
        if m.size != self.d:
            raise ValueError(f"mask has {m.size} entries for a stalk of dimension {self.d}")
        self.restrict(cell, np.diag(m), mediator=mediator)

    def admits(self, a: int, b: int) -> bool:
        """Do two cells still share a live label after their restrictions at every
        mediator they meet through. An indicator sheaf's version of "these glue"."""
        meds = self.mediators(a, b)
        if not meds:
            return False
        return all(np.any(self._transport(a, m) * self._transport(b, m)) for m in meds)


@dataclass(frozen=True)
class ExactGluingObstruction:
    """One failed incidence restriction in an exact sheaf section.

    ``residual`` is ``left - right`` after both local sections have been
    transported to the shared mediator.  It is intentionally retained per
    incidence: a count of failed pairs is useful bookkeeping, but it is not a
    cohomology class and it must not replace the object that failed to glue.
    """

    left_cell: int
    right_cell: int
    mediator: int
    left: tuple[Fraction, ...]
    right: tuple[Fraction, ...]
    residual: tuple[Fraction, ...]


class UndeclaredRestrictionError(ValueError):
    """A strict exact section attempted to inherit an undeclared incidence map."""

    def __init__(self, missing: tuple[tuple[int, int], ...]):
        self.missing = missing
        detail = ", ".join(f"({cell}, {mediator})" for cell, mediator in missing)
        super().__init__(
            "strict exact gluing requires a declared restriction at incidence " + detail
        )


@dataclass(frozen=True)
class ExactGlueResult:
    """Exact local to global reading of one declared sheaf section.

    ``components`` are the connected components of the current agreement graph,
    not certified global sections: a component can contain a failed pair joined
    by another path of successful comparisons. ``obstructions`` preserve
    every failed mediator comparison. The latter are deliberately not named
    ``H1``: they are representatives of the observed restriction failures, not
    a claim that their count is a computed cohomology group.
    """

    gluable: int
    glued: int
    components: tuple[tuple[int, ...], ...]
    obstructions: tuple[ExactGluingObstruction, ...]

    @property
    def ratio(self) -> Fraction:
        """The exact fraction of structural meeting pairs that glue."""
        return Fraction(self.glued, self.gluable) if self.gluable else Fraction(1)

    @property
    def h0(self) -> int:
        """Number of connected components in the exact gluing graph."""
        return len(self.components)

    @property
    def obstruction_count(self) -> int:
        """Number of failed incidence restrictions, not an H1 dimension."""
        return len(self.obstructions)

    @property
    def failed_pairs(self) -> tuple[tuple[int, int], ...]:
        """The meeting pairs with at least one exact failed restriction."""
        return tuple(sorted({(item.left_cell, item.right_cell) for item in self.obstructions}))


@dataclass(frozen=True)
class ExactSectionCheck:
    """Compact exact compatibility certificate for one assigned section.

    Compare each incidence to the lowest index incident cell at its mediator.
    Vanishing of these residuals is equivalent to all pair agreement, without
    constructing a pair graph. Nonzero residuals are anchor representatives,
    NOT the full pair list, a cohomology basis, or a structural complex merge.
    """

    incidence_count: int
    comparison_count: int
    obstructions: tuple[ExactGluingObstruction, ...]

    @property
    def compatible(self) -> bool:
        return not self.obstructions


class ExactSheaf:
    """Rational stalks and incidence restrictions over a relational complex.

    This is the exact counterpart to :class:`Sheaf`, not a tolerance mode of
    it.  Its local sections and restriction maps are integer/rational values;
    approximate values are refused instead of quietly being made into exact
    fractions.  Thus gluing is equality of transported sections at *every*
    shared mediator.  It applies unchanged to pairwise and branching primary
    relations because the mediator lattice is read from the declared boundary
    columns rather than from an expansion.

    The current exact path deliberately reports restriction representatives,
    not a saturated or inferred cohomology class.  A caller that has an
    explicit higher cohomology construction may consume ``obstructions`` as
    its input without losing which incidence supplied the residual.

    ``stalk_dims`` and ``mediator_dims`` optionally declare nonnegative widths
    in canonical cell order. An incidence map has shape (mediator width, stalk
    width). Without a map, identity is allowed only for equal widths in the
    non strict local carrier. ``stalk_dim`` remains the positive uniform default.
    No global square matrix is stored; caller declared local maps are small
    rectangular blocks, and sections are ragged exact coordinate tuples.
    """

    def __init__(self, rex, stalk_dim: int = 1, grade: int = 1, *,
                 require_declared_restrictions: bool = False,
                 stalk_dims=None, mediator_dims=None):
        grade = _grade(grade)
        stalk_dim = _dimension(stalk_dim, context="stalk_dim", minimum=1)
        if not isinstance(require_declared_restrictions, bool):
            raise TypeError("require_declared_restrictions must be a bool")
        rex._ensure_clean()
        self.rex = rex
        self.grade = int(grade)
        self.d = int(stalk_dim)
        self._ncells = _cell_count(rex, self.grade)
        self._inc = _incidences(rex, self.grade, self._ncells)
        nmediators = int(rex.nV) if grade == 1 else int(rex.nE)
        self.stalk_dimensions = self._dimensions(stalk_dims, self._ncells, "stalk_dims")
        self.mediator_dimensions = self._dimensions(mediator_dims, nmediators, "mediator_dims")
        self._stalks = [(Fraction(0),) * width for width in self.stalk_dimensions]
        self._R: dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]] = {}
        # Within one declared state, absent restriction means identity.  A phrase that
        # compares independently selected states instead sets this true, making every
        # correspondence map explicit and preventing an equality by coincidence join.
        self.require_declared_restrictions = require_declared_restrictions
        self._state = self._source_state()

    def _dimensions(self, values, count, context):
        result = (self.d,) * count if values is None else tuple(values)
        if len(result) != count:
            raise ValueError(f"{context} needs {count} entries in the declared cell order")
        return tuple(_dimension(v, context=context) for v in result)

    def _source_state(self):
        self.rex._ensure_clean()
        primary = tuple(tuple(int(v) for v in support) for support in self.rex.relation_supports())
        ids = self.rex.relation_ids
        ids = None if ids is None else tuple(int(v) for v in ids)
        face_state = None
        if self.grade == 2:
            b2 = getattr(self.rex, "_B2_hodge_dual", None)
            if b2 is not None:
                from rexgraph.native_sparse import NativeSparse
                face_state = ((int(b2.nrow), int(b2.ncol)),
                              tuple(tuple(sorted(c.items())) for c in NativeSparse(b2).columns()))
        return (int(self.rex.nV), int(self.rex.nE), _cell_count(self.rex, self.grade),
                primary, ids, face_state)

    def check_state(self) -> None:
        """Reject changed incidence/basis rather than reuse stale restriction addresses.

        Relation weights are not part of these unweighted incidence maps. This
        is a live source guard, not an RCDB transaction or a frozen snapshot.
        """
        if self._source_state() != self._state:
            raise ValueError("sheaf source incidence or basis changed; bind a fresh sheaf")

    @property
    def n_cells(self) -> int:
        return self._ncells

    @staticmethod
    def _scalar(value: object, *, context: str) -> Fraction:
        if isinstance(value, bool):
            raise TypeError(f"{context} must contain exact integers or Fractions, not booleans")
        if isinstance(value, Fraction):
            return value
        if isinstance(value, Integral):
            return Fraction(int(value))
        raise TypeError(
            f"{context} must contain exact integers or Fractions; "
            f"{type(value).__name__} is approximate or unsupported"
        )

    def _vector(self, values: object, *, context: str, width: int) -> tuple[Fraction, ...]:
        try:
            raw = tuple(values.tolist() if isinstance(values, np.ndarray) else values)
        except TypeError as exc:
            raise TypeError(f"{context} must be an iterable of {width} exact entries") from exc
        if len(raw) != width:
            raise ValueError(f"{context} has {len(raw)} entries for a stalk of dimension {width}")
        return tuple(self._scalar(value, context=context) for value in raw)

    def _matrix(self, values: object, *, context: str, shape) -> tuple[tuple[Fraction, ...], ...]:
        height, width = shape
        try:
            raw = values.tolist() if isinstance(values, np.ndarray) else values
            rows = tuple(tuple(row) for row in raw)
        except TypeError as exc:
            raise TypeError(f"{context} must be a {height} by {width} exact matrix") from exc
        if len(rows) != height or any(len(row) != width for row in rows):
            raise ValueError(f"{context} must be a {height} by {width} matrix")
        return tuple(
            tuple(self._scalar(value, context=context) for value in row)
            for row in rows
        )

    def _check_cell(self, cell: int) -> int:
        if isinstance(cell, bool) or not isinstance(cell, Integral):
            raise TypeError("cell must be an integer cell index")
        index = int(cell)
        if not 0 <= index < self._ncells:
            raise ValueError(f"cell {index} is not present at grade {self.grade}")
        return index

    def _check_mediator(self, cell: int, mediator: int) -> int:
        if isinstance(mediator, bool) or not isinstance(mediator, Integral):
            raise TypeError("mediator must be an integer cell index")
        index = int(mediator)
        if index not in self._inc[cell]:
            raise ValueError(
                f"mediator {index} is not incident to cell {cell} at grade {self.grade}"
            )
        return index

    def assign(self, cell: int, values: object) -> None:
        """Assign one exact local section to a declared stalk."""
        index = self._check_cell(cell)
        self._stalks[index] = self._vector(values, context="exact stalk", width=self.stalk_dimensions[index])

    def restrict(self, cell: int, matrix: object, *, mediator: int | None = None) -> None:
        """Set an exact incidence restriction, or one rule at every incidence."""
        index = self._check_cell(cell)
        mediators = self._inc[index] if mediator is None else [self._check_mediator(index, mediator)]
        # Validate the entire cell wide declaration before changing any incidence.
        # Materialize iterators once; different mediator dimensions must not result
        # in a partial update or consume one iterator several times.
        raw = tuple(tuple(row) for row in matrix)
        updates = {(index, item): self._matrix(raw, context="exact restriction", shape=(
            self.mediator_dimensions[item], self.stalk_dimensions[index])) for item in mediators}
        self._R.update(updates)

    def mediators(self, left: int, right: int) -> list[int]:
        """The shared lower or upper cells through which two stalks meet."""
        a, b = self._check_cell(left), self._check_cell(right)
        return sorted(set(self._inc[a]) & set(self._inc[b]))

    def meets(self) -> list[tuple[int, int, list[int]]]:
        """Every structural meeting pair once, retaining all shared mediators."""
        by_mediator: dict[int, list[int]] = {}
        for cell, mediators in enumerate(self._inc):
            for mediator in mediators:
                by_mediator.setdefault(mediator, []).append(cell)
        pairs: dict[tuple[int, int], set[int]] = {}
        for mediator, cells in by_mediator.items():
            for offset, left in enumerate(cells):
                for right in cells[offset + 1:]:
                    pair = (left, right) if left < right else (right, left)
                    pairs.setdefault(pair, set()).add(mediator)
        return [(left, right, sorted(mediators))
                for (left, right), mediators in sorted(pairs.items())]

    def _transport(self, cell: int, mediator: int) -> tuple[Fraction, ...]:
        matrix = self._R.get((cell, mediator))
        section = self._stalks[cell]
        if matrix is None:
            if self.stalk_dimensions[cell] != self.mediator_dimensions[mediator]:
                raise UndeclaredRestrictionError(((cell, mediator),))
            return section
        return tuple(
            sum((coefficient * value for coefficient, value in zip(row, section, strict=True)),
                Fraction(0))
            for row in matrix
        )

    def bind_boundary(self) -> None:
        """Read exact C1 existence/orientation/share restrictions from each boundary.

        Coefficients are reconstructed once from the declared C1 support: a wide relation
        contributes ``-1`` at its head and ``1/(arity-1)`` at every share, including the
        exact cancellation of a deliberate self loop.  This avoids both floating sparse
        B1 and repeated construction of full C0 chains.  Grade zero uses the same
        incidence through its containing relation.  Higher grade restrictions remain
        caller declared until their exact boundary carrier is specified.
        """
        if self.grade not in (0, 1):
            raise ValueError("exact boundary restrictions are currently declared for grades 0 and 1")
        self.check_state()
        if any(self.stalk_dimensions[c] != self.mediator_dimensions[m]
               for c, mediators in enumerate(self._inc) for m in mediators):
            raise ValueError("boundary identity restrictions require equal incidence dimensions")
        # the stored complex's own columns, so a declared head or share restricts as
        # declared; rebuilding them from the support alone answered the canonical column
        from rexgraph.native_rank import primary_columns
        coefficients = primary_columns(self.rex)
        for stalk, mediators in enumerate(self._inc):
            for mediator in mediators:
                edge, vertex = (stalk, mediator) if self.grade == 1 else (mediator, stalk)
                coefficient = coefficients[edge].get(vertex, Fraction(0))
                width = self.stalk_dimensions[stalk]
                matrix = tuple(
                    tuple(coefficient if row == column else Fraction(0) for column in range(width))
                    for row in range(width)
                )
                self._R[(stalk, mediator)] = matrix

    def glue(self) -> ExactGlueResult:
        """Glue exact local sections and retain every failed restriction residual."""
        transported = self._transported()
        parent = list(range(self._ncells))
        pairs = self.meets()

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        glued = 0
        obstructions: list[ExactGluingObstruction] = []
        for left, right, mediators in pairs:
            failures: list[ExactGluingObstruction] = []
            for mediator in mediators:
                left_value = transported[left, mediator]
                right_value = transported[right, mediator]
                residual = tuple(a - b for a, b in zip(left_value, right_value, strict=True))
                if any(residual):
                    failures.append(ExactGluingObstruction(
                        left, right, mediator, left_value, right_value, residual,
                    ))
            if failures:
                obstructions.extend(failures)
                continue
            glued += 1
            a, b = find(left), find(right)
            if a != b:
                parent[a] = b

        groups: dict[int, list[int]] = {}
        for cell_index in range(self._ncells):
            groups.setdefault(find(cell_index), []).append(cell_index)
        components = tuple(tuple(cells) for _root, cells in sorted(groups.items()))
        return ExactGlueResult(
            gluable=len(pairs), glued=glued, components=components,
            obstructions=tuple(obstructions),
        )

    def _transported(self):
        self.check_state()
        missing = tuple((c, m) for c, m in self.undeclared_restrictions()
                        if self.require_declared_restrictions or
                        self.stalk_dimensions[c] != self.mediator_dimensions[m])
        if missing:
            raise UndeclaredRestrictionError(missing)
        return {(c, m): self._transport(c, m)
                for c, mediators in enumerate(self._inc) for m in mediators}

    def check_section(self) -> ExactSectionCheck:
        """Check compatibility using one transport per incidence, no pair graph.

        Work follows local matrix entries and transported widths, plus exact
        coefficient bit growth. A k-way correspondence stays one primary
        relation; its k-1 anchor comparisons are a certificate, not new edges.
        """
        transported = self._transported()
        anchors = {}
        obstructions = []
        comparisons = 0
        for (cell, mediator), right in transported.items():
            if mediator not in anchors:
                anchors[mediator] = (cell, right)
                continue
            left_cell, left = anchors[mediator]
            comparisons += 1
            residual = tuple(a - b for a, b in zip(left, right, strict=True))
            if any(residual):
                obstructions.append(ExactGluingObstruction(
                    left_cell, cell, mediator, left, right, residual))
        return ExactSectionCheck(len(transported), comparisons, tuple(obstructions))

    def section_system(self, *, name="section", stalk_spaces=None, mediator_spaces=None, source=None):
        """Compile retained compatibility equations for exact section completion."""
        from rexgraph.section_calculus import SectionSystem
        return SectionSystem.from_sheaf(self, name=name, stalk_spaces=stalk_spaces,
                                       mediator_spaces=mediator_spaces, source=source)

    def undeclared_restrictions(self) -> tuple[tuple[int, int], ...]:
        """Incidences that would inherit identity if this section were not strict."""
        return tuple(
            (cell, mediator)
            for cell, mediators in enumerate(self._inc)
            for mediator in mediators
            if (cell, mediator) not in self._R
        )


def _rotation(d: int, angle: float) -> np.ndarray:
    """A rotation by `angle` in consecutive 2 blocks, with cos on a trailing odd component.

    The trailing `cos` is not padding: a 1 dimensional stalk has no plane to rotate in,
    and `cos(theta)` is how a connection Laplacian already reads the angle, so a d=1 sheaf
    and that Laplacian agree about what the angle means rather than quietly disagreeing.
    """
    R = np.eye(d, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    for i in range(0, d - 1, 2):
        R[i, i] = c
        R[i, i + 1] = -s
        R[i + 1, i] = s
        R[i + 1, i + 1] = c
    if d % 2 == 1:
        R[d - 1, d - 1] = c
    return R
