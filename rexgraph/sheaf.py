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

import numpy as np

__all__ = ["Sheaf"]


class Sheaf:
    """Stalks of dimension `d` on the cells of one grade, with restrictions per incidence.

    `grade` selects what a cell is: 1 edges (the default, because edges are primary),
    0 vertices, 2 faces.
    """

    def __init__(self, rex, stalk_dim: int = 1, grade: int = 1):
        if int(grade) not in (0, 1, 2):
            raise ValueError(f"grade must be 0, 1 or 2; got {grade!r}")
        if int(stalk_dim) < 1:
            raise ValueError(f"stalk_dim must be >= 1; got {stalk_dim!r}")
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

    #### the cells, and what mediates between them #############################
    def _count_cells(self) -> int:
        return {0: int(self.rex.nV), 1: int(self.rex.nE), 2: int(self.rex.nF)}[self.grade]

    @property
    def n_cells(self) -> int:
        return self._ncells

    def _incidences(self) -> list[list[int]]:
        """cell -> the mediators incident to it, at this grade."""
        rex = self.rex
        if self.grade == 1:
            ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
            idx = np.asarray(rex._boundary_idx, dtype=np.int64)
            return [sorted(set(idx[ptr[e]:ptr[e + 1]].tolist()))
                    for e in range(self._ncells)]
        if self.grade == 0:
            ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
            idx = np.asarray(rex._boundary_idx, dtype=np.int64)
            out: list[list[int]] = [[] for _ in range(self._ncells)]
            for e in range(int(rex.nE)):
                for v in idx[ptr[e]:ptr[e + 1]]:
                    out[int(v)].append(e)          # the mediator IS the edge
            return [sorted(set(s)) for s in out]
        b2 = getattr(rex, "_B2_hodge_dual", None)
        if b2 is None or self._ncells == 0:
            return [[] for _ in range(self._ncells)]
        from rexgraph.core._sparse import to_scipy_csr
        B2 = to_scipy_csr(b2).tocsc()
        return [sorted(set(B2.indices[B2.indptr[f]:B2.indptr[f + 1]].tolist()))
                for f in range(self._ncells)]

    def mediators(self, a: int, b: int) -> list[int]:
        """Every cell one grade away incident to BOTH, which is what makes them meet."""
        return sorted(set(self._inc[int(a)]) & set(self._inc[int(b)]))

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

    #### the data and the inheritance rules ####################################
    def assign(self, cell: int, vec) -> None:
        v = np.asarray(vec, dtype=np.float64).ravel()
        self.stalks[int(cell), :min(self.d, v.size)] = v[:self.d]

    def restrict(self, cell: int, mat, *, mediator: int | None = None) -> None:
        """The inheritance rule. Identity means inherit unchanged, a scale means inherit
        at a factor, zero means do not inherit that component at all.

        `mediator=None` states the rule about the CELL, which applies it at every
        incidence of that cell, which is convenient and exactly the degenerate case that cannot
        express holonomy. Name a mediator to distinguish the ends.
        """
        M = np.asarray(mat, dtype=np.float64).reshape(self.d, self.d)
        meds = self._inc[int(cell)] if mediator is None else [int(mediator)]
        for m in meds:
            self._R[(int(cell), int(m))] = M

    def _transport(self, cell: int, med: int) -> np.ndarray:
        R = self._R.get((int(cell), int(med)))
        s = self.stalks[int(cell)]
        return s if R is None else R @ s

    #### the reading ###########################################################
    def glue(self, *, tol: float = 1e-9) -> dict:
        """How far the section is from agreeing across every meeting, in [0, 1].

        Read relatively: the scale of a restriction belongs to the caller, so the reading
        is normalised by the transported magnitudes rather than by a fixed tolerance.
        """
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
                scale = max(float(np.linalg.norm(ta)), float(np.linalg.norm(tb)))
                if float(np.linalg.norm(ta - tb)) > tol * max(scale, 1.0):
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
        comps = len({find(c) for c in range(self._ncells)}) if self._ncells else 1
        return {"ratio": (glued / gluable) if gluable else 1.0,
                "gluable": gluable, "glued": glued,
                "H0": comps, "H1": len(failed), "failed": failed}

    #### holonomy: the curl of a connection ####################################
    def holonomy(self, theta) -> np.ndarray:
        """`B2^T theta`: the transport accumulated around each face."""
        b2 = getattr(self.rex, "_B2_hodge_dual", None)
        if b2 is None or int(self.rex.nF_hodge) == 0:
            return np.zeros(0, dtype=np.float64)
        from rexgraph.core._sparse import to_scipy_csr
        B2 = to_scipy_csr(b2).tocsr()
        return np.asarray(B2.T @ np.asarray(theta, dtype=np.float64).ravel()).ravel()

    def is_flat(self, theta) -> bool:
        """Is every face's holonomy zero, i.e. is the section parallel."""
        h = self.holonomy(theta)
        return bool(h.size == 0 or np.abs(h).max() == 0.0)

    def gradient_angles(self, potential) -> np.ndarray:
        """The per-incidence angle a bound connection carries."""
        from rexgraph.core._sparse import to_scipy_csr
        B1 = to_scipy_csr(self.rex._B1_dual).tocsr()
        return np.asarray(B1.T @ np.asarray(potential, dtype=np.float64).ravel()).ravel()

    def bind_connection(self, theta) -> None:
        """Bind a U(1) connection to the incidences, signed by the boundary entry.

        Per incidence rather than per cell, so the two ends of a cell carry opposite
        rotations and a cycle can fail to close. A 1-dimensional stalk cannot see this;
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
            from rexgraph.core._sparse import to_scipy_csr
            self._B1c = to_scipy_csr(self.rex._B1_dual).tocsc()
        lo, hi = self._B1c.indptr[int(edge)], self._B1c.indptr[int(edge) + 1]
        rows = self._B1c.indices[lo:hi]
        hit = np.flatnonzero(rows == int(vertex))
        if not hit.size:
            return 0.0
        v = float(self._B1c.data[lo + int(hit[0])])
        return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


    #### global sections #######################################################
    def sections(self, *, tol: float = 1e-9) -> list[list[int]]:
        """The components of the gluing graph: cells that must share an assignment.

        `H0` counts these; this returns the membership, which is what a caller acts on.
        One free value per component, so the space of global sections has dimension
        `len(sections())` for identity restrictions.
        """
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
                scale = max(float(np.linalg.norm(ta)), float(np.linalg.norm(tb)))
                if float(np.linalg.norm(ta - tb)) > tol * max(scale, 1.0):
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
        if self._B1c is None:
            from rexgraph.core._sparse import to_scipy_csr
            self._B1c = to_scipy_csr(self.rex._B1_dual).tocsc()
        B = self._B1c
        for cell, meds in enumerate(self._inc):
            for m in meds:
                edge, vert = (cell, m) if self.grade == 1 else (m, cell)
                if not (0 <= edge < B.shape[1]):
                    continue
                lo, hi = B.indptr[edge], B.indptr[edge + 1]
                rows = B.indices[lo:hi]
                hit = np.flatnonzero(rows == vert)
                if hit.size:
                    self._R[(cell, m)] = float(B.data[lo + int(hit[0])]) * np.eye(self.d)

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
        self._R[(int(cell), int(mediator))] = np.diag(m)

    def admits(self, a: int, b: int) -> bool:
        """Do two cells still share a live label after their restrictions at every
        mediator they meet through. An indicator sheaf's version of "these glue"."""
        meds = self.mediators(a, b)
        if not meds:
            return False
        for m in meds:
            if not np.any(self._transport(a, m) * self._transport(b, m)):
                return False
        return True


def _rotation(d: int, angle: float) -> np.ndarray:
    """A rotation by `angle` in consecutive 2-blocks, with cos on a trailing odd component.

    The trailing `cos` is not padding: a 1-dimensional stalk has no plane to rotate in,
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
