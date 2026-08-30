"""rexgraph.flow.ternary_cochain: the composite-binary cochain and its reduction.

A composite-binary field carries entries in {-1, 0, +1} against one scale. Reducing a
real field to that form has NO free parameter once the scale is derived, and deriving it
is the whole of the construction.

THE REDUCTION. For a code `q` and a scale `s` the residual is `r = x - s q`, and the `s`
that minimises its mass is `s = <x,q>/Q(q)`. Substituting gives

    Q(r)  =  Q(x) - <x,q>^2/Q(q)  =  Q(x) * spread(x, q)

so the residual mass is the field's own quadrance times the SPREAD between the field and
its code. That makes the reduction exact rather than chosen: the right code is the one of
least spread, and `rational_trig.spread` is the quantity already carrying that name.

    dev = spread(x, q)  in [0, 1]     the deviation from the exact composite binary
    Q(r) = Q(x) * dev                 the mass it costs, extensive over cells

Nothing here is a threshold and nothing is a statistic. `1 - spread` is maximised at a
support of the `m` largest magnitudes for each `m`, since `<x,q>` is a sum of magnitudes
once the signs match, so the sweep over `m` is a sort and a prefix sum and its argmax is
an exact comparison. A field already in {-1,0,1} up to scale reduces to itself at `dev =
0`, which a magnitude cutoff does not do.

THE TOWER. Whatever the reduction leaves is a field in its own right, so it reduces the
same way, and `residual_tower` carries that down. Level `k`'s mass is `Q(x) * prod(dev)`
over the levels above it, so the tower reads what each level actually bought. It stops
where the reading repeats, the rule `tower.semantic_closure` uses: more levels stopped
being more field. Two ternary levels cost 4 bits an entry.

WHY PACK IT. The product that matters is the pairing with a +-1 query, which is the
packed product

    score[e] = k[e] - 2 * popcount( P[e] & (S[e] ^ q) )

an integer, with nothing rounded and no vector of floats to read. The float path needs a
dense query vector, which is the embedding a relational model exists to avoid, and it
measured 121.7 Gentry/s against the packed path's 854.6 on the same machine.

WHERE PACKING APPLIES. A field dense in (cells x classes). NOT the co-participation
adjacency, which is weighted rather than ternary, and NOT the boundary, which
`boundary_ptr`/`boundary_idx` already stores without values. Density decides the rest:
planes cost 2 bits an entry whatever the fill, a CSR form about 12 bytes a nonzero, so
packing wins above a fill of 2/(8*12) and branching is what carries an operator across
it. Measured on a 400-edge ring: 0.7% fill at arity 2, 3.7% at arity 8, 31.8% at 64.
"""
from __future__ import annotations

import numpy as np

from rexgraph import ternary as tn

__all__ = ["TernaryCochain", "ternary_reduce", "residual_tower",
           "packing_pays", "packed_bytes", "PACKING_DENSITY"]

#: Fill above which two bitplanes are smaller than a CSR form of the same operator.
#: Exact, not tuned: 2 bits per entry against (8 byte value + 4 byte index) per nonzero.
PACKING_DENSITY = 2.0 / (8 * 12)


def packed_bytes(shape: tuple[int, int]) -> int:
    """What two bitplanes ACTUALLY cost for this shape.

    Not `entries / 4`. A row is padded to a whole 64-bit word, so a short packed axis
    wastes most of every word: 4 classes use 4 bits of 64 and the planes come out 2x
    smaller than float64 rather than 32x. The win needs the packed axis at 64 or more,
    and is only exact at a multiple of 64.
    """
    rows, cols = int(shape[0]), int(shape[1])
    return rows * ((cols + 63) // 64) * 8 * 2


def packing_pays(nnz: int, shape: tuple[int, int]) -> bool:
    """Whether bitplanes are smaller than CSR for an operator of this fill AND shape.

    Both conditions have to hold. A sparse operator is cheaper as CSR because planes
    cost the same whatever the fill, and a narrow one is cheaper dense because the
    padding dominates. Neither is a tuned constant: both come out of the byte counts.
    """
    n = int(shape[0]) * int(shape[1])
    if not n:
        return False
    csr = nnz * 12
    return packed_bytes(shape) < min(csr, n * 8)


def ternary_reduce(values):
    """The composite-binary code of least spread against `values`, row by row.

    Returns `(q, scale, deviation)`: the {-1,0,1} code, the scale `<x,q>/Q(q)` that
    minimises the residual for it, and `spread(x, q)`, which IS the fraction of the row's
    mass the code fails to carry. `x` is recovered as `scale * q` exactly when the
    deviation is 0.

    The support is swept rather than cut. For a support of size `m` the inner product is
    the sum of the `m` largest magnitudes, so the best code of that size is fixed, and the
    objective `(sum of top m)^2 / m` is compared across every `m` exactly. Ties take the
    smaller support, which is the cheaper code for the same fidelity.

    A zero row reduces to a zero code at deviation 0: there was no mass to lose. That is
    an absence and not a perfect fit, and it is why `spread` returning None on a zero
    vector is not carried through here.
    """
    a = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if a.ndim != 2:
        raise ValueError(f"a cochain is (cells, classes), got shape {a.shape}")
    rows, cols = a.shape
    if cols == 0:
        z = np.zeros((rows, 0), dtype=np.int8)
        return z, np.zeros(rows), np.zeros(rows)

    mag = np.abs(a)
    order = np.argsort(-mag, axis=1, kind="stable")
    pref = np.cumsum(np.take_along_axis(mag, order, axis=1), axis=1)
    m = np.arange(1, cols + 1, dtype=np.float64)
    best = np.argmax(pref * pref / m, axis=1)        # first max, so the smallest support

    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(cols), (rows, cols)), axis=1)
    q = np.where(rank <= best[:, None], np.sign(a), 0.0).astype(np.int8)

    ip = pref[np.arange(rows), best]                  # <x, q>, the kept magnitudes
    nnz = (best + 1).astype(np.float64)               # Q(q), a popcount
    qx = np.einsum("ij,ij->i", a, a)                  # Q(x)
    live = qx > 0.0
    scale = np.zeros(rows)
    dev = np.zeros(rows)
    scale[live] = ip[live] / nnz[live]
    dev[live] = 1.0 - (ip[live] * ip[live]) / (qx[live] * nnz[live])
    # a precision guard, not a threshold: spread is in [0, 1] by construction and only
    # float rounding puts it a few ulps outside
    return q, scale, np.clip(dev, 0.0, 1.0)


def residual_tower(values, *, max_levels: int = 8):
    """Reduce, then reduce what is left, until the reading stops moving.

    Each level is a `(code, scale)` pair and the field is the sum of `scale * code` over
    them, so `k` levels cost `2k` bits an entry. `masses[k]` is the mass still unexplained
    below level `k`, which is `Q(x)` times the product of the deviations above it, so the
    tower reports what each level bought rather than asserting it.

    Stops when a level carries no mass, or when the mass reading repeats: more levels
    stopped being more field. That is `tower.semantic_closure`'s rule and it needs no
    tolerance, since the reading either changed or it did not.
    """
    a = np.atleast_2d(np.asarray(values, dtype=np.float64))
    residual = a.copy()
    levels, masses = [], [float(np.einsum("ij,ij->i", a, a).sum())]
    for _ in range(int(max_levels)):
        q, scale, _ = ternary_reduce(residual)
        if not q.any():
            break
        residual = residual - scale[:, None] * q
        mass = float(np.einsum("ij,ij->i", residual, residual).sum())
        if mass == masses[-1]:
            break
        levels.append((q, scale))
        masses.append(mass)
        if mass == 0.0:
            break
    return {"levels": levels, "masses": masses,
            "bits_per_entry": 2 * len(levels),
            "retained": (1.0 - masses[-1] / masses[0]) if masses[0] else 1.0}


class TernaryCochain:
    """A {-1,0,1} cochain over cells, held as bitplanes.

    Carries no float in its product. `score` pairs it with a +-1 query and returns
    integers; `predict` reads the class each cell votes for. Both route through
    `rexgraph.compute`, so the cpu, openmp and any device lane are reachable without a
    call-site change.
    """

    __slots__ = ("_op", "n_cells", "n_classes", "_device", "scale", "deviation")

    def __init__(self, values, *, scale=None, deviation=None):
        """`values` is (n_cells, n_classes) with every entry in {-1, 0, 1}."""
        a = np.asarray(values)
        if a.ndim != 2:
            raise ValueError(f"a cochain is (cells, classes), got shape {a.shape}")
        self._op = tn.pack(a)                      # refuses anything not ternary
        self.n_cells, self.n_classes = self._op.shape
        self._device = None
        #: per-cell scale carrying the code back to the field it reduced, if it reduced one
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        #: per-cell spread against that field: the deviation from the exact composite binary
        self.deviation = None if deviation is None else np.asarray(deviation, dtype=np.float64)

    @classmethod
    def from_float(cls, values) -> TernaryCochain:
        """Reduce a real cochain to composite binary at least spread.

        No threshold and no magnitude cutoff: see `ternary_reduce`. The scale and the
        deviation come back on the result, so what the reduction cost is readable rather
        than assumed.
        """
        q, scale, dev = ternary_reduce(values)
        return cls(q, scale=scale, deviation=dev)

    @property
    def nbytes(self) -> int:
        return self._op.nbytes

    def dense(self) -> np.ndarray:
        """The {-1,0,1} array, for inspection and for checking a lane."""
        return self._op.dense()

    def field(self) -> np.ndarray:
        """`scale * code`, the field this cochain reduced. Needs a scale to exist."""
        if self.scale is None:
            raise ValueError("no scale: this cochain was built from a ternary array, "
                             "not reduced from a field")
        return self.scale[:, None] * self.dense()

    def support(self) -> np.ndarray:
        """How many classes each cell actually carries. A popcount, not a sum."""
        return self._op.arity()

    def score(self, query, *, prefer: str | None = None) -> np.ndarray:
        """Pair every cell with a +-1 query over the classes. Exact integers."""
        q = np.asarray(query)
        if q.ndim != 1 or q.shape[0] != self.n_classes:
            raise ValueError(f"query of length {self.n_classes} required, got {q.shape}")
        return tn.matvec(self._op, q, prefer=prefer)

    def predict(self) -> np.ndarray:
        """The class each cell votes for.

        Pairing with the one-hot `e_c` returns `q[:, c]`, so the vote is the entry itself
        and there is no product to take. Ties go to the lowest class, which `argmax`
        already does; a ternary cochain has many of them by construction and a caller
        wanting them broken has to say how.
        """
        return self.dense().argmax(1)

    def to(self, device: str = "cuda"):
        """Hold the planes on a device. See `rexgraph.ternary` on why residency is
        explicit: the planes ARE the cochain, so re-sending them per product costs more
        than the product."""
        self._device = self._op.to(device)
        return self._device

    def __repr__(self) -> str:
        dense = self.n_cells * self.n_classes * 8
        dev = ("" if self.deviation is None
               else f", dev {float(self.deviation.max()):.3g} worst")
        return (f"TernaryCochain({self.n_cells} cells x {self.n_classes} classes, "
                f"{self.nbytes / 1e6:.3f} MB against {dense / 1e6:.3f} as float64, "
                f"{dense / max(self.nbytes, 1):.1f}x{dev})")


def from_cochain_model(model) -> TernaryCochain:
    """Reduce a trained `CoParticipationCochain` to the composite-binary primitive.

    The model's forward is the identity, so its parameter IS its output and reducing the
    parameter reduces the model. Nothing about the complex changes: the structure lived in
    the optimizer's Green's preconditioner during training and is not carried in the
    cochain at all. Read `.deviation` to see what the reduction cost before trusting it.
    """
    z = model.Z.detach().cpu().numpy() if hasattr(model, "Z") else np.asarray(model)
    return TernaryCochain.from_float(z)
