"""Primary relation overlap through the unsigned weighted boundary tensor.

The C1 axis remains the original relation axis, including document spans.
Stored sentence partitions are not substituted for primary relation identity.
"""
from fractions import Fraction as Q

import numpy as np

from rexgraph.channel_operator import _exact_gram_action
from rexgraph.io.catalog import object_digest
from rexgraph.linear_operator import RexOperator, _exact_array, _numeric_array
from rexgraph.markov import _participation_entries, validate_markov_source
from rexgraph.native_sparse import native_coo


class TextOverlapView(RexOperator):
    """Apply X^T(Xz) minus its diagonal, with X=abs(B1) W.

    This symmetric overlap action is not positive semidefinite in general.
    It is neither a stochastic operator nor a replacement document structure.
    """
    def __init__(self, source):
        weights = validate_markov_source(source)
        n, m = int(source.nV), int(source.nE)
        digest = object_digest(source)
        entries = _participation_entries(source, weights)
        q = np.full(m, Q(0), object)
        columns = [{} for _ in range(m)]
        for v, e, u in entries:
            q[e] += u*u
            columns[e][v] = u
        cache = {}

        def check():
            if object_digest(source) != digest:
                raise ValueError("text overlap source state changed; bind a fresh view")

        def action(values, exact=False):
            check()
            x = _exact_array(values) if exact else _numeric_array(values, operation="text overlap action")
            if exact:
                out = _exact_gram_action(columns, x)
                diagonal = q
            else:
                if "numeric" not in cache:
                    coefficients = _numeric_array([u for v, e, u in entries], operation="text overlap tensor")
                    if np.any(coefficients == 0):
                        raise FloatingPointError("text overlap coefficients underflow float64; use exact action")
                    tensor = native_coo(np.array([v for v, e, u in entries], dtype=np.int64),
                                        np.array([e for v, e, u in entries], dtype=np.int64),
                                        coefficients, (n, m))
                    diagonal = _numeric_array(q, operation="text overlap diagonal")
                    if any(a and b == 0 for a, b in zip(q, diagonal, strict=True)):
                        raise FloatingPointError("text overlap diagonal underflows float64; use exact action")
                    cache["numeric"] = tensor, diagonal
                tensor, diagonal = cache["numeric"]
                out = tensor.transpose_apply(tensor.apply(x))
            out -= diagonal.reshape((m,)+(1,)*(x.ndim-1))*x
            check()
            return out if exact else _numeric_array(out, operation="text overlap result")

        super().__init__("TEXT_OVERLAP_VIEW", (m, m), 1, 1, action, source=source,
            construction="text-overlap", variance="cochain", symmetric=True,
            transpose_matvec=action, exact_matvec=lambda x: action(x, True),
            exact_transpose_matvec=lambda x: action(x, True),
            parameters=(("participation", "abs(B1) W"), ("diagonal", "removed")))
        object.__setattr__(self, "check_state", check)
