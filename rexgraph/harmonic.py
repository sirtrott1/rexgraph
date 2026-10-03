"""Compatibility surface for the retired dense harmonic reference.

The implementation now lives in :mod:`rexgraph.reference.harmonic`; sparse production
code must use :mod:`rexgraph.harmonic_sparse` / :mod:`rexgraph.hodge_coords`.
"""
from rexgraph.reference.harmonic import (
    build_prime_complex,
    get_harmonic_basis,
    harmonic_product_structure,
    prime_coupling_matrix,
    prime_removal_test,
)

__all__ = [
    "build_prime_complex",
    "get_harmonic_basis",
    "harmonic_product_structure",
    "prime_coupling_matrix",
    "prime_removal_test",
]
