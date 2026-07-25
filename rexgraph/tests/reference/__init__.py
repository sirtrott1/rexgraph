"""
Pure-numpy reference implementations of rexgraph.core kernels.

These exist as test fixtures: every Cython kernel in rexgraph.core has a
pure-numpy reference here that the math-correctness tests compare against.

The pattern is:
    rexgraph.core._overlap.build_L_O          <- compiled Cython kernel
    tests.reference.channels_reference.build_L_O   <- pure-numpy oracle
    tests.test_overlap.test_compiled_matches_reference
        ↓
    asserts both produce the same numbers to ~1e-13 (BLAS reordering noise)

Why these exist:
    - Algebraic correctness oracle: if compiled disagrees with reference,
      the compiled kernel has a regression
    - Documentation: the pure-numpy code is human-readable and shows
      exactly what the math is doing
    - Portability: lets users without compiled rexgraph still run the
      framework (slowly) for development and debugging
    - Identity verification: each reference module exports
      verify_*_identities() functions that confirm framework algebraic
      identities hold (tr(RL)=4, χ simplex-valued, κ ∈ [0,1], etc.)

These should NEVER be used in production. The compiled kernels are
~50-100x faster. These exist purely for testing and documentation.

Convention:
    Each reference file mirrors the public functions of one rexgraph.core
    module, with identical signatures. The test files (test_*.py in this
    directory and tests/test_*.py) compare them.
"""
