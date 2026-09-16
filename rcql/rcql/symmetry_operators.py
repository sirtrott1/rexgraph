"""Symmetry adapters reuse Core certificates and sparse group composition."""
from .execution_trace import record_method


def symmetry(source, generators, word=()):
    from rexgraph.chain_map import SymmetryGroup
    result = SymmetryGroup(generators, word)
    if result.domain.source is not source:
        raise ValueError("symmetry generators must be bound to this source")
    record_method("core-exact-chain-symmetry", generators=result.generator_count,
                  word=result.word, metric="euclidean", product="lazy")
    return result


def install(register):
    register("SYMMETRY")(symmetry)
