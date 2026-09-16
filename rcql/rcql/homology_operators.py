"""Read core quotient homology; RCQL introduces no rank or grouping algorithm."""
from .execution_trace import record_method


def _split(source, grade):
    from rexgraph.native_homology import homology_split
    result = homology_split(source, grade)
    record_method("native-exact-homology-quotient", grade=result.grade,
                  algorithms=result.rank_methods, chain_condition="exact-zero",
                  chain_multiplicity=result.chain_multiplicity,
                  quotient_size=result.quotient_size, lower_rank=result.lower_rank,
                  upper_rank=result.upper_rank, quotient_upper_rank=result.quotient_upper_rank)
    return result


def simple_homology(source, grade):
    return _split(source, grade).simple


def multiplicity_homology(source, grade):
    return _split(source, grade).multiplicity


def install(register):
    register("SIMPLE_HOMOLOGY")(simple_homology)
    register("MULTIPLICITY_HOMOLOGY")(multiplicity_homology)
