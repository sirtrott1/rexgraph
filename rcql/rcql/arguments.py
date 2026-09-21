"""Pure syntax argument contracts, pinned against every registered adapter.

Named arguments are bound in declared parameter order before static inference.
Defaults fill interior holes only; omitted trailing options remain omitted.
No numeric package or execution adapter is imported to parse or build a query.
"""
from __future__ import annotations

EXPRESSION_ARGUMENTS = {
    'COUNT': (('values',), ()),
    'SUM': (('values',), ()),
    'MEAN': (('values',), ()),
    'ACCESS': (('value', 'accession', 'exact'), (False,)),
    'ACCESS_TYPES': (('value', 'accessions', 'exact'), (False,)),
    'ACCUMULATE': (('left', 'right'), ()),
    'APPLY': (('action', 'values', 'exact'), (False,)),
    'ADJOINT': (('operator', 'domain_metric', 'codomain_metric'), (None, None)),
    'DIRAC': (('metrics',), (None,)),
    'ANTI_DIRAC': (('metrics',), (None,)),
    'COMMUTATOR': (('left', 'right'), ()),
    'ANTICOMMUTATOR': (('left', 'right'), ()),
    'GRADED_CHAIN': (('components',), ()),
    'GRADE_COMPONENT': (('state', 'grade'), ()),
    'ARITY': (('value',), ()),
    'BETTI': (('grade',), ()),
    'BOUNDARY': (('grade', 'values'), (None,)),
    'CELL': (('grade', 'index'), ()),
    'CELLS': (('grade', 'indices'), (None,)),
    'CHAIN_MAP': (('declaration',), ()),
    'CHANNEL': (('name',), ()),
    'CHARACTER': (('exact',), (False,)),
    'CHARACTER_ENERGY': (('action',), ()),
    'CLOSURE': (('seed', 'max_depth', 'grade'), (8, 0)),
    'COBOUNDARY': (('grade', 'values'), (None,)),
    'COMPOSITE': (('value',), ()),
    'CORELATIONS': (('value',), ()),
    'CO_RELATE': (('left', 'right', 'metric', 'exact'), (None, False)),
    'DESCRIBE': ((), ()),
    'ENCLOSURE': (('value',), ()),
    'EXISTENCE': (('value',), ()),
    'FILES': (('limit', 'offset'), (100, 0)),
    'FILE_HASH': (('name',), ()),
    'FILE_INFO': (('name',), ()),
    'GLUE': (('section',), ()),
    'SECTION_CHECK': (('section',), ()),
    'GRADE': (('value',), (None,)),
    'GREEN': (('values',), (None,)),
    'GREEN_SOLVE': (('action', 'values'), ()),
    'HARMONIC': (('flow',), ()),
    'HASH_FILES': ((), ()),
    'HEAD': (('value',), ()),
    'HODGE': (('flow',), ()),
    'HODGE_COORDS': (('flow',), ()),
    'HODGE_OPERATOR': (('grade', 'alpha'), (1,)),
    'HODGE_DOWN': (('grade', 'metric', 'lower_metric'), (None, None)),
    'HODGE_UP': (('grade', 'metric', 'upper_metric'), (None, None)),
    'HODGE_SUM': (('grade', 'metric', 'lower_metric', 'upper_metric'), (None, None, None)),
    'HODGE_DIFFERENCE': (('grade', 'metric', 'lower_metric', 'upper_metric'), (None, None, None)),
    'INDICATOR': (('value',), ()),
    'INTEGRATE': (('cochain', 'chain', 'exact'), (False,)),
    'METRIC': (('grade', 'weights'), (None,)),
    'METRIC_CURVATURE': (('metric',), ()),
    'MOMENT': (('left', 'right', 'metric', 'exact'), (None, False)),
    'MOMENT_TENSOR': (('family', 'metric', 'exact'), (None, False)),
    'NULLITY': (('value',), ()),
    'QUADRANCE': (('values', 'exact', 'metric'), (False, None)),
    'RANK': (('value',), ()),
    'RCDB_COMMITS': (('record_id', 'limit'), (1000,)),
    'RCDB_GET': (('record_id',), ()),
    'RCDB_HASH': (('record_id',), ()),
    'RCDB_HISTORY': (('record_id',), ()),
    'RCDB_LIST': (('limit', 'offset'), (100, 0)),
    'RCDB_SEARCH': (('text', 'limit'), (100,)),
    'RCDB_SECURITY': ((), ()),
    'RCDB_STATE_HASH': ((), ()),
    'RCDB_STATS': ((), ()),
    'RCDB_VERIFY': (('record_id',), ()),
    'RELATION_SIGNAL': (('signal', 'channel'), ('amplitude',)),
    'RESOLVENT': (('action', 'alpha', 'tol', 'maxiter'), (1.0, 1e-10, 1000)),
    'REX': (('name',), ()),
    'SCALE_MOMENT': (('action', 'order', 'local', 'exact'), (False, False)),
    'SEARCH': (('text', 'limit'), (100,)),
    'SEARCH_TENSORS': (('name', 'text', 'limit'), (100,)),
    'SHARE': (('value',), ()),
    'SHARE_SUPPORT': (('value',), ()),
    'SHOW_OPERATORS': (('limit', 'offset'), (1000, 0)),
    'SIGNAL_AT': (('signal', 'key'), ()),
    'SIGNAL_FLOW': (('signal', 'channel'), ('structural',)),
    'SIGNAL_HODGE': (('signal', 'channel'), ('amplitude',)),
    'SIGNAL_SOURCE': (('signal', 'channel'), ('structural',)),
    'SIGNIFICANCE': (('edge',), ()),
    'SPREAD': (('left', 'right', 'exact', 'metric'), (False, None)),
    'STAR': (('value',), ()),
    'STAR_CHARACTER': (('cell', 'exact'), (False,)),
    'STATE_HASH': ((), ()),
    'TEMPORAL_DELTA': (('step',), ()),
    'TENSORS': (('name', 'limit'), (1000,)),
    'TENSOR_MANIFEST': (('name', 'limit'), (1000,)),
    'WINDING': (('flow',), ()),
    'ZERO': (('grade', 'kind'), ('cochain',)),
}

from .names import insert_unique
from .value_contracts import ARGUMENTS as _VALUE_ARGUMENTS
from .calculus_contracts import ARGUMENTS as _CALCULUS_ARGUMENTS
from .temporal_contracts import ARGUMENTS as _TEMPORAL_ARGUMENTS

for _name, _arguments in _VALUE_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

from .critical_contracts import ARGUMENTS as _CRITICAL_ARGUMENTS
from .certificate_contracts import ARGUMENTS as _CERTIFICATE_ARGUMENTS
from .structure_contracts import ARGUMENTS as _STRUCTURE_ARGUMENTS
from .homology_contracts import ARGUMENTS as _HOMOLOGY_ARGUMENTS
from .partition_contracts import ARGUMENTS as _PARTITION_ARGUMENTS
from .artifact_contracts import ARGUMENTS as _ARTIFACT_ARGUMENTS
from .filling_contracts import ARGUMENTS as _FILLING_ARGUMENTS
from .difference_contracts import ARGUMENTS as _DIFFERENCE_ARGUMENTS
from .coordinate_contracts import ARGUMENTS as _COORDINATE_ARGUMENTS

for _name, _arguments in _COORDINATE_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

from .document_contracts import ARGUMENTS as _DOCUMENT_ARGUMENTS
from .rational_contracts import ARGUMENTS as _RATIONAL_ARGUMENTS
from .markov_contracts import ARGUMENTS as _MARKOV_ARGUMENTS
from .replay_contracts import ARGUMENTS as _REPLAY_ARGUMENTS
from .symmetry_contracts import ARGUMENTS as _SYMMETRY_ARGUMENTS
from .corpus_contracts import ARGUMENTS as _CORPUS_ARGUMENTS
from .turn_contracts import ARGUMENTS as _TURN_ARGUMENTS

for _name, _arguments in _TURN_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _CORPUS_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _SYMMETRY_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _REPLAY_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _MARKOV_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _RATIONAL_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _DOCUMENT_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _DIFFERENCE_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _FILLING_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _ARTIFACT_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _PARTITION_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _HOMOLOGY_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _STRUCTURE_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

for _name, _arguments in _CRITICAL_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)
for _name, _arguments in _CERTIFICATE_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)
for _name, _arguments in _CALCULUS_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)
for _name, _arguments in _TEMPORAL_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

SOURCE_ARGUMENTS = {
    "REX": (("name",), ()),
    "RCDB": (("name",), ()),
    "CATALOG": (("name",), ()),
    "FILE": (("catalog_name", "entry_name"), ()),
    "PHRASE": (("section",), ()),
    "AT": (("source", "version"), ()),
    "AT_TIME": (("source", "time"), ()),
    "RCDB_GET": (("source", "record_id"), ()),
    "RCDB_VERSION": (("source", "record_id", "version"), ()),
    "RCDB_AS_OF": (("source", "record_id", "time"), ()),
    "RCDB_VALID_AT": (("source", "record_id", "time"), ()),
    "VALID_AT": (("source", "record_id", "time"), ()),
    "TRANSACTION_AT": (("source", "record_id", "time"), ()),
}


def bind_arguments(name, args, keywords, *, source=False):
    """Bind syntactic arguments without running an operator or coercing a value."""
    from .ast import Literal
    from .names import canonical_name
    name = canonical_name(name)
    if not keywords:
        return tuple(args)
    catalogue = SOURCE_ARGUMENTS if source else EXPRESSION_ARGUMENTS
    if name not in catalogue:
        raise TypeError(f"{name} has no declared named-argument contract")
    names, defaults = catalogue[name]
    if len(args) > len(names):
        raise TypeError(f"{name} takes at most {len(names)} arguments")
    assigned = {i: arg for i, arg in enumerate(args)}
    for key, value in keywords:
        if key not in names:
            raise TypeError(f"{name} has no argument {key!r}")
        index = names.index(key)
        if index in assigned:
            raise TypeError(f"{name} argument {key!r} supplied twice")
        assigned[index] = value
    required = len(names) - len(defaults)
    end = max(max(assigned, default=-1) + 1, required)
    result = []
    for index in range(end):
        if index in assigned:
            result.append(assigned[index])
        elif index >= required:
            result.append(Literal(defaults[index - required]))
        else:
            raise TypeError(f"{name} missing required argument {names[index]!r}")
    return tuple(result)

from .tensor_contracts import ARGUMENTS as _TENSOR_ARGUMENTS
for _name, _arguments in _TENSOR_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

from .section_contracts import ARGUMENTS as _SECTION_ARGUMENTS
for _name, _arguments in _SECTION_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

from .model_contracts import ARGUMENTS as _MODEL_ARGUMENTS
for _name, _arguments in _MODEL_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

from .molecular_contracts import ARGUMENTS as _MOLECULAR_ARGUMENTS
for _name, _arguments in _MOLECULAR_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

from .program_contracts import ARGUMENTS as _PROGRAM_ARGUMENTS
for _name, _arguments in _PROGRAM_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

from .recursion_contracts import ARGUMENTS as _RECURSION_ARGUMENTS
for _name, _arguments in _RECURSION_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)

from .transformation_contracts import ARGUMENTS as _TRANSFORMATION_ARGUMENTS
for _name, _arguments in _TRANSFORMATION_ARGUMENTS.items():
    insert_unique(EXPRESSION_ARGUMENTS, _name, _arguments)
