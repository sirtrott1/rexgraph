"""Store accession fields with explicit version selectors and projected inputs."""
from .types import Domain, Exactness, PredicateResult, RCType, ValueKind

ARGUMENTS = {"CORPUS_FIELD": (("query", "reading", "exact", "as_of", "valid_at"),
                               ("existence", True, None, None))}


def arguments(args):
    return tuple(args) + ("existence", True, None, None)[max(0, len(args) - 1):]


def result(args):
    exact = arguments(args)[2]
    return RCType("CorpusField", kind=ValueKind.RECORD,
                  domain=Domain.RATIONAL if exact else Domain.REAL,
                  exactness=Exactness.STRUCTURAL)


def refine(typed, children, context):
    from rcdb.corpus import options
    options(*arguments(typed.args))
    return [PredicateResult("corpus_query", "verified",
        "literal vocabulary terms, explicit reading and finite version selectors; no corpus read"),
        PredicateResult("corpus_snapshot", "deferred",
        "runtime captures projected terms on one visible version per record through the store reader")]


def corpus_field(source, query, reading="existence", exact=True, as_of=None, valid_at=None):
    from .execution_trace import current_policy, record_method
    from rcdb.corpus import options
    options(query, reading, exact, as_of, valid_at)
    snapshot = source.corpus_snapshot(as_of=as_of, valid_at=valid_at,
                                     signature_fields=current_policy().record_fields)
    value = snapshot.response(query, reading=reading, exact=exact)
    record_method("rcdb-native-corpus-response", snapshot_digest=snapshot.digest,
                  reading=reading, exact=exact, records=len(snapshot.ids))
    return value


def install(register):
    from .signatures import OperatorSignature, TypePattern
    register(OperatorSignature(name="CORPUS_FIELD", source_kind=ValueKind.RCDB_STORE,
        source_methods=frozenset({"corpus_snapshot"}),
        inputs=(TypePattern("query", literal=(list, tuple)),
                TypePattern("reading", literal=str, optional=True),
                TypePattern("exact", literal=bool, optional=True),
                TypePattern("as_of", literal=(int, float, type(None)), optional=True),
                TypePattern("valid_at", literal=(int, float, type(None)), optional=True)),
        result=result, implementation_key="rcdb.CorpusSnapshot.response",
        requires=frozenset({"read", "records", "search", "identity"}),
        preconditions=("one visible version per record, including append log entries",
                       "project signature fields before scoring; bounded fields exclude metadata labels",
                       "exact rational accession response or one final float conversion; no payload decode")))
