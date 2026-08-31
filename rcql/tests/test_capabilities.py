from rcql import BoundSource, Executor, SourcePolicy, call, query, source


class Record:
    def __init__(self):
        self.id = "patient-17"
        self.version = 2
        self.created = 2.0
        self.tx_from = 2.0
        self.tx_to = None
        self.valid_from = 1.0
        self.valid_to = None
        self.signature = {"nV": 3, "nE": 2, "source": "clinic-a"}


class Store:
    def list(self, limit=100, offset=0):
        return [Record()]

    def get(self, record_id):
        return object()

    def history(self, record_id):
        return [Record()]


def test_record_policy_redacts_identity_and_projects_signature():
    policy = SourcePolicy.allow("records", record_fields={"nV", "nE"})
    result = Executor(sources={"db": BoundSource(Store(), policy)}).execute(
        query(source("db"), call("RCDB_LIST")))
    row = result.values[0][0]
    assert "id" not in row
    assert row["signature"] == {"nE": 2, "nV": 3}


def test_exact_history_requires_identity_capability():
    policy = SourcePolicy.allow("history")
    executor = Executor(sources={"db": BoundSource(Store(), policy)})
    try:
        executor.execute(query(source("db"), call("RCDB_HISTORY", "patient-17")))
    except PermissionError:
        pass
    else:
        raise AssertionError("history by exact record id bypassed identity capability")
