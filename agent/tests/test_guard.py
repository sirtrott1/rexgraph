"""agent.guard: rule-based validity checks, streaming detection, and the guard-bee flow."""
from agent.guard import OutputGuard, GuardRule, relational_complex_guard
from agent import agent_complex, hive


def test_flags_chain_complex_but_not_chain_condition():
    g = relational_complex_guard()
    bad = "A k-rex is a chain complex whose boundary satisfies the chain condition B1 B2 = 0."
    v = g.check(bad)
    assert len(v) == 1                                   # only 'chain complex' flagged
    assert v[0]["matched"].lower() == "chain complex"
    assert "chain condition" not in {x["matched"].lower() for x in v}


def test_autofix_rewrites_the_term():
    g = relational_complex_guard()
    fixed, found = g.fix("The chain complex and the relational-complex are the same object.")
    assert "chain complex" not in fixed.lower()
    assert "relational-complex" not in fixed.lower()
    assert fixed.lower().count("relational complex") == 2
    assert len(found) == 2                               # both violations reported


def test_stream_catches_violation_the_moment_it_completes():
    g = relational_complex_guard()
    # the phrase is split across chunks; it must fire on the chunk that completes it, not before
    chunks = ["A k-rex ", "is a ", "chain ", "complex", " with a boundary."]
    fired_at = None
    for i, (_acc, new) in enumerate(g.scan_stream(chunks)):
        if new and fired_at is None:
            fired_at = i
    assert fired_at == 3                                 # the 'complex' chunk closes 'chain complex'


def test_clean_text_has_no_violations():
    g = relational_complex_guard()
    assert g.check("A relational complex is a graded cell complex; its chain condition is B1 B2 = 0.") == []


def test_guarded_ask_regenerates_then_reports_clean(monkeypatch):
    hive.reset_hive(); agent_complex.reset_live()
    h = hive.get_hive()
    h.attach("writer", "http://x", role="queen", model="m-writer", specialties=["write"])
    replies = iter([
        "A k-rex is a chain complex.",                   # first answer violates
        "A k-rex is a relational complex.",              # the revision is clean
    ])
    monkeypatch.setattr(hive, "_chat", lambda url, model, prompt, **k: next(replies))
    out = h.guarded_ask("writer", "Define a k-rex.", relational_complex_guard())
    assert out["corrected"] is True
    assert out["method"] == "regenerated"
    assert out["violations"] == []
    assert "chain complex" not in out["reply"].lower()


def test_guarded_ask_autofixes_when_regen_still_bad(monkeypatch):
    hive.reset_hive(); agent_complex.reset_live()
    h = hive.get_hive()
    h.attach("writer", "http://x", role="queen", model="m-writer", specialties=["write"])
    monkeypatch.setattr(hive, "_chat", lambda url, model, prompt, **k: "It is a chain complex.")
    out = h.guarded_ask("writer", "Define it.", relational_complex_guard())
    assert out["corrected"] is True
    assert out["method"] == "autofixed"
    assert "relational complex" in out["reply"].lower()
    assert out["violations"] == []


def test_guard_keeps_the_plural_when_fixing():
    """The rule matches 'chain complexes' but its fix was the singular 'relational
    complex', so a plural sentence came out ungrammatical."""
    from agent.guard import relational_complex_guard

    fixed, found = relational_complex_guard().fix(
        "restriction of chain complexes preserves the chain condition")
    assert found
    assert "relational complexes" in fixed
    assert "chain complexes" not in fixed
    assert "chain condition" in fixed          # the axiom keeps its name


def test_guard_leaves_the_chain_condition_alone():
    from agent.guard import relational_complex_guard

    text = "the chain condition B_1 B_2 = 0 holds at every consecutive pair"
    fixed, found = relational_complex_guard().fix(text)
    assert not found and fixed == text
