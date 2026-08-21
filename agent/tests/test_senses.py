"""Sense model."""
import pytest

from agent.senses import SenseModel, extents

# a two-sense word
SYNSETS = {
    "fin": ["bank", "money", "vault"],
    "riv": ["bank", "shore", "levee"],
    "cash": ["money", "cash", "currency"],
    "water": ["shore", "water", "tide"],
}
RELATIONS = {"fin": ["cash"], "riv": ["water"]}
BY_LEMMA = {}
for _s, _ws in SYNSETS.items():
    for _w in _ws:
        BY_LEMMA.setdefault(_w, []).append(_s)


@pytest.fixture
def model():
    return SenseModel.for_word("bank", SYNSETS, BY_LEMMA, RELATIONS, hops=1)


#### extent
def test_membership_alone_carries_nothing_beyond_its_own_synset():
    e = extents(SYNSETS, RELATIONS, ["fin"], hops=0)
    assert e["fin"] == {"bank", "money", "vault"}
    assert "cash" not in e["fin"]


def test_one_hop_admits_the_neighbours_members():
    e = extents(SYNSETS, RELATIONS, ["fin", "riv"], hops=1)
    assert {"cash", "currency"} <= e["fin"]
    assert {"water", "tide"} <= e["riv"]
    assert "water" not in e["fin"] and "cash" not in e["riv"]


def test_the_two_senses_share_only_the_hub(model):
    fin, riv = model.extent["fin"], model.extent["riv"]
    assert fin & riv == {"bank"}


#### restriction
def test_the_mask_says_which_senses_admit_a_lemma(model):
    i_fin, i_riv = model.pos["fin"], model.pos["riv"]
    assert model.mask("currency")[i_fin] == 1.0
    assert model.mask("currency")[i_riv] == 0.0
    assert model.mask("tide")[i_riv] == 1.0
    assert list(model.mask("bank")) == [1.0, 1.0]        # the hub admits both


#### disambiguation
def test_context_selects_the_right_sense(model):
    assert model.disambiguate(["currency", "cash"])["senses"] == ["fin"]
    assert model.disambiguate(["tide", "water"])["senses"] == ["riv"]


def test_the_hub_is_excluded_because_it_decides_nothing(model):
    r = model.disambiguate(["bank"])
    assert r["abstain"] and r["senses"] == []


def test_a_tie_is_reported_not_broken(model):
    r = model.disambiguate(["currency", "tide"])
    assert r["tied"] and set(r["senses"]) == {"fin", "riv"}


def test_a_context_reaching_no_sense_abstains(model):
    r = model.disambiguate(["photosynthesis", "quark"])
    assert r["abstain"] and r["senses"] == []


def test_a_word_with_one_sense_is_not_a_disambiguation_problem():
    m = SenseModel.for_word("vault", SYNSETS, BY_LEMMA, RELATIONS)
    assert m.d == 1
    assert m.disambiguate(["money"])["senses"] == ["fin"]


#### query path
def test_sense_expansion_falls_back_to_every_sense_when_it_cannot_decide(monkeypatch):
    from agent import senses

    monkeypatch.setattr(senses, "inventory", lambda path=None: (SYNSETS, BY_LEMMA, RELATIONS))
    filt = senses.sense_expansion(["bank"])          # no context at all
    blind = senses.sense_expansion(["bank"], blind=True)
    assert set(filt) == set(blind)


def test_sense_expansion_filters_when_the_context_does_reach(monkeypatch):
    from agent import senses

    monkeypatch.setattr(senses, "inventory", lambda path=None: (SYNSETS, BY_LEMMA, RELATIONS))
    filt = senses.sense_expansion(["bank", "currency"])
    blind = senses.sense_expansion(["bank", "currency"], blind=True)
    assert set(filt) < set(blind)                    # strictly narrower
    assert "shore" in blind and "shore" not in filt   # the river sense is dropped


def test_the_share_keeps_a_wide_sense_from_swamping_a_tight_one(monkeypatch):
    from agent import senses

    wide = dict(SYNSETS)
    wide["big"] = ["bank"] + [f"w{i}" for i in range(40)]
    by = {}
    for s, ws in wide.items():
        for w in ws:
            by.setdefault(w, []).append(s)
    monkeypatch.setattr(senses, "inventory", lambda path=None: (wide, by, RELATIONS))
    out = senses.sense_expansion(["bank"], blind=True)
    assert out["w0"][0] < out["money"][0]            # 1/40 against 1/2
