"""Create small lexical files for reader and answerer regression tests."""
import json
from xml.etree import ElementTree as ET


def write_wordnet(path, records, *, verb_forms=()):
    resource = ET.Element("LexicalResource")
    lexicon = ET.SubElement(resource, "Lexicon", id="synthetic", language="en")
    for sid, words, definition, relations in records:
        entries = []
        for index, word in enumerate(words):
            eid = f"{sid}-{index}"
            entry = ET.SubElement(lexicon, "LexicalEntry", id=eid)
            ET.SubElement(entry, "Lemma", writtenForm=word, partOfSpeech="n")
            ET.SubElement(entry, "Sense", id=f"sense-{eid}", synset=sid)
            entries.append(eid)
        synset = ET.SubElement(lexicon, "Synset", id=sid,
                              members=" ".join(entries), partOfSpeech="n")
        ET.SubElement(synset, "Definition").text = definition
        for kind, target in relations:
            ET.SubElement(synset, "SynsetRelation", relType=kind, target=target)
    for word, forms in verb_forms:
        entry = ET.SubElement(lexicon, "LexicalEntry", id=f"verb-{word}")
        ET.SubElement(entry, "Lemma", writtenForm=word, partOfSpeech="v")
        for form in forms:
            ET.SubElement(entry, "Form", writtenForm=form)
    ET.ElementTree(resource).write(path, encoding="utf-8", xml_declaration=True)
    return path


def make_lexical_sources(directory):
    wordnet = write_wordnet(directory / "wordnet.xml", [
        ("harpoon", ["harpoon", "spear"], "a spear used to catch fish", [("holo_member", "point")]),
        ("point", ["point", "tip"], "the sharp end of a tool", []),
        ("are", ["are"], "a unit of area", []),
        ("whale", ["whale", "cetacean"], "a marine mammal", [("hyponym", "orca")]),
        ("orca", ["orca", "killer whale"], "a kind of whale", []),
        ("frigate", ["frigate", "warship"], "a naval vessel", []),
        ("rain", ["rain", "rainfall"], "water falling from clouds", []),
        ("bank-money", ["bank", "vault"], "a place for money", [("hypernym", "finance")]),
        ("bank-river", ["bank", "shore"], "land beside water", [("hypernym", "water")]),
        ("finance", ["currency", "cash"], "money used for payment", [("related", "shared")]),
        ("water", ["tide", "stream"], "moving water", [("related", "shared")]),
        ("shared", ["currency", "tide"], "shared second hop context", []),
    ], verb_forms=[("be", ["are", "is"])])
    roget = directory / "roget.txt"
    roget.write_text("#1. Fishing\nN. harpoon, spear, net.\n", encoding="utf-8")
    entries = [
        {"word": "grief", "synonyms": [{"word": "sorrow"}]},
        {"word": "sorrow", "synonyms": [{"word": "grief"}]},
        {"word": "joy", "antonyms": [{"word": "grief"}]},
        {"word": "whale", "hyponyms": [{"word": "orca"}]},
        {"word": "harpoon", "related": [{"word": "spear"}]},
        {"word": "ship", "synonyms": [{"word": "vessel"}]},
    ]
    wiktionary = directory / "wiktionary.jsonl"
    wiktionary.write_text("".join(json.dumps(dict(entry, lang_code="en", pos="noun")) + "\n"
                                  for entry in entries), encoding="utf-8")
    return {"wordnet": wordnet, "roget": roget, "wiktionary": wiktionary}
