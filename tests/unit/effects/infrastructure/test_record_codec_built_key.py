"""A provenance key's ``built`` member through the record codec.

A key of kind ``keyword`` naming a trait the keyword built (a saga's chapter
trigger) carries ``{"kind", "index"}``; every other key carries no member.
The Java writer's exact string is pinned in ``ProvenanceKeyTest`` too, so the
two sides are held to one spelling.
"""

from __future__ import annotations

import orjson

from effects.domain.provenance import BuiltTrait
from effects.infrastructure.record_codec import (
    _keys_from_json,
    record_from_dict,
    record_to_dict,
)
from tests.unit.effects.builders import gen3_fixture_records

#: ``ProvenanceKeyTest.CHAPTER_III_JSON``: what the Java writer produces for
#: Binding the Old Gods' chapter III trigger.
_CHAPTER_III_JSON = (
    '{"script_file":"cardsfolder/b/binding_the_old_gods.txt","face":0,'
    '"trait_kind":"keyword","index_within_kind":0,'
    '"built":{"kind":"trigger","index":2}}'
)


def _keyword_trigger_record() -> dict:
    """The fixture's first trigger record acting through a keyword's key."""
    return next(
        raw for raw in gen3_fixture_records()
        if raw["kind"] == "trigger"
        and any(key["trait_kind"] == "keyword" for key in raw.get("ability") or ())
    )


def test_the_java_string_decodes_and_re_encodes_unchanged():
    (key,) = _keys_from_json([orjson.loads(_CHAPTER_III_JSON)])
    assert key.built == BuiltTrait("trigger", 2)
    assert orjson.dumps(key.as_dict()).decode() == _CHAPTER_III_JSON


def test_a_record_naming_a_built_trait_round_trips_unchanged():
    raw = _keyword_trigger_record()
    raw["ability"] = [orjson.loads(_CHAPTER_III_JSON)]
    record = record_from_dict(raw)
    assert record.ability[0].built == BuiltTrait("trigger", 2)
    written = record_to_dict(record)
    assert orjson.dumps(written["ability"][0]).decode() == _CHAPTER_III_JSON
    assert record_from_dict(written) == record


def test_a_record_without_the_member_decodes_with_built_none():
    raw = _keyword_trigger_record()
    record = record_from_dict(raw)
    assert all(key.built is None for key in record.ability)
    assert all("built" not in key for key in record_to_dict(record)["ability"])


def test_interning_keeps_a_built_key_apart_from_its_keyword_s():
    built = orjson.loads(_CHAPTER_III_JSON)
    bare = {name: value for name, value in built.items() if name != "built"}
    (a, b, c) = _keys_from_json([bare, built, {**built, "built": {"kind": "trigger", "index": 1}}])
    assert a.built is None
    assert b != a and b.without_built() is not b and b.without_built() == a
    assert c != b
