"""A changed subtype or keyword the head's closed output lists lack is counted
(finding 10.4): the targets drop it, and a reader takes the silence for "none
gained", so ``RecordTargets.unsupervised`` says how many were dropped.

The records are hand-built: no gen-3 record names a value the checked-in lists
lack until the lists are regenerated from a later corpus (``bard``, ``assist``,
``dash`` and ``undying`` stand for those).
"""

from __future__ import annotations

from effects.domain.closed_list_misses import KEYWORD_MISSED, SUBTYPE_MISSED
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.domain.records import (
    ContinuousPayload,
    Contribution,
    RecordKind,
    ResolutionPayload,
)
from tests.unit.effects.builders import make_record


def _resolution(*events):
    return make_record(RecordKind.RESOLUTION, payload=ResolutionPayload(events=tuple(events)))


def _continuous(contribution):
    return make_record(
        RecordKind.CONTINUOUS, payload=ContinuousPayload(contributions=(contribution,)),
        moment=None)


def test_the_names_used_here_are_outside_the_checked_in_lists():
    assert "bard" not in SUBTYPES_CHANGED
    assert not {"assist", "dash", "undying"} & set(KEYWORDS_CHANGED)


def test_a_type_change_to_a_subtype_outside_the_list_is_counted():
    targets = derive_targets(_resolution(
        Event(type=EventType.TYPE_CHANGE, subjects=("E1",),
              params={"types_added": ["creature", "bard", "golem"], "types_removed": []})))
    assert targets.unsupervised == {SUBTYPE_MISSED: 1}


def test_a_keyword_change_outside_the_list_is_counted_and_a_listed_one_is_not():
    targets = derive_targets(_resolution(
        Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
              params={"keywords": ["assist", "flying"], "removed": False})))
    assert targets.unsupervised == {KEYWORD_MISSED: 1}


def test_a_hidden_keyword_is_not_counted_as_a_missing_keyword():
    targets = derive_targets(_resolution(
        Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
              params={"keywords": ["CARDNAME can't block."], "removed": False,
                      "hidden": True})))
    assert targets.unsupervised == {"hidden_keyword_change": 1}


def test_a_token_carrying_outside_values_is_counted():
    targets = derive_targets(_resolution(
        Event(type=EventType.TOKEN_CREATED, subjects=("E1",),
              params={"characteristics": {"types": "Creature - Bard", "keywords": ["dash"]}})))
    assert targets.unsupervised == {SUBTYPE_MISSED: 1, KEYWORD_MISSED: 1}


def test_a_continuous_records_outside_values_are_counted():
    targets = derive_targets(_continuous(
        Contribution(entity="E1", keywords=("undying", "-flying"), types=("bard",))))
    assert targets.unsupervised == {SUBTYPE_MISSED: 1, KEYWORD_MISSED: 1}


def test_a_record_the_lists_cover_counts_nothing():
    targets = derive_targets(_continuous(
        Contribution(entity="E1", keywords=("flying",), types=("golem",))))
    assert not targets.unsupervised
