"""The pending event's subject bit is on its last-known copy too.

A pending event names the card (``E<id>``), the game's object for it now:
Aven Heartstabber's dies trigger names E7, in the graveyard. The event
happened to the object as it last existed, which the snapshot holds as the
card's last-known copy, ``L7`` on the battlefield, linked to E7 by
``last_known_state_of``. Both carry the subject bit: the card the event
names, and the copy that shows the object the event happened to. A
last-known copy of a card the event does not name carries none.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.card_features import card_features, card_segments
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records

#: Aven Heartstabber's dies trigger: the event names E7, L7 is its copy.
_HEARTSTABBER = "3550376f-3c30-4231-90cd-cc0f1b308e87.2-mwqvqdxrusnv.783"


@pytest.fixture(scope="module")
def record():
    return next(found for found in map(record_from_dict, gen3_fixture_records())
                if found.record_id == _HEARTSTABBER)


def _subject_bit(record, entity_id: str) -> float:
    entity = record.state.entity(entity_id)
    # the pending-event part: a pending event, is its subject, is its source
    return card_features(entity, record)[card_segments(entity, record)["pending_event"]][1]


def _without_copy_links(record):
    links = tuple(link for link in record.state.links if link.type != "last_known_state_of")
    return replace(record, state=replace(record.state, links=links))


def test_the_event_names_the_card_and_its_copy_shows_the_object(record):
    assert record.state.pending_event.subjects == ("E7",)
    assert any(link.type == "last_known_state_of" and (link.from_, link.to) == ("L7", "E7")
               for link in record.state.links)


def test_the_copy_of_the_subject_carries_the_subject_bit(record):
    assert _subject_bit(record, "E7") == 1.0
    assert _subject_bit(record, "L7") == 1.0


def test_a_copy_of_a_card_the_event_does_not_name_carries_none(record):
    other = replace(record, state=replace(record.state, pending_event=replace(
        record.state.pending_event, subjects=("E9999",))))
    assert _subject_bit(other, "L7") == 0.0
    assert _subject_bit(_without_copy_links(record), "L7") == 0.0
