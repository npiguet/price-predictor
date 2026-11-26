"""The acting source's ``costs_paid`` is no input on an effect half (finding 4.5, B11).

A forced resolution skips the payment, so its spell reads "paid no mana": on
the pilots, 67% of forced effect halves of a spell with a mana cost against
0.4% of live ones, the one feature that tells them apart (D16: a defect). The
acting source's ``costs_paid`` block is zero on every effect half; the cost
half and every other card keep theirs.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.card_features import card_features, card_segments
from effects.domain.records import RecordOrigin
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"
#: Twitch's forced effect half: the spell on the stack, cast from hand, 0 mana paid.
_FORCED_TWITCH = f"{_RUN}.0-mwqvqdyrb9c0.64"
#: Spell Contortion's effect half, which targets Searing Blaze (E64, cast
#: for {R}{R}) on the stack.
_CONTORTION_EFFECT = f"{_RUN}.0-mwqvqdyrb9c0.406"
#: Benevolent Bodyguard's live pair: the cost half of its sacrifice ability,
#: then its effect half; E28 was cast for {W}.
_BODYGUARD_COST = f"{_RUN}.2-mwqvqdxrusnv.5209"
_BODYGUARD_EFFECT = f"{_RUN}.2-mwqvqdxrusnv.5210"


@pytest.fixture(scope="module")
def records():
    wanted = {_FORCED_TWITCH, _CONTORTION_EFFECT, _BODYGUARD_COST, _BODYGUARD_EFFECT}
    return {record.record_id: record
            for record in map(record_from_dict, gen3_fixture_records())
            if record.record_id in wanted}


def _costs_paid(record, entity_id: str) -> tuple[float, ...]:
    entity = record.state.entity(entity_id)
    return card_features(entity, record)[card_segments(entity, record)["costs_paid"]]


def _live(forced):
    """The forced half as the live game would have written it: not a fork,
    and its spell paid for as cast ({2}{U})."""
    source = forced.state.refs.source
    entities = tuple(
        replace(entity, costs_paid=replace(entity.costs_paid, mana={
            **entity.costs_paid.mana, "U": 1, "C": 2}))
        if entity.id == source else entity
        for entity in forced.state.entities)
    return replace(forced, fork=False, interventional=False, fork_kind=None,
                   state=replace(forced.state, entities=entities))


def test_a_forced_and_a_live_effect_half_differ_in_no_costs_paid_column(records):
    forced = records[_FORCED_TWITCH]
    assert forced.origin is RecordOrigin.FORCED
    live = _live(forced)
    assert live.origin is RecordOrigin.LIVE
    source = forced.state.refs.source
    assert _costs_paid(forced, source) == _costs_paid(live, source)
    assert not any(_costs_paid(live, source))


def test_the_cost_half_keeps_the_source_s_costs(records):
    cost, effect = records[_BODYGUARD_COST], records[_BODYGUARD_EFFECT]
    assert cost.state.refs.source == effect.state.refs.source == "E28"
    assert cost.state.entity("E28").costs_paid.mana["W"] == 1
    assert any(_costs_paid(cost, "E28"))
    assert not any(_costs_paid(effect, "E28"))


def test_another_card_keeps_its_costs_on_an_effect_half(records):
    """Searing Blaze, the target, still says what paid for it."""
    effect = records[_CONTORTION_EFFECT]
    assert effect.state.entity("E64").costs_paid.mana["R"] == 2
    assert any(_costs_paid(effect, "E64"))
