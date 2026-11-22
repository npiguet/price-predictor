"""A card slot reads Forge's hidden "can't block" and "can't attack" (finding 4.8, S14 B16b).

Forge keeps "can't block" and "can't attack" a pump or a static grants as a
hidden keyword, ``granted_temporary.hidden_keywords``, which no line and no
keyword flag reads. The blocker and attacker legality outputs depend on it,
so the card slot carries two bits, "can't block" and "can't attack", from
the three spellings the review pilots carry: ``CARDNAME can't block.``,
``CARDNAME can't attack.`` and ``CARDNAME can't attack or block.``. Any other
hidden keyword sets neither bit.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.effect_head_input import SurfaceBuilder, card_features, card_segments
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

#: A combat record: Skitter of Lizards, E74, can't block this turn.
_SKITTER_COMBAT = "3550376f-3c30-4231-90cd-cc0f1b308e87.0-mwqvqdyrb9c0.59"


@pytest.fixture(scope="module")
def combat():
    return next(record for record in map(record_from_dict, gen3_fixture_records())
                if record.record_id == _SKITTER_COMBAT)


def _with_hidden(record, entity_id: str, hidden):
    entities = tuple(
        replace(entity, granted_temporary=replace(entity.granted_temporary, hidden_keywords=hidden))
        if entity.id == entity_id else entity
        for entity in record.state.entities)
    return replace(record, state=replace(record.state, entities=entities))


def _bits(record, entity_id: str) -> tuple[float, ...]:
    entity = record.state.entity(entity_id)
    return card_features(entity, record)[card_segments(entity, record)["hidden_keywords"]]


def _surface(record):
    return SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)


def test_a_creature_that_cant_block_reads_apart_from_one_that_can(combat):
    hidden = combat.state.entity("E74").granted_temporary.hidden_keywords
    assert hidden == ("CARDNAME can't block.",)
    free = _with_hidden(combat, "E74", ())
    assert _surface(combat).slots != _surface(free).slots
    assert _bits(combat, "E74") == (1.0, 0.0)
    assert _bits(free, "E74") == (0.0, 0.0)


@pytest.mark.parametrize(("hidden", "bits"), [
    (("CARDNAME can't attack.",), (0.0, 1.0)),
    (("CARDNAME can't attack or block.",), (1.0, 1.0)),
    (("This card doesn't untap during your next untap step.",), (0.0, 0.0)),
    (("CARDNAME can only attack alone.",), (0.0, 0.0)),
    (None, (0.0, 0.0)),
])
def test_each_spelling_sets_its_bits(combat, hidden, bits):
    assert _bits(_with_hidden(combat, "E74", hidden), "E74") == bits
