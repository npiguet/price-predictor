"""A slot several keywords share keeps all of them (finding 13.4, S14 B8, S18 B3).

A static granting two keywords writes two keyword lines under its one key, and
the surface reads one slot per key. The slot holds every keyword, so a strip of
one of them leaves the slot (its text still states the others) and says the
keyword's statement could not be removed; only a strip of all of them drops it.
The record is real: Erudite Wizard (E27) under Swiftfoot Boots, haste and
hexproof granted by the boots' one static.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.effect_head_input import SlotKind, SurfaceBuilder, slot_lines
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

_BOOTS = "3550376f-3c30-4231-90cd-cc0f1b308e87.4-mwqw395jccfe.5148"
_CARRIER = "E27"


@pytest.fixture(scope="module")
def record():
    return next(found for found in map(record_from_dict, gen3_fixture_records())
                if found.record_id == _BOOTS)


def _surface(record, *keywords: str):
    strip = {_CARRIER: frozenset(keywords)} if keywords else {}
    return SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record, strip)


def _abilities(surface) -> int:
    return sum(1 for slot in surface.slots
               if slot.kind is SlotKind.ABILITY and slot.entity_id == _CARRIER)


def _shared(record):
    return next(line for line in slot_lines(record.state.entity(_CARRIER))
                if len(line.keywords) > 1)


def test_the_slot_holds_both_keywords_the_static_grants(record):
    assert _shared(record).keywords == frozenset({"haste", "hexproof"})


def test_stripping_one_keyword_keeps_the_slot_and_reports_it_not_removable(record):
    whole = _surface(record)
    stripped = _surface(record, "haste")
    assert _abilities(stripped) == _abilities(whole)
    assert stripped.unremovable == (_CARRIER,)
    assert whole.unremovable == ()


def test_stripping_every_keyword_of_the_slot_drops_it(record):
    stripped = _surface(record, "haste", "hexproof")
    assert _abilities(stripped) == _abilities(_surface(record)) - 1
    assert stripped.unremovable == ()


def test_stripping_the_only_keyword_of_a_slot_drops_the_slot(record):
    shared = _shared(record)
    kept = tuple(line for line in record.state.entity(_CARRIER).lines
                 if line.fragment != "Hexproof")
    assert len(kept) == len(record.state.entity(_CARRIER).lines) - 1
    sole = replace(record, state=replace(record.state, entities=tuple(
        replace(entity, lines=kept) if entity.id == _CARRIER else entity
        for entity in record.state.entities)))
    assert shared.key in {line.key for line in slot_lines(sole.state.entity(_CARRIER))}
    stripped = _surface(sole, "haste")
    assert _abilities(stripped) == _abilities(_surface(sole)) - 1
    assert stripped.unremovable == ()


def _paired(record, keyword: str):
    """The pair's surfaces built for real; gate 2 reads only the twin's
    ``unremovable`` where the observation is refused, so no model runs."""
    from types import SimpleNamespace

    builder = SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0))
    surfaces = [builder.build(record), builder.build(record, {_CARRIER: frozenset({keyword})})]
    return SimpleNamespace(surfaces=surfaces, at_entities=lambda: None)


def test_gate_two_leaves_a_carrier_with_a_shared_statement_unqualified(record):
    from types import SimpleNamespace

    from effects.application.gate_two import KeywordScore, Observation, _Tally
    from effects.application.gate_two import _render_unqualified as render
    from effects.domain.damage_step_keywords import DAMAGE_STEP_KEYWORDS

    row = replace(DAMAGE_STEP_KEYWORDS[0], keyword="haste")
    carrier = SimpleNamespace(carrier=SimpleNamespace(id=_CARRIER))
    scores = {"haste": KeywordScore()}
    _Tally(scores, active=frozenset()).fold(
        [Observation(record, row, carrier)], _paired(record, "haste"))
    assert (scores["haste"].unqualified, scores["haste"].observations) == (1, 0)
    assert "1 carriers unqualified" in render(scores["haste"])
