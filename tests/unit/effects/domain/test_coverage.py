"""The coverage unit's rules (T102).

Satisfaction counts acting hosts, event subjects and referenced refs but
**not** mere snapshot presence; the castability consult ranks a card but never
drops it; and deck weights favour the cards furthest from their target.
"""

from __future__ import annotations

import pytest

from effects.domain.coverage import (
    CardCoverage,
    ConsultVerdict,
    deck_weights,
    qualifying_cards,
    rank_by_consult,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    EffectRecord,
    ResolutionPayload,
)
from effects.domain.state_snapshot import EntityState, Refs
from tests.unit.effects.builders import (
    make_entity,
    make_record,
    make_snapshot,
)


def _entity(entity_id: str, name: str) -> EntityState:
    return make_entity(entity_id, name)


def _record(entities, *, source=None, targets=(), events=()) -> EffectRecord:
    return make_record(
        record_id="r.0.1", run_id="r", game_id="r.0.1",
        state=make_snapshot(
            entities=tuple(entities),
            refs=Refs(source=source, targets=tuple(targets)),
        ),
        payload=ResolutionPayload(events=tuple(events)),
    )


class TestSatisfactionCounting:
    def test_the_acting_lines_host_counts(self):
        record = _record([_entity("E1", "Shock")], source="E1")
        assert qualifying_cards(record) == {"Shock"}

    def test_a_referenced_ref_counts(self):
        record = _record(
            [_entity("E1", "Shock"), _entity("E2", "Grizzly Bears")],
            source="E1", targets=("E2",),
        )
        assert qualifying_cards(record) == {"Shock", "Grizzly Bears"}

    def test_an_event_subject_counts(self):
        record = _record(
            [_entity("E1", "Shock"), _entity("E2", "Grizzly Bears")],
            source="E1",
            events=(Event(
                type=EventType.DAMAGE_DEALT, subjects=("E2",),
                params={"amount": 2},
            ),),
        )
        assert qualifying_cards(record) == {"Shock", "Grizzly Bears"}

    def test_mere_snapshot_presence_does_not_count(self):
        """A card on the battlefield while something else resolves teaches
        the model nothing about that card."""
        record = _record(
            [_entity("E1", "Shock"), _entity("E9", "Bystander")], source="E1",
        )
        assert "Bystander" not in qualifying_cards(record)

    def test_a_vanilla_creature_can_still_be_satisfied(self):
        """The unit is not resolution records: a vanilla creature has no
        acting line and is still coverable through combat."""
        record = _record(
            [_entity("E1", "Grizzly Bears")],
            events=(Event(
                type=EventType.DAMAGE_DEALT, subjects=("E1",),
                params={"amount": 3, "combat": True},
            ),),
        )
        assert qualifying_cards(record) == {"Grizzly Bears"}


class TestConsultRanksButNeverDrops:
    def test_an_uncastable_card_keeps_a_weight(self):
        """Being in a game is the precondition an intervention forks from."""
        weights = rank_by_consult(
            {"Weird Card": 50.0},
            {"Weird Card": ConsultVerdict.UNCASTABLE},
        )
        assert "Weird Card" in weights
        assert weights["Weird Card"] > 0

    def test_a_castable_card_outranks_an_uncastable_one(self):
        weights = rank_by_consult(
            {"a": 50.0, "b": 50.0},
            {"a": ConsultVerdict.CASTABLE, "b": ConsultVerdict.UNCASTABLE},
        )
        assert weights["a"] > weights["b"]

    def test_an_unconsulted_card_ranks_as_castable(self):
        """The safe direction: under-ranking merely slows the run down."""
        weights = rank_by_consult({"a": 50.0, "b": 50.0}, {})
        assert weights["a"] == weights["b"] == 50.0

    def test_no_card_is_ever_removed_by_the_consult(self):
        weights = {"a": 1.0, "b": 2.0, "c": 3.0}
        verdicts = dict.fromkeys(weights, ConsultVerdict.UNCASTABLE)
        assert set(rank_by_consult(weights, verdicts)) == set(weights)


class TestDeckWeighting:
    def test_a_card_with_no_records_weighs_most(self):
        coverage = {
            "empty": CardCoverage("empty", records=0),
            "half": CardCoverage("half", records=25),
        }
        weights = deck_weights(coverage, 50)
        assert weights["empty"] > weights["half"]

    def test_a_satisfied_card_drops_out_of_the_draw(self):
        coverage = {"done": CardCoverage("done", records=50)}
        assert deck_weights(coverage, 50) == {}

    def test_a_retired_card_drops_out_too(self):
        coverage = {"gone": CardCoverage("gone", records=0, retired=True)}
        assert deck_weights(coverage, 50) == {}

    def test_a_card_one_short_still_gets_drawn(self):
        """Linear in the shortfall, so the last records do not take forever."""
        coverage = {"nearly": CardCoverage("nearly", records=49)}
        assert deck_weights(coverage, 50)["nearly"] == 1.0


class TestDeckCandidateFromConvertedText:
    """The cost a converted card hands the manabase heuristic."""

    def test_the_mana_cost_line_is_read(self):
        from effects.domain.coverage import DeckCandidate

        candidate = DeckCandidate.from_converted_text(
            "bolt", "name: bolt\nmana cost: {2}{R}{R}\ntypes: instant\n",
        )
        assert candidate == DeckCandidate("bolt", "{2}{R}{R}")
        assert candidate.mana_cost_line() == "{2}{R}{R}"

    @pytest.mark.parametrize("text", ["name: land\ntypes: land\n", "mana cost:  \n"])
    def test_a_missing_or_empty_cost_is_none(self, text):
        from effects.domain.coverage import DeckCandidate

        assert DeckCandidate.from_converted_text("x", text).mana_cost is None


@pytest.mark.parametrize("records,target,expected", [(0, 50, False), (50, 50, True)])
def test_satisfied_is_records_at_or_above_target(records, target, expected):
    assert CardCoverage("x", records=records).satisfied(target) is expected
