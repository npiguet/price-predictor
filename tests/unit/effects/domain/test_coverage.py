"""The coverage unit's rules (T102).

Satisfaction counts acting hosts, event subjects and referenced refs but
**not** mere snapshot presence; the castability consult ranks a card but never
drops it; and deck weights favour the cards furthest from their target.
"""

from __future__ import annotations

import pytest

from effects.domain.coverage import (
    UNSEEN_RETIRE_FACTOR,
    CardCoverage,
    ConsultVerdict,
    deck_weights,
    qualifying_cards,
    rank_by_consult,
    retire_stalled,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    ContinuousPayload,
    EffectRecord,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import CardLine, Refs
from tests.unit.effects.builders import (
    make_entity,
    make_record,
    make_snapshot,
)

#: The acting key every :func:`_record` carries: build-corpus refuses a
#: resolution with none, so the coverage counts do too.
ACTING_KEY = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)


def _record(entities, *, source=None, targets=(), events=()) -> EffectRecord:
    """A gen-3 resolution record, the only generation coverage counts."""
    return make_record(
        record_id="r.0.1", run_id="r", game_id="r.0.1", generation=3,
        ability=(ACTING_KEY,),
        state=make_snapshot(
            entities=tuple(entities),
            refs=Refs(source=source, targets=tuple(targets)),
        ),
        payload=ResolutionPayload(events=tuple(events)),
    )


class TestSatisfactionCounting:
    def test_the_acting_lines_host_counts(self):
        record = _record([make_entity("E1", "Shock")], source="E1")
        assert qualifying_cards(record) == {"Shock"}

    def test_a_referenced_ref_counts(self):
        record = _record(
            [make_entity("E1", "Shock"), make_entity("E2", "Grizzly Bears")],
            source="E1", targets=("E2",),
        )
        assert qualifying_cards(record) == {"Shock", "Grizzly Bears"}

    def test_an_event_subject_counts(self):
        record = _record(
            [make_entity("E1", "Shock"), make_entity("E2", "Grizzly Bears")],
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
            [make_entity("E1", "Shock"), make_entity("E9", "Bystander")], source="E1",
        )
        assert "Bystander" not in qualifying_cards(record)

    def test_a_vanilla_creature_can_still_be_satisfied(self):
        """The unit is not resolution records: a vanilla creature has no
        acting line and is still coverable through combat."""
        record = _record(
            [make_entity("E1", "Grizzly Bears")],
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


_ANTHEM = "cardsfolder/g/glorious_anthem.txt"


def _anthem_record(host_lines) -> EffectRecord:
    """A ``continuous`` record of the anthem's static: no ``refs.source``, and
    the host's snapshot lacks the acting static's own line."""
    return make_record(
        RecordKind.CONTINUOUS, generation=3,
        ability=(ProvenanceKey(_ANTHEM, 0, "static", 0),),
        state=make_snapshot(entities=(
            make_entity("E1", "glorious anthem", lines=host_lines),
            make_entity("E2", "grizzly bears"),
        )),
        payload=ContinuousPayload(),
    )


class TestAStaticCreditsItsHost:
    """python D5: a ``continuous`` record names no ``refs.source`` and holds
    contributions, not events, so it credited no card at all."""

    def test_the_host_printing_the_acting_script_is_credited(self):
        other_line = CardLine(
            keys=(ProvenanceKey(_ANTHEM, 0, "spell", 0),), kind="spell", granted=False,
        )
        assert qualifying_cards(_anthem_record((other_line,))) == {"glorious anthem"}

    def test_a_host_with_no_other_line_is_credited_through_its_script(self):
        names = {_ANTHEM: "glorious anthem"}
        assert qualifying_cards(_anthem_record(()), card_of_script=names.get) == {
            "glorious anthem",
        }


class TestCreditGoesToTheConvertedCard:
    """python D6: a split card in hand is "Fire // Ice", its half on the stack
    "Ice", a transformed card its back face: none of them a converted card's
    name. Credit goes through the entity's printed script file instead."""

    _FIRE_ICE = "cardsfolder/f/fire_ice.txt"

    def _on_stack(self, name: str, face: int) -> EffectRecord:
        key = ProvenanceKey(self._FIRE_ICE, face, "spell", 0)
        return _record(
            [make_entity("E1", name, zone="stack", lines=(
                CardLine(keys=(key,), kind="spell", granted=False),
            ))],
            source="E1",
        )

    @pytest.mark.parametrize(("name", "face"), [("Fire // Ice", 0), ("Ice", 1)])
    def test_every_face_credits_the_converted_card(self, name, face):
        names = {self._FIRE_ICE: "fire"}
        assert qualifying_cards(self._on_stack(name, face), card_of_script=names.get) == {
            "fire",
        }

    def test_an_entity_with_no_printed_line_keeps_its_name(self):
        record = _record([make_entity("E1", "grizzly bears")], source="E1")
        assert qualifying_cards(record, card_of_script={}.get) == {"grizzly bears"}


class TestOnlyPlayedDecksCountAStalledRound:
    """python D8: workers draw decks with replacement, so about one deck in
    seven written is never played; its cards were still charged a stalled
    round. A round now charges the cards its games showed; a card written but
    never shown retires only after ``UNSEEN_RETIRE_FACTOR`` times as many
    rounds, so a card no game can show still ends the run."""

    def test_a_written_card_no_game_showed_is_not_charged(self):
        coverage = {"unplayed": CardCoverage("unplayed")}

        retire_stalled(
            coverage, {"unplayed": 0}, no_progress_rounds=1,
            decked=frozenset(), written=frozenset({"unplayed"}),
        )

        assert coverage["unplayed"].rounds_without_progress == 0
        assert not coverage["unplayed"].retired

    def test_a_shown_card_is_charged(self):
        coverage = {"shown": CardCoverage("shown")}

        retired = retire_stalled(
            coverage, {"shown": 0}, no_progress_rounds=1,
            decked=frozenset({"shown"}), written=frozenset({"shown"}),
        )

        assert retired == ["shown"]

    def test_a_card_never_shown_still_retires_in_the_end(self):
        coverage = {"never": CardCoverage("never")}
        for _ in range(UNSEEN_RETIRE_FACTOR * 2 - 1):
            retire_stalled(
                coverage, {"never": 0}, no_progress_rounds=2,
                decked=frozenset(), written=frozenset({"never"}),
            )
        assert not coverage["never"].retired

        retire_stalled(
            coverage, {"never": 0}, no_progress_rounds=2,
            decked=frozenset(), written=frozenset({"never"}),
        )

        assert coverage["never"].retired
