"""Which events each payload kind supervises: a combat's, a rewrite's
outgoing event, no trigger's, and a continuous record's contributions read
line by line."""

from __future__ import annotations

from effects.domain.effect_head_layout import OVERLAY_KEYWORDS
from effects.domain.effect_targets import derive_targets, events_of
from effects.domain.event_schema import Event, EventType
from effects.domain.input_vocabularies import CARD_TYPES
from effects.domain.records import (
    CombatPayload,
    ContinuousPayload,
    Contribution,
    RecordKind,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.state_snapshot import COLORS
from tests.unit.effects.domain.effect_targets_support import (
    _record,
    _resolution,
)


class TestPayloadKinds:
    def test_a_combat_records_events_are_read(self):
        record = _record(
            CombatPayload(events=(
                Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                      params={"amount": 4, "combat": True}),
            )),
            kind=RecordKind.COMBAT,
        )
        assert derive_targets(record)["E1"].fields["damage_taken"] == 4.0

    def test_a_rewrite_reads_its_outgoing_event(self):
        """What the replacement produced, not what it received."""
        record = _record(
            RewritePayload(
                incoming=Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                               params={"amount": 5}),
                outgoing=Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                               params={"amount": 0}),
            ),
            kind=RecordKind.REWRITE, moment=None,
        )
        assert derive_targets(record)["E1"].fields["damage_taken"] == 0.0

    def test_a_fired_trigger_supervises_no_entity_with_the_event_that_caused_it(self):
        """The event is the trigger's input overlay; its own effects arrive as the
        resolution record of the ability it put on the stack."""
        record = _record(
            TriggerPayload(
                event=Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                            params={"delta": 3}),
                fired=True,
            ),
            kind=RecordKind.TRIGGER, moment=None,
        )
        assert derive_targets(record) == {}

    def test_an_unfired_trigger_produces_no_entity_targets(self):
        record = _record(
            TriggerPayload(
                event=Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                            params={"delta": 3}),
                fired=False,
            ),
            kind=RecordKind.TRIGGER, moment=None,
        )
        assert derive_targets(record) == {}

    def test_a_continuous_records_contributions_are_its_targets(self):
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", pt_boost=(1, 1), keywords=("flying",)),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].affected
        assert targets["E1"].fields["power_delta"] == 1.0
        assert targets["E1"].fields["keywords_gained"][
            OVERLAY_KEYWORDS.index("flying")
        ] == 1.0

    def test_a_bare_type_token_is_gained_and_a_dashed_one_is_lost(self):
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", types=("creature", "-land")),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["types_gained"][
            CARD_TYPES.index("creature")
        ] == 1.0
        assert targets["E1"].fields["types_lost"][
            CARD_TYPES.index("land")
        ] == 1.0

    def test_an_added_colour_is_gained_and_nothing_is_lost(self):
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", colors=("G",)),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["colors_gained"][COLORS.index("G")] == 1.0
        # Absent, not an empty set: the record says nothing about losses.
        assert "colors_lost" not in targets["E1"].fields

    def test_a_replacement_reports_what_it_sets_and_no_loss(self):
        """``=`` says the static overwrites the line.

        Before gen-3 what it displaced is not in the record, so the lost field
        stays unset rather than being guessed at; a gen-3 record names it
        (``test_a_gen3_static_that_sets_a_line_loses_what_the_board_without_it_had``).
        """
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", colors=("=", "C"), types=("=", "land")),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["colors_gained"][COLORS.index("C")] == 1.0
        assert targets["E1"].fields["types_gained"][CARD_TYPES.index("land")] == 1.0
        assert "colors_lost" not in targets["E1"].fields
        assert "types_lost" not in targets["E1"].fields

    def test_tokens_outside_the_vocabulary_are_dropped_not_misfiled(self):
        """Subtypes and whole-class removals are real, and have no field.

        The record keeps them because they describe what the static did; the
        model has a slot for each core type and none for "every creature type
        at once", so they must fall out here rather than land on a neighbour.
        """
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(
                    entity="E1",
                    types=("elf", "legendary", "all-creature-types",
                           "-all-land-types", "creature"),
                ),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["types_gained"] == [
            1.0 if name == "creature" else 0.0 for name in CARD_TYPES
        ]
        assert "types_lost" not in targets["E1"].fields

    def test_events_of_reads_each_payload_kind(self):
        assert len(events_of(_resolution(
            Event(type=EventType.DESTROYED, subjects=("E1",)),
        ))) == 1
        assert events_of(_record(
            ContinuousPayload(), kind=RecordKind.CONTINUOUS, moment=None,
        )) == ()
