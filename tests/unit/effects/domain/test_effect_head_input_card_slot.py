"""Root spec § 9.4 and § 9.6: the ``[CARD]`` slot's parts, its keyword
flags and what a record's kind changes in them."""

from __future__ import annotations

from effects.domain.card_features import card_features, counter_rows
from effects.domain.effect_head_layout import OVERLAY_KEYWORDS
from effects.domain.event_schema import Event, EventType
from effects.domain.input_layout import (
    THERMOMETER_STEPS,
    SlotKind,
)
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
from effects.domain.records import (
    RecordKind,
    TriggerPayload,
)
from effects.domain.state_snapshot import (
    GlobalState,
    GrantedTemporary,
    PowerToughness,
)
from tests.unit.effects.batch_builders import slots_of_kind
from tests.unit.effects.domain.effect_head_input_support import (
    _ANTHEM,
    _WARD,
    _build,
    _card_part,
    _entity,
    _line,
    _record,
    _snapshot,
)


class TestPerKindVariations:
    """A continuous record's flags: ``test_continuous_keyword_flags.py``."""

    def test_masking_leaves_the_structured_features_alone(self):
        """Gate 2's strip changes keywords, not the card's printed type line."""
        entity = _entity(
            "E1", granted_temporary=GrantedTemporary(keywords=("flying",)),
        )
        record = _record(kind=RecordKind.COMBAT, ability=None,
                         state=_snapshot(entities=(entity,)))
        masked = card_features(entity, record, masked_keywords=frozenset({"flying"}))
        without = card_features(entity, record)
        # Only the flying flag differs.
        differing = [i for i, (a, b) in enumerate(zip(masked, without)) if a != b]
        assert len(differing) == 1

    def test_a_granted_keyword_is_addressable_for_the_gate_two_perturbation(self):
        for keyword in ("first_strike", "double_strike", "deathtouch", "lifelink",
                        "trample", "indestructible", "wither", "infect"):
            assert keyword in OVERLAY_KEYWORDS

    def test_a_trigger_record_overlays_its_pending_event(self):
        entity = _entity("E1")
        pending = Event(type=EventType.LIFE_CHANGE, subjects=("E1",),
                        params={"delta": 3})
        record = _record(
            kind=RecordKind.TRIGGER, moment=None,
            state=_snapshot(entities=(entity,), pending_event=pending),
            payload=TriggerPayload(event=pending, fired=True),
        )
        without = _record(
            kind=RecordKind.TRIGGER, moment=None,
            state=_snapshot(entities=(entity,)),
            payload=TriggerPayload(event=pending, fired=True),
        )
        assert card_features(entity, record) != card_features(entity, without)


class TestKeywordFlags:
    """Root spec § 9.6: only the kept keywords keep a flag."""

    def test_a_keyword_without_a_kept_flag_sets_none(self):
        assert "exalted" not in KEPT_KEYWORD_FLAGS
        entity = _entity("E1", granted_temporary=GrantedTemporary(keywords=("exalted",)))
        record = _record(state=_snapshot(entities=(entity,)))
        assert sum(_card_part(entity, record, "keyword_flags")) == 0.0

    def test_split_second_keeps_its_flag(self):
        entity = _entity("E1", granted_temporary=GrantedTemporary(keywords=("split_second",)))
        record = _record(state=_snapshot(entities=(entity,)))
        flags = _card_part(entity, record, "keyword_flags")
        assert flags[KEPT_KEYWORD_FLAGS.index("split_second")] == 1.0

    def test_an_in_force_keyword_line_sets_its_flag_and_a_face_not_up_none(self):
        up = _entity("E1", lines=(_line(None, kind="keyword", fragment="Flying"),))
        down = _entity("E1", lines=(
            _line(None, kind="keyword", fragment="Flying", active=False),))
        record = _record()
        index = KEPT_KEYWORD_FLAGS.index("flying")
        assert _card_part(up, record, "keyword_flags")[index] == 1.0
        assert _card_part(down, record, "keyword_flags")[index] == 0.0


class TestCardSlotInputs:
    """Root spec § 9.4: the card slot's new inputs."""

    def test_the_face_state_is_one_hot(self):
        from effects.domain.input_vocabularies import FACE_STATES

        record = _record()
        for face in FACE_STATES:
            part = _card_part(_entity("E1", face_state=face), record, "face_state")
            assert part[FACE_STATES.index(face)] == 1.0 and sum(part) == 1.0

    def test_two_face_down_cards_differ_only_in_their_inactive_lines(self):
        from effects.domain.state_snapshot import RealFace

        def face_down(name, key):
            return _entity(
                "E1", name="", face_state="face_down", types=("creature",), colors=(),
                mana_value=0, pt=PowerToughness(base=(2, 2)),
                real_face=RealFace(name=name, types=("creature",), colors=("R",),
                                   mana_value=5, base_pt=(5, 4)),
                lines=(_line(key, kind="trigger", active=False),),
            )

        one = _build(_record(state=_snapshot(entities=(face_down("A", _ANTHEM),))))
        two = _build(_record(state=_snapshot(entities=(face_down("B", _WARD),))))
        assert slots_of_kind(one, SlotKind.CARD) == slots_of_kind(two, SlotKind.CARD)
        one_line, = slots_of_kind(one, SlotKind.ABILITY)
        two_line, = slots_of_kind(two, SlotKind.ABILITY)
        assert one_line.features == two_line.features and one_line.e_vector != two_line.e_vector

    def test_is_a_copy_and_is_a_token(self):
        record = _record()
        plain = _card_part(_entity("E1"), record, "characteristics")
        copy = _card_part(_entity("E1", copy_source="E9"), record, "characteristics")
        token = _card_part(_entity("E1", token_script_id="c_1_1_soldier"), record,
                           "characteristics")
        assert (plain[-2:], copy[-2:], token[-2:]) == ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    def test_mana_cost_reads_each_symbol(self):
        from effects.domain.input_vocabularies import MANA_SYMBOLS

        entity = _entity("E1", mana_cost={"generic": 2, "W/U": 1, "G/P": 2})
        part = _card_part(entity, _record(), "mana_cost")
        assert part[0] == 1.0
        assert part[1 + 2 * MANA_SYMBOLS.index("W/U")] == 1.0
        assert part[1 + 2 * MANA_SYMBOLS.index("G/P")] == 2.0
        assert part[1 + 2 * MANA_SYMBOLS.index("generic")] == 2.0
        assert _card_part(_entity("E1"), _record(), "mana_cost")[0] == 0.0

    def test_printed_loyalty_carries_a_thermometer(self):
        part = _card_part(_entity("E1", printed_loyalty=3), _record(),
                          "printed_loyalty_defense")
        loyalty = part[:3 + THERMOMETER_STEPS]
        assert loyalty[:2] == (1.0, 3.0) and sum(loyalty[3:]) == 3

    def test_the_designations_are_the_collector_s_table(self):
        """``Designation`` writes the closed list the input reads."""
        import re
        from pathlib import Path

        from effects.domain.input_vocabularies import DESIGNATIONS

        java = (Path(__file__).resolve().parents[4] / "forge-connector/src/main/java/com/"
                "pricepredictor/connector/effects/Designation.java").read_text(encoding="utf-8")
        assert tuple(re.findall(r'^\s*[A-Z_]+\("([a-z_]+)",', java, re.MULTILINE)) == DESIGNATIONS

    def test_designations_x_paid_history_and_order_rank(self):
        from effects.domain.input_vocabularies import DESIGNATIONS
        from effects.domain.state_snapshot import CardHistory

        entity = _entity(
            "E1", designations=("goaded", "monstrous"), x_paid=4, order_rank=2,
            history=CardHistory(entered_this_turn=True, loyalty_activated=True),
        )
        record = _record()
        designations = _card_part(entity, record, "designations")
        assert designations[DESIGNATIONS.index("goaded")] == 1.0
        assert sum(designations) == 2.0
        assert _card_part(entity, record, "x_paid")[:2] == (1.0, 4.0)
        assert _card_part(entity, record, "history") == (1.0, 0.0, 0.0, 1.0)
        assert _card_part(entity, record, "order_rank")[:2] == (1.0, 2.0)

    def test_chosen_values_and_costs_paid(self):
        from effects.domain.state_snapshot import COLORS, CardChoices, CostsPaid

        entity = _entity(
            "E1", chosen=CardChoices(colors=("white",), type="Human", number=3),
            costs_paid=CostsPaid(mana={"B": 2}, cast_from="graveyard", mana_rider=True),
        )
        record = _record()
        chosen = _card_part(entity, record, "chosen")
        assert chosen[COLORS.index("W")] == 1.0
        paid = _card_part(entity, record, "costs_paid")
        assert paid[0] == 1.0 and paid[1 + 2 * COLORS.index("B")] == 2.0 and paid[-1] == 1.0

    def test_subtypes_and_chosen_types_share_the_subtype_rows(self):
        from effects.domain.input_vocabularies import SUBTYPE_ROWS, subtype_id
        from effects.domain.state_snapshot import CardChoices

        entity = _entity("E1", subtypes=("human", "soldier"),
                         chosen=CardChoices(type="Human"))
        card, = slots_of_kind(_build(_record(state=_snapshot(entities=(entity,)))),
                              SlotKind.CARD)
        assert card.subtypes == (subtype_id("human"), subtype_id("soldier"))
        assert card.chosen_types == (subtype_id("Human"),) == card.subtypes[:1]
        assert subtype_id("not a subtype") == SUBTYPE_ROWS - 1

    def test_a_chosen_card_type_is_no_subtype(self):
        from effects.domain.card_features import chosen_subtypes
        from effects.domain.state_snapshot import CardChoices

        assert chosen_subtypes(_entity("E1", chosen=CardChoices(type="Artifact"))) == ()

    def test_every_named_counter_has_a_row_of_its_own(self):
        from effects.domain.input_vocabularies import COUNTER_ROWS, counter_id

        rows = dict(counter_rows(_entity("E1", counters={"OIL": 2, "P1P1": 1, "ZZZ": 4})))
        assert rows[counter_id("OIL")] == 2.0 and rows[counter_id("P1P1")] == 1.0
        assert rows[COUNTER_ROWS - 1] == 4.0
        assert len({counter_id("OIL"), counter_id("P1P1"), counter_id("ACORN")}) == 3

    def test_turn_and_control_inputs(self):
        record = _record(state=_snapshot(global_=GlobalState(
            turn=2, phase="main1", active="P1", priority="P0", stack_size=0)))
        stolen = _entity("E1", controller="P0", owner="P1")
        part = _card_part(stolen, record, "relation_to_act")
        # controller active, controller holds priority, owner acting, owner differs
        assert part[3:] == (0.0, 1.0, 0.0, 1.0)
