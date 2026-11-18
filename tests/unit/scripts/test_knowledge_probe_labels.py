"""The knowledge probes' line and record labels, read from real scripts and records."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from effect_knowledge_probes import (
    common,
    labels,
)

from effects.domain.effect_head_input import MAGNITUDE_CAP
from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS
from tests.unit.scripts.knowledge_probe_support import (
    fixture_line,
    fixture_lines,
    join_triggers,
)

# ── line labels from real scripts ───────────────────────────────────────


def _is_battlefield_creature(entity) -> bool:
    return entity.zone == "battlefield" and "creature" in entity.types


def _resolution_with_huge_damage(records):
    """A resolution record, one of its creatures, and the record with 9999 damage dealt to it."""
    from effects.domain.event_schema import Event, EventType
    from effects.domain.records import Moment

    record = next(r for r in records if r.moment is Moment.RESOLUTION
                  and any(_is_battlefield_creature(e) for e in r.state.entities))
    creature = next(e for e in record.state.entities if _is_battlefield_creature(e))
    event = Event(EventType.DAMAGE_DEALT, subjects=(creature.id,),
                  params={"amount": 9999, "combat": False})
    huge = replace(record, payload=replace(record.payload,
                                           events=(*record.payload.events, event)))
    return record, creature, huge


class TestLineLabels:
    def test_an_activated_damage_ability(self):
        line = fixture_line("AB$ DealDamage", "NumDmg$ 3", "Sac<1/CARDNAME>")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["damage"] == 3
        assert found["cost_r"] == 0 and found["cost_generic"] == 3
        assert found["cost_total"] == 3
        assert found["damage_per_mana"] == pytest.approx(1.0)
        assert found["sacrifice_cost"] == 1.0 and found["tap_cost"] == 1.0
        assert found["repeatable"] == 0.0
        assert found["instant_speed"] == 1.0

    def test_a_spell_line_has_no_mana_cost_labels(self):
        line = fixture_line("SP$ DealDamage", "NumDmg$ 3")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type,
                                   "instant")
        assert found["damage"] == 3
        assert "cost_total" not in found and "damage_per_mana" not in found
        assert found["instant_speed"] == 1.0
        assert found["repeatable"] == 0.0

    def test_a_variable_amount_is_left_unlabelled(self):
        line = fixture_line("NumAtt$ -X")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert "power_change" not in found and "toughness_change" not in found
        assert found["until_end_of_turn"] == 1.0

    def test_a_restricted_any_colour_mana_ability(self):
        # No fixture mana ability carries a restriction: a real one is given the
        # member Forge writes for it.
        line = fixture_line("Cost$ T | Produced$ Any")
        script = f"{line.script_text} | RestrictValid$ Spell"
        found = labels.line_labels(script, line.line_kind, line.script_api_type)
        assert found["any_colour"] == 1.0 and found["restricted"] == 1.0
        assert found["mana_amount"] == 1.0 and found["tap_cost"] == 1.0
        assert found["produces_g"] == 0.0

    def test_a_two_colour_mana_ability(self):
        line = fixture_line("Produced$ Combo U R")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["produces_u"] == found["produces_r"] == 1.0
        assert found["produces_w"] == 0.0 and found["any_colour"] == 0.0

    def test_a_sorcery_speed_activation(self):
        line = fixture_line("SorcerySpeed$ True", "KW$ Vigilance")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["instant_speed"] == 0.0 and found["repeatable"] == 1.0

    def test_a_counter_is_not_until_end_of_turn(self):
        line = fixture_line("AB$ PutCounter |")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["as_counter"] == 1.0 and found["until_end_of_turn"] == 0.0

    def test_a_chain_sums_its_segments(self):
        line = next(line for line in fixture_lines()
                    if line.script_text.count("[SEG]") and "NumDmg$ 4" in line.script_text)
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        segments = labels.segments(line.script_text)
        stated = [s["NumDmg"] for s in segments if "NumDmg" in s]
        assert all(v.lstrip("+-").isdigit() for v in stated)
        assert found["damage"] == sum(int(v) for v in stated)

    def test_a_charm_root_states_no_amounts(self):
        root = "CharmNum$ 2 | Choices$ SV1,SV2,SV3,SV4 | SP$ Charm"
        found = labels.line_labels(root, "spell", "Charm")
        assert not {"damage", "power_change", "counters_placed"} & set(found)

    # The real scripts, as their sidecars in output/cardsfolder/ carry them.
    @pytest.mark.parametrize(("script", "kind_and_api", "target", "sign"), [
        pytest.param(
            "Execute$ SV1 | Mode$ Attacks | TriggerDescription$ 10,000 Needles — Whenever "
            "this creature attacks, it gets +9999/+0 until end of turn. | ValidCard$ "
            "Card.Self [SEG] SV1: DB$ Pump | Defined$ Self | NumAtt$ +9999",
            ("triggered", "Pump"), "power_change", 1, id="jumbo-cactuar"),
        pytest.param(
            "IsCurse$ True | NumDef$ -9999 | SP$ Pump | SpellDescription$ Target creature "
            "gets -0/-9999 until end of turn. | ValidTgts$ Creature",
            ("spell", "Pump"), "toughness_change", -1, id="overkill"),
        pytest.param(
            "Execute$ SV1 | IsPresent$ Card.Self+counters_GE1000_TIME | Mode$ Always | "
            "TriggerDescription$ When there are 1,000 or more time counters on CARDNAME, "
            "sacrifice it and each opponent loses 1,000 life. | TriggerZones$ Battlefield "
            "[SEG] SV1: DB$ Sacrifice | SubAbility$ SV2 [SEG] SV2: DB$ LoseLife | "
            "Defined$ Opponent | LifeAmount$ 1000",
            ("triggered", "Sacrifice"), "life_amount", 1, id="the-millennium-calendar"),
        pytest.param(
            "AB$ GainLife | Cost$ SubCounter<8/LOYALTY> | LifeAmount$ 100 | "
            "Planeswalker$ True | SpellDescription$ You gain 100 life. | Ultimate$ True",
            ("planeswalker", "GainLife"), "life_amount", 1, id="ajani-mentor-of-heroes"),
    ])
    def test_an_amount_is_capped_where_the_model_s_targets_are(
            self, script, kind_and_api, target, sign):
        line_kind, api = kind_and_api
        assert labels.line_labels(script, line_kind, api)[target] == sign * MAGNITUDE_CAP

    def test_damage_per_mana_is_read_off_the_capped_damage(self):
        found = labels.line_labels(
            "AB$ DealDamage | Cost$ 2 R T | NumDmg$ 100 | ValidTgts$ Any", "activated",
            "DealDamage")
        assert found["damage"] == MAGNITUDE_CAP
        assert found["damage_per_mana"] == pytest.approx(MAGNITUDE_CAP / 3)

    def test_every_amount_label_is_within_the_cap_and_no_flag_moves(self):
        for line in fixture_lines():
            found = labels.line_labels(line.script_text, line.line_kind,
                                       line.script_api_type)
            for name, value in found.items():
                if labels.TARGETS_BY_NAME[name].kind == labels.AMOUNT:
                    assert abs(value) <= MAGNITUDE_CAP
                else:
                    assert value in (0.0, 1.0)


# ── record labels from real records ─────────────────────────────────────


class TestRecordLabels:
    @pytest.fixture(scope="class")
    def items(self, records, items_by_record):
        """``(record, item)`` for every board item of the fixture records."""
        by_id = {r.record_id: r for r in records}
        return [(by_id[i.record_id], i) for found in items_by_record.values() for i in found]

    def test_every_item_names_a_known_target_and_a_finite_label(self, items):
        assert items
        for _, item in items:
            assert item.target in labels.TARGETS_BY_NAME
            assert np.isfinite(item.label)

    def test_a_blocker_label_follows_the_legal_blockers(self, items):
        blocks = [(r, i) for r, i in items if i.target == "may_block"]
        assert blocks
        assert any(i.label == 0.0 for _, i in blocks)
        for record, item in blocks:
            assert item.label == float(item.entity in record.payload.legal_blockers)
            assert item.partner_entity == record.payload.anchor_attacker

    def test_a_decision_yields_its_affordable_verdict(self, items):
        decisions = [(r, i) for r, i in items
                     if i.target == "affordable"]
        assert decisions
        for record, item in decisions:
            assert item.label == float(record.payload.candidates[0].affordable)

    def test_a_trigger_yields_whether_it_fired(self, items):
        fires = [(r, i) for r, i in items if i.target == "fires"]
        assert {i.label for _, i in fires} == {0.0, 1.0}
        for record, item in fires:
            assert item.label == float(record.payload.fired)

    def test_affected_items_cover_the_battlefield_of_an_effect_half(self, items):
        from effects.domain.effect_targets import derive_targets

        affected = [(r, i) for r, i in items if i.target == "affected"]
        assert affected
        for record, item in affected:
            entry = derive_targets(record).get(item.entity)
            assert item.label == float(bool(entry and entry.affected))

    def test_damage_taken_is_capped_where_the_model_s_targets_are(self, records, sidecars,
                                                                  acting):
        # The fixture's resolutions damage no creature, so one is given the
        # damage event a real record carries, at an amount far past the cap.
        record, creature, huge = _resolution_with_huge_damage(records)
        found = [i for i in labels.record_items(huge, sidecars, join_triggers([huge]), acting)
                 if i.target == "damage_taken" and i.entity == creature.id]
        assert [i.label for i in found] == [MAGNITUDE_CAP]

    @staticmethod
    def _producing(records, *mana_by_color):
        """A resolution whose events produce the given ``mana_by_color`` maps.

        The map is what the collector writes: Forge's produced-mana string as a
        count per colour letter, ``{"C": 2}`` for a Sol Ring's ``2``.
        """
        from effects.domain.event_schema import Event, EventType
        from effects.domain.records import ResolutionPayload

        record = next(r for r in records if isinstance(r.payload, ResolutionPayload))
        events = tuple(
            Event(EventType.MANA_PRODUCED, subjects=(record.actor_player,),
                  params={"mana_by_color": mana})
            for mana in mana_by_color
        )
        return replace(record, payload=ResolutionPayload(events=events))

    def test_observed_mana_sums_the_counts_each_event_carries(self, records):
        """A Sol Ring makes two mana in one event, a Tinder Farm one of each of
        two colours."""
        assert labels.mana_observed(self._producing(records, {"C": 2})) == 2.0
        assert labels.mana_observed(self._producing(records, {"R": 1, "W": 1})) == 2.0
        assert labels.mana_observed(
            self._producing(records, {"G": 1}, {"G": 1}, {"U": 3})) == 5.0

    def test_observed_mana_is_capped_where_the_model_s_targets_are(self, records):
        produced = self._producing(records, *({"G": 1},) * (int(MAGNITUDE_CAP) + 10))
        assert labels.mana_observed(produced) == MAGNITUDE_CAP

    def test_an_event_that_names_no_mana_says_nothing(self, records):
        """An empty map is mana the collector could not read, not zero mana."""
        assert labels.mana_observed(self._producing(records, {})) is None
        assert labels.mana_observed(self._producing(records, {}, {"B": 1})) == 1.0

    def test_the_join_finds_causes_in_the_same_game(self, records):
        result = join_triggers(records)
        by_id = {r.record_id: r for r in records}
        assert 0.0 <= result.join_rate <= 1.0 or np.isnan(result.join_rate)
        for record_id, key in result.cause.items():
            game = by_id[record_id].game_id
            assert any(r.game_id == game and r.ability and r.ability[0] == key
                       for r in records if r.kind.value == "resolution")

    def test_state_dependence_is_the_spread_of_a_text_s_outcomes(self, records):
        profiles = [p for p in (labels.effect_profile(r) for r in records) if p is not None]
        assert len(profiles) >= 2
        same = labels.state_dependence({"t": [profiles[0], profiles[0]]})
        assert same["t"] == (0.0, 2)
        mixed = labels.state_dependence({"t": profiles[:2], "once": profiles[:1]})
        assert "once" not in mixed and mixed["t"][1] == 2
        assert labels.state_weight(5) == pytest.approx(0.5)

    def test_the_mined_pairs_join_a_death_trigger_to_a_sacrifice_outlet(self):
        items = common.line_items([GEN3_SIDECAR_FOLDERS[0]], "script")
        pairs = labels.mined_pairs(items)
        by_key = {item.key: item for item in items}
        assert pairs
        for trigger, cause in pairs:
            assert by_key[trigger].line_kind == "triggered"
        assert len({t for t, _ in pairs}) == len(pairs)
