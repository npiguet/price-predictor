"""The knowledge probes' target restrictions, colour pips and zone effects."""

from __future__ import annotations

from dataclasses import replace

import pytest
from effect_knowledge_probes import (
    colour_pips,
    sweeps,
    target_restrictions,
    zone_effects,
)

from tests.unit.effects.record_builders import player_of
from tests.unit.scripts.knowledge_probe_support import decision_record


class TestTargetRestrictions:
    def test_numbers_are_masked(self):
        assert target_restrictions.masked("Creature.powerLE2") == "Creature.powerLE#"

    @pytest.mark.parametrize("restriction, expected", [
        ("Creature", "card type only"),
        ("Artifact,Enchantment", "card type only"),
        ("Creature.OppCtrl", "controller"),
        ("Creature.nonLegendary", "supertype"),
        ("Creature.nonBlack", "colour"),
        ("Creature.withFlying", "keyword"),
        ("Creature.powerLE2", "number"),
        ("Creature.attacking", "combat or tapped"),
        ("Card.Creature+namedSomething", "other"),
    ])
    def test_family(self, restriction, expected):
        assert target_restrictions.family(restriction) == expected

    def test_flips_break_exactly_the_stated_condition(self, records):
        from effects.domain.records import PlayabilityDecisionPayload

        assert target_restrictions.flip_for("Creature.OppCtrl") == "to the acting player"
        assert target_restrictions.flip_for("Artifact") == "loses artifact"
        assert target_restrictions.flip_for("Artifact,Creature") is None
        assert target_restrictions.flip_for("Creature.OppCtrl+nonLegendary") is None
        record = next(r for r in records if isinstance(r.payload, PlayabilityDecisionPayload)
                      and r.payload.candidates and r.payload.candidates[0].legal_targets)
        entity = next(e for e in record.state.entities if e.zone == "battlefield")
        mine = target_restrictions.flipped(record, entity.id, "to the acting player")
        assert mine.state.entity(entity.id).controller == record.actor_player
        theirs = target_restrictions.flipped(record, entity.id, "to the opponent")
        assert theirs.state.entity(entity.id).controller != record.actor_player
        legendary = target_restrictions.flipped(record, entity.id, "made legendary")
        assert "legendary" in legendary.state.entity(entity.id).supertypes
        lost = target_restrictions.flipped(record, entity.id, "loses creature")
        assert "creature" not in lost.state.entity(entity.id).types


class TestColourPips:
    def test_the_cost_parses_into_coloured_pips_and_generic(self):
        assert sweeps.parse_cost("{2}{R}{W/U}") == ([("R",), ("W", "U")], 2)
        assert colour_pips.shape([("U",)]) == "one pip"
        assert colour_pips.shape([("U",), ("U",)]) == "repeated pip"
        assert colour_pips.shape([("W",), ("U",)]) == "several colours"
        assert colour_pips.shape([("W", "U")]) == "hybrid"

    @pytest.mark.parametrize("cost, lands, expected", [
        ("{2}{R}", ["R", "G", "G"], True),
        ("{2}{R}", ["G", "G", "G"], False),
        ("{2}{R}", ["R", "G"], False),
        ("{U}{U}", ["U", "G"], False),
        ("{1}{U}", ["U", "U"], True),
        ("{W/U}{W/U}", ["W", "U"], True),
        ("{W/U}{W/U}", ["W", "G"], False),
    ])
    def test_the_rule_matches_pips_to_lands(self, cost, lands, expected):
        assert colour_pips.payable(*sweeps.parse_cost(cost), lands) is expected

    @pytest.mark.parametrize("cost", ["{2}{R}", "{U}{U}", "{1}{U}{U}", "{W}{U}", "{W/U}",
                                      "{G}"])
    def test_configurations_keep_their_rule_answers(self, cost):
        configs = colour_pips.configurations(cost)
        coloured, generic = sweeps.parse_cost(cost)
        total = len(coloured) + generic
        assert configs["exact"] == (configs["exact"][0], True)
        assert configs["surplus"][1] is True
        for name in ("one short", "pip missing", "pip missing, +2", "all off colour"):
            assert configs[name][1] is False, name
        assert len(configs["exact"][0]) == total
        assert len(configs["one short"][0]) == total - 1
        assert len(configs["pip missing"][0]) == total
        assert len(configs["pip missing, +2"][0]) == total + 2

    def test_the_missing_colour_is_still_present_only_for_a_repeated_pip(self):
        assert colour_pips.missing_colour_present("{1}{U}{U}")
        assert not colour_pips.missing_colour_present("{1}{U}")
        assert not colour_pips.missing_colour_present("{W}{U}")

    def test_a_configuration_edits_the_lands_to_its_colours(self, records):
        record = decision_record(records, "{U}")
        actor = record.actor_player
        lands, _ = colour_pips.configurations("{2}{U}")["pip missing"]
        edited = sweeps.set_untapped_lands(record, actor, list(lands), len(lands))
        production = player_of(edited.state, actor).untapped_production
        assert {c: n for c, n in production.items() if n} == {"W": 3}


class TestZoneEffects:
    def test_chain_apis_read_every_segment(self):
        text = "SP$ Destroy | ValidTgts$ Creature [SEG] SV1 | DB$ Draw | NumCards$ 2"
        assert zone_effects.chain_apis(text) == {"Destroy", "Draw"}
        assert zone_effects.chain_apis(None) == set()

    def test_retargeting_moves_only_the_player_target(self, records):
        record = next(r for r in records if len(r.state.players) == 2)
        one, two = (p.id for p in record.state.players)
        aimed = replace(record, state=replace(record.state, refs=replace(
            record.state.refs, targets=(one, "card-1"))))
        assert zone_effects.player_targets(aimed) == [one]
        moved = zone_effects.retargeted(aimed, one, two)
        assert moved.state.refs.targets == (two, "card-1")
        assert zone_effects.player_targets(moved) == [two]
