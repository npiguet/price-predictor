"""The snapshot's reader rules (T024).

Two of these are the ones a collector or a consumer gets wrong quietly: an
absent tier read as an empty board, and a perspective baked in at collection
time.
"""

from __future__ import annotations

from dataclasses import replace

from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import (
    CardChoices,
    CardHistory,
    CardLine,
    CostsPaid,
    EffectDuration,
    EntityRole,
    EntityState,
    FaceState,
    GlobalState,
    GrantedTemporary,
    LineKind,
    LineRef,
    Link,
    PlayerState,
    PowerToughness,
    StateSnapshot,
    TurnCounts,
)
from tests.unit.effects.builders import (
    java_golden_entities,
    java_golden_records,
    make_entity,
    make_snapshot,
)


def _snapshot(**overrides) -> StateSnapshot:
    defaults = {
        "global_": GlobalState(
            turn=7, phase="combat_damage", active="P0", priority="P1",
            stack_size=1, combat_substep="first_strike",
        ),
        "players": (
            PlayerState(id="P0", life=14, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        ),
    }
    defaults.update(overrides)
    return make_snapshot(**defaults)


class TestEntityArithmetic:
    def test_an_entity_reads_its_totals_less_its_damage(self):
        entity = EntityState(
            id="E1", name="A", zone="battlefield", controller="P0", damage=1,
            pt=PowerToughness(base=(2, 2), boosts=(1, 0), counters=(0, 1)),
        )
        assert (entity.power, entity.toughness, entity.remaining_toughness) == (3, 3, 2)

    def test_an_entity_without_power_and_toughness_reads_zero(self):
        entity = EntityState(id="E1", name="A", zone="hand", controller="P0")
        assert (entity.power, entity.toughness) == (0, 0)

    def test_a_new_total_toughness_moves_the_base_and_clears_damage(self):
        entity = EntityState(
            id="E1", name="A", zone="battlefield", controller="P0", damage=2,
            pt=PowerToughness(base=(2, 2), boosts=(0, 1), counters=(0, 1)),
        ).with_total_toughness(6)
        assert entity.pt == PowerToughness(base=(2, 4), boosts=(0, 1), counters=(0, 1))
        assert (entity.toughness, entity.damage) == (6, 0)

    def test_ability_keys_hold_printed_then_attached_lines_only(self):
        from effects.domain.provenance import ProvenanceKey

        printed, attached, temporary = (
            ProvenanceKey("cardsfolder/a/a.txt", 0, "spell", i) for i in range(3))
        entity = EntityState(
            id="E1", name="A", zone="battlefield", controller="P0",
            printed=(printed,), granted_attached=(attached,),
            granted_temporary=GrantedTemporary(abilities=(temporary,)),
        )
        assert entity.ability_keys == (printed, attached)


class TestLookups:
    def test_an_entity_resolves_by_id(self):
        entity = EntityState(id="E12", name="A", zone="battlefield", controller="P0")
        assert _snapshot(entities=(entity,)).entity("E12") is entity

    def test_a_missing_entity_returns_none(self):
        assert _snapshot().entity("E99") is None


def _line(name: str, kind: str = "static", index: int = 0, **members) -> CardLine:
    from effects.domain.provenance import ProvenanceKey

    key = ProvenanceKey(f"cardsfolder/{name[0]}/{name}.txt", 0, kind, index)
    return CardLine(keys=(key,), kind=kind, granted=members.pop("granted", False), **members)


class TestAGen3EntityReadsAsTheGen2ReadersExpect:
    """The one mapping from a gen-3 entity's ``lines`` onto ``printed`` and
    ``ability_keys`` (T074)."""

    def test_printed_is_the_lines_printed_on_the_card_in_force_or_not(self):
        own, unmet = _line("angel", "keyword"), _line("angel", index=1, active=False)
        granted = _line("levitation", granted=True, active=True)
        entity = make_entity("E1", "A", lines=(own, granted, unmet))
        assert entity.printed == (*own.keys, *unmet.keys)
        assert entity.granted_attached == ()

    def test_the_lines_replace_whatever_printed_was_passed(self):
        stale = _line("stale")
        own = _line("angel", "keyword")
        entity = make_entity("E1", "A", printed=stale.keys, granted_attached=stale.keys,
                             lines=(own,))
        assert (entity.printed, entity.granted_attached) == (own.keys, ())
        assert replace(entity, lines=()).printed == ()

    def test_ability_keys_are_the_lines_in_force_printed_or_granted(self):
        own, granted = _line("angel", "keyword"), _line("levitation", granted=True, active=True)
        unmet = _line("enforcer", active=False)
        other_face = _line("bloodline", "trigger", active=False)
        entity = make_entity("E1", "A", lines=(own, granted, unmet, other_face))
        assert entity.ability_keys == (*own.keys, *granted.keys)
        assert entity.line_keys == (*own.keys, *granted.keys, *unmet.keys, *other_face.keys)

    def test_a_key_two_lines_share_is_one_token(self):
        flying = _line("levitation", "static", granted=True, fragment="Flying")
        vigilance = replace(flying, fragment="Vigilance")
        entity = make_entity("E1", "A", lines=(flying, vigilance))
        assert entity.ability_keys == flying.keys

    def test_a_temporary_grant_rides_the_overlay_not_the_tokens(self):
        """FR-073 on gen-3: the overlay counts it, so it is no token too."""
        pumped = _line("jitte", "activated", granted=True)
        own = _line("bear", "activated")
        entity = make_entity(
            "E1", "A", lines=(own, pumped),
            granted_temporary=GrantedTemporary(abilities=pumped.keys),
        )
        assert entity.ability_keys == own.keys
        assert entity.line_keys == (*own.keys, *pumped.keys)

    def test_a_keyword_line_reads_its_keyword_without_the_value(self):
        assert _line("a", "keyword", fragment="Ward:2").keyword == "ward"
        assert _line("a", "keyword", fragment="First Strike").keyword == "first_strike"
        assert _line("a", "keyword", fragment="Protection:Goblin").keyword == "protection"
        assert _line("a", "static").keyword is None

    def test_only_keyword_lines_are_keyword_lines(self):
        ward = _line("a", "keyword", fragment="Ward:2")
        entity = make_entity("E1", "A", lines=(_line("a", "static"), ward))
        assert entity.keyword_lines == (ward,)
        assert make_entity("E2", "B").keyword_lines == ()

    def test_a_line_in_force_is_any_but_an_inactive_one(self):
        assert _line("a", active=True).in_force and _line("a", "spell").in_force
        assert not _line("a", active=False).in_force


class TestTheJavaGoldensRead:
    """The gen-3 JSON the connector's tests emit, through the codec (T074)."""

    @staticmethod
    def _entities() -> list[EntityState]:
        from effects.infrastructure.record_codec import _entity_from_json

        return [_entity_from_json(e) for e in java_golden_entities()]

    @staticmethod
    def _named(name: str) -> list[EntityState]:
        return [e for e in TestTheJavaGoldensRead._entities() if e.name == name]

    def test_every_entity_reads_in_the_gen3_shape(self):
        entities = self._entities()
        assert len(entities) > 100
        assert all(e.lines is not None for e in entities)
        assert all(e.granted_attached == () for e in entities)
        assert any(e.lines for e in entities)

    def test_power_and_toughness_read_base_temporary_and_counters(self):
        for data, entity in zip(java_golden_entities(), self._entities(), strict=True):
            if data["base_pt"] is None:
                assert entity.pt is None
                continue
            boosts = data["pt_boosts"]
            assert entity.pt == PowerToughness(
                base=tuple(data["base_pt"]), boosts=tuple(boosts["temporary"]),
                counters=tuple(boosts["counters"]))

    def test_an_anthem_boost_is_a_temporary_boost(self):
        assert any(bear.pt.boosts == (1, 1) and (bear.power, bear.toughness) == (3, 3)
                   for bear in self._named("Grizzly Bears"))

    def test_a_static_s_flying_is_a_granted_keyword_line_keyed_to_the_static(self):
        bears = [b for b in self._named("Grizzly Bears")
                 if [line.keyword for line in b.keyword_lines] == ["flying"]]
        assert bears
        for bear in bears:
            (line,) = bear.keyword_lines
            assert line.granted
            assert line.keys[0].script_file == "cardsfolder/l/levitation.txt"
            assert bear.printed == ()
            assert line.keys[0] in bear.ability_keys
            assert "flying" in bear.granted_temporary.keywords

    def test_a_printed_keyword_is_keyed_to_the_card_s_keyword_line(self):
        for angel in self._named("Serra Angel"):
            assert [line.keyword for line in angel.keyword_lines] == ["flying", "vigilance"]
            assert [(k.trait_kind, k.index_within_kind) for k in angel.printed] == [
                ("keyword", 0), ("keyword", 1)]

    def test_a_modal_spell_on_the_stack_names_its_chosen_modes(self):
        chosen = {line.chosen for e in self._named("Fiery Confluence") for line in e.lines}
        assert (0, 1, 1) in chosen

    def test_a_static_carries_whether_its_condition_holds(self):
        statics = [line for e in self._entities() for line in e.lines
                   if line.kind == LineKind.STATIC]
        assert statics and all(line.active is not None for line in statics)


class TestTheGen3FactsRead:
    """A card's, a player's and the board's gen-3 facts, from the goldens (T074)."""

    @staticmethod
    def _states() -> list[StateSnapshot]:
        from effects.infrastructure.record_codec import record_from_dict

        return [record_from_dict(raw).state for raw in java_golden_records()]

    @staticmethod
    def _entities() -> list[EntityState]:
        from effects.infrastructure.record_codec import _entity_from_json

        return [_entity_from_json(e) for e in java_golden_entities()]

    def test_every_card_names_its_owner_face_and_history(self):
        entities = self._entities()
        assert all(e.owner in ("P0", "P1") for e in entities)
        assert all(e.face_state in set(FaceState) for e in entities)
        assert all(isinstance(e.history, CardHistory) for e in entities)
        assert all(e.chosen == CardChoices() and e.role is None and e.in_tiers
                   for e in entities)
        assert any(e.history.entered_this_turn for e in entities)

    def test_a_mana_cost_counts_its_symbols_and_a_land_has_none(self):
        entities = self._entities()
        assert any(e.mana_cost == {"generic": 1, "W": 2} for e in entities)
        assert any(e.mana_cost is None and "land" in e.types for e in entities)

    def test_a_planeswalker_prints_its_loyalty(self):
        assert any(e.printed_loyalty in (3, 4) and "planeswalker" in e.types
                   for e in self._entities())

    def test_a_cast_card_says_what_paid_for_it(self):
        paid = [e.costs_paid for e in self._entities() if e.costs_paid is not None]
        assert CostsPaid(mana={"W": 1, "U": 0, "B": 0, "R": 1, "G": 0, "C": 0}) in paid
        assert any(each.cast_from == "hand" for each in paid)

    def test_an_order_rank_is_set_on_the_battlefield(self):
        ranks = [e.order_rank for e in self._entities() if e.on_battlefield]
        assert ranks and all(rank is not None for rank in ranks)

    def test_a_static_names_the_permanents_it_affects(self):
        lines = [line for e in self._entities() for line in e.lines
                 if line.affected is not None]
        assert lines and all(line.kind == LineKind.STATIC for line in lines)
        assert any(len(line.affected) > 4 for line in lines)

    def test_a_player_carries_the_turn_counts(self):
        players = [p for state in self._states() for p in state.players]
        assert all(isinstance(p.turn_counts, TurnCounts) for p in players)
        assert any(p.turn_counts.life_gained == 3 for p in players)
        assert any(p.monarch for p in players)

    def test_a_floating_effect_names_the_line_that_made_it(self):
        effects = [made for state in self._states() for made in state.global_.floating_effects]
        assert {made.duration for made in effects} == {
            EffectDuration.END_OF_TURN, EffectDuration.PERMANENT}
        made = next(made for made in effects if made.source_lines)
        assert made.source_lines[0].script_file == "cardsfolder/p/prodigal_pyromancer.txt"

    def test_links_end_at_refs_or_lines(self):
        links = [link for state in self._states() for link in state.links]
        targets = [link for link in links if link.type == "targets"]
        assert targets and all(isinstance(link.from_, LineRef) and link.index == 0
                               for link in targets)
        assert all(link.index is None for link in links if link.type != "targets")
        assert {"controls", "owns", "attacks", "blocks"} <= {link.type for link in links}

    def test_a_link_s_line_ends_give_their_keys(self):
        key = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
        link = Link(from_=LineRef("E1", key), to="E2", type="targets", index=0)
        assert link.line_keys() == (key,)
        assert Link(from_="P0", to="E2", type="owns").line_keys() == ()

    def test_an_extra_entity_is_not_one_of_the_tiers(self):
        assert not make_entity("L7", "Bear", role=EntityRole.LAST_KNOWN).in_tiers
        assert make_entity("E7", "Bear").in_tiers

    def test_a_gen2_snapshot_reads_no_gen3_member(self):
        snapshot = _snapshot()
        assert snapshot.links is None and snapshot.global_.floating_effects == ()
        assert all(player.turn_counts is None for player in snapshot.players)
