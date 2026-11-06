"""Protection copies (FR-058, research R5): which real records can carry a rare
protection quality, and that each copy's payload, rewritten by rule 702.16, gives
``derive_targets`` the rule's labels.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest

from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.protection_copies import (
    EVERYTHING_RULE,
    PROTECTION_RULES,
    ProtectionOutcome,
    protection_copy,
    qualify,
    rare_qualities,
    rules_offered,
)
from effects.domain.records import (
    Candidate,
    CombatPayload,
    EffectRecord,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.rule_outcomes import ProtectionQuality, QualityKind
from effects.domain.state_snapshot import (
    CardLine,
    EntityState,
    GlobalState,
    GrantedTemporary,
    PlayerState,
    PowerToughness,
    Refs,
)
from effects.infrastructure.record_codec import record_from_dict, record_to_dict
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

REPO_ROOT = Path(__file__).parents[4]

DIED = ZONE_OUTCOMES.index("died")
PLAYERS = (PlayerState(id="P0", life=20, hand=3, library=30, graveyard=0),
           PlayerState(id="P1", life=20, hand=3, library=30, graveyard=0))


def _creature(entity_id: str, toughness: int = 2, *, subtypes=("goblin",), colours=("R",),
              keywords=(), **overrides) -> EntityState:
    return make_entity(
        entity_id, entity_id, controller="P1", types=("creature",), subtypes=subtypes,
        colors=colours, pt=PowerToughness(base=(2, toughness)), mana_value=2, lines=(),
        granted_temporary=GrantedTemporary(keywords=tuple(keywords)), **overrides,
    )


def _spell(entity_id: str = "S1", *, types=("sorcery",), subtypes=(), colours=("R",),
           mana_value: int = 2) -> EntityState:
    return make_entity(entity_id, entity_id, zone="stack", types=types, subtypes=subtypes,
                       colors=colours, mana_value=mana_value, lines=())


def _state(entities, **refs):
    return make_snapshot(
        global_=GlobalState(turn=3, phase="main1", active="P0", priority="P0", stack_size=1),
        players=PLAYERS, entities=tuple(entities), refs=Refs(**refs),
    )


def _damage(subject: str, amount: int, source: str, *, combat: bool = False) -> Event:
    params = {"amount": amount, "combat": combat, "cause": source}
    if combat:
        params["source"] = source
    return Event(type=EventType.DAMAGE_DEALT, subjects=(subject,), params=params,
                 attributed_to="root")


def _died(subject: str) -> Event:
    return Event(type=EventType.ZONE_CHANGE, subjects=(subject,),
                 params={"from_zone": "battlefield", "to_zone": "graveyard"})


def _sweeper(*creatures: EntityState, source: EntityState | None = None,
             events=None, targets=()) -> EffectRecord:
    """A resolution dealing 2 damage to each creature, killing the 2-toughness ones."""
    source = source or _spell()
    if events is None:
        events = []
        for creature in creatures:
            events.append(_damage(creature.id, 2, source.id))
            if creature.toughness <= 2:
                events.append(_died(creature.id))
    return make_record(
        record_id="g.0-0.1", game_id="g.0-0.1", generation=3,
        state=_state((source, *creatures), source=source.id, targets=targets),
        payload=ResolutionPayload(events=tuple(events), conditions_met=()),
    )


def _copy(record: EffectRecord, rule: str, seed: int = 0):
    copy = protection_copy(record, qualify(record), rule=rule, rng=random.Random(seed), index=0)
    assert copy is not None
    return copy


# ── qualities ────────────────────────────────────────────────────────────


class TestRareQualities:
    def test_a_creature_offers_its_types_creature_types_and_conditions(self):
        found = rare_qualities(_creature("E1", subtypes=("goblin", "warrior")))
        assert found == (
            ProtectionQuality(QualityKind.CARD_TYPE, "creature"),
            ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin"),
            ProtectionQuality(QualityKind.CREATURE_TYPE, "warrior"),
            ProtectionQuality(QualityKind.CONDITION, "monocolored"),
            ProtectionQuality(QualityKind.CONDITION, "mana value 3 or less"),
            ProtectionQuality(QualityKind.CONDITION, "each color"),
        )

    def test_a_non_creature_offers_no_creature_type(self):
        aura = _spell(types=("enchantment",), subtypes=("aura",), colours=("W", "U"),
                      mana_value=4)
        assert {q.kind for q in rare_qualities(aura)} == {
            QualityKind.CARD_TYPE, QualityKind.CONDITION}
        assert ProtectionQuality(QualityKind.CONDITION, "multicolored") in rare_qualities(aura)

    def test_everything_and_colours_are_never_rare(self):
        kinds = {q.kind for q in rare_qualities(_creature("E1"))}
        assert QualityKind.EVERYTHING not in kinds and QualityKind.COLOUR not in kinds


def test_the_rules_named_are_the_catalogue_s_trained_protection_rules():
    import json

    lines = (REPO_ROOT / "rules" / "heads" / "702.16.jsonl").read_text(encoding="utf-8")
    trained = {row["id"] for row in map(json.loads, lines.splitlines())
               if row["output"] == "trained"}
    assert PROTECTION_RULES <= trained


# ── damage prevented (702.16e) ───────────────────────────────────────────


class TestDamagePrevented:
    def test_an_untargeted_sweeper_qualifies_each_creature_it_damaged(self):
        record = _sweeper(_creature("E1"), _creature("E2", 4))
        found = qualify(record)
        assert [(o.outcome, o.protected, o.covered) for o in found] == [
            (ProtectionOutcome.DAMAGE_PREVENTED, "E1", ("S1",)),
            (ProtectionOutcome.DAMAGE_PREVENTED, "E2", ("S1",)),
        ]
        assert rules_offered(found) == ("702.16e", EVERYTHING_RULE)

    def test_the_copy_prevents_the_damage_and_undoes_the_death(self):
        record = _sweeper(_creature("E1"), _creature("E2"))
        copy = protection_copy(record, qualify(record)[:1], rule="702.16e",
                               rng=random.Random(3), index=2)
        assert copy.record.record_id == "g.0-0.1~pc2" and copy.record.synthetic
        assert copy.rule == "702.16e" and copy.outcome_entities == ("E1",)
        assert copy.quality in qualify(record)[0].qualities
        targets = derive_targets(copy.record)
        assert targets["E1"].fields == {HeadField.DAMAGE_PREVENTED: 2.0}
        assert targets["E2"].fields[HeadField.ZONE_OUTCOME] == DIED
        prevented = [e for e in copy.record.payload.events
                     if e.type is EventType.DAMAGE_PREVENTED]
        assert prevented[0].params == {"amount": 2, "source": "S1", "cause": "S1"}

    def test_the_quality_is_granted_as_a_line_and_its_kept_flag(self):
        copy = _copy(_sweeper(_creature("E1")), "702.16e")
        protected = copy.record.state.entity("E1")
        assert copy.quality.keyword in protected.granted_temporary.keywords
        assert protected.lines == (CardLine(keys=(), kind="keyword", granted=True,
                                            fragment=copy.quality.fragment),)

    def test_protection_from_everything_for_702_16j(self):
        copy = _copy(_sweeper(_creature("E1")), EVERYTHING_RULE)
        assert copy.quality == ProtectionQuality.everything()
        protected = copy.record.state.entity("E1")
        assert "protection_from_everything" in protected.granted_temporary.keywords
        assert protected.lines[-1].fragment == "Protection from everything"

    def test_a_targeted_creature_is_not_copied(self):
        assert qualify(_sweeper(_creature("E1"), targets=("E1",))) == ()

    @pytest.mark.parametrize("keyword", ["lifelink", "infect", "wither"])
    def test_a_source_whose_damage_does_more_is_refused(self, keyword):
        source = replace(_spell(), granted_temporary=GrantedTemporary(keywords=(keyword,)))
        assert qualify(_sweeper(_creature("E1"), source=source)) == ()

    def test_a_death_the_damage_did_not_cause_is_refused(self):
        events = [_damage("E1", 1, "S1"), _died("E1")]
        assert qualify(_sweeper(_creature("E1", 4), events=events)) == ()

    def test_a_deathtouch_death_is_the_damage_s(self):
        source = replace(_spell(), granted_temporary=GrantedTemporary(keywords=("deathtouch",)))
        events = [_damage("E1", 1, "S1"), _died("E1")]
        assert len(qualify(_sweeper(_creature("E1", 4), source=source, events=events))) == 1

    def test_any_other_event_on_the_creature_is_refused(self):
        shrink = Event(type=EventType.PT_CHANGE, subjects=("E1",),
                       params={"power_delta": -1, "toughness_delta": -1})
        events = [_damage("E1", 2, "S1"), shrink]
        assert qualify(_sweeper(_creature("E1", 4), events=events)) == ()

    def test_the_quality_covers_every_source_of_the_damage(self):
        goblin, elf = _creature("A1", subtypes=("goblin",)), _creature("A2", subtypes=("elf",))
        blocker = _creature("B1", 6, subtypes=("wall",), colours=("W",))
        record = make_record(
            RecordKind.COMBAT, generation=3, state=_state((goblin, elf, blocker)),
            payload=CombatPayload(events=(
                _damage("B1", 2, "A1", combat=True), _damage("B1", 2, "A2", combat=True),
            )),
        )
        (found,) = qualify(record)
        assert found.covered == ("A1", "A2")
        assert ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin") not in found.qualities
        assert ProtectionQuality(QualityKind.CARD_TYPE, "creature") in found.qualities
        copy = _copy(record, "702.16e")
        assert derive_targets(copy.record)["B1"].fields == {HeadField.DAMAGE_PREVENTED: 4.0}

    def test_forks_and_synthetic_records_are_never_copied(self):
        record = _sweeper(_creature("E1"))
        assert qualify(replace(record, synthetic=True)) == ()
        assert qualify(replace(record, fork=True)) == ()


# ── a target made illegal (702.16b) ──────────────────────────────────────


def _decision(*targets: str, source: EntityState | None = None) -> EffectRecord:
    source = source or make_entity("H1", "bolt", zone="hand", types=("instant",),
                                   colors=("R",), mana_value=1, lines=())
    candidate = Candidate(ability=(), can_play=True, affordable=True, has_legal_target=True,
                          legal_targets=targets)
    return make_record(
        RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION, generation=3,
        state=_state((source, _creature("E1"), _creature("E2")), source=source.id),
        payload=PlayabilityDecisionPayload(candidates=(candidate,)),
    )


class TestTargetIllegal:
    def test_one_permanent_leaves_the_legal_targets(self):
        record = _decision("E1", "E2", "P1")
        assert [o.protected for o in qualify(record)] == ["E1", "E2"]
        copy = _copy(record, "702.16b")
        targets = derive_targets(copy.record)
        legal = {ref: targets[ref].fields[HeadField.TARGET_LEGAL] for ref in ("E1", "E2", "P1")}
        assert legal[copy.protected] == 0 and sum(legal.values()) == 2
        assert copy.record.payload.candidates[0].has_legal_target

    def test_a_candidate_with_one_legal_target_is_not_copied(self):
        assert qualify(_decision("E1")) == ()

    def test_the_quality_is_the_candidate_card_s(self):
        record = _decision("E1", "E2")
        assert all(q.covers(_char(record.state.entity("H1"))) for q in qualify(record)[0].qualities)


def _char(entity):
    from effects.domain.rule_outcomes import Characteristics

    return Characteristics.of(entity)


# ── a block made illegal (702.16f) ───────────────────────────────────────


def _blockers(*blockers: EntityState, must_block=()) -> EffectRecord:
    attacker = _creature("A1", subtypes=("human",), colours=("W",))
    return make_record(
        RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS, generation=3,
        what_if=False, state=_state((attacker, *blockers)),
        payload=PlayabilityBlockersPayload(
            anchor_attacker="A1", legal_blockers=tuple(b.id for b in blockers),
            must_block=tuple(must_block)),
    )


class TestBlockIllegal:
    def test_every_blocker_with_the_quality_becomes_forbidden(self):
        record = _blockers(_creature("B1", subtypes=("goblin",)),
                           _creature("B2", subtypes=("elf",), colours=("G",)),
                           _creature("B3", subtypes=("goblin",), colours=("B",)),
                           must_block=("B3",))
        goblins = ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin")
        found = qualify(record)
        assert [o.protected for o in found] == ["A1"] * 3
        only_goblins = tuple(
            replace(o, qualities=(goblins,)) for o in found if o.covered == ("B1",))
        copy = protection_copy(record, only_goblins, rule="702.16f", rng=random.Random(0), index=0)
        assert copy.outcome_entities == ("B1", "B3")
        payload = copy.record.payload
        assert payload.legal_blockers == ("B2",) and payload.must_block == ()
        targets = derive_targets(copy.record)
        assert [targets[b].fields[HeadField.BLOCKER_LEGAL] for b in ("B1", "B2", "B3")] == [0, 1, 0]
        attacker = copy.record.state.entity("A1")
        assert attacker.lines[-1].fragment == "Protection:Goblin"

    def test_protection_from_everything_forbids_every_blocker(self):
        record = _blockers(_creature("B1"), _creature("B2", subtypes=("elf",)))
        copy = _copy(record, EVERYTHING_RULE)
        assert copy.record.payload.legal_blockers == ()

    def test_a_what_if_query_is_not_copied(self):
        assert qualify(replace(_blockers(_creature("B1")), what_if=True)) == ()


# ── an Aura put into the graveyard (702.16c) ─────────────────────────────


def _aura_spell(extra=()) -> EffectRecord:
    aura = _spell("S1", types=("enchantment",), subtypes=("aura",), colours=("B",))
    events = (
        Event(type=EventType.ATTACHED, subjects=("S1",), params={"cause": "S1"}),
        Event(type=EventType.PT_CHANGE, subjects=("E1",),
              params={"power_delta": 2, "toughness_delta": -2, "cause": "S1"}),
        Event(type=EventType.ZONE_CHANGE, subjects=("S1",), attributed_to="root",
              params={"from_zone": "stack", "to_zone": "battlefield", "cause": "S1"}),
        *extra,
    )
    return make_record(
        generation=3, state=_state((aura, _creature("E1", 4)), source="S1", targets=("E1",)),
        payload=ResolutionPayload(events=events, conditions_met=()),
    )


class TestAuraToGraveyard:
    def test_the_aura_goes_to_the_graveyard_and_does_nothing(self):
        record = _aura_spell()
        (found,) = qualify(record)
        assert found.outcome is ProtectionOutcome.AURA_TO_GRAVEYARD and found.protected == "E1"
        copy = _copy(record, "702.16c")
        (moved,) = copy.record.payload.events
        assert moved.params == {"from_zone": "stack", "to_zone": "graveyard", "cause": "S1"}
        targets = derive_targets(copy.record)
        assert targets["S1"].fields[HeadField.ZONE_OUTCOME] == DIED
        assert "E1" not in targets or not targets["E1"].affected
        assert copy.outcome_entities == ("S1", "E1")

    def test_an_aura_record_touching_anything_else_is_not_copied(self):
        other = Event(type=EventType.LIFE_CHANGE, subjects=("P0",), params={"delta": 1})
        assert qualify(_aura_spell(extra=(other,))) == ()


# ── the draw ─────────────────────────────────────────────────────────────


class TestDraw:
    def test_a_rule_no_opportunity_shows_draws_nothing(self):
        record = _sweeper(_creature("E1"))
        assert protection_copy(record, qualify(record), rule="702.16f",
                               rng=random.Random(0), index=0) is None

    def test_no_shared_rare_quality_leaves_only_protection_from_everything(self):
        artifact = make_entity("S2", "rod", types=("artifact",), mana_value=5, lines=())
        events = [_damage("E1", 1, "S1"), _damage("E1", 1, "S2")]
        record = _sweeper(_creature("E1", 4), events=events)
        record = replace(record, state=replace(
            record.state, entities=(*record.state.entities, artifact)))
        (found,) = qualify(record)
        assert found.qualities == () and rules_offered((found,)) == (EVERYTHING_RULE,)

    def test_the_same_seed_draws_the_same_copy(self):
        record = _sweeper(_creature("E1", subtypes=("goblin", "elf")), _creature("E2"))
        first, again = _copy(record, "702.16e", seed=7), _copy(record, "702.16e", seed=7)
        assert first == again

    def test_a_copy_writes_and_reads_back(self):
        copy = _copy(_sweeper(_creature("E1")), "702.16e")
        assert record_from_dict(record_to_dict(copy.record)) == copy.record
