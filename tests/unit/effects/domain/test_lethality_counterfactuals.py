"""Counterfactual copies of damage and -X/-X resolutions (FR-063e).

A copy redraws the amount and one hit creature's toughness and is labelled by
rule 704.5f-g. The tests pin which records qualify, that every copy's labels
follow the rule whatever was drawn, and that the edited text, key and X agree
with the amount the labels used.
"""

from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from effects.domain.effect_head_layout import ZONE_OUTCOMES
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.lethality_counterfactuals import (
    SOURCE_EXCLUSIONS,
    BoardReading,
    CopyMaterial,
    EffectKind,
    counterfactual,
    edit_text,
    lethal,
    qualify,
)
from effects.domain.provenance import ProvenanceKey, SourceTree
from effects.domain.records import EffectRecord, ResolutionPayload
from effects.domain.state_snapshot import (
    EntityState,
    GlobalState,
    PlayerState,
    PowerToughness,
    Refs,
)
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

DIED = ZONE_OUTCOMES.index("died")
BOLT = ("NumDmg$ 3 | SP$ DealDamage | SpellDescription$ CARDNAME deals 3 damage to "
        "target creature. | ValidTgts$ Creature")
SHRINK = ("IsCurse$ True | NumAtt$ -2 | NumDef$ -2 | SP$ Pump | SpellDescription$ "
          "Target creature gets -2/-2 until end of turn. | ValidTgts$ Creature")
KEY = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)


def _creature(entity_id: str, toughness: int, *, damage: int = 0, **overrides) -> EntityState:
    return make_entity(
        entity_id, entity_id, controller="P1", types=("creature",),
        pt=PowerToughness(base=(2, toughness)), damage=damage, **overrides,
    )


def _record(events, entities, *, x=None, targets=("E1",), source_x_paid=None) -> EffectRecord:
    state = make_snapshot(
        global_=GlobalState(turn=3, phase="main1", active="P0", priority="P0", stack_size=1),
        players=(PlayerState(id="P0", life=20, hand=3, library=30, graveyard=0),
                 PlayerState(id="P1", life=20, hand=3, library=30, graveyard=0)),
        entities=(make_entity("S1", "spell", zone="stack", x_paid=source_x_paid), *entities),
        refs=Refs(targets=targets, source="S1", x=x),
    )
    return make_record(
        record_id="r.0-0.1", run_id="r", game_id="r.0-0.1", state=state,
        payload=ResolutionPayload(events=tuple(events)), ability=(KEY,),
    )


def _damage(subject: str, amount: int) -> Event:
    return Event(type=EventType.DAMAGE_DEALT, subjects=(subject,), params={"amount": amount})


def _died(subject: str) -> Event:
    return Event(type=EventType.ZONE_CHANGE, subjects=(subject,),
                 params={"from_zone": "battlefield", "to_zone": "graveyard"})


def _shrunk(subject: str, amount: int) -> Event:
    return Event(type=EventType.PT_CHANGE, subjects=(subject,),
                 params={"power_delta": -amount, "toughness_delta": -amount},
                 duration="end_of_turn")


def _line(text: str, role_spans=()):
    return SimpleNamespace(script_text=text, role_spans=role_spans)


def _no_keywords(entity) -> set[str]:
    return set()


def _qualify(record, text=BOLT, keywords_of=_no_keywords):
    return qualify(record, _line(text), derive_targets(record), BoardReading(keywords_of))


class TestTheRule:
    def test_damage_kills_at_toughness_counting_marked_damage(self):
        assert lethal(EffectKind.DAMAGE, 3, _creature("E1", 3))
        assert not lethal(EffectKind.DAMAGE, 2, _creature("E1", 3))
        assert lethal(EffectKind.DAMAGE, 2, _creature("E1", 3, damage=1))

    def test_a_shrink_kills_at_zero_toughness_or_at_marked_damage(self):
        assert lethal(EffectKind.SHRINK, 2, _creature("E1", 2))
        assert not lethal(EffectKind.SHRINK, 2, _creature("E1", 3))
        assert lethal(EffectKind.SHRINK, 2, _creature("E1", 3, damage=1))


class TestQualify:
    def test_a_lethal_bolt_qualifies(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        qualified = _qualify(record)
        assert qualified is not None
        effect = qualified.effect
        assert (effect.kind, effect.amount, effect.stated) == ("damage", 3, "text")
        assert qualified.hit == ("E1",)

    def test_a_survivor_qualifies(self):
        assert _qualify(_record([_damage("E1", 3)], [_creature("E1", 5)])) is not None

    def test_an_outcome_the_rule_contradicts_is_not_copied(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 5)])
        assert _qualify(record) is None

    def test_a_chain_that_replaces_the_death_is_not_copied(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        text = BOLT + " | ReplaceDyingDefined$ Targeted"
        assert _qualify(record, text) is None

    def test_a_harmless_sub_ability_is_allowed(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        text = BOLT + " | SubAbility$ SV1 [SEG] SV1: DB$ Draw | NumCards$ 1"
        assert _qualify(record, text) is not None

    def test_a_hit_creature_moved_where_no_zone_outcome_names_is_not_copied(self):
        """Its outcome is unknown, never "stayed": the rule cannot be checked."""
        onto_stack = Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                           params={"from_zone": "battlefield", "to_zone": "stack"})
        record = _record([_damage("E1", 3), onto_stack], [_creature("E1", 5)])
        assert _qualify(record) is None

    def test_a_second_effect_on_the_hit_creature_is_not_copied(self):
        tapped = Event(type=EventType.TAPPED, subjects=("E1",), params={})
        record = _record([_damage("E1", 3), tapped], [_creature("E1", 5)])
        assert _qualify(record) is None

    def test_a_deathtouch_source_is_not_copied(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        def source_deathtouch(entity) -> set[str]:
            return {"deathtouch"} if entity.id == "S1" else set()

        assert _qualify(record, keywords_of=source_deathtouch) is None

    def test_an_indestructible_or_enchanted_creature_is_not_copied(self):
        """Each on a record that qualifies without the one difference."""
        survivor = _record([_damage("E1", 3)], [_creature("E1", 5)])
        assert _qualify(survivor) is not None
        assert _qualify(survivor, keywords_of=lambda e: (
            {"indestructible"} if e.id == "E1" else set())) is None
        aura = make_entity("A1", "aura", attached_to="E1")
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2), aura])
        assert _qualify(_record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)]))
        assert _qualify(record) is None

    @pytest.mark.parametrize("keyword", sorted(SOURCE_EXCLUSIONS))
    def test_a_source_whose_keyword_changes_what_damage_does_is_not_copied(self, keyword):
        """Deathtouch, wither, infect and lifelink each change what the hit
        does or what else happens beside it."""
        record = _record([_damage("E1", 3)], [_creature("E1", 5)])
        assert _qualify(record) is not None
        assert _qualify(record, keywords_of=lambda e: (
            {keyword} if e.id == "S1" else set())) is None

    def test_a_shield_counter_on_the_hit_creature_is_not_copied(self):
        """The counter is spent instead of the damage, so no lethality rule holds."""
        shielded = _creature("E1", 5, counters={"SHIELD": 1})
        assert _qualify(_record([_damage("E1", 3)], [shielded])) is None

    def test_a_hit_that_is_no_creature_is_not_copied(self):
        walker = make_entity("E1", "E1", controller="P1", types=("planeswalker",))
        assert _qualify(_record([_damage("E1", 3)], [walker])) is None

    def test_a_cost_half_is_not_copied(self):
        from effects.domain.records import ActivationPayload, Costs, Moment

        effect = _record([_damage("E1", 3)], [_creature("E1", 5)])
        cost = replace(effect, moment=Moment.ACTIVATION,
                       payload=ActivationPayload(costs=Costs(), events=()))
        assert _qualify(cost) is None

    @pytest.mark.parametrize("text", [
        BOLT + " | SubAbility$ SV1 [SEG] SV1: DB$ DealDamage | NumDmg$ 1 | Defined$ Remembered",
        BOLT + " | DamageSource$ Targeted",
        BOLT + " | DividedAsYouChoose$ 3",
    ])
    def test_a_chain_of_two_hits_or_a_blocked_parameter_is_not_copied(self, text):
        record = _record([_damage("E1", 3)], [_creature("E1", 5)])
        assert _qualify(record, text) is None

    def test_a_death_a_shield_or_a_replacement_may_turn_is_not_copied(self):
        """P3a #14: a regeneration shield (a floating effect from a Regenerate
        line affecting the hit creature) or a replacement of a move or a
        destruction on the board (an exile-instead) changes what the death
        does, which the copies' labels would not show."""
        from effects.domain.state_snapshot import CardLine, CommandZoneObject

        regen_key = ProvenanceKey("cardsfolder/r/regenerate.txt", 0, "spell", 0)
        rip_key = ProvenanceKey("cardsfolder/r/rest_in_peace.txt", 0, "replacement", 0)

        class Lines:
            def line_or_none(self, key):
                text = {regen_key: "SP$ Regenerate | ValidTgts$ Creature",
                        rip_key: "Event$ Moved | Destination$ Graveyard | ReplaceWith$ Exile"}
                return _line(text[key]) if key in text else None

        record = _record([_damage("E1", 3)], [_creature("E1", 5)])
        assert qualify(record, _line(BOLT), derive_targets(record),
                       BoardReading(_no_keywords, Lines())) is not None
        shield = CommandZoneObject(id="E50", source_lines=(regen_key,), affected=("E1",))
        shielded = replace(record, state=replace(record.state, global_=replace(
            record.state.global_, floating_effects=(shield,))))
        assert qualify(shielded, _line(BOLT), derive_targets(shielded),
                       BoardReading(_no_keywords, Lines())) is None
        rip = make_entity("R1", "rest in peace", types=("enchantment",), lines=(
            CardLine(keys=(rip_key,), kind="replacement", granted=False),))
        exiling = replace(record, state=replace(
            record.state, entities=(*record.state.entities, rip)))
        assert qualify(exiling, _line(BOLT), derive_targets(exiling),
                       BoardReading(_no_keywords, Lines())) is None

    def test_an_aura_no_longer_on_the_battlefield_is_no_attachment(self):
        """A last-known copy (no zone) of an Aura or a phased-out one attaches
        nothing the rules see."""
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        for gone in ({"zone": None, "id": "L7"}, {"designations": ("phased_out",)}):
            aura = make_entity("A1", "aura", attached_to="E1", **gone)
            with_aura = replace(record, state=replace(
                record.state, entities=(*record.state.entities, aura)))
            assert _qualify(with_aura) is not None, gone

    def test_a_player_hit_by_the_same_amount_comes_along(self):
        life = Event(type=EventType.LIFE_CHANGE, subjects=("P1",), params={"delta": -3})
        record = _record([_damage("E1", 3), _died("E1"), life], [_creature("E1", 2)])
        assert _qualify(record).players == ("P1",)

    def test_a_paid_x_is_the_amount(self):
        record = _record([_damage("E1", 4), _died("E1")], [_creature("E1", 4)], x=4,
                         source_x_paid=4)
        qualified = _qualify(record, BOLT.replace("NumDmg$ 3", "NumDmg$ X"))
        assert (qualified.effect.amount, qualified.effect.stated) == (4, "x")

    def test_a_symmetric_shrink_qualifies(self):
        record = _record([_shrunk("E1", 2), _died("E1")], [_creature("E1", 2)])
        qualified = _qualify(record, SHRINK)
        effect = qualified.effect
        assert (effect.kind, effect.amount, effect.symmetric) == ("shrink", 2, True)

    def test_no_fork_intervention_directed_fork_or_synthetic_record_is_copied(self):
        """Only natural records are copy material, as for the other two
        derivations (``protection_opportunities.qualify``, the damage grid)."""
        from effects.domain.records import DirectedTag

        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        assert _qualify(record) is not None
        directed = DirectedTag(variant="v", operation="script_edit")
        for flags in ({"fork": True}, {"fork": True, "interventional": True},
                      {"fork": True, "directed": directed, "aimed_at": ("702.2b",)},
                      {"synthetic": True}):
            assert _qualify(replace(record, **flags)) is None, flags


class TestCopies:
    def test_every_copy_follows_the_rule_whatever_was_drawn(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        targets = derive_targets(record)
        qualified = _qualify(record)
        rng = random.Random(7)
        outcomes = set()
        for number in range(200):
            material = CopyMaterial(record, qualified, targets, _line(BOLT), BOLT)
            copy = counterfactual(material, rng, number)
            amount = int(copy.edit.text.split("NumDmg$ ")[1].split()[0])
            creature = copy.record.state.entity("E1")
            fields = copy.targets["E1"].fields
            dies = fields.get("zone_outcome") == DIED
            assert dies == (amount >= creature.pt.base[1])
            assert fields["damage_taken"] == amount
            assert f"deals {amount} damage" in copy.edit.text
            outcomes.add(dies)
        assert outcomes == {True, False}

    def test_a_copy_reads_its_text_under_a_key_of_its_own(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        copy = counterfactual(CopyMaterial(record, _qualify(record), derive_targets(record),
                                           _line(BOLT), BOLT), random.Random(1), 0)
        assert copy.edit.key.tree == SourceTree.COUNTERFACTUAL
        assert copy.record.ability == (copy.edit.key,)
        assert copy.record.record_id == "r.0-0.1~cf0"
        assert record.state.entity("E1").pt.base == (2, 2)

    def test_a_paid_x_is_edited_in_the_references_and_on_the_source_not_the_text(self):
        record = _record([_damage("E1", 4), _died("E1")], [_creature("E1", 4)], x=4,
                         source_x_paid=4)
        text = BOLT.replace("NumDmg$ 3", "NumDmg$ X")
        qualified = _qualify(record, text)
        for seed in range(6):
            copy = counterfactual(CopyMaterial(record, qualified, derive_targets(record),
                                               _line(text), text), random.Random(seed), 0)
            assert copy.edit.key is None and copy.edit.text is None
            x = copy.record.state.refs.x
            assert copy.targets["E1"].fields["damage_taken"] == x
            # Both inputs the model reads the paid X through move together.
            assert copy.record.state.entity("S1").x_paid == x

    def test_an_x_the_board_computes_keeps_its_amount_in_every_copy(self):
        """``refs.x`` from an ``SVar:X`` ("damage equal to the number of
        creatures you control"): no copy redraws it; the toughness moves."""
        record = _record([_damage("E1", 4), _died("E1")], [_creature("E1", 4)], x=4)
        text = BOLT.replace("NumDmg$ 3", "NumDmg$ X")
        qualified = _qualify(record, text)
        assert qualified.effect.stated == "computed"
        toughnesses = set()
        for seed in range(30):
            copy = counterfactual(CopyMaterial(record, qualified, derive_targets(record),
                                               _line(text), text), random.Random(seed), 0)
            assert copy.record.state.refs.x == copy.edit.amount == 4
            assert copy.edit.key is None
            assert copy.record.state.entity("S1").x_paid is None
            toughness = copy.record.state.entity("E1").pt.base[1]
            assert copy.targets["E1"].fields["damage_taken"] == 4
            assert (copy.targets["E1"].fields.get("zone_outcome") == DIED) == (toughness <= 4)
            toughnesses.add(toughness)
        assert len(toughnesses) > 1

    def test_a_player_s_life_follows_the_amount(self):
        life = Event(type=EventType.LIFE_CHANGE, subjects=("P1",), params={"delta": -3})
        record = _record([_damage("E1", 3), _died("E1"), life], [_creature("E1", 2)])
        copy = counterfactual(CopyMaterial(record, _qualify(record), derive_targets(record),
                                           _line(BOLT), BOLT), random.Random(5), 0)
        assert copy.targets["P1"].fields["life_delta"] == -copy.targets["E1"].fields[
            "damage_taken"]

    def test_a_shrink_edits_power_and_toughness_together(self):
        qualified = _qualify(
            _record([_shrunk("E1", 2), _died("E1")], [_creature("E1", 2)]), SHRINK)
        edited = edit_text(SHRINK, qualified.effect, 5)
        assert "NumAtt$ -5" in edited and "NumDef$ -5" in edited and "-5/-5" in edited


class TestTheProseSurface:
    """On the prose surface the encoded text is the converted prose, which holds
    no ``NumDmg$``: a copy edits the amount in the prose too, or not at all."""

    PROSE = "cardname deals 3 damage to target creature."

    def test_a_copy_edits_the_amount_in_the_prose_and_the_script_alike(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        copy = counterfactual(CopyMaterial(record, _qualify(record), derive_targets(record),
                                           _line(BOLT), self.PROSE), random.Random(1), 0)
        amount = copy.targets["E1"].fields["damage_taken"]
        assert copy.edit.text == f"cardname deals {int(amount)} damage to target creature."
        assert f"NumDmg$ {int(amount)}" in copy.edit.script

    def test_prose_naming_no_amount_keeps_the_amount_and_moves_the_toughness(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        prose = "cardname deals damage to target creature."
        copy = counterfactual(CopyMaterial(record, _qualify(record), derive_targets(record),
                                           _line(BOLT), prose), random.Random(1), 0)
        assert copy.edit.key is None
        assert copy.targets["E1"].fields["damage_taken"] == 3.0

    def test_role_spans_move_with_an_edit_that_changes_the_prose_length(self):
        from effects.domain.provenance import RoleSpan

        record = _record([_damage("E1", 10), _died("E1")], [_creature("E1", 2)])
        script = BOLT.replace("3", "10")
        prose = "cardname deals 10 damage to target creature."
        target = prose.index("target")
        spans = (RoleSpan(0, target, "effect"), RoleSpan(target, len(prose), "target-spec"))
        copy = counterfactual(CopyMaterial(
            record, _qualify(record, script), derive_targets(record),
            _line(script, spans), prose), random.Random(1), 0)
        moved = copy.edit.text.index("target")
        assert [(s.start, s.end) for s in copy.edit.role_spans] == [
            (0, moved), (moved, len(copy.edit.text)),
        ]


class TestThePinnedOutput:
    """The whole of ``counterfactual``'s output for a seeded draw, on four
    real-shaped records: what any restructuring of it must leave unchanged."""

    SWEEP = ("NumDmg$ 10 | SP$ DamageAll | SpellDescription$ CARDNAME deals 10 damage to "
             "each creature and each player. | ValidCards$ Creature | ValidPlayers$ Player")
    SWEEP_PROSE = "cardname deals 10 damage to each creature and each player."
    WEAKEN = ("IsCurse$ True | NumDef$ -3 | SP$ PumpAll | SpellDescription$ Creatures your "
              "opponents control get -0/-3 until end of turn. | ValidCards$ Creature.OppCtrl")

    @staticmethod
    def _summary(record: EffectRecord, copy) -> tuple:
        """Everything the copy holds, having checked it moved nothing else."""
        state = copy.record.state
        assert replace(
            copy.record, record_id=record.record_id, ability=record.ability,
            state=replace(state, entities=record.state.entities, refs=record.state.refs),
            payload=record.payload,
        ) == record
        assert copy.record.payload == replace(record.payload, events=())
        assert [replace(entity, pt=None, damage=0, x_paid=None) for entity in state.entities] == [
            replace(entity, pt=None, damage=0, x_paid=None) for entity in record.state.entities
        ]
        assert replace(state.refs, x=None) == replace(record.state.refs, x=None)
        if copy.edit.key is None:
            assert copy.record.ability == record.ability
        else:
            assert copy.record.ability == (copy.edit.key,)
            assert replace(copy.edit.key, script_file=KEY.script_file) == KEY
        return (
            copy.record.record_id,
            None if copy.edit.key is None else copy.edit.key.script_file,
            state.refs.x,
            tuple((entity.id, entity.x_paid) for entity in state.entities
                  if entity.x_paid is not None),
            tuple((entity.id, entity.pt.base, entity.damage)
                  for entity in state.entities if entity.pt is not None),
            copy.edit.text, copy.edit.script,
            tuple((span.start, span.end, span.role) for span in copy.edit.role_spans),
            tuple((subject, entry.affected, sorted(entry.fields.items()))
                  for subject, entry in copy.targets.items()),
        )

    def _copies(self, record: EffectRecord, text: str, line, seed: int) -> list[tuple]:
        targets = derive_targets(record)
        before = {subject: replace(entry, fields=dict(entry.fields))
                  for subject, entry in targets.items()}
        qualified = qualify(record, line, targets, BoardReading(_no_keywords))
        assert qualified is not None
        rng = random.Random(seed)
        material = CopyMaterial(record, qualified, targets, line, text)
        out = [self._summary(record, counterfactual(material, rng, n)) for n in range(3)]
        assert targets == before
        return out

    def test_a_sweep_on_the_prose_surface(self):
        from effects.domain.provenance import RoleSpan

        life = [Event(type=EventType.LIFE_CHANGE, subjects=(player,), params={"delta": -10})
                for player in ("P0", "P1")]
        boosted = make_entity(
            "E2", "E2", controller="P1", types=("creature",),
            pt=PowerToughness(base=(1, 1), boosts=(1, 1), counters=(0, 1)), damage=1,
        )
        record = _record(
            [_damage("E1", 10), _died("E1"), _damage("E2", 10), _died("E2"),
             _damage("E3", 10), *life],
            [_creature("E1", 2), boosted, _creature("E3", 12)], targets=(),
        )
        prose = self.SWEEP_PROSE
        each = prose.index("each")
        spans = (RoleSpan(0, each, "effect"), RoleSpan(each, len(prose), "target-spec"))
        got = self._copies(record, prose, _line(self.SWEEP, spans), seed=23)
        assert got == PINNED_SWEEP

    def test_a_one_sided_shrink_on_the_script_surface(self):
        weakened = [Event(type=EventType.PT_CHANGE, subjects=(subject,),
                          params={"toughness_delta": -3}, duration="end_of_turn")
                    for subject in ("E1", "E2")]
        record = _record([weakened[0], _died("E1"), weakened[1]],
                         [_creature("E1", 3), _creature("E2", 6)])
        got = self._copies(record, self.WEAKEN, _line(self.WEAKEN), seed=4)
        assert got == PINNED_WEAKEN

    def test_a_symmetric_shrink_on_the_script_surface(self):
        record = _record([_shrunk("E1", 2), _died("E1")], [_creature("E1", 2)])
        got = self._copies(record, SHRINK, _line(SHRINK), seed=9)
        assert got == PINNED_SHRINK

    def test_a_paid_x_with_a_player_hit(self):
        text = BOLT.replace("NumDmg$ 3", "NumDmg$ X")
        life = Event(type=EventType.LIFE_CHANGE, subjects=("P1",), params={"delta": -4})
        record = _record([_damage("E1", 4), life], [_creature("E1", 6)], x=4, source_x_paid=4)
        got = self._copies(record, text, _line(text), seed=31)
        assert got == PINNED_X


def _sweep(amount: int) -> tuple[str, str]:
    return (f"cardname deals {amount} damage to each creature and each player.",
            f"NumDmg$ {amount} | SP$ DamageAll | SpellDescription$ CARDNAME deals {amount} "
            "damage to each creature and each player. | ValidCards$ Creature | "
            "ValidPlayers$ Player")


def _weaken(amount: int) -> tuple[str, str]:
    text = (f"IsCurse$ True | NumDef$ -{amount} | SP$ PumpAll | SpellDescription$ Creatures "
            f"your opponents control get -0/-{amount} until end of turn. | "
            "ValidCards$ Creature.OppCtrl")
    return text, text


def _shrink(amount: int) -> tuple[str, str]:
    text = SHRINK.replace("-2", f"-{amount}")
    return text, text


def _cf(amount: int) -> str:
    return f"counterfactual/{amount}/{KEY.script_file}"


def _hit(kind: str, amount: int, *, dies: bool, symmetric: bool = False) -> list:
    if kind == "damage":
        fields = [("damage_taken", float(amount))]
    else:
        fields = [("power_delta", -float(amount) if symmetric else 0.0), ("pt_duration", 1),
                  ("toughness_delta", -float(amount))]
    return fields + [("zone_outcome", DIED)] * dies


def _life(amount: int) -> list:
    return [("life_delta", -float(amount))]


_SPANS = ((0, 27, "effect"), (27, 57, "target-spec"))

PINNED_SWEEP = [
    ("r.0-0.1~cf0", _cf(5), None, (), (("E1", (2, 1), 0), ("E2", (1, 1), 1), ("E3", (2, 12), 0)),
     *_sweep(5), _SPANS,
     (("E1", True, _hit("damage", 5, dies=True)), ("E2", True, _hit("damage", 5, dies=True)),
      ("E3", True, _hit("damage", 5, dies=False)), ("P0", True, _life(5)),
      ("P1", True, _life(5)))),
    ("r.0-0.1~cf1", _cf(5), None, (), (("E1", (2, 2), 0), ("E2", (1, 5), 0), ("E3", (2, 12), 0)),
     *_sweep(5), _SPANS,
     (("E1", True, _hit("damage", 5, dies=True)), ("E2", True, _hit("damage", 5, dies=False)),
      ("E3", True, _hit("damage", 5, dies=False)), ("P0", True, _life(5)),
      ("P1", True, _life(5)))),
    ("r.0-0.1~cf2", _cf(6), None, (), (("E1", (2, 4), 0), ("E2", (1, 1), 1), ("E3", (2, 12), 0)),
     *_sweep(6), _SPANS,
     (("E1", True, _hit("damage", 6, dies=True)), ("E2", True, _hit("damage", 6, dies=True)),
      ("E3", True, _hit("damage", 6, dies=False)), ("P0", True, _life(6)),
      ("P1", True, _life(6)))),
]
PINNED_WEAKEN = [
    ("r.0-0.1~cf0", _cf(4), None, (), (("E1", (2, 3), 0), ("E2", (2, 2), 0)), *_weaken(4), (),
     (("E1", True, _hit("shrink", 4, dies=True)), ("E2", True, _hit("shrink", 4, dies=True)))),
    ("r.0-0.1~cf1", _cf(7), None, (), (("E1", (2, 3), 0), ("E2", (2, 3), 0)), *_weaken(7), (),
     (("E1", True, _hit("shrink", 7, dies=True)), ("E2", True, _hit("shrink", 7, dies=True)))),
    ("r.0-0.1~cf2", _cf(2), None, (), (("E1", (2, 1), 0), ("E2", (2, 6), 0)), *_weaken(2), (),
     (("E1", True, _hit("shrink", 2, dies=True)),
      ("E2", True, _hit("shrink", 2, dies=False)))),
]
PINNED_SHRINK = [
    ("r.0-0.1~cf0", _cf(8), None, (), (("E1", (2, 5), 0),), *_shrink(8), (),
     (("E1", True, _hit("shrink", 8, dies=True, symmetric=True)),)),
    ("r.0-0.1~cf1", _cf(3), None, (), (("E1", (2, 1), 0),), *_shrink(3), (),
     (("E1", True, _hit("shrink", 3, dies=True, symmetric=True)),)),
    ("r.0-0.1~cf2", _cf(6), None, (), (("E1", (2, 2), 0),), *_shrink(6), (),
     (("E1", True, _hit("shrink", 6, dies=True, symmetric=True)),)),
]
PINNED_X = [
    ("r.0-0.1~cf0", None, 1, (("S1", 1),), (("E1", (2, 2), 0),), None, None, (),
     (("E1", True, _hit("damage", 1, dies=False)), ("P1", True, _life(1)))),
    ("r.0-0.1~cf1", None, 7, (("S1", 7),), (("E1", (2, 1), 0),), None, None, (),
     (("E1", True, _hit("damage", 7, dies=True)), ("P1", True, _life(7)))),
    ("r.0-0.1~cf2", None, 3, (("S1", 3),), (("E1", (2, 4), 0),), None, None, (),
     (("E1", True, _hit("damage", 3, dies=False)), ("P1", True, _life(3)))),
]


class TestStepWiring:
    """``StepPreparer._counterfactuals``: copies per qualifying record, staged."""

    class _Batcher:
        def __init__(self):
            self.rng = random.Random(11)
            self.overrides = {}

        def line_of(self, key):
            return _line(BOLT)

        def text_of(self, key):
            return BOLT

        def override_text(self, key, text, like, *, script=None, role_spans=None):
            self.overrides[key] = (text, like)

    def test_each_qualifying_record_gets_its_copies_and_their_texts(self):
        from effects.application.step_preparation import StepPreparer

        preparer = SimpleNamespace(
            config=SimpleNamespace(lethality_copies=3),
            _keywords=SimpleNamespace(keywords_of=_no_keywords), sidecars=None,
        )
        lethal_bolt = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        contradicted = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 9)])
        records = [lethal_bolt, contradicted]
        batcher = self._Batcher()
        copies = StepPreparer._counterfactuals(
            preparer, records, [derive_targets(r) for r in records], batcher)
        assert len(copies) == 3
        assert {copy.edit.key for copy in copies} == set(batcher.overrides)
        assert all(batcher.overrides[copy.edit.key] == (copy.edit.text, KEY) for copy in copies)
