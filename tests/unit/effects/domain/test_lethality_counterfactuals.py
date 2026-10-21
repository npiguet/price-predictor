"""Counterfactual copies of damage and -X/-X resolutions (FR-063e).

A copy redraws the amount and one hit creature's toughness and is labelled by
rule 704.5f-g. The tests pin which records qualify, that every copy's labels
follow the rule whatever was drawn, and that the edited text, key and X agree
with the amount the labels used.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

from effects.domain.effect_head_layout import ZONE_OUTCOMES
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.lethality_counterfactuals import (
    COUNTERFACTUAL_TREE,
    counterfactual,
    edit_text,
    lethal,
    qualify,
)
from effects.domain.provenance import ProvenanceKey
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
    return EntityState(
        id=entity_id, name=entity_id, zone="battlefield", controller="P1",
        types=("creature",), pt=PowerToughness(base=(2, toughness)), damage=damage,
        **overrides,
    )


def _record(events, entities, *, x=None, targets=("E1",)) -> EffectRecord:
    state = make_snapshot(
        global_=GlobalState(turn=3, phase="main1", active="P0", priority="P0", stack_size=1),
        players=(PlayerState(id="P0", life=20, hand=3, library=30, graveyard=0),
                 PlayerState(id="P1", life=20, hand=3, library=30, graveyard=0)),
        entities=(make_entity("S1", "spell", zone="stack"), *entities),
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
    return qualify(record, _line(text), derive_targets(record), keywords_of)


class TestTheRule:
    def test_damage_kills_at_toughness_counting_marked_damage(self):
        assert lethal("damage", 3, _creature("E1", 3))
        assert not lethal("damage", 2, _creature("E1", 3))
        assert lethal("damage", 2, _creature("E1", 3, damage=1))

    def test_a_shrink_kills_at_zero_toughness_or_at_marked_damage(self):
        assert lethal("shrink", 2, _creature("E1", 2))
        assert not lethal("shrink", 2, _creature("E1", 3))
        assert lethal("shrink", 2, _creature("E1", 3, damage=1))


class TestQualify:
    def test_a_lethal_bolt_qualifies(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        qualified = _qualify(record)
        assert qualified is not None
        assert (qualified.kind, qualified.amount, qualified.stated) == ("damage", 3, "text")
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
        record = _record([_damage("E1", 3)], [_creature("E1", 2)])
        assert _qualify(record, keywords_of=lambda e: {"indestructible"}) is None
        aura = EntityState(id="A1", name="aura", zone="battlefield", controller="P0",
                           attached_to="E1")
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2), aura])
        assert _qualify(record) is None

    def test_a_player_hit_by_the_same_amount_comes_along(self):
        life = Event(type=EventType.LIFE_CHANGE, subjects=("P1",), params={"delta": -3})
        record = _record([_damage("E1", 3), _died("E1"), life], [_creature("E1", 2)])
        assert _qualify(record).players == ("P1",)

    def test_a_paid_x_is_the_amount(self):
        record = _record([_damage("E1", 4), _died("E1")], [_creature("E1", 4)], x=4)
        qualified = _qualify(record, BOLT.replace("NumDmg$ 3", "NumDmg$ X"))
        assert (qualified.amount, qualified.stated) == (4, "x")

    def test_a_symmetric_shrink_qualifies(self):
        record = _record([_shrunk("E1", 2), _died("E1")], [_creature("E1", 2)])
        qualified = _qualify(record, SHRINK)
        assert (qualified.kind, qualified.amount, qualified.symmetric) == ("shrink", 2, True)


class TestCopies:
    def test_every_copy_follows_the_rule_whatever_was_drawn(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        targets = derive_targets(record)
        qualified = _qualify(record)
        rng = random.Random(7)
        outcomes = set()
        for number in range(200):
            copy = counterfactual(record, qualified, targets, _line(BOLT), BOLT, rng, number)
            amount = int(copy.text.split("NumDmg$ ")[1].split()[0])
            creature = copy.record.state.entity("E1")
            fields = copy.targets["E1"].fields
            dies = fields.get("zone_outcome") == DIED
            assert dies == (amount >= creature.pt.base[1])
            assert fields["damage_taken"] == amount
            assert f"deals {amount} damage" in copy.text
            outcomes.add(dies)
        assert outcomes == {True, False}

    def test_a_copy_reads_its_text_under_a_key_of_its_own(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        copy = counterfactual(record, _qualify(record), derive_targets(record), _line(BOLT), BOLT,
                              random.Random(1), 0)
        assert copy.key.script_file.startswith(f"{COUNTERFACTUAL_TREE}/")
        assert copy.record.ability == (copy.key,)
        assert copy.record.record_id == "r.0-0.1~cf0"
        assert record.state.entity("E1").pt.base == (2, 2)

    def test_a_paid_x_is_edited_in_the_references_not_the_text(self):
        record = _record([_damage("E1", 4), _died("E1")], [_creature("E1", 4)], x=4)
        text = BOLT.replace("NumDmg$ 3", "NumDmg$ X")
        qualified = _qualify(record, text)
        copy = counterfactual(record, qualified, derive_targets(record), _line(text), text,
                              random.Random(3), 0)
        assert copy.key is None and copy.text is None
        assert copy.targets["E1"].fields["damage_taken"] == copy.record.state.refs.x

    def test_a_player_s_life_follows_the_amount(self):
        life = Event(type=EventType.LIFE_CHANGE, subjects=("P1",), params={"delta": -3})
        record = _record([_damage("E1", 3), _died("E1"), life], [_creature("E1", 2)])
        copy = counterfactual(record, _qualify(record), derive_targets(record), _line(BOLT), BOLT,
                              random.Random(5), 0)
        assert copy.targets["P1"].fields["life_delta"] == -copy.targets["E1"].fields[
            "damage_taken"]

    def test_a_shrink_edits_power_and_toughness_together(self):
        qualified = _qualify(
            _record([_shrunk("E1", 2), _died("E1")], [_creature("E1", 2)]), SHRINK)
        edited = edit_text(SHRINK, qualified, 5)
        assert "NumAtt$ -5" in edited and "NumDef$ -5" in edited and "-5/-5" in edited


class TestTheProseSurface:
    """On the prose surface the encoded text is the converted prose, which holds
    no ``NumDmg$``: a copy edits the amount in the prose too, or not at all."""

    PROSE = "cardname deals 3 damage to target creature."

    def test_a_copy_edits_the_amount_in_the_prose_and_the_script_alike(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        copy = counterfactual(record, _qualify(record), derive_targets(record),
                              _line(BOLT), self.PROSE, random.Random(1), 0)
        amount = copy.targets["E1"].fields["damage_taken"]
        assert copy.text == f"cardname deals {int(amount)} damage to target creature."
        assert f"NumDmg$ {int(amount)}" in copy.script

    def test_prose_naming_no_amount_keeps_the_amount_and_moves_the_toughness(self):
        record = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        prose = "cardname deals damage to target creature."
        copy = counterfactual(record, _qualify(record), derive_targets(record),
                              _line(BOLT), prose, random.Random(1), 0)
        assert copy.key is None
        assert copy.targets["E1"].fields["damage_taken"] == 3.0

    def test_role_spans_move_with_an_edit_that_changes_the_prose_length(self):
        from effects.domain.provenance import RoleSpan

        record = _record([_damage("E1", 10), _died("E1")], [_creature("E1", 2)])
        script = BOLT.replace("3", "10")
        prose = "cardname deals 10 damage to target creature."
        target = prose.index("target")
        spans = (RoleSpan(0, target, "effect"), RoleSpan(target, len(prose), "target-spec"))
        copy = counterfactual(
            record, _qualify(record, script), derive_targets(record),
            _line(script, spans), prose, random.Random(1), 0,
        )
        moved = copy.text.index("target")
        assert [(s.start, s.end) for s in copy.role_spans] == [
            (0, moved), (moved, len(copy.text)),
        ]


class TestStepWiring:
    """``StepPreparer._counterfactuals``: copies per qualifying record, staged."""

    class _Batcher:
        def __init__(self):
            self.rng = random.Random(11)
            self.staged = {}

        def line_of(self, key):
            return _line(BOLT)

        def text_of(self, key):
            return BOLT

        def stand_in(self, key, text, like, *, script=None, role_spans=None):
            self.staged[key] = (text, like)

    def test_each_qualifying_record_gets_its_copies_and_their_texts(self):
        from effects.application.step_preparation import StepPreparer

        preparer = SimpleNamespace(
            config=SimpleNamespace(lethality_copies=3),
            _keywords=SimpleNamespace(keywords_of=_no_keywords),
        )
        lethal_bolt = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])
        contradicted = _record([_damage("E1", 3), _died("E1")], [_creature("E1", 9)])
        records = [lethal_bolt, contradicted]
        batcher = self._Batcher()
        copies = StepPreparer._counterfactuals(
            preparer, records, [derive_targets(r) for r in records], batcher)
        assert len(copies) == 3
        assert {copy.key for copy in copies} == set(batcher.staged)
        assert all(batcher.staged[copy.key] == (copy.text, KEY) for copy in copies)
