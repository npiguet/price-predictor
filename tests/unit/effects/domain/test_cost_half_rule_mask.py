"""A cost half's tag names the entities its payment trains at, so an added
copy of it trains what the payment did under the hard rule mask (task G11b,
finding 10.5, decision D7).

G11 made a cost half train, without the gate, ``zone_outcome``, ``tap_state``,
the counter deltas and ``life_delta`` at the entities ``payload.costs`` names.
A rule's effective outputs are the map outputs its tagged records carry at the
tags' outcome entities (``rule_output_carriage.carried_outputs``), and the
hard mask trains an added copy's drawn rule's effective outputs alone; a tag
naming no outcome entity on a cost half would leave the payment's fields out
of both, and an added copy of a sacrifice would not learn that the creature
dies. So a cost half's tags name the paid entities as their outcome entities
(research R1).

The records are real (the gen-3 fixture and ``cost-halves.jsonl``, as G11's
tests read them). No shipped detector family reads a cost half: each reads an
effect half, so the tag is hand-built, from a stand-in family matching every
resolution record with no outcome entity of its own.
"""

from __future__ import annotations

from dataclasses import replace

from effects.domain.cost_targets import is_cost_half, paid_entities
from effects.domain.effect_head_layout import (
    PER_ENTITY_FIELDS,
    ZONE_OUTCOMES,
    HeadField,
    ZoneOutcome,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.records import Curation
from effects.domain.rule_catalogue import DetectorDeclaration, Polarity, Scenario
from effects.domain.rule_detectors import DetectorIndex, FamilyMatch, tag_record
from effects.domain.rule_mask import GATE_OUTPUT, RuleMask, copy_weights
from effects.domain.surface_collation import entity_target_tensors
from tests.unit.effects.domain.test_cost_half_targets import (
    _BODYGUARD,
    _QUICKSAND,
    _batch,
    _column,
    _cost_halves,
    _record,
    _records,
    _scored,
)

RULE = "701.21a"
#: The map outputs a sacrifice rule might name: the gate and the zone outcome.
MAP_OUTPUTS = (GATE_OUTPUT, HeadField.ZONE_OUTCOME)
_DIED = ZONE_OUTCOMES.index(ZoneOutcome.TO_GRAVEYARD)


class _EveryResolution:
    """A stand-in family: every resolution record fires, its condition the
    acting source, no outcome entity of its own (what a family reading the
    cost half's acting line would find)."""

    id = "every_resolution"
    params = frozenset()
    general = frozenset()
    kinds = frozenset({"resolution"})

    def compile(self, params):
        return self.match

    def match(self, record, lines, params=None):
        source = record.state.refs.source
        return FamilyMatch(Polarity.FIRES, True, condition=(source,) if source else ())


def _index() -> DetectorIndex:
    declaration = DetectorDeclaration("every_resolution", "resolution", "", "")
    scenario = Scenario(id=f"{RULE}-1", unit="701.21", rules=(RULE,),
                        record_kind="resolution", variants=(), detector=declaration)
    return DetectorIndex({"resolution": ((scenario, declaration, _EveryResolution()),)})


def _tag(record):
    (tag,) = tag_record(record, _index()).tags
    return tag


def _added_copy(record, tag):
    """``record`` as rule budgets write an added copy drawn for :data:`RULE`."""
    return replace(record, curation=Curation(
        natural_share=1, copies=2, drawn_rule=RULE, drawn_rules=(RULE,),
        condition_entities=tag.condition, outcome_entities=tag.outcome))


class TestACostHalfsTagNamesWhoPaid:
    def test_the_paid_entities_are_those_its_targets_train_at(self):
        halves = _cost_halves()
        assert len(halves) == 46 + 7
        for record in halves:
            assert set(paid_entities(record)) >= set(_scored(record)), record.record_id

    def test_no_other_record_names_a_paid_entity(self):
        assert not [r.record_id for r in _records().values()
                    if not is_cost_half(r) and paid_entities(r)]

    def test_a_sacrifice_cost_halfs_tag_names_the_sacrificed_creature(self):
        tag = _tag(_record(_BODYGUARD))
        assert tag.outcome == ("E28",)

    def test_a_tapped_and_sacrificed_permanent_is_named_once(self):
        assert _tag(_record(_QUICKSAND)).outcome == ("E73",)

    def test_an_effect_halfs_tag_keeps_its_familys_outcome(self):
        effect = next(r for r in _records().values()
                      if r.kind.value == "resolution" and not is_cost_half(r))
        assert _tag(effect).outcome == ()


class TestAnAddedCopyTrainsWhatThePaymentDid:
    def test_the_rules_effective_outputs_hold_the_zone_outcome(self):
        """What the survey writes to the manifest for the rule (``_note_carried``)."""
        from effects.domain.rule_output_carriage import carried_outputs

        record = _record(_BODYGUARD)
        assert carried_outputs(record, _tag(record).outcome, MAP_OUTPUTS) == {
            HeadField.ZONE_OUTCOME}

    def test_the_hard_mask_trains_the_sacrificed_creatures_zone_outcome(self):
        from effects.domain.rule_output_carriage import carried_outputs

        natural = _record(_BODYGUARD)
        tag = _tag(natural)
        effective = {RULE: tuple(carried_outputs(natural, tag.outcome, MAP_OUTPUTS))}
        copy = _added_copy(natural, tag)
        weights = copy_weights(copy, RuleMask.HARD, effective)
        (surface,), _ = _batch(copy)
        targets = entity_target_tensors(
            [surface], [derive_targets(copy)], PER_ENTITY_FIELDS, [weights])
        column = _column(surface, "E28")
        zone = targets.scored(HeadField.ZONE_OUTCOME)[0]
        assert zone.nonzero().flatten().tolist() == [column]
        assert targets.fields.values[HeadField.ZONE_OUTCOME][0, column] == _DIED
        assert not targets.gate_scored.any()

    def test_the_soft_mask_trains_it_at_full_weight(self):
        from effects.domain.rule_output_carriage import carried_outputs

        natural = _record(_BODYGUARD)
        tag = _tag(natural)
        effective = {RULE: tuple(carried_outputs(natural, tag.outcome, MAP_OUTPUTS))}
        weights = copy_weights(_added_copy(natural, tag), RuleMask.SOFT, effective)
        assert weights.weight(HeadField.ZONE_OUTCOME) == 1.0
        assert weights.weight(HeadField.TAP_STATE) == 0.5
