"""The control score reads a gated field conditionally on a record that trains
no gate, as training reads it (task G11b, finding 10.5, decision D7).

A cost half trains no affected gate and trains what its payment did at the
entities ``payload.costs`` names (G11); a rule test of such a row reads the
field head alone there, never ``P(affected) ×`` it, and its targets are the
payment's. The records are the real gen-3 fixture's cost halves; the rule-test
rows are written by hand, as the control score's own tests write them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from effects.application.control_score import _OutputReading, decoder_of
from effects.domain.cost_targets import is_cost_half, paid_entities
from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    GATE_INDEX,
    PER_ENTITY_WIDTH,
    ZONE_OUTCOMES,
    HeadField,
    ZoneOutcome,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.rule_output_targets import entity_indicators
from effects.domain.selection_set import RULE_TESTS, SelectionSet, TaggedRecord
from effects.domain.surface_builder import has_card_slot
from tests.unit.effects.application.test_control_score import (  # noqa: F401
    STRATA,
    _epoch,
    _prepare,
    run,
)
from tests.unit.effects.domain.test_cost_half_targets import (
    _BODYGUARD,
    _batch,
    _column,
    _record,
)

_DIED = ZONE_OUTCOMES.index(ZoneOutcome.DIED)
_ZONE = HeadField.ZONE_OUTCOME
RULE = "r.paid"


def _sacrifice():
    record = _record(_BODYGUARD)
    (surface,), targets = _batch(record)
    return targets, _column(surface, "E28")


class TestTheSacrificedCreaturesZoneOutcome:
    def test_its_target_is_the_move_to_the_graveyard(self):
        hot = [0.0] * len(ZONE_OUTCOMES)
        hot[_DIED] = 1.0
        assert entity_indicators(_ZONE, derive_targets(_record(_BODYGUARD)), "E28") == hot

    def test_an_entity_the_payment_did_not_reach_is_read_nowhere(self):
        record = _record(_BODYGUARD)
        (surface,), _ = _batch(record)
        other = next(surface.slots[slot].subject_id for slot in surface.entity_slots
                     if surface.slots[slot].subject_id != "E28")
        assert entity_indicators(_ZONE, derive_targets(record), other) is None

    def test_it_is_read_conditionally(self):
        targets, column = _sacrifice()
        reading = _OutputReading(decoder_of(_ZONE))
        assert reading.at(targets, 0, column) is reading.conditional

    def test_the_conditional_reading_leaves_the_gate_out(self):
        outputs = torch.zeros(1, 1, PER_ENTITY_WIDTH)
        outputs[0, 0, GATE_INDEX] = -2.0
        start, _ = FIELD_SLICES[_ZONE]
        outputs[0, 0, start + _DIED] = 6.0
        heads = SimpleNamespace(gathered=outputs)
        reading = _OutputReading(decoder_of(_ZONE))
        reading.candidate = _DIED
        rows = columns = torch.tensor([0])
        alone = float(reading.conditional(heads, rows, columns)[0])
        gated = float(reading(heads, rows, columns)[0])
        assert alone == pytest.approx(gated / float(torch.sigmoid(torch.tensor(-2.0))))
        assert alone > 0.9


def test_a_gated_records_entity_keeps_the_gated_reading(records):
    record = next(r for r in records if derive_targets(r).gated and r.state.entities)
    (surface,), targets = _batch(record)
    column = _column(surface, record.state.entities[0].id)
    reading = _OutputReading(decoder_of(_ZONE))
    assert reading.at(targets, 0, column) is reading


def _paid_rows(records):
    """Fixture cost halves whose payment moved a card, at the paid entities."""
    return [TaggedRecord(r.record_id, r.game_id, paid_entities(r)) for r in records
            if is_cost_half(r) and _ZONE in _trained_fields(r)]


def _trained_fields(record):
    return {name for entry in derive_targets(record).values() if entry.affected
            for name in entry.fields}


def _unaffected_rows(records):
    """Gated resolutions at an entity nothing affected: the zone stayed."""
    rows = []
    for record in records:
        targets = derive_targets(record)
        if not targets.gated or is_cost_half(record):
            continue
        entity = next((e.id for e in record.state.entities
                       if has_card_slot(e) and not _affected(targets, e.id)), None)
        if entity is not None:
            rows.append(TaggedRecord(record.record_id, record.game_id, (entity,)))
    return rows[:8]


def _affected(targets, entity: str) -> bool:
    entry = targets.get(entity)
    return entry is not None and entry.affected


def _selection(records) -> SelectionSet:
    """Every game in the reporting half, one rule test whose ``fires`` rows
    are the payments."""
    fires, does_not = _paid_rows(records), _unaffected_rows(records)
    assert len(fires) >= 2 and len(does_not) >= 2
    games = tuple(sorted({r.game_id for r in records}))
    tests = {RULE: {"fires": tuple(fires), "does_not_fire": tuple(does_not)}}
    return SelectionSet(
        corpus_digest="fixture", r=4,
        halves={s: {"selection": (), "reporting": games} for s in STRATA},
        probes={s: {"reporting": {}} for s in STRATA}, probe_kinds={},
        grids={s: {"reporting": {}} for s in STRATA},
        rule_tests={s: {"reporting": tests} for s in STRATA},
        deviance_samples={s: {"reporting": ()} for s in STRATA},
        standard_errors={}, left_out={}, settings={"max_item_error": 0.1},
    )


def test_a_rule_read_at_the_payments_is_scored(run, records):  # noqa: F811
    scorer, _, _ = _prepare(run, records, _selection(records), half="reporting",
                            strict=False, components=(RULE_TESTS,),
                            rule_outputs={RULE: (_ZONE,)})
    score = _epoch(run, scorer)
    for stratum in STRATA:
        assert set(score.strata[stratum].items[RULE_TESTS]) == {RULE}
