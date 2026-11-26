"""A cost half trains no affected gate, and trains what its payment did
(task G11, finding 10.5, decision D7).

Whether an entity was affected is the effect half's question, so a cost half
trains the gate nowhere, as a playability record does not. Its payment still
moved permanents and cards, and those train without the gate at the entities
``payload.costs`` names: a sacrificed or discarded card's move to the
graveyard, an exiled card's to exile, a returned permanent's to its owner's
hand, a tapped permanent's tap state 1 and an untapped one's 0, the source's
counter delta, the payer's life. Nothing else of a cost half trains.

The records are real. The gen-3 fixture holds Benevolent Bodyguard's
sacrifice, Vithian Stinger's tap, Quicksand's tap-and-sacrifice and Skitter of
Lizards' mana-only cast; ``cost-halves.jsonl`` holds seven lines of the review
pilots, each as the collector wrote it. Only the unnamed-channel and
off-battlefield cases edit a real record, since no record pays those channels
alone.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest
import torch

from effects.domain.effect_head_layout import (
    DISPERSED_FIELDS,
    FIELDS_BY_NAME,
    PER_ENTITY_FIELDS,
    PER_ENTITY_WIDTH,
    ZONE_OUTCOMES,
    FieldScope,
    HeadField,
    ZoneOutcome,
)
from effects.domain.effect_losses import EntityLossScoring, per_entity_loss
from effects.domain.effect_targets import derive_targets
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.sampling import SamplingClass
from effects.domain.surface_builder import SurfaceBuilder
from effects.domain.surface_collation import entity_target_tensors
from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import read_shard
from tests.unit.effects.builders import GEN3_RECORDS
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

#: Cut from the review pilots (``output/effects/gen3/scratch/2026-11-2*-review-pilot*``):
#: pilot 5's Tyvar, Jubilant Brawler -2, Firewild Borderpost's returned land
#: and Soul Spike's two exiled cards; pilot 4's Champion of the Weird
#: (1 life and two -1/-1 counters), Grimclaw Bats (1 life), Deadshot
#: Minotaur's cycling discard and Silkbind Faerie's untap symbol. No byte of
#: a line was changed.
_LINES = Path(__file__).parents[3] / "fixtures" / "effects" / "cost-halves.jsonl"

_FIXTURE_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"
_PILOT_5_RUN = "cd2b8419-91d0-4b47-975c-5a3d77b88315"
_BODYGUARD = f"{_FIXTURE_RUN}.2-mwqvqdxrusnv.5209"
_STINGER = f"{_FIXTURE_RUN}.0-mwqw740eu02u.217"
_QUICKSAND = f"{_FIXTURE_RUN}.0-mwqvqdyrb9c0.314"
_SKITTER = f"{_FIXTURE_RUN}.0-mwqvqdyrb9c0.11"
_TYVAR = f"{_PILOT_5_RUN}.2-mwr51yg3i0vh.1486"
_CHAMPION = f"{_FIXTURE_RUN}.5-mwqw5tpydi06.838"
_BATS = f"{_FIXTURE_RUN}.1-mwqw0ogy54jx.2937"
_MINOTAUR = f"{_FIXTURE_RUN}.2-mwqvwtly1zvq.13805"
_SOUL_SPIKE = f"{_PILOT_5_RUN}.3-mwr4qdjt90ol.9717"
_BORDERPOST = f"{_PILOT_5_RUN}.3-mwr4qdjt90ol.3471"
_SILKBIND = f"{_FIXTURE_RUN}.1-mwqw0ogy54jx.8677"

_DIED = ZONE_OUTCOMES.index(ZoneOutcome.DIED)
_EXILED = ZONE_OUTCOMES.index(ZoneOutcome.EXILED)
_TO_HAND = ZONE_OUTCOMES.index(ZoneOutcome.TO_HAND)
_LOYALTY = "counters_delta_loyalty"
_M1M1 = "counters_delta_m1m1"


def _is_cost_half(record: EffectRecord) -> bool:
    return record.kind is RecordKind.RESOLUTION and record.moment is Moment.ACTIVATION


@cache
def _records() -> dict[str, EffectRecord]:
    with _LINES.open(encoding="utf-8") as lines:
        pilots = [record_from_dict(json.loads(line)) for line in lines if line.strip()]
    return {record.record_id: record for record in (*read_shard(GEN3_RECORDS), *pilots)}


def _record(record_id: str) -> EffectRecord:
    return _records()[record_id]


def _cost_halves() -> list[EffectRecord]:
    return [record for record in _records().values() if _is_cost_half(record)]


def _scored(record: EffectRecord) -> dict[str, dict[str, object]]:
    """``entity -> {field: target}``: every field the record trains at each
    entity, written or silent, where the field trains (a gated field only at
    an affected entity)."""
    out: dict[str, dict[str, object]] = {}
    for subject, entry in derive_targets(record).items():
        trained = {spec.name: _trained(entry, spec) for spec in PER_ENTITY_FIELDS}
        trained = {name: value for name, value in trained.items() if value is not None}
        if trained:
            out[subject] = trained
    return out


def _trained(entry, spec):
    """The target ``spec`` trains on at ``entry``, None where it trains none."""
    if spec.gated and not entry.affected:
        return None
    return entry.supervised(spec)


def _batch(*records: EffectRecord):
    """The records' real surfaces and their per-entity targets, collated."""
    builder = SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0))
    surfaces = [builder.build(record) for record in records]
    targets = entity_target_tensors(
        surfaces, [derive_targets(record) for record in records], PER_ENTITY_FIELDS)
    return surfaces, targets


def _column(surface, subject: str) -> int:
    return next(column for column, slot in enumerate(surface.entity_slots)
                if surface.slots[slot].subject_id == subject)


class TestACostHalfTrainsNoGate:
    def test_no_cost_half_trains_the_gate(self):
        halves = _cost_halves()
        assert len(halves) == 46 + 7
        assert not [record.record_id for record in halves if derive_targets(record).gated]

    def test_a_sacrifice_scores_no_gate_anywhere(self):
        _, targets = _batch(_record(_BODYGUARD))
        assert targets.real.any()
        assert not targets.gate_scored.any()

    def test_the_gate_term_is_zero_and_the_zone_term_is_not(self):
        """The sacrificed creature's move reaches the loss with the gate's
        weight zero and the field's mask one."""
        _, targets = _batch(_record(_BODYGUARD))
        torch.manual_seed(0)
        outputs = torch.randn(1, targets.mask.shape[1], PER_ENTITY_WIDTH)
        fields = (FIELDS_BY_NAME[HeadField.ZONE_OUTCOME],)
        _, parts = per_entity_loss(
            outputs, targets,
            EntityLossScoring(
                fields=fields,
                log_dispersion=torch.zeros(len(DISPERSED_FIELDS)),
                report_parts=True,
            ),
        )
        by_name = parts.as_dict()
        assert by_name["gate"] == 0.0
        assert by_name[HeadField.ZONE_OUTCOME] > 0.0


class TestWhatThePaymentDidTrains:
    def test_a_sacrificed_creature_is_scored_as_dying(self):
        record = _record(_BODYGUARD)
        assert _scored(record) == {"E28": {HeadField.ZONE_OUTCOME: _DIED}}
        (surface,), targets = _batch(record)
        zone = targets.scored(HeadField.ZONE_OUTCOME)[0]
        assert zone.nonzero().flatten().tolist() == [_column(surface, "E28")]
        assert targets.fields.values[HeadField.ZONE_OUTCOME][0, _column(surface, "E28")] == _DIED

    def test_a_planeswalkers_minus_two_scores_its_loyalty_delta(self):
        record = _record(_TYVAR)
        tyvar = record.state.refs.source
        assert record.state.entity(tyvar).name == "Tyvar, Jubilant Brawler"
        assert _scored(record) == {tyvar: {_LOYALTY: -2}}

    def test_counters_put_as_a_cost_and_life_paid_score_source_and_payer(self):
        record = _record(_CHAMPION)
        champion = record.state.entity(record.state.refs.source)
        assert _scored(record) == {
            champion.id: {_M1M1: 2}, champion.controller: {HeadField.LIFE_DELTA: -1},
        }

    def test_life_paid_scores_the_payers_life_alone(self):
        record = _record(_BATS)
        payer = record.state.entity(record.state.refs.source).controller
        assert _scored(record) == {payer: {HeadField.LIFE_DELTA: -1}}

    def test_a_tapped_permanent_is_scored_tapped(self):
        assert _scored(_record(_STINGER)) == {"E2": {HeadField.TAP_STATE: 1}}

    def test_an_untap_symbol_scores_untapped(self):
        record = _record(_SILKBIND)
        (untapped,) = record.payload.costs.untapped
        assert _scored(record) == {untapped: {HeadField.TAP_STATE: 0}}

    def test_a_permanent_tapped_and_sacrificed_scores_both(self):
        assert _scored(_record(_QUICKSAND)) == {
            "E73": {HeadField.ZONE_OUTCOME: _DIED, HeadField.TAP_STATE: 1},
        }

    def test_a_discarded_card_goes_to_the_graveyard_and_its_player_trains_nothing(self):
        record = _record(_MINOTAUR)
        (discarded,) = record.payload.costs.discarded
        assert _scored(record) == {discarded: {HeadField.ZONE_OUTCOME: _DIED}}

    def test_exiled_cards_are_scored_exiled(self):
        record = _record(_SOUL_SPIKE)
        exiled = record.payload.costs.exiled
        assert len(exiled) == 2
        assert _scored(record) == {ref: {HeadField.ZONE_OUTCOME: _EXILED} for ref in exiled}

    def test_a_returned_permanent_is_scored_to_hand(self):
        record = _record(_BORDERPOST)
        (returned,) = record.payload.costs.returned
        assert _scored(record) == {returned: {HeadField.ZONE_OUTCOME: _TO_HAND}}


class TestNothingElseOfACostHalfTrains:
    def test_a_mana_only_cast_scores_no_field(self):
        """The cast card's own move to the stack is the cast, not an outcome."""
        record = _record(_SKITTER)
        assert record.payload.costs.mana_by_color and record.payload.events
        assert _scored(record) == {}
        _, targets = _batch(record)
        assert not targets.gate_scored.any()
        assert not any(targets.scored(spec.name).any() for spec in PER_ENTITY_FIELDS)

    @pytest.mark.parametrize("channel", ["revealed", "exerted", "convoked", "delved"])
    def test_a_channel_no_field_names_trains_nothing(self, channel):
        """Hand-built: Skitter of Lizards' mana-only cast, given one card of
        the board in a channel the task names no field for."""
        record = _record(_SKITTER)
        some = record.state.entities[0].id
        costs = replace(record.payload.costs, **{channel: (some,)})
        edited = replace(record, payload=replace(record.payload, costs=costs))
        assert _scored(edited) == {}

    def test_counters_whose_source_left_the_battlefield_train_nothing(self):
        """Hand-built: Tyvar's -2 with Tyvar in the graveyard, where no counter
        sits, so the record cannot say which permanent paid them."""
        record = _record(_TYVAR)
        source = record.state.refs.source
        entities = tuple(
            replace(entity, zone="graveyard") if entity.id == source else entity
            for entity in record.state.entities)
        edited = replace(record, state=replace(record.state, entities=entities))
        assert _scored(edited) == {}

    def test_a_cost_half_takes_no_silent_value(self):
        for spec in PER_ENTITY_FIELDS:
            for scope in (FieldScope.PERMANENT, FieldScope.PLAYER):
                assert spec.supervised_value(
                    {}, set(), record_class=SamplingClass.RESOLUTION_COST, scope=scope,
                ) is None, spec.name

    def test_every_cost_half_scores_only_what_its_payment_names(self):
        for record in _cost_halves():
            costs = record.payload.costs
            named = {*costs.tapped, *costs.untapped, *costs.sacrificed, *costs.discarded,
                     *costs.exiled, *costs.returned}
            if costs.counters:
                named.add(record.state.refs.source)
            if costs.life:
                named.add(record.state.entity(record.state.refs.source).controller)
            assert set(_scored(record)) <= named, record.record_id
