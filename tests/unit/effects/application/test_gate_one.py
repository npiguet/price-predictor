"""Gate 1's stratum and its plumbing (T099).

Gate 1 blocks shipping, and until this existed it was reported only when its
baseline was *missing* — supplying the baseline produced no gate-1 line at all,
so the acceptance criterion for User Story 1 silently never ran.

The stratum is where the gate is won or lost. The identity baseline holds a free
vector per ability text, so on a text it saw in training it recalls the answer
without reading a word; comparing the two over such texts measures memorization
on both sides. Only texts absent from training separate them.
"""

from __future__ import annotations

import pytest

from effects.application.gate_one import (
    evaluate_gate_one,
    training_texts,
    unique_text_records,
)
from effects.domain.evaluation_report import CheckStatus
from effects.domain.gate_metrics import GateOneMetrics
from effects.domain.provenance import ProvenanceKey, first_text
from effects.domain.records import (
    EffectRecord,
    RecordKind,
    acting_keys,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot
from tests.unit.effects.builders import make_record

_SNAPSHOT = StateSnapshot(
    global_=GlobalState(
        turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
    ),
    players=(),
    entities=(),
)


class _Batcher:
    """Stands in for the real batcher: a key maps to a text by its index."""

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = texts

    def text_of(self, key):
        return self._texts.get(key.index_within_kind)


def _key(index: int) -> ProvenanceKey:
    return ProvenanceKey("cardsfolder/a/a.txt", 0, "spell", index)


def _record(
    index: int | None, *, kind: RecordKind = RecordKind.RESOLUTION,
    game: str = "g1",
) -> EffectRecord:
    return make_record(
        kind, record_id=f"r.{index}", run_id="r", game_id=game,
        # None, not (): the envelope requires the kinds without an acting line
        # to carry no ability field at all.
        ability=(_key(index),) if index is not None else None,
    )


def _acting(records) -> set:
    return {acting_keys(record) for record in records}


def _text(record, batcher) -> str | None:
    return first_text(acting_keys(record), batcher.text_of)


@pytest.fixture
def batcher() -> _Batcher:
    return _Batcher({0: "deal 3 damage", 1: "draw a card", 2: "gain 2 life"})


class TestTheStratum:
    def test_a_text_seen_in_training_is_excluded(self, batcher):
        """The baseline recalls it, so the comparison measures nothing."""
        training = [_record(0)]
        held_out = [_record(0), _record(1)]
        stratum = unique_text_records(
            held_out, batcher.text_of, seen=training_texts(_acting(training), batcher.text_of),
        )
        assert [_text(r, batcher) for r in stratum] == ["draw a card"]

    def test_a_text_absent_from_training_is_kept(self, batcher):
        stratum = unique_text_records(
            [_record(2)], batcher.text_of,
            seen=training_texts(_acting([_record(0)]), batcher.text_of),
        )
        assert len(stratum) == 1

    def test_the_same_text_appearing_twice_is_still_kept(self, batcher):
        """The rule is "absent from training", not "seen once" — a text held out
        twice is held out twice, and both records score."""
        stratum = unique_text_records(
            [_record(1), _record(1)], batcher.text_of, seen=set(),
        )
        assert len(stratum) == 2

    def test_a_combat_record_is_not_in_the_stratum(self, batcher):
        """The envelope forbids it an acting line, so it could not be
        stratified by text even if the kind filter were removed."""
        combat = _record(None, kind=RecordKind.COMBAT)
        assert _text(combat, batcher) is None
        assert unique_text_records([combat], batcher.text_of, seen=set()) == []

    def test_a_record_whose_text_does_not_resolve_is_skipped(self, batcher):
        assert unique_text_records([_record(99)], batcher.text_of, seen=set()) == []

    def test_an_empty_training_set_keeps_every_resolution_record(self, batcher):
        stratum = unique_text_records(
            [_record(0), _record(1), _record(2)], batcher.text_of, seen=set(),
        )
        assert len(stratum) == 3


class TestTrainingTexts:
    def test_it_collects_every_acting_text(self, batcher):
        assert training_texts(_acting([_record(0), _record(1)]), batcher.text_of) == {
            "deal 3 damage", "draw a card",
        }

    def test_records_without_an_acting_line_contribute_nothing(self, batcher):
        assert training_texts(_acting([_record(None)]), batcher.text_of) == set()


class TestGateOne:
    """All three margins must hold, on the card-disjoint unique-text stratum."""

    def _identity(self) -> GateOneMetrics:
        return GateOneMetrics(
            affected_gate_f1=0.60,
            zone_outcome_accuracy=0.50,
            mean_poisson_deviance=1.00,
        )

    def _passing(self) -> GateOneMetrics:
        return GateOneMetrics(
            affected_gate_f1=0.70, zone_outcome_accuracy=0.60,
            mean_poisson_deviance=0.90,
        )

    def test_all_three_margins_holding_passes(self):
        result = evaluate_gate_one(self._passing(), self._identity())
        assert result.status is CheckStatus.PASS
        assert not result.blocks

    def test_a_short_gate_f1_gain_fails(self):
        model = GateOneMetrics(0.64, 0.60, 0.90)
        result = evaluate_gate_one(model, self._identity())
        assert result.status is CheckStatus.FAIL
        assert "affected-gate F1" in result.detail

    def test_a_short_zone_accuracy_gain_fails(self):
        model = GateOneMetrics(0.70, 0.54, 0.90)
        result = evaluate_gate_one(model, self._identity())
        assert result.status is CheckStatus.FAIL
        assert "zone-outcome accuracy" in result.detail

    def test_a_short_deviance_reduction_fails(self):
        model = GateOneMetrics(0.70, 0.60, 0.96)
        result = evaluate_gate_one(model, self._identity())
        assert result.status is CheckStatus.FAIL
        assert "deviance" in result.detail

    def test_exactly_meeting_a_threshold_passes(self):
        model = GateOneMetrics(0.65, 0.55, 0.95)
        assert evaluate_gate_one(model, self._identity()).status is CheckStatus.PASS

    def test_failing_two_margins_reports_both(self):
        model = GateOneMetrics(0.60, 0.50, 0.90)
        result = evaluate_gate_one(model, self._identity())
        assert "affected-gate F1" in result.detail
        assert "zone-outcome accuracy" in result.detail

    def test_it_blocks_shipping(self):
        model = GateOneMetrics(0.60, 0.50, 1.00)
        assert evaluate_gate_one(model, self._identity()).blocks

    def test_the_reduction_is_relative_not_absolute(self):
        identity = GateOneMetrics(0.60, 0.50, 0.10)
        model = GateOneMetrics(0.70, 0.60, 0.094)  # 6% relative
        assert evaluate_gate_one(model, identity).status is CheckStatus.PASS

    def test_a_margin_that_could_not_be_measured_fails(self):
        """No zone target on the stratum makes the accuracy NaN; a NaN margin
        compared against its floor is False, which used to read as a pass."""
        model = GateOneMetrics(0.70, float("nan"), 0.90)
        result = evaluate_gate_one(model, self._identity())
        assert result.status is CheckStatus.FAIL
        assert "zone-outcome accuracy" in result.detail
        assert "not measured" in result.detail

    def test_an_unmeasured_model_deviance_fails(self):
        model = GateOneMetrics(0.70, 0.60, float("nan"))
        assert evaluate_gate_one(model, self._identity()).status is CheckStatus.FAIL
