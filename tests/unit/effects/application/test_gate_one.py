"""Gate 1's stratum and its plumbing.

Gate 1 blocks shipping, and until this existed it was reported only when its
baseline was *missing* — supplying the baseline produced no gate-1 line at all,
so the acceptance criterion for User Story 1 silently never ran.

The stratum is where the gate is won or lost. The identity baseline holds a free
vector per ability text, so on a text it saw in training it recalls the answer
without reading a word; comparing the two over such texts measures memorization
on both sides. Only texts absent from training separate them.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from effects.application.ability_text import KeyResolver
from effects.application.gate_one import (
    evaluate_gate_one,
    measure,
    run_gate_one,
    training_texts,
    unique_text_records,
    unique_text_stratum,
)
from effects.application.model_loading import Runnable, evaluation_fields
from effects.domain.evaluation_report import CheckStatus
from effects.domain.gate_metrics import GATE1_MIN_ZONE_CLASS_CELLS, GateOneMetrics
from effects.domain.provenance import ProvenanceKey, first_text
from effects.domain.records import (
    EffectRecord,
    RecordKind,
    acting_keys,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot
from effects.domain.strata import Stratum
from effects.infrastructure.effect_model_store import ScoringFlags
from effects.infrastructure.shard_io import write_shard
from effects.infrastructure.vocabulary_file import vocabulary_surface
from tests.unit.effects.builders import fixture_run, make_record

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
        import dataclasses

        stratum = unique_text_records(
            [_record(1), dataclasses.replace(_record(1), record_id="r.1b")],
            batcher.text_of, seen=set(),
        )
        assert len(stratum) == 2

    def test_a_records_repeated_row_is_read_once(self, batcher):
        """A curated stratum repeats a record's row for every added copy;
        gate 1 reads each distinct record once (FR-068a)."""
        stratum = unique_text_records(
            [_record(1), _record(2), _record(1)], batcher.text_of, seen=set(),
        )
        assert [r.record_id for r in stratum] == ["r.1", "r.2"]

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


def _metrics(
    affected_gate_f1: float, zone_outcome_accuracy: float, mean_deviance: float,
) -> GateOneMetrics:
    """One model's scores, its own dispersions the main model's."""
    return GateOneMetrics(
        affected_gate_f1, zone_outcome_accuracy, mean_deviance, mean_deviance,
    )


class TestGateOne:
    """All three margins must hold, on the card-disjoint unique-text stratum."""

    def _identity(self) -> GateOneMetrics:
        return _metrics(
            affected_gate_f1=0.60,
            zone_outcome_accuracy=0.50,
            mean_deviance=1.00,
        )

    def _passing(self) -> GateOneMetrics:
        return _metrics(
            affected_gate_f1=0.70, zone_outcome_accuracy=0.60,
            mean_deviance=0.90,
        )

    def test_all_three_margins_holding_passes(self):
        result = evaluate_gate_one(self._passing(), self._identity())
        assert result.status is CheckStatus.PASS
        assert not result.blocks

    def test_a_short_gate_f1_gain_fails(self):
        model = _metrics(0.64, 0.60, 0.90)
        result = evaluate_gate_one(model, self._identity())
        assert result.status is CheckStatus.FAIL
        assert "affected-gate F1" in result.detail

    def test_a_short_zone_accuracy_gain_fails(self):
        model = _metrics(0.70, 0.54, 0.90)
        result = evaluate_gate_one(model, self._identity())
        assert result.status is CheckStatus.FAIL
        assert "zone-outcome accuracy" in result.detail

    def test_a_short_deviance_reduction_fails(self):
        model = _metrics(0.70, 0.60, 0.96)
        result = evaluate_gate_one(model, self._identity())
        assert result.status is CheckStatus.FAIL
        assert "deviance" in result.detail

    def test_exactly_meeting_a_threshold_passes(self):
        model = _metrics(0.65, 0.55, 0.95)
        assert evaluate_gate_one(model, self._identity()).status is CheckStatus.PASS

    def test_failing_two_margins_reports_both(self):
        model = _metrics(0.60, 0.50, 0.90)
        result = evaluate_gate_one(model, self._identity())
        assert "affected-gate F1" in result.detail
        assert "zone-outcome accuracy" in result.detail

    def test_it_blocks_shipping(self):
        model = _metrics(0.60, 0.50, 1.00)
        assert evaluate_gate_one(model, self._identity()).blocks

    def test_the_reduction_is_relative_not_absolute(self):
        identity = _metrics(0.60, 0.50, 0.10)
        model = _metrics(0.70, 0.60, 0.094)  # 6% relative
        assert evaluate_gate_one(model, identity).status is CheckStatus.PASS

    def test_a_margin_that_could_not_be_measured_fails(self):
        """No zone target on the stratum makes the accuracy NaN; a NaN margin
        compared against its floor is False, which used to read as a pass."""
        model = _metrics(0.70, float("nan"), 0.90)
        result = evaluate_gate_one(model, self._identity())
        assert result.status is CheckStatus.FAIL
        assert "zone-outcome accuracy" in result.detail
        assert "not measured" in result.detail

    def test_an_unmeasured_model_deviance_fails(self):
        model = _metrics(0.70, 0.60, float("nan"))
        assert evaluate_gate_one(model, self._identity()).status is CheckStatus.FAIL

    def test_an_unmeasured_zone_accuracy_says_why(self):
        model = _metrics(0.70, float("nan"), 0.90)
        detail = evaluate_gate_one(model, self._identity()).detail
        assert f"fewer than two zone outcomes of {GATE1_MIN_ZONE_CLASS_CELLS} cells" in detail

    def test_an_unmeasured_deviance_says_why(self):
        model = _metrics(0.70, 0.60, float("nan"))
        detail = evaluate_gate_one(model, self._identity()).detail
        assert "no magnitude field's constant-prediction deviance" in detail

    def test_each_model_s_own_dispersion_figure_is_printed_beside_the_margin(self):
        identity = GateOneMetrics(0.60, 0.50, 1.00, 0.40)
        values = evaluate_gate_one(self._passing(), identity).values
        assert values["deviance_reduction"] == pytest.approx(0.10)
        assert values["identity_deviance"] == 1.00
        assert values["identity_own_dispersion_deviance"] == 0.40
        assert values["model_own_dispersion_deviance"] == 0.90


# ── two tiny models over the fixture stratum ─────────────────────────────

#: The checkpoints' default --deviance-floor.
FLOOR = 0.04


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("gate-one"))


def _runnable(run, model, records) -> Runnable:
    batcher = run.preparer.scoring_batcher()
    batcher.materializer = run.scoring
    return Runnable(run.encoder, model, batcher, evaluation_fields(records))


def _wider(model):
    """``model`` with every log-dispersion lowered: the same predictions, a
    wider spread around them."""
    other = copy.deepcopy(model)
    with torch.no_grad():
        other.per_entity_dispersion.log_dispersion -= 3.0
    return other


class _Context:
    """The evaluation context gate 1 reads: the fixture's records as the
    card-disjoint stratum, no training text, the tiny run as the main model."""

    def __init__(self, run, corpus, records, shard) -> None:
        write_shard(shard, records)
        self._run = run
        self.main = SimpleNamespace(
            provenance=SimpleNamespace(card_disjoint_games=sorted({r.game_id for r in records})),
            scored_under=lambda: ScoringFlags("hard", 1.0, FLOOR),
        )
        self.tokenizer_files = SimpleNamespace(vocab=corpus.vocab)
        self.shards = {str(Stratum.TRAINING): (), str(Stratum.CARD_DISJOINT): (shard,)}
        self.sidecars = run.preparer.scoring_batcher().sidecars

    def main_runnable(self, records) -> Runnable:
        return _runnable(self._run, self._run.model, records)

    def runnable_of(self, identity, records) -> Runnable:
        return _runnable(self._run, identity, records)


@pytest.fixture(scope="module")
def context(run, corpus, records, tmp_path_factory):
    return _Context(run, corpus, records, tmp_path_factory.mktemp("cd") / "cd.jsonl.gz")


@pytest.fixture(scope="module")
def stratum(context):
    texts = KeyResolver(context.sidecars, vocabulary_surface(context.tokenizer_files.vocab))
    found = unique_text_stratum(context.shards, context.main.provenance.card_disjoint_games,
                                texts.text_of)
    assert found
    return found


def test_measure_reads_the_fixture_stratum(run, stratum):
    own = run.model.per_entity_dispersion().detach()
    metrics = measure(stratum, _runnable(run, run.model, stratum),
                      log_dispersion=own, deviance_floor=FLOOR)
    assert math.isfinite(metrics.affected_gate_f1)
    assert math.isfinite(metrics.mean_deviance)
    assert metrics.own_dispersion_deviance == metrics.mean_deviance


def test_measure_reads_the_deviance_under_the_dispersions_it_is_given(run, stratum):
    main = run.model.per_entity_dispersion().detach()
    wider = _wider(run.model)
    metrics = measure(stratum, _runnable(run, wider, stratum),
                      log_dispersion=main, deviance_floor=FLOOR)
    alike = measure(stratum, _runnable(run, run.model, stratum),
                    log_dispersion=main, deviance_floor=FLOOR)
    assert metrics.mean_deviance == alike.mean_deviance
    assert metrics.own_dispersion_deviance != metrics.mean_deviance


def test_two_models_differing_only_in_dispersion_have_a_zero_deviance_margin(run, context):
    """Each read under its own dispersions, the wider one alone moved the
    margin."""
    result = run_gate_one(context, _wider(run.model))
    assert result.values["deviance_reduction"] == 0.0
    assert result.values["affected_gate_f1_gain"] == 0.0
    assert (result.values["identity_own_dispersion_deviance"]
            != result.values["identity_deviance"])
    assert result.status is CheckStatus.FAIL
