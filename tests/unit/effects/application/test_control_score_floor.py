"""The control score marks down what it cannot read and reads only outputs
whose targets differ between a rule's polarities (12.9).

The scorer runs over the gen-3 fixture records and the hand-frozen selection
set of ``test_control_score``; a reading is made infinite by hand where the
model's weights would overflow.
"""

from __future__ import annotations

import math

import pytest

from effects.application.control_items import explained_deviance, probe_item, rule_item
from effects.application.control_reads import OutputReading
from effects.application.control_score import ControlScore, StratumScore
from effects.application.half_score_report import HalfReading, half_score_checks
from effects.application.run_reporter import format_non_finite
from effects.domain.head_decoding import decoder_of
from effects.domain.score_scales import grid_score
from effects.domain.selection_set import PROBES, RULE_TESTS
from tests.unit.effects.application.test_control_score import (
    STRATA,
    _epoch,
    _prepare,
    _selection,
)
from tests.unit.effects.builders import fixture_run

INF = float("inf")


# ── an item with a reading it cannot read scores at the floor ───────────


def test_a_probe_with_one_infinite_row_scores_the_floor_not_the_rest():
    labels = [1, 1, 0, 0, 1, 0]
    reads = [0.9, 0.8, 0.1, 0.2, INF, 0.3]
    assert probe_item("binary", labels, reads) == -1.0
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert probe_item("amount", y, [1.0, 2.0, 3.0, 4.0, INF]) == -1.0
    assert probe_item("amount", y, [1.0, 2.0, 3.0, 4.0, math.nan]) == -1.0


def test_a_row_without_a_label_is_still_nothing_to_score_not_the_floor():
    assert probe_item("binary", [1, 0, math.nan], [0.9, 0.1, 0.5]) == pytest.approx(1.0)
    assert math.isnan(probe_item("binary", [1, 1], [0.9, 0.5]))


def test_a_rule_with_one_infinite_reading_scores_the_floor():
    perfect = ([0.9, 0.8], [0.1, 0.2])
    assert rule_item([perfect, ([0.9, INF], [0.1, 0.2])]) == -1.0


def test_a_grid_with_one_infinite_reading_scores_the_floor():
    answers = [[0.0, 1.0], [1.0, 0.0]]
    assert grid_score(answers, answers) == 1.0
    assert grid_score([[0.0, 1.0], [INF, 0.0]], answers) == -1.0
    assert grid_score([[math.nan, 1.0], [1.0, 0.0]], answers) == -1.0


def test_a_deviance_term_with_a_non_finite_loss_scores_the_floor():
    parts = {"a": INF, "b": math.nan, "c": 0.5}
    floor = {"a": 1.0, "b": 1.0, "c": 1.0}
    got = explained_deviance(parts, floor, kept=("a", "b", "c"), deviance_floor=0.0)
    assert got == pytest.approx({"a": -1.0, "b": -1.0, "c": 0.5})


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("control-floor"))


@pytest.fixture(scope="module")
def scorer(run, records):
    return _prepare(run, records, _selection(records))[0]


def _epoch_with(run, scorer, spoil) -> object:
    """An epoch whose readings ``spoil`` edits in place before they are scored."""
    real = scorer._values

    def spoiled(cache, forward):
        values = real(cache, forward)
        if cache.stratum == STRATA[0]:
            spoil(cache, values)
        return values

    scorer._values = spoiled
    try:
        return _epoch(run, scorer)
    finally:
        del scorer._values


def test_an_epoch_with_one_infinite_probe_reading_marks_that_item_down_and_counts_it(run, scorer):
    clean = _epoch(run, scorer).strata[STRATA[0]]
    assert clean.non_finite == 0

    def spoil(cache, values):
        values[cache.probes["dies"].positions[3]] = INF

    spoiled = _epoch_with(run, scorer, spoil).strata[STRATA[0]]
    assert spoiled.items[PROBES]["dies"] == -1.0
    assert spoiled.items[PROBES]["affordable"] == clean.items[PROBES]["affordable"]
    assert spoiled.non_finite == 1
    assert _epoch(run, scorer).strata[STRATA[1]].non_finite == 0


def test_an_epoch_with_one_infinite_rule_reading_marks_that_rule_down(run, scorer):
    def spoil(cache, values):
        reading = next(iter(cache.rules["r.board"].values()))
        values[reading.fires[0][0]] = INF

    spoiled = _epoch_with(run, scorer, spoil).strata[STRATA[0]]
    assert spoiled.items[RULE_TESTS]["r.board"] == -1.0
    assert spoiled.non_finite == 1


# ── an output whose targets do not differ is not read ───────────────────


def test_an_output_whose_targets_are_the_same_in_both_polarities_is_not_settled():
    reading = OutputReading(decoder_of("can_play"))
    reading.observe("fires", [1.0])
    reading.observe("does_not_fire", [1.0 + 1e-9])
    assert not reading.settle()


def test_an_output_whose_targets_differ_is_settled():
    reading = OutputReading(decoder_of("can_play"))
    reading.observe("fires", [1.0])
    reading.observe("does_not_fire", [0.0])
    assert reading.settle()


def _rules(run, records, outputs, **overrides):
    scorer, _, _ = _prepare(run, records, _selection(records),
                            rule_outputs={"r.board": outputs}, **overrides)
    return scorer


def test_a_rule_scores_as_its_informative_output_alone(run, records):
    """In the fixture ``can_play`` differs between the polarities and ``gate``
    is the same on both."""
    alone = _rules(run, records, ("can_play",))
    both = _rules(run, records, ("gate", "can_play"))
    for stratum in STRATA:
        assert set(both.strata[stratum].rules["r.board"]) == {"can_play"}
        assert "r.board/gate" in both.strata[stratum].left_out[RULE_TESTS]
    assert (_epoch(run, both).strata[STRATA[0]].items[RULE_TESTS]["r.board"]
            == pytest.approx(_epoch(run, alone).strata[STRATA[0]].items[RULE_TESTS]["r.board"]))


def test_a_rule_whose_every_output_is_constant_is_left_out_as_a_whole(run, records):
    scorer = _rules(run, records, ("gate", "damage_taken"), strict=False)
    cache = scorer.strata[STRATA[0]]
    assert "r.board" not in cache.rules
    assert "r.board" in cache.left_out[RULE_TESTS]


# ── the count is printed ────────────────────────────────────────────────


def _control(card: int, game: int) -> ControlScore:
    return ControlScore({STRATA[0]: StratumScore({}, {}, non_finite=card),
                         STRATA[1]: StratumScore({}, {}, non_finite=game)}, 0.5)


def test_the_epoch_log_prints_each_stratums_non_finite_readings():
    assert format_non_finite(_control(0, 3)) == (
        "non-finite readings: card-disjoint 0 | game-disjoint 3")


def test_the_evaluation_report_names_a_stratums_non_finite_readings():
    components = dict.fromkeys(("probes", "grids", "rule_tests", "deviance"), 0.5)
    score = ControlScore({STRATA[0]: StratumScore(components, {}, non_finite=4),
                          STRATA[1]: StratumScore(components, {})}, 0.5)
    checks = {c.name: c for c in half_score_checks(score, HalfReading("reporting", {}))}
    assert "4 non-finite reading(s) scored at -1" in checks["reporting-half card-disjoint"].detail
    assert "non-finite" not in checks["reporting-half game-disjoint"].detail
