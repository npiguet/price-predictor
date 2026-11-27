"""Gate 2's move report: how much of the right move each keyword's prediction makes.

Each combat scores the share ``m / t`` of its right move; the shares are
averaged per effect with a standard error clustered by game, a combat whose
right move is 0 is counted apart, and a keyword's share is the mean of its
effects'. The report is a second line per keyword and changes no verdict.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from effects.application import gate_two
from effects.application.gate_two import KeywordScore, evaluate_keyword
from effects.application.gate_two_moves import EffectMoveTally, MoveTally, PredictedMove
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KEYWORDS_BY_NAME,
    combat_participant,
)
from effects.domain.evaluation_report import (
    EffectMoves,
    EvaluationReport,
    KeywordMoves,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.records import CombatPayload, RecordKind
from effects.domain.score_scales import clustered_mean_standard_error
from tests.unit.effects.domain.test_damage_step_predicates import _unblocked_attacker
from tests.unit.effects.record_builders import make_record

_LIFELINK = KEYWORDS_BY_NAME["lifelink"]
_GAIN = _LIFELINK.effects[0]


def _lifelink_combat(dealt: int, *, game: str = "g1"):
    """A 3-power lifelinker unblocked at P1, which the record shows dealt ``dealt``."""
    state, carrier = _unblocked_attacker(3, 3, keywords=("lifelink",))
    events = (Event(EventType.DAMAGE_DEALT, ("P1",),
                    {"amount": dealt, "combat": True, "source": "A"}),) if dealt else ()
    record = make_record(RecordKind.COMBAT, state=state,
                         payload=CombatPayload(events=events))
    record = dataclasses.replace(record, game_id=game)
    participant = combat_participant(
        record, carrier, lambda entity: set(entity.granted_temporary.keywords))
    return record, participant


class TestTheShare:
    def test_a_combat_scores_its_predicted_move_over_the_right_one(self):
        record, participant = _lifelink_combat(3)
        tally = MoveTally()
        tally.fold(record, participant, [PredictedMove(_GAIN, "P0", 1.5)])
        assert tally.per_effect[_GAIN.label].shares == [0.5]

    def test_an_effect_read_at_several_subjects_averages_their_shares(self):
        tally = EffectMoveTally()
        tally.fold("g1", [(2.0, 2.0), (0.0, 1.0)])
        assert (tally.shares, tally.games) == ([0.5], ["g1"])

    def test_a_combat_whose_right_move_is_zero_is_counted_apart_with_its_move(self):
        record, participant = _lifelink_combat(0)
        tally = MoveTally()
        tally.fold(record, participant, [PredictedMove(_GAIN, "P0", -0.4)])
        summary = tally.summary(_LIFELINK).effects[0]
        assert (summary.combats, summary.zero_size) == (0, 1)
        assert summary.zero_size_move == pytest.approx(0.4)
        assert math.isnan(summary.mean_share)

    def test_an_effects_summary_holds_the_mean_error_and_quartiles(self):
        tally = EffectMoveTally()
        for game, share in [("g1", 1.0), ("g1", 0.5), ("g2", 0.0), ("g3", 0.5)]:
            tally.fold(game, [(share, 1.0)])
        summary = tally.summary("controller.life_delta")
        assert summary.combats == 4
        assert summary.mean_share == pytest.approx(0.5)
        assert summary.quartiles == pytest.approx((0.375, 0.5, 0.625))
        assert summary.standard_error == pytest.approx(math.sqrt(0.5 * 1.5) / 4)


class TestTheClusteredStandardError:
    def test_it_sums_each_games_residuals_before_squaring(self):
        """Mean 0.5; residuals 0.5, 0 (g1), -0.5 (g2), 0 (g3); game sums
        0.5, -0.5, 0: √(0.25 + 0.25) / 4, times √(3/2) for three games."""
        error = clustered_mean_standard_error([1.0, 0.5, 0.0, 0.5], ["g1", "g1", "g2", "g3"])
        assert error == pytest.approx(math.sqrt(0.5 * 1.5) / 4)

    def test_one_game_per_value_is_the_unclustered_error_of_a_mean(self):
        values = [1.0, 2.0, 4.0]
        mean = sum(values) / 3
        expected = math.sqrt(1.5) * math.sqrt(sum((v - mean) ** 2 for v in values)) / 3
        assert clustered_mean_standard_error(values, ["a", "b", "c"]) == pytest.approx(expected)

    def test_no_value_has_no_error(self):
        assert math.isnan(clustered_mean_standard_error([], []))

    def test_one_game_has_no_error_whatever_the_spread(self):
        assert math.isnan(clustered_mean_standard_error(
            [0.1, 0.9, 0.2, 1.5], ["g1"] * 4))

    def test_two_games_are_corrected_by_root_two(self):
        """Mean 0.675; g1 residuals -0.575, 0.225 sum -0.35, g2's 0.35:
        √(0.1225 + 0.1225) · √2 / 4."""
        error = clustered_mean_standard_error([0.1, 0.9, 0.2, 1.5], ["g1", "g1", "g2", "g2"])
        assert error == pytest.approx(math.sqrt(0.245) * math.sqrt(2) / 4)

    def test_a_single_game_effect_prints_n_a(self):
        tally = EffectMoveTally()
        for share in (0.1, 0.9):
            tally.fold("g1", [(share, 1.0)])
        assert "± n/a (q" in tally.summary("a").render()


def _effect(label: str, share: float, combats: int = 10) -> EffectMoves:
    return EffectMoves(label, combats, share, 0.05, (share - 0.1, share, share + 0.1))


class TestTheKeywordShare:
    def test_each_effect_with_a_share_counts_equally(self):
        moves = KeywordMoves((_effect("a", 0.2, 900), _effect("b", 0.6, 10),
                              EffectMoves("c", 0, math.nan, math.nan, (math.nan,) * 3, 4, 0.1)))
        assert moves.share == pytest.approx(0.4)

    def test_a_keyword_with_no_sized_combat_has_no_share(self):
        moves = KeywordMoves((EffectMoves("c", 0, math.nan, math.nan, (math.nan,) * 3, 4, 0.1),))
        assert moves.share is None
        assert moves.render() == "moves: share none; c no share, t=0: 4 (mean |m| 0.10)"


class TestTheReportLine:
    def test_the_moves_line_follows_its_keywords_verdict_line(self):
        moves = KeywordMoves((
            EffectMoves("opponent.damage_taken", 812, 0.35, 0.03, (0.18, 0.39, 0.66), 14, 0.6),
            _effect("opponent.zone_outcome/died", 0.52, 790),
        ))
        verdict = evaluate_keyword(KEYWORDS_BY_NAME["first_strike"], KeywordScore())
        report = EvaluationReport(keyword_verdicts=[dataclasses.replace(verdict, moves=moves)])
        rendered = report.render_check(gate_two.evaluate_gate_two(report.keyword_verdicts))
        keyword_line, moves_line = rendered.splitlines()[-2:]
        assert keyword_line.startswith("    first_strike")
        assert moves_line == (
            "      moves: share 0.43; opponent.damage_taken 0.35 ± 0.03 (q 0.18/0.39/0.66)"
            " over 812, t=0: 14 (mean |m| 0.60); opponent.zone_outcome/died 0.52 ± 0.05"
            " (q 0.42/0.52/0.62) over 790")
        # scripts/effect_gen2_report/build_data.gate2 reads only lines holding n=.
        assert "n=" not in moves_line

    def test_a_verdict_without_moves_renders_its_line_alone(self):
        verdict = evaluate_keyword(KEYWORDS_BY_NAME["trample"], KeywordScore())
        report = EvaluationReport(
            keyword_verdicts=[dataclasses.replace(verdict, moves=None)])
        rendered = report.render_check(gate_two.evaluate_gate_two(report.keyword_verdicts))
        assert rendered.splitlines()[-1].startswith("    trample")


def test_the_moves_change_no_verdict():
    """Decision D11: the verdict reads the direction alone."""
    row = _LIFELINK
    bare = KeywordScore(qualifying=300, agreeing=250, observations=300)
    moved = dataclasses.replace(bare, moves=MoveTally())
    record, participant = _lifelink_combat(3)
    moved.moves.fold(record, participant, [PredictedMove(_GAIN, "P0", -9.0)])
    with_moves, without = evaluate_keyword(row, moved), evaluate_keyword(row, bare)
    assert with_moves.moves.share == pytest.approx(-3.0)
    assert dataclasses.replace(with_moves, moves=None) == dataclasses.replace(without, moves=None)


# ── on the fixture's real combats, through the one forward ───────────────


@pytest.fixture(scope="module")
def fixture_verdicts(corpus, records, tmp_path_factory):
    from effects.application.model_loading import Runnable
    from effects.infrastructure.record_codec import record_from_dict
    from tests.unit.effects.builders import gen3_fixture_records
    from tests.unit.effects.training_builders import fixture_run

    run = fixture_run(corpus, records, tmp_path_factory.mktemp("gate-two-moves"))
    batcher = run.preparer.scoring_batcher()
    batcher.materializer = run.scoring
    runnable = Runnable(run.encoder, run.model, batcher, tuple(run.fields[1]))
    scores = gate_two.score_keywords(
        map(record_from_dict, gen3_fixture_records()), DAMAGE_STEP_KEYWORDS, runnable)
    return {row.keyword: evaluate_keyword(row, scores[row.keyword])
            for row in DAMAGE_STEP_KEYWORDS}


def test_the_fixtures_first_strike_and_trample_combats_each_score_a_share(fixture_verdicts):
    first_strike = {e.label: e for e in fixture_verdicts["first_strike"].moves.effects}
    assert set(first_strike) == {"opponent.damage_taken", "opponent.zone_outcome/died"}
    assert all(e.combats == 1 and e.zero_size == 0 for e in first_strike.values())
    (trample,) = fixture_verdicts["trample"].moves.effects
    assert (trample.label, trample.combats) == ("defending_player.life_delta", 1)
    assert fixture_verdicts["lifelink"].moves == KeywordMoves()
    for verdict in fixture_verdicts.values():
        assert "n=" not in verdict.moves.render()
