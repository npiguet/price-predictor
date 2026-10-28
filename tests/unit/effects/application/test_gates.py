"""The three gates and their thresholds (T042).

Every threshold here is contract carried from FR-118, FR-119 and FR-123, so
these tests are as much about the numbers as about the logic: a gate that
silently loosened would let a model ship that the spec says must not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from effects.application.evaluate_effect_model import (
    NOT_IMPLEMENTED_CHECKS,
    not_implemented,
)
from effects.application.evaluation_context import parse_baseline_checkpoint
from effects.application.gate_one import GateOneMetrics, evaluate_gate_one
from effects.application.gate_two import evaluate_gate_two, evaluate_keyword
from effects.application.geometry_checks import (
    CacheGeometry,
    evaluate_ward_canary,
    gate_three_verdict,
)
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KEYWORDS_BY_NAME,
    MIN_DIRECTION_AGREEMENT,
    MIN_QUALIFYING_RECORDS,
    KeywordEffect,
    Subject,
)
from effects.domain.effect_head_layout import FIELDS_BY_NAME
from effects.domain.evaluation_report import CheckName, CheckStatus, EvaluationReport
from effects.domain.gate_metrics import (
    balanced_accuracy,
    cosine_distance,
    f1_score,
    mean_pairwise_cosine,
    poisson_deviance,
    top_component_share,
)


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


class TestGateOneMetrics:
    def test_f1_is_one_for_a_perfect_gate(self):
        observed = np.array([1, 0, 1, 1])
        assert f1_score(observed, observed) == pytest.approx(1.0)

    def test_f1_is_zero_when_nothing_true_is_caught(self):
        assert f1_score(np.array([0, 0]), np.array([1, 1])) == 0.0

    def test_deviance_is_zero_for_an_exact_prediction(self):
        observed = np.array([1.0, 3.0, 5.0])
        assert poisson_deviance(observed, observed) == pytest.approx(0.0, abs=1e-9)

    def test_deviance_grows_with_error(self):
        observed = np.array([3.0, 3.0])
        near = poisson_deviance(np.array([3.1, 3.1]), observed)
        far = poisson_deviance(np.array([9.0, 9.0]), observed)
        assert far > near

    def test_deviance_handles_zero_observations(self):
        value = poisson_deviance(np.array([0.5, 0.5]), np.array([0.0, 0.0]))
        assert math.isfinite(value)

    def test_an_empty_slice_is_nan_rather_than_zero(self):
        assert math.isnan(poisson_deviance(np.array([]), np.array([])))

    def test_zone_accuracy_weighs_every_observed_outcome_the_same(self):
        """Answering the dominant class every time scores one over the number
        of classes observed, not that class's share (FR-109)."""
        observed = np.array([0] * 9 + [1])
        always_stayed = np.zeros(10, dtype=int)
        assert balanced_accuracy(always_stayed, observed) == pytest.approx(0.5)

    def test_zone_accuracy_of_nothing_is_nan(self):
        assert math.isnan(balanced_accuracy(np.array([]), np.array([])))


class TestGateTwo:
    """Per keyword, routing only — it blocks nothing."""

    def test_the_table_covers_the_eight_damage_step_keywords(self):
        assert {row.keyword for row in DAMAGE_STEP_KEYWORDS} == {
            "first_strike", "double_strike", "deathtouch", "lifelink",
            "trample", "indestructible", "wither", "infect",
        }

    def test_every_affected_field_is_a_real_head_output(self):
        for row in DAMAGE_STEP_KEYWORDS:
            for effect in row.effects:
                assert effect.field in FIELDS_BY_NAME, (
                    f"{row.keyword} names {effect.field}, which the per-entity "
                    "head does not predict"
                )

    def test_enough_records_and_enough_agreement_passes(self):
        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["deathtouch"],
            qualifying_records=500, agreeing=400,
        )
        assert verdict.passed
        assert not verdict.routed_to_probe

    def test_too_few_records_routes_to_a_probe(self):
        """Untested is not passed — that is how a canary stops being one."""
        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["wither"],
            qualifying_records=40, agreeing=40,
        )
        assert verdict.routed_to_probe
        assert "under-sampled" in verdict.reason

    def test_too_little_agreement_routes_to_a_probe(self):
        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["trample"],
            qualifying_records=1000, agreeing=600,
        )
        assert verdict.routed_to_probe
        assert "direction agreement" in verdict.reason

    def test_exactly_meeting_both_thresholds_passes(self):
        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["lifelink"],
            qualifying_records=MIN_QUALIFYING_RECORDS,
            agreeing=int(MIN_QUALIFYING_RECORDS * MIN_DIRECTION_AGREEMENT),
        )
        assert verdict.passed

    def test_the_gate_blocks_nothing(self):
        verdicts = [
            evaluate_keyword(row, qualifying_records=10, agreeing=0)
            for row in DAMAGE_STEP_KEYWORDS
        ]
        result = evaluate_gate_two(verdicts)
        assert result.status is CheckStatus.REPORTED
        assert not result.blocks

    def test_it_names_the_keywords_it_routed(self):
        verdicts = [
            evaluate_keyword(
                KEYWORDS_BY_NAME["infect"], qualifying_records=10,
                agreeing=10,
            ),
            evaluate_keyword(
                KEYWORDS_BY_NAME["lifelink"], qualifying_records=500,
                agreeing=500,
            ),
        ]
        detail = evaluate_gate_two(verdicts).detail
        assert "infect" in detail
        assert "lifelink" not in detail

    def test_all_passing_says_no_probe_machinery_is_needed(self):
        """That output is the whole reason the gate runs before stage three."""
        verdicts = [
            evaluate_keyword(row, qualifying_records=500, agreeing=500)
            for row in DAMAGE_STEP_KEYWORDS
        ]
        assert "no stage-three probe machinery" in evaluate_gate_two(verdicts).detail

    def test_an_effect_with_a_nonsense_direction_is_rejected(self):
        with pytest.raises(ValueError, match="direction must be"):
            KeywordEffect("damage_taken", Subject.CARRIER, direction=0)

    @pytest.mark.parametrize("keyword", ["first_strike", "double_strike"])
    def test_striking_first_predicts_more_damage_to_its_opponent(self, keyword):
        """Within the first-strike step the carrier's hit lands with the
        keyword and not without it."""
        row = KEYWORDS_BY_NAME[keyword]
        damage = next(e for e in row.effects if e.field == "damage_taken")
        assert damage.subject is Subject.OPPONENT
        assert damage.direction == 1

    @pytest.mark.parametrize("keyword", ["first_strike", "double_strike"])
    def test_striking_first_unblocked_predicts_the_defender_loses_more_life(
        self, keyword,
    ):
        row = KEYWORDS_BY_NAME[keyword]
        life = next(e for e in row.effects if e.field == "life_delta")
        assert life.subject is Subject.DEFENDING_PLAYER
        assert life.direction == -1

    def test_lifelink_predicts_the_controller_gains_life(self):
        row = KEYWORDS_BY_NAME["lifelink"]
        life = next(e for e in row.effects if e.field == "life_delta")
        assert life.subject is Subject.CONTROLLER
        assert life.direction == 1

    def test_trample_predicts_the_defender_loses_more_life(self):
        row = KEYWORDS_BY_NAME["trample"]
        life = next(e for e in row.effects if e.field == "life_delta")
        assert life.subject is Subject.DEFENDING_PLAYER
        assert life.direction == -1


class TestGateThree:
    def test_a_spread_space_passes(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(400, 64))
        result = gate_three_verdict(CacheGeometry.of(vectors))
        assert result.status is CheckStatus.PASS

    def test_a_collapsed_space_fails(self):
        """Every ability on top of every other passes every average-case metric."""
        rng = np.random.default_rng(0)
        vectors = np.ones((400, 64)) + 0.001 * rng.normal(size=(400, 64))
        result = gate_three_verdict(CacheGeometry.of(vectors))
        assert result.status is CheckStatus.FAIL
        assert "collapsed" in result.detail

    def test_a_one_dimensional_space_fails_on_the_top_component(self):
        direction = np.zeros(64)
        direction[0] = 1.0
        vectors = np.outer(np.linspace(-1, 1, 400), direction)
        result = gate_three_verdict(CacheGeometry.of(vectors))
        assert result.status is CheckStatus.FAIL
        assert "principal component" in result.detail

    def test_it_blocks_shipping(self):
        vectors = np.ones((100, 8))
        assert gate_three_verdict(CacheGeometry.of(vectors)).blocks

    def test_an_all_zero_space_fails(self):
        """The effect head once trained on an all-zero e: no variance at all is
        the most collapsed a space can be, not an unmeasurable one."""
        result = gate_three_verdict(CacheGeometry.of(np.zeros((100, 64))))
        assert result.status is CheckStatus.FAIL
        assert "principal component" in result.detail

    def test_too_few_vectors_to_measure_fails(self):
        result = gate_three_verdict(CacheGeometry.of(np.ones((1, 8))))
        assert result.status is CheckStatus.FAIL
        assert "not measured" in result.detail

    def test_mean_cosine_is_one_for_identical_vectors(self):
        vectors = np.tile(np.arange(1.0, 9.0), (50, 1))
        assert mean_pairwise_cosine(vectors, pairs=100) == pytest.approx(1.0)

    def test_mean_cosine_is_near_zero_for_random_high_dimensional_vectors(self):
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(500, 128))
        assert abs(mean_pairwise_cosine(vectors, pairs=2000)) < 0.1

    def test_a_pair_never_compares_a_vector_with_itself(self):
        """Self-pairs would drag the mean toward 1 and mask a real collapse."""
        vectors = np.eye(3)
        assert mean_pairwise_cosine(vectors, pairs=500) == pytest.approx(0.0)

    def test_the_sample_is_reproducible_under_a_seed(self):
        rng = np.random.default_rng(2)
        vectors = rng.normal(size=(200, 32))
        assert mean_pairwise_cosine(vectors, pairs=500, seed=7) == (
            mean_pairwise_cosine(vectors, pairs=500, seed=7)
        )

    def test_a_single_vector_is_nan_rather_than_a_score(self):
        assert math.isnan(mean_pairwise_cosine(np.ones((1, 8))))

    def test_the_top_component_of_a_line_explains_everything(self):
        vectors = np.outer(np.linspace(-1, 1, 100), np.array([1.0, 0.0, 0.0]))
        assert top_component_share(vectors) == pytest.approx(1.0)

    def test_the_top_component_of_an_isotropic_cloud_is_small(self):
        rng = np.random.default_rng(3)
        assert top_component_share(rng.normal(size=(500, 32))) < 0.15


class TestWardCanary:
    def _vectors(self, twin_distance: float):
        ward = np.array([1.0, 0.0, 0.0])
        # Twins sit at the requested angle; bare keywords sit further away.
        angle = twin_distance
        twins = [np.array([1.0 - angle, math.sqrt(max(0.0, 2 * angle - angle ** 2)), 0.0])]
        bare = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        return ward, twins, bare

    def test_ward_nearer_its_twins_than_the_median_keyword_reads_as_such(self):
        ward, twins, bare = self._vectors(0.05)
        result = evaluate_ward_canary(ward, twins, bare)
        assert "every functional twin is nearer" in result.detail

    def test_a_twin_farther_than_the_median_is_reported(self):
        ward = np.array([1.0, 0.0, 0.0])
        twins = [np.array([-1.0, 0.0, 0.0])]  # opposite: distance 2
        bare = [np.array([0.0, 1.0, 0.0])]    # orthogonal: distance 1
        result = evaluate_ward_canary(ward, twins, bare)
        assert "sit farther from ward" in result.detail

    def test_the_canary_reports_and_does_not_block(self):
        ward, twins, bare = self._vectors(0.05)
        assert evaluate_ward_canary(ward, twins, bare).status is (
            CheckStatus.REPORTED
        )

    def test_an_empty_cache_skips_rather_than_failing(self):
        result = evaluate_ward_canary(np.ones(3), [], [])
        assert result.status is CheckStatus.SKIPPED

    def test_cosine_distance_is_zero_for_a_vector_and_itself(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == pytest.approx(0.0)

    def test_cosine_distance_of_a_zero_vector_is_nan(self):
        assert math.isnan(cosine_distance(np.zeros(3), np.ones(3)))


class TestUnimplementedChecks:
    """FR-116 names three checks that were never built.

    The corpus holds forks and mana records, so a line blaming missing records
    would send an operator looking for data that is already there.
    """

    def test_each_reports_not_implemented_and_blocks_nothing(self):
        for name in (
            CheckName.MATCHED_REAL_VS_FORK, CheckName.PROBE_DIFF,
            CheckName.ROLE_POLARITY,
        ):
            result = not_implemented(name)
            assert result.status is CheckStatus.NOT_IMPLEMENTED
            assert not result.blocks
            assert result.detail.startswith("not implemented")

    def test_none_claims_the_corpus_lacks_its_records(self):
        for name in NOT_IMPLEMENTED_CHECKS:
            assert "does not have" not in not_implemented(name).detail

    def test_the_report_marks_them_apart_from_a_skip(self):
        report = EvaluationReport()
        report.add(not_implemented(CheckName.PROBE_DIFF))
        assert report.render().startswith("[todo] probe-diff: not implemented")


class TestReport:
    def test_a_report_ships_when_no_gate_blocks(self):
        report = EvaluationReport()
        report.add(evaluate_gate_one(
            GateOneMetrics(0.70, 0.60, 0.90), GateOneMetrics(0.60, 0.50, 1.00),
        ))
        report.add(not_implemented(CheckName.ROLE_POLARITY))
        assert report.ships

    def test_a_blocking_gate_stops_it(self):
        report = EvaluationReport()
        report.add(gate_three_verdict(CacheGeometry.of(np.ones((50, 8)))))
        assert not report.ships
        assert [c.name for c in report.blocking_failures] == ["gate-3"]

    def test_gate_two_never_stops_it(self):
        report = EvaluationReport()
        report.keyword_verdicts = [
            evaluate_keyword(row, qualifying_records=1, agreeing=0)
            for row in DAMAGE_STEP_KEYWORDS
        ]
        report.add(evaluate_gate_two(report.keyword_verdicts))
        assert report.ships

    def test_the_render_names_every_check_and_the_verdict(self):
        report = EvaluationReport()
        report.add(gate_three_verdict(CacheGeometry.of(np.ones((50, 8)))))
        rendered = report.render()
        assert "gate-3" in rendered
        assert "GATES FAILED: gate-3" in rendered

    def test_a_passing_gate_still_prints_every_figure_it_measured(self):
        """The design record needs all three gate-1 margins, pass or fail."""
        report = EvaluationReport()
        report.add(evaluate_gate_one(
            GateOneMetrics(0.70, 0.60, 0.90), GateOneMetrics(0.60, 0.50, 1.00),
        ))
        rendered = report.render()
        assert "affected_gate_f1_gain=+0.1000" in rendered
        assert "zone_outcome_accuracy_gain=+0.1000" in rendered
        assert "poisson_deviance_reduction=+0.1000" in rendered

    def test_a_failing_gate_prints_the_figure_that_held_too(self):
        report = EvaluationReport()
        report.add(gate_three_verdict(CacheGeometry.of(np.ones((50, 8)))))
        rendered = report.render()
        assert "mean_pairwise_cosine=" in rendered
        assert "top_component_share=" in rendered


class TestVariantCheckpointFlag:
    def test_it_parses_name_equals_path(self):
        name, path = parse_baseline_checkpoint("identity=models/x/latest.pt")
        assert name == "identity"
        assert path.name == "latest.pt"

    def test_a_bare_path_is_rejected(self):
        with pytest.raises(ValueError, match="NAME=PATH"):
            parse_baseline_checkpoint("models/x/latest.pt")

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="NAME=PATH"):
            parse_baseline_checkpoint("=models/x/latest.pt")

    def test_a_misspelt_baseline_is_rejected(self):
        """Otherwise gate 1 skips for want of an identity checkpoint."""
        with pytest.raises(ValueError, match="unknown baseline 'identiy'"):
            parse_baseline_checkpoint("identiy=models/x/latest.pt")

    def test_the_shipping_model_is_not_a_baseline(self):
        with pytest.raises(ValueError, match="unknown baseline 'full'"):
            parse_baseline_checkpoint("full=models/x/latest.pt")
