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
    GATE1_MIN_DEVIANCE_REDUCTION,
    GATE1_MIN_GATE_F1_GAIN,
    GATE1_MIN_ZONE_ACCURACY_GAIN,
    GATE3_COSINE_PAIRS,
    GATE3_MAX_MEAN_COSINE,
    GATE3_MAX_TOP_COMPONENT,
    STAGE_GATED_CHECKS,
    CheckStatus,
    EvaluationReport,
    GateOneMetrics,
    Stratum,
    cosine_distance,
    evaluate_gate_one,
    evaluate_gate_three,
    evaluate_gate_two,
    evaluate_keyword,
    evaluate_ward_canary,
    f1_score,
    mean_pairwise_cosine,
    parse_variant_checkpoint,
    poisson_deviance,
    skip_unavailable,
    top_component_share,
)
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KEYWORDS_BY_NAME,
    MIN_DIRECTION_AGREEMENT,
    MIN_QUALIFYING_RECORDS,
    KeywordEffect,
    Subject,
)
from effects.domain.effect_model import FIELDS_BY_NAME


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

    def test_the_thresholds_are_the_specs(self):
        assert GATE1_MIN_GATE_F1_GAIN == 0.05
        assert GATE1_MIN_ZONE_ACCURACY_GAIN == 0.05
        assert GATE1_MIN_DEVIANCE_REDUCTION == 0.05

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


class TestGateTwo:
    """Per keyword, routing only — it blocks nothing."""

    def test_the_table_covers_the_eight_damage_step_keywords(self):
        assert {row.keyword for row in DAMAGE_STEP_KEYWORDS} == {
            "first_strike", "double_strike", "deathtouch", "lifelink",
            "trample", "indestructible", "wither", "infect",
        }

    def test_every_row_names_a_predicate_fields_and_a_direction(self):
        for row in DAMAGE_STEP_KEYWORDS:
            assert row.qualifies_when
            assert row.effects
            for effect in row.effects:
                assert effect.direction in (-1, 1)

    def test_every_affected_field_is_a_real_head_output(self):
        for row in DAMAGE_STEP_KEYWORDS:
            for effect in row.effects:
                assert effect.field in FIELDS_BY_NAME, (
                    f"{row.keyword} names {effect.field}, which the per-entity "
                    "head does not predict"
                )

    def test_the_thresholds_are_the_specs(self):
        assert MIN_QUALIFYING_RECORDS == 200
        assert MIN_DIRECTION_AGREEMENT == 0.70

    def test_enough_records_and_enough_agreement_passes(self):
        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["deathtouch"],
            qualifying_records=500, agreeing_records=400,
        )
        assert verdict.passed
        assert not verdict.routed_to_probe

    def test_too_few_records_routes_to_a_probe(self):
        """Untested is not passed — that is how a canary stops being one."""
        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["wither"],
            qualifying_records=40, agreeing_records=40,
        )
        assert verdict.routed_to_probe
        assert "under-sampled" in verdict.reason

    def test_too_little_agreement_routes_to_a_probe(self):
        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["trample"],
            qualifying_records=1000, agreeing_records=600,
        )
        assert verdict.routed_to_probe
        assert "direction agreement" in verdict.reason

    def test_exactly_meeting_both_thresholds_passes(self):
        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["lifelink"],
            qualifying_records=MIN_QUALIFYING_RECORDS,
            agreeing_records=int(MIN_QUALIFYING_RECORDS * MIN_DIRECTION_AGREEMENT),
        )
        assert verdict.passed

    def test_the_gate_blocks_nothing(self):
        verdicts = [
            evaluate_keyword(row, qualifying_records=10, agreeing_records=0)
            for row in DAMAGE_STEP_KEYWORDS
        ]
        result = evaluate_gate_two(verdicts)
        assert result.status is CheckStatus.REPORTED
        assert not result.blocks

    def test_it_names_the_keywords_it_routed(self):
        verdicts = [
            evaluate_keyword(
                KEYWORDS_BY_NAME["infect"], qualifying_records=10,
                agreeing_records=10,
            ),
            evaluate_keyword(
                KEYWORDS_BY_NAME["lifelink"], qualifying_records=500,
                agreeing_records=500,
            ),
        ]
        detail = evaluate_gate_two(verdicts).detail
        assert "infect" in detail
        assert "lifelink" not in detail

    def test_all_passing_says_no_probe_machinery_is_needed(self):
        """That output is the whole reason the gate runs before stage three."""
        verdicts = [
            evaluate_keyword(row, qualifying_records=500, agreeing_records=500)
            for row in DAMAGE_STEP_KEYWORDS
        ]
        assert "no stage-three probe machinery" in evaluate_gate_two(verdicts).detail

    def test_an_effect_with_a_nonsense_direction_is_rejected(self):
        with pytest.raises(ValueError, match="direction must be"):
            KeywordEffect("damage_taken", Subject.CARRIER, direction=0)

    def test_first_strike_predicts_less_damage_back_to_the_carrier(self):
        row = KEYWORDS_BY_NAME["first_strike"]
        damage = next(e for e in row.effects if e.field == "damage_taken")
        assert damage.subject is Subject.CARRIER
        assert damage.direction == -1

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
    def test_the_thresholds_are_the_specs(self):
        assert GATE3_MAX_MEAN_COSINE == 0.5
        assert GATE3_COSINE_PAIRS == 10_000
        assert GATE3_MAX_TOP_COMPONENT == 0.30

    def test_a_spread_space_passes(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(400, 64))
        result = evaluate_gate_three(vectors)
        assert result.status is CheckStatus.PASS

    def test_a_collapsed_space_fails(self):
        """Every ability on top of every other passes every average-case metric."""
        rng = np.random.default_rng(0)
        vectors = np.ones((400, 64)) + 0.001 * rng.normal(size=(400, 64))
        result = evaluate_gate_three(vectors)
        assert result.status is CheckStatus.FAIL
        assert "collapsed" in result.detail

    def test_a_one_dimensional_space_fails_on_the_top_component(self):
        direction = np.zeros(64)
        direction[0] = 1.0
        vectors = np.outer(np.linspace(-1, 1, 400), direction)
        result = evaluate_gate_three(vectors)
        assert result.status is CheckStatus.FAIL
        assert "principal component" in result.detail

    def test_it_blocks_shipping(self):
        vectors = np.ones((100, 8))
        assert evaluate_gate_three(vectors).blocks

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


class TestStageGatedChecks:
    def test_the_three_later_stage_checks_are_named(self):
        assert set(STAGE_GATED_CHECKS) == {
            "matched-real-vs-fork", "probe-diff", "role-polarity",
        }

    def test_a_check_with_no_records_skips_rather_than_failing(self):
        """A check that measured nothing is not a failure."""
        result = skip_unavailable("role-polarity")
        assert result.status is CheckStatus.SKIPPED
        assert not result.blocks
        assert "stage two" in result.detail

    def test_the_fork_checks_name_stage_three(self):
        for name in ("matched-real-vs-fork", "probe-diff"):
            assert "stage three" in skip_unavailable(name).detail


class TestReport:
    def test_a_report_ships_when_no_gate_blocks(self):
        report = EvaluationReport()
        report.add(evaluate_gate_one(
            GateOneMetrics(0.70, 0.60, 0.90), GateOneMetrics(0.60, 0.50, 1.00),
        ))
        report.add(skip_unavailable("role-polarity"))
        assert report.ships

    def test_a_blocking_gate_stops_it(self):
        report = EvaluationReport()
        report.add(evaluate_gate_three(np.ones((50, 8))))
        assert not report.ships
        assert [c.name for c in report.blocking_failures] == ["gate-3"]

    def test_gate_two_never_stops_it(self):
        report = EvaluationReport()
        report.keyword_verdicts = [
            evaluate_keyword(row, qualifying_records=1, agreeing_records=0)
            for row in DAMAGE_STEP_KEYWORDS
        ]
        report.add(evaluate_gate_two(report.keyword_verdicts))
        assert report.ships

    def test_the_render_names_every_check_and_the_verdict(self):
        report = EvaluationReport()
        report.add(evaluate_gate_three(np.ones((50, 8))))
        rendered = report.render()
        assert "gate-3" in rendered
        assert "BLOCKED" in rendered

    def test_the_four_strata_are_named(self):
        assert {s.value for s in Stratum} == {
            "unique-text", "shared-text", "novel-combination",
            "numeric-extrapolation",
        }


class TestVariantCheckpointFlag:
    def test_it_parses_name_equals_path(self):
        name, path = parse_variant_checkpoint("identity=models/x/latest.pt")
        assert name == "identity"
        assert path.name == "latest.pt"

    def test_a_bare_path_is_rejected(self):
        with pytest.raises(ValueError, match="NAME=PATH"):
            parse_variant_checkpoint("models/x/latest.pt")

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="NAME=PATH"):
            parse_variant_checkpoint("=models/x/latest.pt")
