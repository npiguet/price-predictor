"""Gate 2's routing: per keyword a verdict, and the table it reads (FR-119 – FR-122).

A keyword with too few qualifying records or too little direction agreement is
routed to a probe; the gate blocks nothing.
"""

from __future__ import annotations

import pytest

from effects.application.gate_two import (
    evaluate_gate_two,
    evaluate_keyword,
)
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KEYWORDS_BY_NAME,
    MIN_DIRECTION_AGREEMENT,
    MIN_QUALIFYING_RECORDS,
    KeywordEffect,
    Subject,
)
from effects.domain.effect_head_layout import (
    FIELDS_BY_NAME,
)
from effects.domain.evaluation_report import CheckStatus
from tests.unit.effects.selection_builders import keyword_score


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
        verdict = evaluate_keyword(KEYWORDS_BY_NAME["deathtouch"], keyword_score(500, 400))
        assert verdict.passed
        assert not verdict.routed_to_probe

    def test_too_few_records_routes_to_a_probe(self):
        """Untested is not passed — that is how a canary stops being one."""
        verdict = evaluate_keyword(KEYWORDS_BY_NAME["wither"], keyword_score(40, 40))
        assert verdict.routed_to_probe
        assert "under-sampled" in verdict.reason

    def test_too_little_agreement_routes_to_a_probe(self):
        verdict = evaluate_keyword(KEYWORDS_BY_NAME["trample"], keyword_score(1000, 600))
        assert verdict.routed_to_probe
        assert "direction agreement" in verdict.reason

    def test_exactly_meeting_both_thresholds_passes(self):
        verdict = evaluate_keyword(KEYWORDS_BY_NAME["lifelink"], keyword_score(
            MIN_QUALIFYING_RECORDS,
            int(MIN_QUALIFYING_RECORDS * MIN_DIRECTION_AGREEMENT),
        ))
        assert verdict.passed

    def test_the_gate_blocks_nothing(self):
        verdicts = [
            evaluate_keyword(row, keyword_score(10, 0))
            for row in DAMAGE_STEP_KEYWORDS
        ]
        result = evaluate_gate_two(verdicts)
        assert result.status is CheckStatus.REPORTED
        assert not result.blocks

    def test_it_names_the_keywords_it_routed(self):
        verdicts = [
            evaluate_keyword(KEYWORDS_BY_NAME["infect"], keyword_score(10, 10)),
            evaluate_keyword(KEYWORDS_BY_NAME["lifelink"], keyword_score(500, 500)),
        ]
        detail = evaluate_gate_two(verdicts).detail
        assert "infect" in detail
        assert "lifelink" not in detail

    def test_all_passing_says_no_probe_machinery_is_needed(self):
        """That output is the whole reason the gate runs before stage three."""
        verdicts = [
            evaluate_keyword(row, keyword_score(500, 500))
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
