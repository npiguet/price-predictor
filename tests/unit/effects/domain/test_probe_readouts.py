"""The model's answers to probe questions and how an item of them scores
(research E1, E6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from effects.domain.probe_readouts import READOUTS, item_score, read


class TestItemScore:
    def test_a_yes_no_target_scores_as_an_auc(self):
        assert item_score("binary", [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75

    def test_an_amount_scores_as_an_r_squared(self):
        y = np.array([1.0, 2.0, 3.0])
        assert item_score("amount", y, y) == 1.0

    def test_rows_not_finite_in_both_are_dropped(self):
        assert item_score("binary", [0, 1, np.nan], [0.1, 0.9, 0.5]) == 1.0
        assert item_score("binary", [0, 1, 1], [0.1, 0.9, np.nan]) == 1.0

    def test_undefined_scores_are_nan(self):
        assert math.isnan(item_score("binary", [1], [0.3]))
        assert math.isnan(item_score("binary", [1, 1], [0.3, 0.4]))
        assert math.isnan(item_score("amount", [2.0, 2.0], [1.0, 3.0]))

    def test_weights_reach_the_auc(self):
        y, prediction = [0, 1, 1], [0.5, 0.9, 0.1]
        assert item_score("binary", y, prediction) == 0.5
        assert item_score("binary", y, prediction, [1.0, 3.0, 1.0]) == 0.75


class TestReadouts:
    def test_every_probe_question_has_a_readout(self):
        assert set(READOUTS) == {
            "affected", "dies", "damage_taken", "pt_until_end_of_turn", "may_block",
            "legal_target", "legal_target_protected", "affordable", "fires"}

    def test_a_verdict_is_read_from_the_record_and_ignores_the_slot(self):
        import torch

        verdict = torch.zeros(2, 4)
        verdict[1, -1] = 2.0
        expected = float(torch.sigmoid(torch.tensor(2.0)))
        assert read("fires", None, verdict, 1, 0) == pytest.approx(expected)
        assert read("fires", None, verdict, 1, 7) == pytest.approx(expected)
