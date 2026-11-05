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
        from effects.domain.effect_head_layout import FAILED_CHECKS

        assert set(READOUTS) == {
            "affected", "dies", "damage_taken", "pt_until_end_of_turn", "may_block",
            "legal_target", "legal_target_protected", "affordable", "fires",
            "target_legal", "may_attack", "can_play", "timing",
            *(f"failed_{check}" for check in FAILED_CHECKS)}

    def test_a_verdict_is_read_from_the_record_and_ignores_the_slot(self):
        import torch

        from effects.domain.effect_head_layout import VERDICT_LAYOUT, VERDICT_WIDTH

        verdict = torch.zeros(2, VERDICT_WIDTH)
        verdict[1, VERDICT_LAYOUT.slices["trigger_fired"][0]] = 2.0
        expected = float(torch.sigmoid(torch.tensor(2.0)))
        assert read("fires", None, verdict, 1, 0) == pytest.approx(expected)
        assert read("fires", None, verdict, 1, 7) == pytest.approx(expected)
        assert read("fires", None, verdict, 0, 0) == pytest.approx(0.5)

    def test_a_failed_check_reads_its_own_member_column(self):
        import torch

        from effects.domain.effect_head_layout import (
            FAILED_CHECKS,
            VERDICT_LAYOUT,
            VERDICT_WIDTH,
        )

        verdict = torch.zeros(1, VERDICT_WIDTH)
        start, end = VERDICT_LAYOUT.slices["failed_checks"]
        assert end - start == len(FAILED_CHECKS)
        verdict[0, start + FAILED_CHECKS.index("casting_speed")] = 3.0
        assert read("failed_casting_speed", None, verdict, 0, 0) == pytest.approx(
            float(torch.sigmoid(torch.tensor(3.0))))
        assert read("failed_whose_turn", None, verdict, 0, 0) == pytest.approx(0.5)

    def test_target_legality_reads_the_field_at_the_slot_ungated(self):
        import torch

        from effects.domain.effect_head_layout import FIELD_SLICES, PER_ENTITY_WIDTH

        outputs = torch.full((1, 3, PER_ENTITY_WIDTH), -9.0)
        outputs[0, 2, FIELD_SLICES["target_legal"][0]] = 0.0
        assert read("target_legal", outputs, None, 0, 2) == pytest.approx(0.5)
        assert read("target_legal", outputs, None, 0, 1) < 0.001


class TestEntitySlots:
    def test_players_are_read_beside_cards(self):
        from types import SimpleNamespace

        from effects.domain.effect_head_input import SlotKind
        from effects.domain.probe_readouts import card_slots, entity_slots

        slots = [SimpleNamespace(kind=SlotKind.GLOBAL, entity_id=None, player_id=None),
                 SimpleNamespace(kind=SlotKind.PLAYER, entity_id=None, player_id="P0"),
                 SimpleNamespace(kind=SlotKind.CARD, entity_id="E5", player_id=None),
                 SimpleNamespace(kind=SlotKind.ABILITY, entity_id="E5", player_id=None)]
        surface = SimpleNamespace(slots=slots)
        assert card_slots(surface) == {"E5": 2}
        assert entity_slots(surface) == {"E5": 2, "P0": 1}
