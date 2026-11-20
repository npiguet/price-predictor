"""Unit tests for ``train-encoder``'s training math: weighted MSE, the
per-head correlation accumulator and the MLM mask draw.
"""

from __future__ import annotations

import pytest
import torch

from sealed.application.encoder_evaluation import _HeadCorrAccumulator
from sealed.application.encoder_heads import N_HEADS
from sealed.application.encoder_losses import _per_batch_weighted_mse
from sealed.application.encoder_mlm import MlmMaskSpec, _draw_mlm_mask
from sealed.domain.encoder_model import COLOR_ORDER

# ── Per-batch weighted MSE ───────────────────────────────────────────


def _predictions(values: list[float]) -> dict[str, torch.Tensor]:
    """Build a model-output dict whose every cell equals ``values[h]``.

    Lengths of ``values`` must equal the number of heads (9 = 4 signed +
    5 color_lift). The resulting dict has shape (B=1, ...) ready for
    ``_per_batch_weighted_mse``.
    """
    assert len(values) == 9
    return {
        "score_play": torch.tensor([values[0]]),
        "score_draw": torch.tensor([values[1]]),
        "played_rate": torch.tensor([values[2]]),
        "cast_lift": torch.tensor([values[3]]),
        "color_lift": torch.tensor([[values[4], values[5], values[6], values[7], values[8]]]),
    }


class TestPerBatchWeightedMSE:
    def test_single_card_single_active_head(self):
        # Only score_play active (head 0). Pred = 0.5, label = 0.0,
        # weight = 1.0, head_mask = 1.0.
        preds = _predictions([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        labels = torch.zeros(1, 9)
        weights = torch.zeros(1, 9)
        weights[0, 0] = 1.0
        head_mask = torch.zeros(1, 9)
        head_mask[0, 0] = 1.0
        loss = _per_batch_weighted_mse(preds, labels, weights, head_mask)
        # weighted average MSE for the one card / one head: 0.25
        assert float(loss.item()) == pytest.approx(0.25)

    def test_zero_head_mask_short_circuits_to_zero(self):
        # All heads have zero head_mask. Loss must be exactly 0.
        preds = _predictions([0.5] * 9)
        labels = torch.zeros(1, 9)
        weights = torch.zeros(1, 9)
        head_mask = torch.zeros(1, 9)
        loss = _per_batch_weighted_mse(preds, labels, weights, head_mask)
        assert float(loss.item()) == 0.0

    def test_color_lift_block_uses_one_fifth_prefactor(self):
        # Activate only color_lift_W (head 4) with MSE = 1.
        preds = _predictions([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        labels = torch.zeros(1, 9)
        weights = torch.zeros(1, 9)
        weights[0, 4] = 1.0
        head_mask = torch.zeros(1, 9)
        head_mask[0, 4] = 1.0
        loss = _per_batch_weighted_mse(preds, labels, weights, head_mask)
        # weighted-avg = 1.0; with 1/5 prefactor, contribution = 0.2.
        assert float(loss.item()) == pytest.approx(0.2)


# ── Per-head correlation accumulator (val diagnostics) ───────────────


def _all_active_mask(n_rows: int) -> torch.Tensor:
    return torch.ones(n_rows, N_HEADS, dtype=torch.float32)


class TestHeadCorrAccumulator:
    def test_perfect_correlation_is_one(self):
        acc = _HeadCorrAccumulator()
        targets = torch.arange(10, dtype=torch.float32).unsqueeze(1).repeat(1, N_HEADS)
        preds = 3.0 * targets + 1.0  # perfectly (positively) linear
        acc.add(preds, targets, _all_active_mask(10))
        corrs = acc.correlations()
        assert corrs["score_play"] == pytest.approx(1.0, abs=1e-6)
        assert corrs["color_lift_G"] == pytest.approx(1.0, abs=1e-6)

    def test_anti_correlation_is_minus_one(self):
        acc = _HeadCorrAccumulator()
        targets = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, N_HEADS)
        preds = -2.0 * targets
        acc.add(preds, targets, _all_active_mask(8))
        assert acc.correlations()["score_play"] == pytest.approx(-1.0, abs=1e-6)

    def test_accumulates_across_calls(self):
        acc = _HeadCorrAccumulator()
        t1 = torch.tensor([[0.0], [1.0], [2.0]]).repeat(1, N_HEADS)
        t2 = torch.tensor([[3.0], [4.0], [5.0]]).repeat(1, N_HEADS)
        acc.add(t1 * 2.0, t1, _all_active_mask(3))
        acc.add(t2 * 2.0, t2, _all_active_mask(3))
        # Combined: pred = 2*target over 0..5 → perfectly correlated.
        assert acc.correlations()["score_play"] == pytest.approx(1.0, abs=1e-6)

    def test_constant_target_gives_none(self):
        acc = _HeadCorrAccumulator()
        targets = torch.full((5, N_HEADS), 0.5)
        preds = torch.arange(5, dtype=torch.float32).unsqueeze(1).repeat(1, N_HEADS)
        acc.add(preds, targets, _all_active_mask(5))
        assert acc.correlations()["score_play"] is None

    def test_too_few_active_cells_gives_none(self):
        acc = _HeadCorrAccumulator()
        targets = torch.tensor([[0.0], [1.0], [2.0]]).repeat(1, N_HEADS)
        preds = targets * 2.0
        head_mask = torch.zeros(3, N_HEADS)
        head_mask[0, 0] = 1.0  # only one active cell for head 0
        acc.add(preds, targets, head_mask)
        assert acc.correlations()["score_play"] is None

    def test_empty_when_nothing_added(self):
        corrs = _HeadCorrAccumulator().correlations()
        assert set(corrs) == {
            "score_play", "score_draw", "played_rate", "cast_lift",
            *(f"color_lift_{c}" for c in COLOR_ORDER),
        }
        assert all(v is None for v in corrs.values())


# ── MLM mask draw ─────────────────────────────────────────────────────


class TestDrawMlmMask:
    def test_special_tokens_never_masked(self):
        # ids contain special token ids 0,1,2,3 and a few real ids 5,6.
        torch.manual_seed(0)
        ids = torch.tensor([[0, 1, 2, 3, 5, 6, 5, 6]], dtype=torch.long)
        attention_mask = torch.ones_like(ids)
        mask_token_id = 3
        special_token_ids = (0, 1, 2, 3)
        masked, positions = _draw_mlm_mask(
            ids, attention_mask, MlmMaskSpec(
                mask_prob=1.0, mask_token_id=mask_token_id,
                special_token_ids=special_token_ids,
            ),
        )
        # mask_prob=1 → every eligible position must flip; specials must not.
        for col, original in enumerate(ids[0].tolist()):
            if original in special_token_ids:
                assert positions[0, col].item() == 0
                assert masked[0, col].item() == original
            else:
                assert positions[0, col].item() == 1
                assert masked[0, col].item() == mask_token_id

    def test_pad_positions_never_masked(self):
        torch.manual_seed(0)
        ids = torch.tensor([[5, 6, 0, 0]], dtype=torch.long)  # PAD = 0
        attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        masked, positions = _draw_mlm_mask(
            ids, attention_mask, MlmMaskSpec(
                mask_prob=1.0, mask_token_id=3, special_token_ids=(0, 1, 2, 3),
            ),
        )
        # The two PAD positions are not eligible.
        assert positions[0, 2].item() == 0
        assert positions[0, 3].item() == 0

    def test_mask_prob_drives_approximate_fraction(self):
        torch.manual_seed(42)
        n_real = 1000
        ids = torch.full((1, n_real), 5, dtype=torch.long)
        attention_mask = torch.ones_like(ids)
        _, positions = _draw_mlm_mask(
            ids, attention_mask, MlmMaskSpec(
                mask_prob=0.15, mask_token_id=3, special_token_ids=(0, 1, 2, 3),
            ),
        )
        fraction = positions.float().mean().item()
        # Rough sanity check; CLT gives stddev ~ sqrt(0.15*0.85/1000) ≈ 0.011
        assert 0.10 < fraction < 0.20
