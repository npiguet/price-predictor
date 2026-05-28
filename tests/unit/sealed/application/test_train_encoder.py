"""Unit tests for ``train-encoder``: aggregation, label arithmetic,
weighted MSE, MLM mask draw, stratification, and missing-card handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sealed.application.train_encoder import (
    N_HEADS,
    CardCounters,
    CardLabels,
    _aggregate,
    _build_label_map,
    _colors_from_mana_cost,
    _draw_mlm_mask,
    _drop_missing_cards,
    _HeadCorrAccumulator,
    _LengthBucketBatchSampler,
    _pad_collate,
    _per_batch_weighted_mse,
    _split_cards,
    _write_win_rates,
)
from sealed.domain.encoder_model import COLOR_ORDER
from sealed.domain.match import Side
from sealed.infrastructure.cards_played_reader import CardsPlayedRow
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

# Per-color counter lists on CardCounters are indexed by WUBRG position.
_CI: dict[str, int] = {c: i for i, c in enumerate(COLOR_ORDER)}


def _row(
    *,
    cp_a: list[str],
    cp_b: list[str],
    cnp_a: list[str],
    cnp_b: list[str],
    winner: str = "A",
    starter: str = "A",
) -> CardsPlayedRow:
    return CardsPlayedRow(
        timestamp="2026-05-03T14:22:01Z",
        run_id="run",
        set_code="BLB",
        method_a="forge-best",
        method_b="forge-3sub",
        cards_played_a=cp_a,
        cards_played_b=cp_b,
        cards_not_played_a=cnp_a,
        cards_not_played_b=cnp_b,
        winner=Side(winner),
        starter=Side(starter),
    )


def _seed_card(folder: Path, card_name: str, mana_cost: str | None) -> None:
    sanitized = card_name.lower().replace(" ", "_").replace(",", "")
    letter = sanitized[0]
    (folder / letter).mkdir(parents=True, exist_ok=True)
    cost_line = f"mana cost: {mana_cost}\n" if mana_cost is not None else ""
    (folder / letter / f"{sanitized}.txt").write_text(
        f"name: {card_name}\n{cost_line}types: instant\n",
        encoding="utf-8",
    )


# ── Aggregation (primary + @play + per-color, single pass) ────────────


class TestAggregate:
    def test_winning_and_losing_sides_both_counted(self, tmp_path: Path):
        loc = ConvertedCardLocator(tmp_path)
        rows = [
            _row(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[],
                 winner="A", starter="A"),
            _row(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[],
                 winner="B", starter="A"),
        ]
        counters = _aggregate(rows, loc)
        # LB: A side, won game 1 / lost game 2
        assert counters["LB"].wins_when_played == 1
        assert counters["LB"].losses_when_played == 1
        assert counters["LB"].wins_when_in_deck == 1
        assert counters["LB"].losses_when_in_deck == 1
        # GB: B side, lost game 1 / won game 2
        assert counters["GB"].wins_when_played == 1
        assert counters["GB"].losses_when_played == 1

    def test_in_deck_includes_not_played(self, tmp_path: Path):
        rows = [
            _row(cp_a=["LB"], cp_b=[], cnp_a=["GB"], cnp_b=[],
                 winner="A", starter="A"),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        assert counters["LB"].wins_when_played == 1
        assert counters["LB"].wins_when_in_deck == 1
        assert counters["GB"].wins_when_played == 0
        assert counters["GB"].wins_when_in_deck == 1

    def test_starter_drives_at_play_subset(self, tmp_path: Path):
        # Card on side A is starter (at play); same card later on side A
        # but starter is B (at draw).
        rows = [
            _row(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="A"),
            _row(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="B"),
        ]
        c = _aggregate(rows, ConvertedCardLocator(tmp_path))["LB"]
        assert c.wins_when_played_at_play == 1   # only the first row
        assert c.wins_when_in_deck_at_play == 1
        # On-draw counterparts: derivable by subtraction (FR-010a).
        assert c.wins_when_played - c.wins_when_played_at_play == 1
        assert c.wins_when_in_deck - c.wins_when_in_deck_at_play == 1

    def test_multi_game_match_accumulates(self, tmp_path: Path):
        rows = [
            _row(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="A"),
            _row(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="A"),
            _row(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="B", starter="A"),
        ]
        c = _aggregate(rows, ConvertedCardLocator(tmp_path))["LB"]
        assert c.wins_when_played == 2
        assert c.losses_when_played == 1
        assert c.wins_when_in_deck == 2
        assert c.losses_when_in_deck == 1

    def test_per_color_counters_built_from_deck_color_set(self, tmp_path: Path):
        # Side A deck: LB ({R}) + GB ({G}). Deck colors = {R, G}.
        # Both LB and GB get +1 wins_when_in_deck_with_{R,G}; LB is
        # "played", GB is "not played".
        _seed_card(tmp_path, "LB", "{R}")
        _seed_card(tmp_path, "GB", "{G}")
        rows = [
            _row(cp_a=["LB"], cp_b=[], cnp_a=["GB"], cnp_b=[],
                 winner="A", starter="A"),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        c_lb = counters["LB"]
        assert c_lb.wins_when_played_with[_CI["R"]] == 1
        assert c_lb.wins_when_played_with[_CI["G"]] == 1
        assert c_lb.wins_when_in_deck_with[_CI["R"]] == 1
        assert c_lb.wins_when_in_deck_with[_CI["G"]] == 1
        assert c_lb.wins_when_played_with[_CI["W"]] == 0
        c_gb = counters["GB"]
        assert c_gb.wins_when_in_deck_with[_CI["R"]] == 1
        assert c_gb.wins_when_in_deck_with[_CI["G"]] == 1
        assert c_gb.wins_when_played_with[_CI["R"]] == 0  # GB was not played

    def test_losing_side_per_color_counters_increment(self, tmp_path: Path):
        _seed_card(tmp_path, "LB", "{R}")
        _seed_card(tmp_path, "GB", "{G}")
        rows = [
            _row(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[],
                 winner="A", starter="A"),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        # GB (loser): losses_when_played_with_G should be 1.
        assert counters["GB"].losses_when_played_with[_CI["G"]] == 1
        assert counters["GB"].losses_when_in_deck_with[_CI["G"]] == 1

    def test_card_with_no_mana_cost_contributes_no_color(self, tmp_path: Path):
        # Land-like card: no "mana cost:" line; deck-color set ignores it.
        _seed_card(tmp_path, "Plains-ish", None)
        _seed_card(tmp_path, "LB", "{R}")
        rows = [
            _row(cp_a=["LB"], cp_b=[], cnp_a=["Plains-ish"], cnp_b=[],
                 winner="A", starter="A"),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        # Plains-ish is "not played" but still in deck running color R.
        assert counters["Plains-ish"].wins_when_in_deck_with[_CI["R"]] == 1
        # It contributed no color of its own.
        assert counters["Plains-ish"].wins_when_in_deck_with[_CI["W"]] == 0

    def test_card_missing_from_corpus_is_still_counted(self, tmp_path: Path):
        # A card with no .txt: still gets primary/@play counters (it's
        # dropped later by _drop_missing_cards), and an empty color set.
        rows = [
            _row(cp_a=["Mystery Card"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="A"),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        c = counters["Mystery Card"]
        assert c.wins_when_played == 1
        assert all(v == 0 for v in c.wins_when_in_deck_with)


# ── Color extraction ─────────────────────────────────────────────────


class TestColorsFromManaCost:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("{R}", {"R"}),
            ("{1}{G}", {"G"}),
            ("{2}{U}{U}", {"U"}),
            ("{W/U}", {"W", "U"}),         # hybrid
            ("{W/P}", {"W"}),               # Phyrexian
            ("{2/W}", {"W"}),               # mono-hybrid
            ("{2}", set()),
            ("{C}", set()),
            ("{X}", set()),
            ("{2}{R}{G}", {"R", "G"}),
            ("", set()),
            (None, set()),
        ],
    )
    def test_dispatches_correctly(self, line, expected):
        assert _colors_from_mana_cost(line) == expected


# ── Label arithmetic ─────────────────────────────────────────────────


def _make_counter(
    *,
    wp: int = 0, lp: int = 0, wd: int = 0, ld: int = 0,
    wp_play: int = 0, lp_play: int = 0,
    wd_play: int = 0, ld_play: int = 0,
    color_in_deck: dict[str, int] | None = None,
    color_played: dict[str, int] | None = None,
    color_loss_in_deck: dict[str, int] | None = None,
    color_loss_played: dict[str, int] | None = None,
) -> CardCounters:
    c = CardCounters()
    c.wins_when_played = wp
    c.losses_when_played = lp
    c.wins_when_in_deck = wd
    c.losses_when_in_deck = ld
    c.wins_when_played_at_play = wp_play
    c.losses_when_played_at_play = lp_play
    c.wins_when_in_deck_at_play = wd_play
    c.losses_when_in_deck_at_play = ld_play
    for letter_map, slots in (
        (color_in_deck, c.wins_when_in_deck_with),
        (color_played, c.wins_when_played_with),
        (color_loss_in_deck, c.losses_when_in_deck_with),
        (color_loss_played, c.losses_when_played_with),
    ):
        for letter, value in (letter_map or {}).items():
            slots[_CI[letter]] = value
    return c


class TestBuildLabelMap:
    def test_excludes_zero_in_deck(self):
        counters = {"X": _make_counter(wd=0, ld=0)}
        labels = _build_label_map(counters, shrinkage_k=20.0)
        assert "X" not in labels

    def test_score_play_raw_and_shrunk_with_k_zero_match(self):
        # 4 in-deck @play games: 3 wins (all played), 1 loss (played).
        counters = {"X": _make_counter(
            wp=3, lp=1, wd=3, ld=1,
            wp_play=3, lp_play=1, wd_play=3, ld_play=1,
        )}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        # raw_score_play = (3 - 1) / (3 + 1) = 0.5
        assert lbl.raw_score_play == pytest.approx(0.5)
        # k=0 → shrunk == raw.
        assert lbl.shrunk_score_play == pytest.approx(0.5)

    def test_score_play_shrinkage_pulls_toward_zero(self):
        counters = {"X": _make_counter(
            wp=2, lp=0, wd=2, ld=0,
            wp_play=2, lp_play=0, wd_play=2, ld_play=0,
        )}
        labels = _build_label_map(counters, shrinkage_k=20.0)
        lbl = labels["X"]
        # raw = 2/2 = 1.0; shrunk = 2 / (2 + 20) ≈ 0.0909
        assert lbl.raw_score_play == pytest.approx(1.0)
        assert lbl.shrunk_score_play == pytest.approx(2.0 / 22.0)

    def test_zero_at_play_denominator_yields_none(self):
        # Card never observed on the play.
        counters = {"X": _make_counter(
            wp=2, lp=1, wd=2, ld=1,
            wp_play=0, lp_play=0, wd_play=0, ld_play=0,
        )}
        labels = _build_label_map(counters, shrinkage_k=20.0)
        lbl = labels["X"]
        assert lbl.raw_score_play is None
        assert lbl.shrunk_score_play is None
        # score_draw cell should be populated from the @draw subset.
        assert lbl.raw_score_draw is not None

    def test_played_rate_always_present_when_card_kept(self):
        counters = {"X": _make_counter(wp=1, lp=0, wd=2, ld=2)}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        # raw_played_rate = 1 / 4 = 0.25
        assert lbl.raw_played_rate == pytest.approx(0.25)
        assert lbl.shrunk_played_rate == pytest.approx(0.25)

    def test_cast_lift_empty_when_never_played(self):
        # Card is in deck 4 times but never played.
        counters = {"X": _make_counter(wp=0, lp=0, wd=2, ld=2)}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        assert lbl.raw_cast_lift is None

    def test_cast_lift_empty_when_always_played(self):
        # Card is in deck and played every time.
        counters = {"X": _make_counter(wp=2, lp=2, wd=2, ld=2)}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        assert lbl.raw_cast_lift is None

    def test_color_lift_empty_when_no_observations_with_color(self):
        # Card in deck once with color R and never with W.
        counters = {"X": _make_counter(
            wp=1, lp=0, wd=1, ld=0,
            color_in_deck={"R": 1}, color_played={"R": 1},
        )}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        assert lbl.raw_color_lift["R"] is not None
        assert lbl.raw_color_lift["W"] is None


# ── cards-win-rates.txt writer ───────────────────────────────────────


def _label(name: str, **overrides) -> CardLabels:
    base = dict(
        card_name=name,
        counters=CardCounters(),
        raw_score_play=0.5, shrunk_score_play=0.5,
        raw_score_draw=0.4, shrunk_score_draw=0.4,
        raw_played_rate=0.6, shrunk_played_rate=0.6,
        raw_cast_lift=0.1, shrunk_cast_lift=0.1,
        raw_color_lift={c: 0.0 for c in COLOR_ORDER},
        shrunk_color_lift={c: 0.0 for c in COLOR_ORDER},
    )
    base.update(overrides)
    return CardLabels(**base)


class TestWriteWinRates:
    def test_header_matches_fr013a(self, tmp_path: Path):
        labels = {"X": _label("X")}
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == (
            "card_name;wins_when_played;wins_when_in_deck;"
            "losses_when_played;losses_when_in_deck;"
            "raw_score_play;shrunk_score_play;"
            "raw_score_draw;shrunk_score_draw;"
            "raw_played_rate;shrunk_played_rate;"
            "raw_cast_lift;shrunk_cast_lift;"
            "raw_color_lift_W;shrunk_color_lift_W;"
            "raw_color_lift_U;shrunk_color_lift_U;"
            "raw_color_lift_B;shrunk_color_lift_B;"
            "raw_color_lift_R;shrunk_color_lift_R;"
            "raw_color_lift_G;shrunk_color_lift_G"
        )
        # 1 (card_name) + 4 (primary counters) + 9 heads × 2 (raw+shrunk)
        # = 23 columns. (Supporting docs count "24" — typo; spec.md
        # FR-013a's column list adds to 23.)
        assert len(first.split(";")) == 23

    def test_row_has_23_fields(self, tmp_path: Path):
        labels = {"X": _label("X")}
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        rows = path.read_text(encoding="utf-8").splitlines()
        assert len(rows[1].split(";")) == 23

    def test_empty_cells_render_as_empty_string(self, tmp_path: Path):
        no_play = _label(
            "Y",
            raw_score_play=None, shrunk_score_play=None,
            raw_cast_lift=None, shrunk_cast_lift=None,
            raw_color_lift={"W": None, "U": 0.1, "B": None, "R": None, "G": None},
            shrunk_color_lift={"W": None, "U": 0.1, "B": None, "R": None, "G": None},
        )
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates({"Y": no_play}, path)
        row = path.read_text(encoding="utf-8").splitlines()[1].split(";")
        # raw_score_play (idx 5) and shrunk_score_play (idx 6) should be empty.
        assert row[5] == ""
        assert row[6] == ""
        # raw_color_lift_U (idx 15) is non-empty; raw_color_lift_W (idx 13) empty.
        assert row[13] == ""
        assert row[15] == "0.10000"

    def test_sorted_by_shrunk_score_play_desc_with_empties_at_end(
        self, tmp_path: Path,
    ):
        labels = {
            "Top": _label("Top", shrunk_score_play=0.9),
            "Mid": _label("Mid", shrunk_score_play=0.5),
            "Bot": _label("Bot", shrunk_score_play=-0.4),
            "Empty": _label("Empty", shrunk_score_play=None),
        }
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        rows = path.read_text(encoding="utf-8").splitlines()
        assert rows[1].startswith("Top;")
        assert rows[2].startswith("Mid;")
        assert rows[3].startswith("Bot;")
        assert rows[4].startswith("Empty;")


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
            ids, attention_mask, mask_prob=1.0,
            mask_token_id=mask_token_id, special_token_ids=special_token_ids,
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
            ids, attention_mask, mask_prob=1.0,
            mask_token_id=3, special_token_ids=(0, 1, 2, 3),
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
            ids, attention_mask, mask_prob=0.15,
            mask_token_id=3, special_token_ids=(0, 1, 2, 3),
        )
        fraction = positions.float().mean().item()
        # Rough sanity check; CLT gives stddev ~ sqrt(0.15*0.85/1000) ≈ 0.011
        assert 0.10 < fraction < 0.20


# ── Stratification ───────────────────────────────────────────────────


def _label_map_with_score_play(values: dict[str, float | None]) -> dict[str, CardLabels]:
    labels: dict[str, CardLabels] = {}
    for name, v in values.items():
        labels[name] = _label(
            name,
            shrunk_score_play=v,
            shrunk_score_draw=0.0,
            shrunk_cast_lift=0.0,
            shrunk_color_lift={c: 0.0 for c in COLOR_ORDER},
        )
    return labels


class TestSplitCards:
    def test_disjoint_split(self):
        labels = _label_map_with_score_play(
            {f"c{i}": (i / 100.0) for i in range(50)},
        )
        train, val = _split_cards(labels, val_fraction=0.2, seed=42)
        assert set(train).isdisjoint(set(val))
        assert set(train) | set(val) == set(labels.keys())

    def test_fallback_chain_used_when_score_play_empty(self):
        # Card "a" has empty score_play but non-empty score_draw → its
        # stratification key falls back to score_draw.
        labels = {
            "a": _label("a",
                        shrunk_score_play=None, shrunk_score_draw=0.5,
                        shrunk_cast_lift=None,
                        shrunk_color_lift={c: None for c in COLOR_ORDER}),
            "b": _label("b", shrunk_score_play=0.7),
            "c": _label("c", shrunk_score_play=-0.3),
        }
        train, val = _split_cards(labels, val_fraction=0.2, seed=42)
        # All three cards land somewhere; no exception, set complete.
        assert set(train) | set(val) == {"a", "b", "c"}

    def test_catch_all_stratum_for_fully_degenerate(self):
        # Every signed cell empty → catch-all stratum.
        labels = {
            f"x{i}": _label(
                f"x{i}",
                shrunk_score_play=None, shrunk_score_draw=None,
                shrunk_cast_lift=None,
                shrunk_color_lift={c: None for c in COLOR_ORDER},
            )
            for i in range(10)
        }
        train, val = _split_cards(labels, val_fraction=0.2, seed=42)
        assert set(train).isdisjoint(set(val))
        assert len(val) == 2  # 20% of 10


# ── Missing-card handling (report-but-don't-block) ───────────────────


class TestDropMissingCards:
    def test_noop_when_all_cards_present(self, tmp_path: Path):
        for name in ("Lightning Bolt", "Grizzly Bears"):
            _seed_card(tmp_path, name, "{R}")
        counters = {n: CardCounters() for n in ("Lightning Bolt", "Grizzly Bears")}
        dropped = _drop_missing_cards(counters, ConvertedCardLocator(tmp_path))
        assert dropped == 0
        assert set(counters) == {"Lightning Bolt", "Grizzly Bears"}

    def test_drops_missing_cards_and_reports(self, tmp_path: Path, capsys):
        _seed_card(tmp_path, "Lightning Bolt", "{R}")
        counters = {
            n: CardCounters()
            for n in ("Lightning Bolt", "Counterspell", "Llanowar Elves")
        }
        dropped = _drop_missing_cards(counters, ConvertedCardLocator(tmp_path))
        assert dropped == 2
        # Missing cards are removed; the resolvable one survives.
        assert set(counters) == {"Lightning Bolt"}
        # ... and they're reported on stdout (not raised).
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "2 card(s)" in out
        assert "Counterspell" in out
        assert "Llanowar Elves" in out
        assert "python -m price_predictor convert" in out

    def test_caps_displayed_card_names_at_20(self, tmp_path: Path, capsys):
        counters = {f"Card {i}": CardCounters() for i in range(30)}
        dropped = _drop_missing_cards(counters, ConvertedCardLocator(tmp_path))
        assert dropped == 30
        assert counters == {}
        assert "and 10 more" in capsys.readouterr().out


# ── Length-bucketed batching + per-batch padding ─────────────────────


def _item(length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A dataset-shaped item: input_ids of the given length + zero label tensors."""
    return (
        torch.arange(1, length + 1, dtype=torch.long),
        torch.zeros(N_HEADS), torch.zeros(N_HEADS), torch.zeros(N_HEADS),
    )


class TestPadCollate:
    def test_pads_to_batch_max_and_builds_mask(self):
        batch = [_item(2), _item(5), _item(3)]
        ids, mask, labels, weights, head_mask = _pad_collate(batch)
        assert ids.shape == (3, 5)        # padded to the longest item
        assert mask.shape == (3, 5)
        # row 0: real positions 0,1 then PAD(=0); mask 1,1,0,0,0.
        assert ids[0].tolist() == [1, 2, 0, 0, 0]
        assert mask[0].tolist() == [1, 1, 0, 0, 0]
        assert ids[1].tolist() == [1, 2, 3, 4, 5]
        assert mask[1].tolist() == [1, 1, 1, 1, 1]
        assert labels.shape == (3, N_HEADS)
        assert weights.shape == (3, N_HEADS)
        assert head_mask.shape == (3, N_HEADS)

    def test_single_item_batch(self):
        ids, mask, *_ = _pad_collate([_item(4)])
        assert ids.shape == (1, 4)
        assert mask.tolist() == [[1, 1, 1, 1]]


class TestLengthBucketBatchSampler:
    def test_covers_every_index_once_per_epoch(self):
        lengths = [10, 1, 7, 3, 9, 2, 8]  # 7 items, batch_size 3 → 3 batches
        sampler = _LengthBucketBatchSampler(lengths, batch_size=3, shuffle=True)
        assert len(sampler) == 3
        seen = sorted(i for batch in sampler for i in batch)
        assert seen == list(range(7))

    def test_chunks_group_similar_lengths(self):
        lengths = [100, 1, 99, 2, 98, 3]  # sorted: 1,2,3 | 98,99,100
        sampler = _LengthBucketBatchSampler(lengths, batch_size=3, shuffle=False)
        batches = list(sampler)
        # not shuffled → length order; each chunk's lengths are contiguous.
        chunk_lengths = [sorted(lengths[i] for i in b) for b in batches]
        assert chunk_lengths == [[1, 2, 3], [98, 99, 100]]

    def test_shuffle_varies_batch_order_across_epochs(self):
        lengths = list(range(50))
        sampler = _LengthBucketBatchSampler(lengths, batch_size=5, shuffle=True)
        e1 = list(sampler)
        e2 = list(sampler)
        # Same coverage, but a different order (vanishingly unlikely to match).
        assert sorted(i for b in e1 for i in b) == sorted(i for b in e2 for i in b)
        assert e1 != e2

    def test_no_shuffle_is_deterministic(self):
        lengths = [5, 3, 8, 1, 9, 2]
        s = _LengthBucketBatchSampler(lengths, batch_size=2, shuffle=False)
        assert list(s) == list(s)

    def test_empty(self):
        s = _LengthBucketBatchSampler([], batch_size=4, shuffle=True)
        assert len(s) == 0
        assert list(s) == []
