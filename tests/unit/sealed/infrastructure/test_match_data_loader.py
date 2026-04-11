"""Unit tests for the match data loader (parse match-outcomes.txt, build datasets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sealed.infrastructure.match_data_loader import (
    parse_match_outcome,
    MatchOutcome,
    load_match_outcomes,
    build_training_examples,
    collate_training_examples,
    BASIC_LANDS,
)


class TestParseMatchOutcome:
    def test_basic_parse(self):
        line = "CardA|CardB;CardC|CardD;2;1"
        outcome = parse_match_outcome(line)
        assert outcome.deck_a_names == ["CardA", "CardB"]
        assert outcome.deck_b_names == ["CardC", "CardD"]
        assert outcome.wins_a == 2
        assert outcome.wins_b == 1

    def test_winner_is_deck_with_two_wins(self):
        outcome = parse_match_outcome("A|B;C|D;2;1")
        assert outcome.winner_names == ["A", "B"]
        assert outcome.loser_names == ["C", "D"]

    def test_loser_wins_two(self):
        outcome = parse_match_outcome("A|B;C|D;0;2")
        assert outcome.winner_names == ["C", "D"]
        assert outcome.loser_names == ["A", "B"]


class TestBasicLandFiltering:
    def test_basic_lands_filtered(self):
        """Basic lands should be filtered from deck card lists."""
        line = "Lightning Bolt|Mountain|Mountain|Forest;Llanowar Elves|Island|Plains;2;0"
        outcome = parse_match_outcome(line)
        winner_non_basic = [n for n in outcome.winner_names if n.lower() not in BASIC_LANDS]
        assert "Lightning Bolt" in winner_non_basic
        assert "Mountain" not in winner_non_basic


class TestLoadMatchOutcomes:
    def test_load_from_file(self, tmp_path):
        f = tmp_path / "outcomes.txt"
        f.write_text("A|B;C|D;2;1\nE|F;G|H;1;2\n", encoding="utf-8")
        outcomes = load_match_outcomes(f)
        assert len(outcomes) == 2
        assert outcomes[0].wins_a == 2
        assert outcomes[1].wins_b == 2

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "outcomes.txt"
        f.write_text("A|B;C|D;2;1\n\n\nE|F;G|H;1;2\n", encoding="utf-8")
        outcomes = load_match_outcomes(f)
        assert len(outcomes) == 2


class TestBuildTrainingExamples:
    def _make_embeddings(self, tmp_path, cards: dict[str, np.ndarray]) -> Path:
        """Create .npz embedding files for test cards (in first-letter subdirs)."""
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        for name, emb in cards.items():
            letter_dir = cards_dir / name[0]
            letter_dir.mkdir(exist_ok=True)
            np.savez_compressed(letter_dir / f"{name}.npz", embedding=emb)
        return cards_dir

    def test_builds_winner_loser_tensors(self, tmp_path):
        emb = np.random.randn(544).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {
            "card_a": emb,
            "card_b": emb + 1,
        })
        outcomes = [MatchOutcome(
            deck_a_names=["card_a"], deck_b_names=["card_b"],
            wins_a=2, wins_b=0,
        )]
        examples = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1
        assert examples[0].winner_cards.shape == (1, 544)
        assert examples[0].loser_cards.shape == (1, 544)

    def test_filters_basic_lands(self, tmp_path):
        emb = np.random.randn(544).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {"card_a": emb})
        outcomes = [MatchOutcome(
            deck_a_names=["card_a", "Mountain", "Mountain"],
            deck_b_names=["card_a"],
            wins_a=2, wins_b=0,
        )]
        examples = build_training_examples(outcomes, cards_dir)
        # Mountain should be filtered, so winner has only 1 card
        assert examples[0].winner_cards.shape[0] == 1

    def test_double_slash_card_name_resolved(self, tmp_path):
        """Glassworks // Shattered Yard resolves to glassworks_shattered_yard.npz."""
        emb = np.random.randn(544).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {
            "glassworks_shattered_yard": emb,
        })
        outcomes = [MatchOutcome(
            deck_a_names=["Glassworks // Shattered Yard"],
            deck_b_names=["Glassworks // Shattered Yard"],
            wins_a=2, wins_b=0,
        )]
        examples = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1

    def test_accented_card_name_resolved(self, tmp_path):
        """Dandân (with accent) resolves to dandan.npz on disk."""
        emb = np.random.randn(544).astype(np.float32)
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        d_dir = cards_dir / "d"
        d_dir.mkdir()
        np.savez_compressed(d_dir / "dandan.npz", embedding=emb)
        outcomes = [MatchOutcome(
            deck_a_names=["Dand\u00e2n"],
            deck_b_names=["Dand\u00e2n"],
            wins_a=2, wins_b=0,
        )]
        examples = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1

    def test_double_faced_card_found_by_prefix(self, tmp_path):
        """A DFC stored as 'frontface_backface.npz' is found by front-face name."""
        emb = np.random.randn(544).astype(np.float32)
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        m_dir = cards_dir / "m"
        m_dir.mkdir()
        np.savez_compressed(
            m_dir / "mosswood_dreadknight_dread_whispers.npz",
            embedding=emb,
        )
        outcomes = [MatchOutcome(
            deck_a_names=["Mosswood Dreadknight"],
            deck_b_names=["Mosswood Dreadknight"],
            wins_a=2, wins_b=0,
        )]
        examples = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1
        np.testing.assert_array_equal(
            examples[0].winner_cards[0].numpy(), emb,
        )

    def test_missing_card_skips_match(self, tmp_path):
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        outcomes = [MatchOutcome(
            deck_a_names=["nonexistent"], deck_b_names=["also_missing"],
            wins_a=2, wins_b=0,
        )]
        examples = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 0


class TestCollateFunction:
    def test_variable_length_padding(self):
        """Collate should pad to max length with boolean masks."""
        from sealed.infrastructure.match_data_loader import TrainingExample

        ex1 = TrainingExample(
            winner_cards=torch.randn(5, 544),
            loser_cards=torch.randn(3, 544),
        )
        ex2 = TrainingExample(
            winner_cards=torch.randn(8, 544),
            loser_cards=torch.randn(6, 544),
        )

        batch = collate_training_examples([ex1, ex2])
        assert batch.winner_cards.shape == (2, 8, 544)  # padded to max=8
        assert batch.loser_cards.shape == (2, 6, 544)    # padded to max=6
        assert batch.winner_mask.shape == (2, 8)
        assert batch.loser_mask.shape == (2, 6)

    def test_mask_marks_real_cards_true(self):
        from sealed.infrastructure.match_data_loader import TrainingExample

        ex1 = TrainingExample(
            winner_cards=torch.randn(3, 544),
            loser_cards=torch.randn(2, 544),
        )
        ex2 = TrainingExample(
            winner_cards=torch.randn(5, 544),
            loser_cards=torch.randn(4, 544),
        )

        batch = collate_training_examples([ex1, ex2])
        # First example: 3 real cards, padded to 5
        assert batch.winner_mask[0, :3].all()
        assert not batch.winner_mask[0, 3:].any()
        # Second example: 5 real cards, no padding needed
        assert batch.winner_mask[1, :5].all()
