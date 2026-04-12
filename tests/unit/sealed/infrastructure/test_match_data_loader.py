"""Unit tests for the match data loader (parse match-outcomes.txt, build datasets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sealed.domain.card_embedding_layout import total_dim
from sealed.infrastructure.converted_card_locator import BASIC_LAND_NAMES
from sealed.infrastructure.match_data_loader import (
    EmbeddingTable,
    MatchOutcome,
    TrainingExample,
    build_training_examples,
    collate_training_examples,
    load_match_outcomes,
    parse_match_outcome,
)

D_MODEL = total_dim(256)


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
        winner_non_basic = [n for n in outcome.winner_names if n.lower() not in BASIC_LAND_NAMES]
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

    def test_builds_winner_loser_index_tensors(self, tmp_path):
        emb = np.random.randn(D_MODEL).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {
            "card_a": emb,
            "card_b": emb + 1,
        })
        outcomes = [MatchOutcome(
            deck_a_names=["card_a"], deck_b_names=["card_b"],
            wins_a=2, wins_b=0,
        )]
        examples, table = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1
        assert examples[0].winner_indices.shape == (1,)
        assert examples[0].loser_indices.shape == (1,)
        assert table.num_cards == 2

    def test_filters_basic_lands(self, tmp_path):
        emb = np.random.randn(D_MODEL).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {"card_a": emb})
        outcomes = [MatchOutcome(
            deck_a_names=["card_a", "Mountain", "Mountain"],
            deck_b_names=["card_a"],
            wins_a=2, wins_b=0,
        )]
        examples, _ = build_training_examples(outcomes, cards_dir)
        assert examples[0].winner_indices.shape[0] == 1

    def test_double_slash_card_name_resolved(self, tmp_path):
        """Glassworks // Shattered Yard resolves to glassworks_shattered_yard.npz."""
        emb = np.random.randn(D_MODEL).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {
            "glassworks_shattered_yard": emb,
        })
        outcomes = [MatchOutcome(
            deck_a_names=["Glassworks // Shattered Yard"],
            deck_b_names=["Glassworks // Shattered Yard"],
            wins_a=2, wins_b=0,
        )]
        examples, _ = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1

    def test_accented_card_name_resolved(self, tmp_path):
        """Dandân (with accent) resolves to dandan.npz on disk."""
        emb = np.random.randn(D_MODEL).astype(np.float32)
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
        examples, _ = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1

    def test_double_faced_card_found_by_prefix(self, tmp_path):
        """A DFC stored as 'frontface_backface.npz' is found by front-face name."""
        emb = np.random.randn(D_MODEL).astype(np.float32)
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
        examples, table = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1
        idx = examples[0].winner_indices[0].item()
        np.testing.assert_array_equal(table.embedding.weight[idx].numpy(), emb)

    def test_missing_card_skips_match(self, tmp_path):
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        outcomes = [MatchOutcome(
            deck_a_names=["nonexistent"], deck_b_names=["also_missing"],
            wins_a=2, wins_b=0,
        )]
        examples, _ = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 0

    def test_shared_table_across_examples(self, tmp_path):
        """Two matches reusing the same card share one table row."""
        emb = np.random.randn(D_MODEL).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {
            "card_a": emb,
            "card_b": emb + 1,
        })
        outcomes = [
            MatchOutcome(["card_a"], ["card_b"], 2, 0),
            MatchOutcome(["card_a"], ["card_b"], 2, 1),
        ]
        examples, table = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 2
        assert table.num_cards == 2
        assert examples[0].winner_indices.tolist() == examples[1].winner_indices.tolist()


class TestEmbeddingTable:
    def test_frozen_by_default(self):
        table = EmbeddingTable(torch.randn(3, D_MODEL), {"a": 0, "b": 1, "c": 2})
        assert table.is_frozen()
        assert not table.embedding.weight.requires_grad

    def test_unfreeze_flips_requires_grad(self):
        table = EmbeddingTable(torch.randn(3, D_MODEL), {"a": 0, "b": 1, "c": 2})
        table.unfreeze()
        assert not table.is_frozen()
        assert table.embedding.weight.requires_grad

    def test_lookup_returns_seeded_vectors(self):
        vecs = torch.randn(3, D_MODEL)
        table = EmbeddingTable(vecs, {"a": 0, "b": 1, "c": 2})
        out = table(torch.tensor([0, 2]))
        torch.testing.assert_close(out[0], vecs[0])
        torch.testing.assert_close(out[1], vecs[2])


class TestCollateFunction:
    def test_variable_length_padding(self):
        """Collate should pad to max length with boolean masks."""
        ex1 = TrainingExample(
            winner_indices=torch.arange(5, dtype=torch.long),
            loser_indices=torch.arange(3, dtype=torch.long),
        )
        ex2 = TrainingExample(
            winner_indices=torch.arange(8, dtype=torch.long),
            loser_indices=torch.arange(6, dtype=torch.long),
        )

        batch = collate_training_examples([ex1, ex2])
        assert batch.winner_indices.shape == (2, 8)
        assert batch.loser_indices.shape == (2, 6)
        assert batch.winner_mask.shape == (2, 8)
        assert batch.loser_mask.shape == (2, 6)

    def test_mask_marks_real_cards_true(self):
        ex1 = TrainingExample(
            winner_indices=torch.arange(3, dtype=torch.long),
            loser_indices=torch.arange(2, dtype=torch.long),
        )
        ex2 = TrainingExample(
            winner_indices=torch.arange(5, dtype=torch.long),
            loser_indices=torch.arange(4, dtype=torch.long),
        )

        batch = collate_training_examples([ex1, ex2])
        assert batch.winner_mask[0, :3].all()
        assert not batch.winner_mask[0, 3:].any()
        assert batch.winner_mask[1, :5].all()
