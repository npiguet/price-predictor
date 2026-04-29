"""Unit tests for the match data loader (parse match-outcomes.txt, build datasets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sealed.domain.card_embedding_layout import total_dim
from sealed.infrastructure.converted_card_locator import BASIC_LAND_NAMES
from sealed.infrastructure.match_data_loader import (
    EmbeddingTable,
    MatchOutcome,
    MatchTrainingExample,
    build_training_examples,
    collate_training_examples,
    load_match_outcomes,
    parse_match_outcome,
)

D_MODEL = total_dim(256)

SAMPLE_LINE = (
    "2026-04-22T14:30:05Z;run-xyz;RVR;forge-best;gen-2;"
    "CardA|CardB;CardC|CardD;"
    "ABA;BAB;47"
)


def _outcome(
    deck_a: list[str],
    deck_b: list[str],
    games: str,
    *,
    timestamp: str = "2026-04-22T14:30:05Z",
    run_id: str = "run-xyz",
    set_code: str = "RVR",
    method_a: str = "forge-best",
    method_b: str = "forge-3sub",
    play: str | None = None,
    duration_s: int = 47,
) -> MatchOutcome:
    """Build a MatchOutcome fixture; play defaults to all-A of matching length."""
    if play is None:
        play = "A" * len(games)
    return MatchOutcome(
        timestamp=timestamp,
        run_id=run_id,
        set_code=set_code,
        method_a=method_a,
        method_b=method_b,
        deck_a_names=deck_a,
        deck_b_names=deck_b,
        games=games,
        play=play,
        duration_s=duration_s,
    )


class TestParseMatchOutcome:
    def test_basic_parse(self):
        outcome = parse_match_outcome(SAMPLE_LINE)
        assert outcome.deck_a_names == ["CardA", "CardB"]
        assert outcome.deck_b_names == ["CardC", "CardD"]
        assert outcome.wins_a == 2
        assert outcome.wins_b == 1

    def test_metadata_fields_populated(self):
        outcome = parse_match_outcome(SAMPLE_LINE)
        assert outcome.timestamp == "2026-04-22T14:30:05Z"
        assert outcome.run_id == "run-xyz"
        assert outcome.set_code == "RVR"
        assert outcome.method_a == "forge-best"
        assert outcome.method_b == "gen-2"
        assert outcome.games == "ABA"
        assert outcome.play == "BAB"
        assert outcome.duration_s == 47

    def test_winner_is_deck_with_two_wins(self):
        outcome = parse_match_outcome(SAMPLE_LINE)
        assert outcome.winner_names == ["CardA", "CardB"]
        assert outcome.loser_names == ["CardC", "CardD"]

    def test_loser_wins_two(self):
        line = (
            "2026-04-22T14:30:05Z;run-xyz;RVR;forge-best;gen-2;"
            "A|B;C|D;BB;AA;12"
        )
        outcome = parse_match_outcome(line)
        assert outcome.winner_names == ["C", "D"]
        assert outcome.loser_names == ["A", "B"]

    def test_two_game_sweep_parses(self):
        line = (
            "2026-04-22T14:30:05Z;run-xyz;RVR;forge-best;forge-3sub;"
            "X|Y;Z|W;AA;BA;12"
        )
        outcome = parse_match_outcome(line)
        assert outcome.wins_a == 2
        assert outcome.wins_b == 0

    def test_wrong_field_count_rejected(self):
        legacy_line = "A|B;C|D;2;1"
        with pytest.raises(ValueError, match="10 semicolon"):
            parse_match_outcome(legacy_line)


class TestBasicLandFiltering:
    def test_basic_lands_filtered(self):
        """Basic lands should be filtered from deck card lists during training-example build."""
        line = (
            "2026-04-22T14:30:05Z;run-xyz;RVR;forge-best;forge-3sub;"
            "Lightning Bolt|Mountain|Mountain|Forest;Llanowar Elves|Island|Plains;"
            "AA;BA;30"
        )
        outcome = parse_match_outcome(line)
        winner_non_basic = [
            n for n in outcome.winner_names if n.lower() not in BASIC_LAND_NAMES
        ]
        assert "Lightning Bolt" in winner_non_basic
        assert "Mountain" not in winner_non_basic


class TestLoadMatchOutcomes:
    def test_load_from_file(self, tmp_path):
        f = tmp_path / "outcomes.txt"
        f.write_text(SAMPLE_LINE + "\n" + SAMPLE_LINE + "\n", encoding="utf-8")
        outcomes = load_match_outcomes(f)
        assert len(outcomes) == 2
        assert outcomes[0].wins_a == 2

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "outcomes.txt"
        f.write_text(SAMPLE_LINE + "\n\n\n" + SAMPLE_LINE + "\n", encoding="utf-8")
        outcomes = load_match_outcomes(f)
        assert len(outcomes) == 2


class TestBuildMatchTrainingExamples:
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
        outcomes = [_outcome(["card_a"], ["card_b"], "AA")]
        examples, table = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1
        assert examples[0].winner_indices.shape == (1,)
        assert examples[0].loser_indices.shape == (1,)
        assert table.num_cards == 2

    def test_filters_basic_lands(self, tmp_path):
        emb = np.random.randn(D_MODEL).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {"card_a": emb})
        outcomes = [_outcome(
            ["card_a", "Mountain", "Mountain"],
            ["card_a"],
            "AA",
        )]
        examples, _ = build_training_examples(outcomes, cards_dir)
        assert examples[0].winner_indices.shape[0] == 1

    def test_double_slash_card_name_resolved(self, tmp_path):
        """Glassworks // Shattered Yard resolves to glassworks_shattered_yard.npz."""
        emb = np.random.randn(D_MODEL).astype(np.float32)
        cards_dir = self._make_embeddings(tmp_path, {
            "glassworks_shattered_yard": emb,
        })
        outcomes = [_outcome(
            ["Glassworks // Shattered Yard"],
            ["Glassworks // Shattered Yard"],
            "AA",
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
        outcomes = [_outcome(
            ["Dandân"],
            ["Dandân"],
            "AA",
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
        outcomes = [_outcome(
            ["Mosswood Dreadknight"],
            ["Mosswood Dreadknight"],
            "AA",
        )]
        examples, table = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 1
        idx = examples[0].winner_indices[0].item()
        np.testing.assert_array_equal(table.embedding.weight[idx].numpy(), emb)

    def test_missing_card_skips_match(self, tmp_path):
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        outcomes = [_outcome(["nonexistent"], ["also_missing"], "AA")]
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
            _outcome(["card_a"], ["card_b"], "AA"),
            _outcome(["card_a"], ["card_b"], "ABA"),
        ]
        examples, table = build_training_examples(outcomes, cards_dir)
        assert len(examples) == 2
        assert table.num_cards == 2
        assert examples[0].winner_indices.tolist() == examples[1].winner_indices.tolist()


class TestEmbeddingTable:
    def test_lookup_returns_seeded_vectors(self):
        vecs = torch.randn(3, D_MODEL)
        table = EmbeddingTable(vecs, {"a": 0, "b": 1, "c": 2})
        out = table(torch.tensor([0, 2]))
        torch.testing.assert_close(out[0], vecs[0])
        torch.testing.assert_close(out[1], vecs[2])


class TestSetTextVectors:
    """``set_text_vectors`` splices encoder text vectors into the leading
    ``2 * encoder_d_model`` columns while leaving the trailing
    deterministic-feature slice untouched, and gradients flow back through
    the source tensor (FR-007, T002, T012)."""

    def _table(self):
        encoder_d_model = 8
        text_dim = 2 * encoder_d_model
        d_model = text_dim + 32  # FEATURE_COUNT
        baseline = torch.randn(4, d_model)
        return baseline, EmbeddingTable(baseline, {"a": 0, "b": 1, "c": 2, "d": 3})

    def test_text_columns_overwritten(self):
        baseline, table = self._table()
        encoder_d_model = 8
        text_dim = 2 * encoder_d_model
        indices = torch.tensor([1, 3], dtype=torch.long)
        text_vectors = torch.zeros(2, text_dim)

        table.set_text_vectors(indices, text_vectors)
        out = table(indices)
        torch.testing.assert_close(out[:, :text_dim], text_vectors)

    def test_deterministic_slice_untouched(self):
        baseline, table = self._table()
        encoder_d_model = 8
        text_dim = 2 * encoder_d_model
        indices = torch.tensor([1, 3], dtype=torch.long)
        text_vectors = torch.zeros(2, text_dim)

        table.set_text_vectors(indices, text_vectors)
        out = table(indices)
        torch.testing.assert_close(out[:, text_dim:], baseline[indices, text_dim:])

    def test_gradient_flows_back_through_text_vectors(self):
        baseline, table = self._table()
        encoder_d_model = 8
        text_dim = 2 * encoder_d_model
        indices = torch.tensor([1, 3], dtype=torch.long)
        # ``text_vectors`` is a derived tensor with grad-tracked source so we
        # can check that the source's grad gets populated through the splice.
        source = torch.randn(2, text_dim, requires_grad=True)
        text_vectors = source * 2.0

        table.set_text_vectors(indices, text_vectors)
        out = table(indices)
        loss = out[:, :text_dim].sum()
        loss.backward()

        assert source.grad is not None
        assert source.grad.shape == source.shape
        assert source.grad.abs().sum().item() > 0


class TestCollateFunction:
    def test_variable_length_padding(self):
        """Collate should pad to max length with boolean masks."""
        ex1 = MatchTrainingExample(
            winner_indices=torch.arange(5, dtype=torch.long),
            loser_indices=torch.arange(3, dtype=torch.long),
        )
        ex2 = MatchTrainingExample(
            winner_indices=torch.arange(8, dtype=torch.long),
            loser_indices=torch.arange(6, dtype=torch.long),
        )

        batch = collate_training_examples([ex1, ex2])
        assert batch.winner_indices.shape == (2, 8)
        assert batch.loser_indices.shape == (2, 6)
        assert batch.winner_mask.shape == (2, 8)
        assert batch.loser_mask.shape == (2, 6)

    def test_mask_marks_real_cards_true(self):
        ex1 = MatchTrainingExample(
            winner_indices=torch.arange(3, dtype=torch.long),
            loser_indices=torch.arange(2, dtype=torch.long),
        )
        ex2 = MatchTrainingExample(
            winner_indices=torch.arange(5, dtype=torch.long),
            loser_indices=torch.arange(4, dtype=torch.long),
        )

        batch = collate_training_examples([ex1, ex2])
        assert batch.winner_mask[0, :3].all()
        assert not batch.winner_mask[0, 3:].any()
        assert batch.winner_mask[1, :5].all()
