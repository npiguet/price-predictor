"""Integration tests for the vocabulary build → tokenizer pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE_CORPUS = Path(__file__).parent.parent / "fixtures" / "converted_cards_training"


class TestVocabularyBuildPipeline:
    """T008: Run build_vocabulary against fixture corpus and verify round-trip."""

    def test_build_and_round_trip_via_save_load(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer, save_vocabulary

        if not FIXTURE_CORPUS.exists():
            pytest.skip("Fixture corpus not found")

        result = build_vocabulary(FIXTURE_CORPUS, freq_threshold=1)
        vocab_path = tmp_path / "vocab.txt"
        save_vocabulary(result.vocab, vocab_path)

        tok = load_tokenizer(vocab_path)
        assert tok.vocab_size == result.vocab_size

    def test_known_tokens_from_fixtures_appear_in_vocab(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer, save_vocabulary

        if not FIXTURE_CORPUS.exists():
            pytest.skip("Fixture corpus not found")

        result = build_vocabulary(FIXTURE_CORPUS, freq_threshold=1)
        vocab_path = tmp_path / "vocab.txt"
        save_vocabulary(result.vocab, vocab_path)

        tok = load_tokenizer(vocab_path)

        # Tokens that must appear given fixture corpus content
        expected_tokens = ["cardname", "flying", "{R}", "{W}"]
        for token in expected_tokens:
            assert token in tok._vocab, f"Expected token not found: {token!r}"

    def test_first_strike_as_single_token_in_vocab(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer, save_vocabulary

        if not FIXTURE_CORPUS.exists():
            pytest.skip("Fixture corpus not found")

        result = build_vocabulary(FIXTURE_CORPUS, freq_threshold=1)
        vocab_path = tmp_path / "vocab.txt"
        save_vocabulary(result.vocab, vocab_path)

        tok = load_tokenizer(vocab_path)
        # first_strike should be a single token (from fixtures/first_striker.txt)
        assert "first_strike" in tok._vocab


class TestVocabularyToDatasetPipeline:
    """T018: Full pipeline: build vocabulary → save → load tokenizer → construct dataset → verify batch shapes."""

    def test_full_pipeline_batch_shapes(self, tmp_path: Path):
        import torch
        from price_predictor.application.build_vocabulary import build_vocabulary
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer, save_vocabulary
        from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset

        if not FIXTURE_CORPUS.exists():
            pytest.skip("Fixture corpus not found")

        # Build vocabulary from fixtures
        result = build_vocabulary(FIXTURE_CORPUS, freq_threshold=1)
        vocab_path = tmp_path / "vocab.txt"
        save_vocabulary(result.vocab, vocab_path)

        # Load tokenizer
        tok = load_tokenizer(vocab_path)

        # Construct dataset
        card_tuples = [
            ("Lightning Bolt", "name: lightning bolt\nmana cost: {R}\ntypes: instant\n", 2.50),
            ("Serra Angel", "name: serra angel\nmana cost: {3}{W}{W}\ntypes: creature angel\n", 1.00),
        ]
        max_seq_len = 64
        ds = TransformerTrainingDataset(card_tuples, max_seq_len=max_seq_len, tokenizer=tok)

        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids"].shape == (max_seq_len,)
        assert item["attention_mask"].shape == (max_seq_len,)
        assert item["input_ids"].dtype == torch.long

    def test_vocab_size_from_custom_tokenizer_not_30522(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer, save_vocabulary

        if not FIXTURE_CORPUS.exists():
            pytest.skip("Fixture corpus not found")

        result = build_vocabulary(FIXTURE_CORPUS, freq_threshold=1)
        vocab_path = tmp_path / "vocab.txt"
        save_vocabulary(result.vocab, vocab_path)

        tok = load_tokenizer(vocab_path)
        assert tok.vocab_size != 30522
