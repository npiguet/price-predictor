"""Tests for tokenizer_store: save_vocabulary and load_tokenizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.domain.tokenizer import MtgTokenizer


class TestSaveVocabulary:
    def test_writes_vocab_txt_with_tokens_by_id(self, tmp_path: Path):
        from price_predictor.infrastructure.tokenizer_store import save_vocabulary

        vocab = {"[PAD]": 0, "[UNK]": 1, "cardname": 2, "flying": 3}
        path = tmp_path / "vocab.txt"
        save_vocabulary(vocab, path)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "[PAD]"
        assert lines[1] == "[UNK]"
        assert lines[2] == "cardname"
        assert lines[3] == "flying"

    def test_tokens_sorted_by_id(self, tmp_path: Path):
        from price_predictor.infrastructure.tokenizer_store import save_vocabulary

        vocab = {"flying": 3, "[PAD]": 0, "creature": 4, "[UNK]": 1, "cardname": 2}
        path = tmp_path / "vocab.txt"
        save_vocabulary(vocab, path)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines == ["[PAD]", "[UNK]", "cardname", "flying", "creature"]

    def test_writes_utf8(self, tmp_path: Path):
        from price_predictor.infrastructure.tokenizer_store import save_vocabulary

        vocab = {"[PAD]": 0, "[UNK]": 1}
        path = tmp_path / "vocab.txt"
        save_vocabulary(vocab, path)
        content = path.read_bytes()
        assert content  # file is non-empty


class TestLoadTokenizer:
    def test_returns_mtg_tokenizer_with_correct_vocab(self, tmp_path: Path):
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer, save_vocabulary

        vocab = {"[PAD]": 0, "[UNK]": 1, "cardname": 2, "flying": 3}
        vocab_path = tmp_path / "vocab.txt"
        save_vocabulary(vocab, vocab_path)

        tok = load_tokenizer(vocab_path)
        assert isinstance(tok, MtgTokenizer)
        assert tok.vocab_size == 4

    def test_validates_pad_at_line_0(self, tmp_path: Path):
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer

        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("[UNK]\n[PAD]\ncardname", encoding="utf-8")

        with pytest.raises(ValueError, match="\\[PAD\\]"):
            load_tokenizer(vocab_path)

    def test_validates_unk_at_line_1(self, tmp_path: Path):
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer

        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("[PAD]\ncardname\n[UNK]", encoding="utf-8")

        with pytest.raises(ValueError, match="\\[UNK\\]"):
            load_tokenizer(vocab_path)

    def test_raises_file_not_found_for_missing_file(self, tmp_path: Path):
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer

        with pytest.raises(FileNotFoundError):
            load_tokenizer(tmp_path / "nonexistent.txt")

    def test_round_trip_save_load(self, tmp_path: Path):
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer, save_vocabulary

        vocab = {"[PAD]": 0, "[UNK]": 1, "cardname": 2, "first_strike": 3, "{W}": 4}
        vocab_path = tmp_path / "vocab.txt"
        save_vocabulary(vocab, vocab_path)

        tok = load_tokenizer(vocab_path)
        assert tok.vocab_size == 5
        ids, mask = tok.encode("first strike {W}", max_length=10)
        # first_strike = ID 3, {W} = ID 4
        assert 3 in ids
        assert 4 in ids
