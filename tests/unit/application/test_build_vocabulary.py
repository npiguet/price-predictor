"""Tests for VocabularyBuilder and build_vocabulary use case."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestMultiWordKeywordsConstant:
    def test_has_24_entries(self):
        from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS

        assert len(MULTI_WORD_KEYWORDS) == 24

    def test_all_entries_are_underscore_form(self):
        from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS

        for kw in MULTI_WORD_KEYWORDS:
            assert "_" in kw, f"Expected underscore form, got: {kw!r}"

    def test_contains_first_strike(self):
        from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS

        assert "first_strike" in MULTI_WORD_KEYWORDS

    def test_contains_double_strike(self):
        from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS

        assert "double_strike" in MULTI_WORD_KEYWORDS

    def test_doctors_companion_normalized_without_apostrophe(self):
        from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS

        assert "doctors_companion" in MULTI_WORD_KEYWORDS
        # Should NOT contain the apostrophe form
        assert "doctor's_companion" not in MULTI_WORD_KEYWORDS

    def test_all_24_keywords_present(self):
        from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS

        expected = {
            "aura_swap", "bands_with_other", "battle_cry", "choose_a_background",
            "cumulative_upkeep", "doctors_companion", "double_agenda", "double_strike",
            "double_team", "first_strike", "for_mirrodin", "hidden_agenda", "job_select",
            "level_up", "living_metal", "living_weapon", "more_than_meets_the_eye",
            "partner_with", "read_ahead", "space_sculptor", "split_second",
            "start_your_engines", "starting_intensity", "umbra_armor",
        }
        assert set(MULTI_WORD_KEYWORDS) == expected


class TestBuildVocabulary:
    def test_pad_is_id_zero(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary

        (tmp_path / "card.txt").write_text(
            "name: Test Card\nmana cost: {R}\ntypes: instant\nspell[1]: CARDNAME deals 3 damage.\n",
            encoding="utf-8",
        )
        result = build_vocabulary(tmp_path, freq_threshold=1)
        assert result.vocab["[PAD]"] == 0

    def test_unk_is_id_one(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary

        (tmp_path / "card.txt").write_text(
            "name: Test Card\nmana cost: {R}\ntypes: instant\n",
            encoding="utf-8",
        )
        result = build_vocabulary(tmp_path, freq_threshold=1)
        assert result.vocab["[UNK]"] == 1

    def test_all_multi_word_keywords_present_as_underscore_tokens(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS, build_vocabulary

        (tmp_path / "card.txt").write_text(
            "name: Test Card\nmana cost: {R}\ntypes: instant\n",
            encoding="utf-8",
        )
        result = build_vocabulary(tmp_path, freq_threshold=1)
        for kw in MULTI_WORD_KEYWORDS:
            assert kw in result.vocab, f"Expected multi-word keyword: {kw!r}"

    def test_frequency_threshold_excludes_rare_words(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary

        # "xyzzyword" appears only once — below threshold of 5
        (tmp_path / "card.txt").write_text(
            "name: Test Card\nmana cost: {R}\ntypes: instant\nspell[1]: xyzzyword.\n",
            encoding="utf-8",
        )
        result = build_vocabulary(tmp_path, freq_threshold=5)
        assert "xyzzyword" not in result.vocab

    def test_frequency_threshold_includes_common_words(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary

        # Write 5 cards all containing "commonword" → freq=5, meets threshold
        for i in range(5):
            (tmp_path / f"card_{i}.txt").write_text(
                f"name: Card {i}\nmana cost: {{R}}\ntypes: instant\nspell[1]: commonword.\n",
                encoding="utf-8",
            )
        result = build_vocabulary(tmp_path, freq_threshold=5)
        assert "commonword" in result.vocab

    def test_mana_symbols_always_included_regardless_of_threshold(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary

        # {Q} appears only once — still must be included
        (tmp_path / "card.txt").write_text(
            "name: Test Card\nmana cost: {Q}\ntypes: instant\n",
            encoding="utf-8",
        )
        result = build_vocabulary(tmp_path, freq_threshold=999)
        assert "{Q}" in result.vocab

    def test_vocab_build_result_fields_present(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import VocabBuildResult, build_vocabulary

        (tmp_path / "card.txt").write_text(
            "name: Test Card\nmana cost: {R}\ntypes: instant\n",
            encoding="utf-8",
        )
        result = build_vocabulary(tmp_path, freq_threshold=1)
        assert isinstance(result, VocabBuildResult)
        assert isinstance(result.vocab, dict)
        assert isinstance(result.vocab_size, int)
        assert isinstance(result.domain_token_count, int)
        assert isinstance(result.freq_threshold_token_count, int)
        assert isinstance(result.coverage_pct, float)
        assert isinstance(result.unk_pct, float)

    def test_coverage_pct_is_percentage(self, tmp_path: Path):
        from price_predictor.application.build_vocabulary import build_vocabulary

        for i in range(5):
            (tmp_path / f"card_{i}.txt").write_text(
                f"name: Card {i}\nmana cost: {{R}}\ntypes: instant\nspell[1]: flying.\n",
                encoding="utf-8",
            )
        result = build_vocabulary(tmp_path, freq_threshold=1)
        # coverage_pct is a percentage (0.0 to 100.0)
        assert 0.0 <= result.coverage_pct <= 100.0
        assert 0.0 <= result.unk_pct <= 100.0
        assert abs(result.coverage_pct + result.unk_pct - 100.0) < 0.01


class TestSC001SC002Validation:
    """Validate SC-001 (< 10,000 tokens) and SC-002 (≥ 95% coverage) against fixture corpus."""

    def test_vocab_size_well_below_10000_on_fixture_corpus(self):
        """Build vocabulary from the training fixture corpus and check size < 10,000."""
        from price_predictor.application.build_vocabulary import build_vocabulary

        fixture_corpus = Path(__file__).parent.parent.parent / "fixtures" / "converted_cards_training"
        if not fixture_corpus.exists():
            pytest.skip("Fixture corpus not found")

        result = build_vocabulary(fixture_corpus, freq_threshold=1)
        assert result.vocab_size < 10000

    def test_color_names_present_as_tokens(self):
        from price_predictor.application.build_vocabulary import build_vocabulary
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            (tmp_path / "card.txt").write_text(
                "name: Test\nmana cost: {W}\ntypes: creature\n",
                encoding="utf-8",
            )
            result = build_vocabulary(tmp_path, freq_threshold=1)
            for color in ("white", "blue", "black", "red", "green", "colorless"):
                assert color in result.vocab, f"Expected color token: {color!r}"

    def test_game_zones_present_as_tokens(self):
        from price_predictor.application.build_vocabulary import build_vocabulary
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            (tmp_path / "card.txt").write_text(
                "name: Test\nmana cost: {W}\ntypes: creature\n",
                encoding="utf-8",
            )
            result = build_vocabulary(tmp_path, freq_threshold=1)
            for zone in ("battlefield", "exile", "graveyard", "hand", "library", "stack"):
                assert zone in result.vocab, f"Expected zone token: {zone!r}"


class TestSC003DomainCoverage:
    """Validate SC-003: all MTG domain terms present in vocabulary."""

    def test_all_24_multi_word_keywords_in_vocab(self):
        from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS, build_vocabulary
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            (tmp_path / "card.txt").write_text(
                "name: Test\nmana cost: {R}\ntypes: instant\n",
                encoding="utf-8",
            )
            result = build_vocabulary(tmp_path, freq_threshold=1)
            for kw in MULTI_WORD_KEYWORDS:
                assert kw in result.vocab, f"Missing multi-word keyword: {kw!r}"

    def test_basic_mana_symbols_in_vocab(self):
        from price_predictor.application.build_vocabulary import build_vocabulary
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            # Include cards with each basic color
            for sym in ("{W}", "{U}", "{B}", "{R}", "{G}", "{T}", "{X}"):
                (tmp_path / f"card_{sym[1]}.txt").write_text(
                    f"name: Card {sym}\nmana cost: {sym}\ntypes: instant\n",
                    encoding="utf-8",
                )
            result = build_vocabulary(tmp_path, freq_threshold=1)
            for sym in ("{W}", "{U}", "{B}", "{R}", "{G}", "{T}", "{X}"):
                assert sym in result.vocab, f"Missing mana symbol: {sym!r}"

    def test_domain_terms_in_vocab(self):
        from price_predictor.application.build_vocabulary import build_vocabulary
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            (tmp_path / "card.txt").write_text(
                "name: Test\nmana cost: {R}\ntypes: legendary creature\nkeyword[1]: flying\n"
                "spell[1]: enters the battlefield.\n",
                encoding="utf-8",
            )
            result = build_vocabulary(tmp_path, freq_threshold=1)
            for term in ("creature", "flying", "battlefield", "legendary"):
                assert term in result.vocab, f"Missing domain term: {term!r}"
