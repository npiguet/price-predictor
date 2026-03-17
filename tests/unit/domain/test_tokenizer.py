"""Tests for MtgTokenizer domain class."""

from __future__ import annotations

import pytest

from price_predictor.domain.tokenizer import MtgTokenizer


def _make_tokenizer() -> MtgTokenizer:
    """Build a small tokenizer from the fixture vocab."""
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "cardname": 2,
        "creature": 3,
        "flying": 4,
        "first_strike": 5,
        "{W}": 6,
        "{R}": 7,
        "{2}": 8,
        "vigilance": 9,
        "double_strike": 10,
        "strike": 11,
        "double": 12,
    }
    return MtgTokenizer(vocab)


class TestMtgTokenizerConstruction:
    def test_pad_id_is_zero(self):
        tok = _make_tokenizer()
        assert tok.PAD_ID == 0

    def test_unk_id_is_one(self):
        tok = _make_tokenizer()
        assert tok.UNK_ID == 1

    def test_vocab_size_property(self):
        tok = _make_tokenizer()
        assert tok.vocab_size == 13

    def test_pad_constant_string(self):
        assert MtgTokenizer.PAD == "[PAD]"

    def test_unk_constant_string(self):
        assert MtgTokenizer.UNK == "[UNK]"


class TestMtgTokenizerTokenize:
    def test_lowercases_plain_text(self):
        tok = _make_tokenizer()
        tokens = tok._tokenize("Flying Vigilance")
        assert "flying" in tokens
        assert "vigilance" in tokens

    def test_mana_symbols_stay_uppercase(self):
        tok = _make_tokenizer()
        tokens = tok._tokenize("{W}{R}")
        assert "{W}" in tokens
        assert "{R}" in tokens
        assert "{w}" not in tokens
        assert "{r}" not in tokens

    def test_multi_word_keyword_replaced_before_split(self):
        tok = _make_tokenizer()
        tokens = tok._tokenize("first strike")
        assert "first_strike" in tokens
        assert "first" not in tokens
        assert "strike" not in tokens

    def test_double_strike_takes_priority_over_strike(self):
        tok = _make_tokenizer()
        tokens = tok._tokenize("double strike")
        assert "double_strike" in tokens
        assert "double" not in tokens
        assert "strike" not in tokens

    def test_cardname_lowercased(self):
        tok = _make_tokenizer()
        tokens = tok._tokenize("CARDNAME deals damage")
        assert "cardname" in tokens

    def test_numbers_extracted(self):
        tok = _make_tokenizer()
        tokens = tok._tokenize("deals 3 damage")
        assert "3" in tokens

    def test_punctuation_extracted(self):
        tok = _make_tokenizer()
        tokens = tok._tokenize("flying, vigilance")
        assert "," in tokens


class TestMtgTokenizerEncode:
    def test_flying_vigilance_produces_two_domain_tokens(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("flying, vigilance", max_length=10)
        # flying (4) and vigilance (9) are both in-vocab
        assert 4 in ids  # flying
        assert 9 in ids  # vigilance

    def test_first_strike_encodes_as_single_token(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("first strike", max_length=10)
        assert 5 in ids  # first_strike ID

    def test_mana_symbols_encoded_uppercase(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("{W}{W}", max_length=10)
        assert ids[0] == 6  # {W}
        assert ids[1] == 6  # {W} again

    def test_cardname_encodes_correctly(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("CARDNAME", max_length=10)
        assert 2 in ids  # cardname ID

    def test_unknown_word_maps_to_unk_id(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("xyzzy", max_length=10)
        assert ids[0] == tok.UNK_ID

    def test_padding_fills_to_max_length(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("flying", max_length=10)
        assert len(ids) == 10
        assert len(mask) == 10
        assert ids[-1] == tok.PAD_ID

    def test_attention_mask_ones_for_real_tokens(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("flying", max_length=10)
        assert mask[0] == 1
        assert mask[-1] == 0

    def test_attention_mask_zeros_for_padding(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("flying", max_length=10)
        n_real = sum(1 for m in mask if m == 1)
        n_pad = sum(1 for m in mask if m == 0)
        assert n_real >= 1
        assert n_pad >= 1
        assert n_real + n_pad == 10

    def test_output_lengths_match_max_length(self):
        tok = _make_tokenizer()
        for ml in [4, 8, 16]:
            ids, mask = tok.encode("flying", max_length=ml)
            assert len(ids) == ml
            assert len(mask) == ml


class TestMtgTokenizerEncodeEdgeCases:
    def test_truncation_at_max_length(self):
        tok = _make_tokenizer()
        long_text = "flying " * 100
        ids, mask = tok.encode(long_text, max_length=8)
        assert len(ids) == 8
        assert len(mask) == 8

    def test_empty_string_input(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("", max_length=8)
        assert len(ids) == 8
        assert all(i == tok.PAD_ID for i in ids)
        assert all(m == 0 for m in mask)

    def test_text_with_only_mana_symbols(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("{W}{R}", max_length=8)
        assert ids[0] == 6  # {W}
        assert ids[1] == 7  # {R}
        assert ids[2] == tok.PAD_ID

    def test_double_strike_not_split_as_double_plus_strike(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("double strike", max_length=10)
        # double_strike should be ID 10, not double (12) + strike (11)
        assert 10 in ids
        assert 12 not in ids
        assert 11 not in ids


class TestMtgTokenizerDecode:
    def test_decode_stops_at_pad(self):
        tok = _make_tokenizer()
        token_ids = [4, 9, 0, 6, 7]  # flying, vigilance, PAD, {W}, {R}
        result = tok.decode(token_ids)
        assert "{W}" not in result
        assert "flying" in result

    def test_decode_maps_unk_to_unk_string(self):
        tok = _make_tokenizer()
        token_ids = [tok.UNK_ID]
        result = tok.decode(token_ids)
        assert "[UNK]" in result

    def test_round_trip_for_in_vocab_text(self):
        tok = _make_tokenizer()
        ids, mask = tok.encode("flying vigilance", max_length=20)
        result = tok.decode(ids)
        assert "flying" in result
        assert "vigilance" in result
