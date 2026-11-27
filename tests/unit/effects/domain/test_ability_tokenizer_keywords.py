"""Keyword lines on the prose surface and the multi-word keywords' token forms. Definitions are the
real ones in ``keyword-definitions.json``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.domain.ability_tokenizer import AbilityTokenizer, keyword_token
from effects.domain.keyword_definitions import KeywordDefinition
from price_predictor.domain.keyword_tokens import multi_word_keywords

_DEFINITIONS_PATH = (
    Path(__file__).resolve().parents[3] / "fixtures" / "effects" / "keyword-definitions.json"
)


def _definitions() -> dict[str, KeywordDefinition]:
    raw = json.loads(_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    return {
        name: KeywordDefinition(
            keyword=name,
            reminder_template=entry.get("reminder_template"),
            formatter=entry.get("formatter"),
        )
        for name, entry in raw.items()
    }


def _tokenizer(extra: tuple[str, ...] = ()) -> AbilityTokenizer:
    """A vocabulary of the specials, the multi-word keywords and every word of
    every template, and the punctuation they read."""
    definitions = _definitions()
    words = {
        word
        for d in definitions.values()
        for word in (d.reminder_template or "").lower().replace("%", " ").split()
    }
    tokens = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", *multi_word_keywords(definitions)]
    tokens += sorted(words) + ["{2}", "{w/b}", "-", *extra]
    vocab = {t: i for i, t in enumerate(dict.fromkeys(tokens))}
    return AbilityTokenizer(vocab, definitions, surface="prose")


class TestProseKeywordLines:
    def test_a_ward_line_on_the_prose_surface_carries_its_values(self):
        line = _tokenizer().keywords.line_of("ward {2}")
        assert (line.name, line.details, line.name_end) == ("Ward", "2", len("ward"))

    def test_a_hybrid_ward_line_reads_its_shard_back_to_forges_spelling(self):
        assert _tokenizer().keywords.line_of("ward {W/B}").details == "W/B"

    def test_a_ward_line_expands_whole_with_its_cost_filled(self):
        tokenizer = _tokenizer()
        expanded = tokenizer.expand_keywords(tokenizer.tokenize("ward {2}"), probability=1.0)
        assert [t.text for t in expanded][-3:] == ["pays", "{2}", "."]

    def test_protection_from_red_is_a_line_with_its_quality(self):
        line = _tokenizer().keywords.line_of("protection from red")
        assert (line.name, line.details, line.name_end) == ("Protection", "red", len("protection"))

    def test_hexproof_from_black_does_not_expand_to_plain_hexproof(self):
        tokenizer = _tokenizer()
        line = tokenizer.keywords.line_of("hexproof from black")
        assert (line.name, line.details) == ("Hexproof", "black")
        expanded = tokenizer.expand_keywords(
            tokenizer.tokenize("hexproof from black"), probability=1.0,
        )
        text = " ".join(t.text for t in expanded)
        assert "black spells" in text
        assert "opponents control" in text and "target of spells" not in text

    @pytest.mark.parametrize("text", [
        "flying", "flying, first strike", "ward {2} and trample",
        "creatures you control have ward {2}", "target creature gains protection from red",
    ])
    def test_prose_that_only_starts_with_a_keyword_is_no_line_of_it(self, text):
        line = _tokenizer().keywords.line_of(text)
        assert line is None or text == "flying"


class TestMultiWordTokenForms:
    @pytest.mark.parametrize(("name", "form"), [
        ("Beam me up", "beam_me_up"),
        ("Jump-start", "jump_start"),
        ("Web-slinging", "web_slinging"),
    ])
    def test_the_three_keywords_have_a_form_the_grammar_can_produce(self, name, form):
        assert keyword_token(name) == form
        assert form in multi_word_keywords(_definitions())

    @pytest.mark.parametrize("text", ["beam me up", "jump-start", "web-slinging"])
    def test_the_three_keywords_merge_and_expand(self, text):
        tokenizer = _tokenizer()
        tokens = tokenizer.tokenize(text)
        assert len(tokens) == 1 and tokenizer.keywords.name_of_token(tokens[0].text)
        assert not tokenizer.must_expand(tokens[0])
        expanded = tokenizer.expand_keywords(tokens, probability=1.0)
        assert expanded and all(t.expanded_from for t in expanded)
