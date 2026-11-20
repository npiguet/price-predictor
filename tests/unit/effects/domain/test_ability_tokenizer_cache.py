"""The tokenize cache (equal but distinct lists, keyed on the spans, capped)
and the shape of a token."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.ability_tokenizer import AbilityTokenizer, Token
from effects.domain.provenance import RoleSpan
from tests.unit.effects.domain import ability_tokenizer_support

# The shared fixture: pytest collects it from this module's namespace.
tokenizer = ability_tokenizer_support.tokenizer


class TestTokenizeCache:
    def test_repeated_tokenizing_returns_equal_but_distinct_lists(self, tokenizer):
        first = tokenizer.tokenize("draw a card")
        second = tokenizer.tokenize("draw a card")
        assert first == second
        assert first is not second

    def test_mutating_one_result_does_not_affect_the_other(self, tokenizer):
        first = tokenizer.tokenize("draw a card")
        second = tokenizer.tokenize("draw a card")
        first.append(Token(text="extra", token_id=0))
        assert second[-1].text != "extra"
        assert len(second) == 3

    def test_the_cache_keys_on_the_spans_as_well_as_the_text(self, tokenizer, monkeypatch):
        """The prose surface tokenizes with spans on every step (FR-069), so
        they are cached too, and never mistaken for another span set's."""
        calls = []
        original = AbilityTokenizer._split_with_offsets

        def counting(self, text):
            calls.append(text)
            return original(self, text)

        monkeypatch.setattr(AbilityTokenizer, "_split_with_offsets", counting)
        spans = (RoleSpan(0, 4, "cost"),)

        tokenizer.tokenize("draw a card")
        tokenizer.tokenize("draw a card")
        assert calls == ["draw a card"]

        assert tokenizer.tokenize("draw a card", spans)[0].role == "cost"
        assert tokenizer.tokenize("draw a card", spans)[0].role == "cost"
        assert calls == ["draw a card", "draw a card"]

        assert tokenizer.tokenize("draw a card")[0].role is None
        assert calls == ["draw a card", "draw a card"]

    def test_the_cache_is_capped(self, tokenizer, monkeypatch):
        monkeypatch.setattr(AbilityTokenizer, "_TOKENIZE_CACHE_CAP", 2)
        first = tokenizer.tokenize("draw a card")
        tokenizer.tokenize("deal damage")
        tokenizer.tokenize("put counter")
        assert len(tokenizer._tokenize_cache) <= 2
        assert [t.text for t in tokenizer.tokenize("draw a card")] == [
            t.text for t in first
        ]

    def test_an_expansion_is_split_once_and_takes_each_callers_role(
        self, tokenizer, monkeypatch,
    ):
        calls: list[str] = []
        original = AbilityTokenizer._split_with_offsets

        def counting(self, text):
            calls.append(text)
            return original(self, text)

        monkeypatch.setattr(AbilityTokenizer, "_split_with_offsets", counting)
        flying = tokenizer.tokenize("flying")
        spanned = tokenizer.tokenize("flying", (RoleSpan(0, 6, "effect"),))
        calls.clear()

        plain = tokenizer.expand_keywords(flying, probability=1.0)
        roled = tokenizer.expand_keywords(spanned, probability=1.0)
        again = tokenizer.expand_keywords(flying, probability=1.0)

        assert len(calls) == 1
        assert again == plain
        assert {t.role for t in plain} == {None}
        assert {t.role for t in roled} == {"effect"}
        assert [replace(t, role=None) for t in roled] == plain
        assert {t.expanded_from for t in plain} == {"flying"}


class TestTokenShape:
    def test_a_token_is_immutable(self, tokenizer):
        token = tokenizer.tokenize("draw")[0]
        with pytest.raises(AttributeError):
            token.text = "discard"

    def test_a_bare_token_defaults_every_optional_field(self):
        token = Token(text="draw", token_id=5)
        assert (token.start, token.end, token.role) == (None, None, None)
        assert (token.expanded_from, token.number) == (None, None)
