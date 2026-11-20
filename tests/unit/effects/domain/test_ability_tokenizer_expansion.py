"""Keyword expansion (forced, probabilistic, host-bodied, unknown) and the
multi-word index the merge step looks candidates up in."""

from __future__ import annotations

import random

import pytest

from effects.domain.ability_tokenizer import HOST_BODIED_KEYWORDS, AbilityTokenizer
from effects.domain.provenance import RoleSpan
from tests.unit.effects.domain import ability_tokenizer_support
from tests.unit.effects.domain.ability_tokenizer_support import _definitions, _vocab

# The shared fixture: pytest collects it from this module's namespace.
tokenizer = ability_tokenizer_support.tokenizer


class TestKeywordExpansion:
    def test_a_known_keyword_is_left_alone_at_probability_zero(self, tokenizer):
        tokens = tokenizer.tokenize("flying")
        assert [t.text for t in tokenizer.expand_keywords(tokens)] == ["flying"]

    def test_a_known_keyword_expands_at_probability_one(self, tokenizer):
        tokens = tokenizer.tokenize("flying")
        expanded = tokenizer.expand_keywords(tokens, probability=1.0)
        assert [t.text for t in expanded][:3] == ["cant", "be", "blocked"]

    def test_a_probability_the_draws_decide_needs_an_rng(self, tokenizer):
        with pytest.raises(ValueError, match="needs an rng"):
            tokenizer.expand_keywords(tokenizer.tokenize("flying"), probability=0.5)

    def test_a_forced_keyword_expands_at_probability_zero(self, tokenizer):
        """``--withhold-keyword``: the model reads the definition, never the token."""
        tokens = tokenizer.tokenize("flying")
        expanded = tokenizer.expand_keywords(tokens, forced=frozenset({"flying"}))
        assert [t.text for t in expanded][:3] == ["cant", "be", "blocked"]
        assert all(t.expanded_from == "flying" for t in expanded)

    def test_forcing_one_keyword_leaves_the_others_alone(self, tokenizer):
        tokens = tokenizer.tokenize("flying")
        kept = tokenizer.expand_keywords(tokens, forced=frozenset({"vigilance"}))
        assert [t.text for t in kept] == ["flying"]

    def test_an_unknown_keyword_always_expands(self, tokenizer):
        """Forced expansion is what makes a set the vocabulary predates readable."""
        unknown = AbilityTokenizer(_vocab(), _definitions())
        assert not unknown.is_known("vigilance")
        tokens = unknown.tokenize("vigilance")
        expanded = unknown.expand_keywords(tokens, probability=0.0)
        assert [t.text for t in expanded][:2] == ["attacking", "doesnt"]

    def test_an_expanded_token_records_the_keyword_it_came_from(self, tokenizer):
        expanded = tokenizer.expand_keywords(
            tokenizer.tokenize("flying"), probability=1.0,
        )
        assert all(t.expanded_from == "flying" for t in expanded)

    def test_an_expanded_token_has_no_offset_in_the_source(self, tokenizer):
        expanded = tokenizer.expand_keywords(
            tokenizer.tokenize("flying"), probability=1.0,
        )
        assert all(t.start is None and t.end is None for t in expanded)

    def test_an_expanded_token_inherits_the_keywords_role(self, tokenizer):
        text = "flying"
        tokens = tokenizer.tokenize(text, (RoleSpan(0, len(text), "effect"),))
        expanded = tokenizer.expand_keywords(tokens, probability=1.0)
        assert all(t.role == "effect" for t in expanded)

    def test_a_keyword_inside_an_expansion_stays_a_token(self, tokenizer):
        """Flying's own reminder text names flying; expanding it again would
        bury the line's text under nested definitions."""
        once = tokenizer.expand_keywords(
            tokenizer.tokenize("flying"), probability=1.0,
        )
        twice = tokenizer.expand_keywords(once, probability=1.0)
        assert [t.text for t in twice] == [t.text for t in once]

    def test_a_host_bodied_keyword_never_expands(self, tokenizer):
        """Its body is printed on the card; a generic sentence would replace it."""
        assert "chapter" in HOST_BODIED_KEYWORDS
        tokens = tokenizer.tokenize("chapter")
        assert tokenizer.expandable(tokens[0]) is False
        expanded = tokenizer.expand_keywords(tokens, probability=1.0)
        assert [t.text for t in expanded] == ["chapter"]

    def test_a_keyword_with_no_template_is_not_expandable(self, tokenizer):
        tokens = tokenizer.tokenize("wither")
        assert tokenizer.expandable(tokens[0]) is False

    def test_without_an_instance_the_placeholder_is_dropped(self, tokenizer):
        """The generic wording, not a tokenized "%s"."""
        expanded = tokenizer.expand_keywords(
            tokenizer.tokenize("cycling"), probability=1.0,
        )
        texts = [t.text for t in expanded]
        assert "%" not in texts
        assert texts[:2] == [",", "discard"]

    def test_expansion_is_reproducible_under_a_seeded_rng(self, tokenizer):
        tokens = tokenizer.tokenize("flying vigilance")
        first = tokenizer.expand_keywords(
            tokens, probability=0.5, rng=random.Random(7),
        )
        second = tokenizer.expand_keywords(
            tokens, probability=0.5, rng=random.Random(7),
        )
        assert [t.text for t in first] == [t.text for t in second]

    def test_a_tokenizer_without_definitions_expands_nothing(self):
        bare = AbilityTokenizer(_vocab(("flying",)))
        tokens = bare.tokenize("flying")
        assert bare.expandable(tokens[0]) is False
        assert bare.expand_keywords(tokens, probability=1.0) == tokens


class TestMultiWordIndexing:
    """``_multi_word_at`` looks candidates up by first word (perf fix).

    The vocabulary below stacks four multi-word entries that share a first
    word ("first"), plus the two-word entries the base fixture already
    carries, so a bucket has to preserve its longest-first order and a
    mismatch on the second word has to fall through to the next candidate in
    the same bucket rather than a different one.
    """

    @pytest.fixture
    def indexed(self) -> AbilityTokenizer:
        return AbilityTokenizer(
            _vocab(("first_blood", "first_strike_damage", "damage", "blood")),
            _definitions(),
        )

    def test_the_longest_same_bucket_entry_wins(self, indexed):
        text = "you gain first strike damage"
        texts = [t.text for t in indexed.tokenize(text)]
        assert "first_strike_damage" in texts
        assert "first_strike" not in texts

    def test_a_first_word_match_with_a_differing_rest_falls_through(self, indexed):
        """"first" also opens "first_blood", but the text says "first strike":
        that candidate must be rejected and "first_strike" tried next."""
        texts = [t.text for t in indexed.tokenize("gains first strike")]
        assert "first_strike" in texts
        assert "first_blood" not in texts

    def test_an_entry_that_would_overrun_the_text_end_is_skipped(self, indexed):
        """"first_strike_damage" needs a third word "damage" that isn't there;
        the shorter "first_strike" in the same bucket must still match."""
        texts = [t.text for t in indexed.tokenize("creature gains first strike")]
        assert texts[-1] == "first_strike"

    def test_a_merged_three_part_tokens_span_covers_first_to_last_word(self, indexed):
        text = "gains first strike damage now"
        merged = next(
            t for t in indexed.tokenize(text) if t.text == "first_strike_damage"
        )
        assert text[merged.start:merged.end] == "first strike damage"

    def test_a_lone_word_with_no_multi_word_bucket_is_unaffected(self, indexed):
        texts = [t.text for t in indexed.tokenize("deal damage")]
        assert texts == ["deal", "damage"]
