"""The ability tokenizer (T063).

Three properties the shared ``MtgTokenizer`` cannot provide and the encoder
depends on: offsets that line up with the sidecar's role spans, whole-token
lookup with no subword fallback, and forced expansion of keywords the
vocabulary has never seen.
"""

from __future__ import annotations

import random

import pytest

from effects.application.extract_keyword_definitions import KeywordDefinition
from effects.domain.ability_tokenizer import (
    HOST_BODIED_KEYWORDS,
    AbilityTokenizer,
    Token,
)
from effects.domain.provenance import RoleSpan

# "flying" and "vigilance" are deliberately absent: the fixture adds them as
# known keywords, and a tokenizer built without them is how the forced-expansion
# path (a keyword the vocabulary predates) is exercised.
_WORDS = (
    "reach draw a card deal damage to target creature or "
    "player whenever you gain life put counter on this discard from hand "
    "blocked except by creatures with cant be attacking doesnt cause tap "
    "cycling"
).split()


def _vocab(extra: tuple[str, ...] = ()) -> dict[str, int]:
    tokens = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]"]
    tokens += ["first_strike", "double_strike", "+1/+1"]
    tokens += _WORDS
    tokens += list(extra)
    tokens += ["{R}", "{T}", "{2}", ":", ",", ".", "1", "2", "3"]
    seen: dict[str, int] = {}
    for token in tokens:
        seen.setdefault(token, len(seen))
    return seen


def _definitions() -> dict[str, KeywordDefinition]:
    return {
        "Flying": KeywordDefinition(
            keyword="Flying",
            reminder_template=(
                "cant be blocked except by creatures with flying or reach"
            ),
        ),
        "Vigilance": KeywordDefinition(
            keyword="Vigilance",
            reminder_template="attacking doesnt cause this creature to tap",
        ),
        "Cycling": KeywordDefinition(
            keyword="Cycling",
            reminder_template="%s, discard this card: draw a card",
        ),
        "Chapter": KeywordDefinition(
            keyword="Chapter", reminder_template="see the card",
        ),
        "Wither": KeywordDefinition(keyword="Wither", reminder_template=None),
    }


@pytest.fixture
def tokenizer() -> AbilityTokenizer:
    return AbilityTokenizer(_vocab(("flying", "vigilance")), _definitions())


class TestSpecialTokens:
    def test_mask_and_cls_are_addressable(self, tokenizer):
        assert tokenizer.mask_id != tokenizer.cls_id
        assert tokenizer.pad_id == 0
        assert tokenizer.unk_id == 1

    def test_vocab_size_is_the_vocabularys(self, tokenizer):
        assert tokenizer.vocab_size == len(_vocab(("flying", "vigilance")))


class TestOffsets:
    def test_every_token_carries_its_span_in_the_source(self, tokenizer):
        text = "draw a card"
        for token in tokenizer.tokenize(text):
            assert text[token.start:token.end].lower() == token.text

    def test_offsets_survive_a_multi_word_merge(self, tokenizer):
        text = "gains first strike until end of turn"
        merged = next(t for t in tokenizer.tokenize(text) if t.text == "first_strike")
        assert text[merged.start:merged.end] == "first strike"

    def test_a_longer_multi_word_entry_wins(self, tokenizer):
        text = "has double strike"
        texts = [t.text for t in tokenizer.tokenize(text)]
        assert "double_strike" in texts
        assert "first_strike" not in texts

    def test_mana_symbols_keep_their_case(self, tokenizer):
        token = tokenizer.tokenize("{R}")[0]
        assert token.text == "{R}"

    def test_a_digit_token_carries_its_value(self, tokenizer):
        token = next(t for t in tokenizer.tokenize("deal 3 damage") if t.number)
        assert token.number == 3.0


class TestRoleSpans:
    def test_a_token_takes_the_role_of_the_span_covering_it(self, tokenizer):
        text = "{T}: draw a card"
        spans = (RoleSpan(0, 4, "cost"), RoleSpan(4, len(text), "effect"))
        tokens = tokenizer.tokenize(text, spans)
        assert tokens[0].role == "cost"
        assert tokens[-1].role == "effect"

    def test_the_more_specific_span_wins_where_two_overlap(self, tokenizer):
        text = "deal damage to target creature"
        spans = (
            RoleSpan(0, len(text), "effect"),
            RoleSpan(text.index("target"), len(text), "target-spec"),
        )
        roles = {t.text: t.role for t in tokenizer.tokenize(text, spans)}
        assert roles["deal"] == "effect"
        assert roles["creature"] == "target-spec"

    def test_a_token_outside_every_span_has_no_role(self, tokenizer):
        text = "draw a card"
        tokens = tokenizer.tokenize(text, (RoleSpan(0, 4, "cost"),))
        assert tokens[0].role == "cost"
        assert tokens[-1].role is None

    def test_no_spans_means_no_roles(self, tokenizer):
        assert all(t.role is None for t in tokenizer.tokenize("draw a card"))


class TestWholeTokenLookup:
    def test_an_unknown_word_becomes_unk(self, tokenizer):
        token = tokenizer.tokenize("bushido")[0]
        assert token.token_id == tokenizer.unk_id
        assert token.text == "bushido"

    def test_there_is_no_subword_fallback(self, tokenizer):
        """"drawing" shares a prefix with "draw" and is still one [UNK]."""
        tokens = tokenizer.tokenize("drawing")
        assert len(tokens) == 1
        assert tokens[0].token_id == tokenizer.unk_id

    def test_a_known_word_resolves_to_its_own_id(self, tokenizer):
        token = tokenizer.tokenize("draw")[0]
        assert token.token_id == tokenizer.id_of("draw")
        assert token.token_id != tokenizer.unk_id


class TestKeywordExpansion:
    def test_a_known_keyword_is_left_alone_at_probability_zero(self, tokenizer):
        tokens = tokenizer.tokenize("flying")
        assert [t.text for t in tokenizer.expand_keywords(tokens)] == ["flying"]

    def test_a_known_keyword_expands_at_probability_one(self, tokenizer):
        tokens = tokenizer.tokenize("flying")
        expanded = tokenizer.expand_keywords(tokens, probability=1.0)
        assert [t.text for t in expanded][:3] == ["cant", "be", "blocked"]

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

    def test_a_parameterized_template_uses_the_instances_values(self, tokenizer):
        expanded = tokenizer.expand_keywords(
            tokenizer.tokenize("cycling"), probability=1.0,
            instance_values={"cycling": ["{2}"]},
        )
        assert expanded[0].text == "{2}"

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

    def test_role_spans_bypass_the_cache(self, tokenizer, monkeypatch):
        calls = []
        original = AbilityTokenizer._split_with_offsets

        def counting(self, text):
            calls.append(text)
            return original(self, text)

        monkeypatch.setattr(AbilityTokenizer, "_split_with_offsets", counting)

        tokenizer.tokenize("draw a card")
        tokenizer.tokenize("draw a card")
        assert calls == ["draw a card"]

        tokenizer.tokenize("draw a card", (RoleSpan(0, 4, "cost"),))
        assert calls == ["draw a card", "draw a card"]

        tokenizer.tokenize("draw a card")
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


class TestTokenShape:
    def test_a_token_is_immutable(self, tokenizer):
        token = tokenizer.tokenize("draw")[0]
        with pytest.raises(AttributeError):
            token.text = "discard"

    def test_a_bare_token_defaults_every_optional_field(self):
        token = Token(text="draw", token_id=5)
        assert (token.start, token.end, token.role) == (None, None, None)
        assert (token.expanded_from, token.number) == (None, None)
