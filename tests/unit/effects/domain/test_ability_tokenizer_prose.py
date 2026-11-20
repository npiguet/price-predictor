"""The prose surface's tokens: special ids, offsets into the source, role
spans and whole-token lookup with no subword fallback."""

from __future__ import annotations

from effects.domain.provenance import RoleSpan
from tests.unit.effects.domain import ability_tokenizer_support
from tests.unit.effects.domain.ability_tokenizer_support import _vocab

# The shared fixture: pytest collects it from this module's namespace.
tokenizer = ability_tokenizer_support.tokenizer


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
