"""The ability tokenizer (T063).

Three properties the shared ``MtgTokenizer`` cannot provide and the encoder
depends on: offsets that line up with the sidecar's role spans, whole-token
lookup with no subword fallback, and forced expansion of keywords the
vocabulary has never seen.
"""

from __future__ import annotations

import random
from pathlib import Path

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


# ── gen-2: the script surface (FR-007–FR-010) ───────────────────────────


def _script_tokenizer(definitions=None) -> AbilityTokenizer:
    return AbilityTokenizer(
        _vocab(("first_strike",)), definitions or {}, surface="script",
    )


def _texts(tokenizer: AbilityTokenizer, text: str) -> list[str]:
    return [t.text for t in tokenizer.tokenize(text)]


class TestScriptSurfaceTokenization:
    """Spec Story 1 scenarios 4 and 5."""

    def test_keys_and_selectors_split_on_camel_case(self):
        assert _texts(_script_tokenizer(), "ValidTgts$ Creature.nonDragon+YouCtrl") == [
            "valid", "tgts", "$", "creature", ".", "non", "dragon", "+", "you", "ctrl",
        ]

    def test_a_chain_reference_stays_one_token(self):
        assert _texts(_script_tokenizer(), "SubAbility$ SV2") == [
            "sub", "ability", "$", "sv2",
        ]

    def test_a_counter_type_stays_one_token(self):
        assert _texts(_script_tokenizer(), "CounterType$ P1P1") == [
            "counter", "type", "$", "p1p1",
        ]

    def test_a_comparison_threshold_splits_letters_from_digits(self):
        assert _texts(_script_tokenizer(), "ConditionCompare$ GE3") == [
            "condition", "compare", "$", "ge", "3",
        ]

    def test_a_dollar_prefix_glued_to_its_value_is_its_own_token(self):
        assert _texts(_script_tokenizer(), "Count$Valid Creature.YouCtrl") == [
            "count", "$", "valid", "creature", ".", "you", "ctrl",
        ]

    def test_a_token_script_name_splits_at_its_underscores(self):
        texts = _texts(_script_tokenizer(), "TokenScript$ w_1_1_soldier")
        assert texts == ["token", "script", "$", "w", "1", "1", "soldier"]
        assert not any("_" in text for text in texts)

    def test_an_underscore_elsewhere_stays_a_word_character(self):
        assert "c_a_food" in _texts(_script_tokenizer(), "Defined$ c_a_food")

    def test_every_choices_item_of_a_charm_stays_one_token(self):
        """Cryptic Command's real gen-2 root line."""
        assert _texts(
            _script_tokenizer(), "CharmNum$ 2 | Choices$ SV1,SV2,SV3,SV4 | SP$ Charm",
        ) == [
            "charm", "num", "$", "2", "|", "choices", "$",
            "sv1", ",", "sv2", ",", "sv3", ",", "sv4", "|", "sp", "$", "charm",
        ]

    def test_choices_on_a_non_chooser_api_is_a_selector(self):
        """Abzan Advantage's real segment: ``Choices$`` here names creatures,
        not chain labels, so it splits like any value."""
        texts = _texts(
            _script_tokenizer(),
            "SV1: Choices$ Creature.leastToughnessControlledByYou | "
            "CounterType$ P1P1 | DB$ PutCounter",
        )
        assert texts[:12] == [
            "sv1", ":", "choices", "$", "creature", ".",
            "least", "toughness", "controlled", "by", "you", "|",
        ]
        assert "p1p1" in texts

    def test_the_chooser_test_reads_the_choices_own_segment(self):
        text = "Choices$ SV1,SV2 | SP$ Charm [SEG] SV1: Choices$ Creature.YouCtrl | DB$ Pump"
        texts = _texts(_script_tokenizer(), text)
        assert texts[2:5] == ["sv1", ",", "sv2"]
        assert "creature" in texts and "you" in texts

    def test_segment_openers_stay_one_token(self):
        text = (
            "Execute$ SV1 | Mode$ ChangesZone [SEG] SV1: DB$ Draw | "
            "SubAbility$ SV2 [SEG] SV2: DB$ ChangeZone"
        )
        assert _texts(_script_tokenizer(), text) == [
            "execute", "$", "sv1", "|", "mode", "$", "changes", "zone",
            "[SEG]", "sv1", ":", "db", "$", "draw", "|",
            "sub", "ability", "$", "sv2",
            "[SEG]", "sv2", ":", "db", "$", "change", "zone",
        ]

    def test_an_option_lines_opening_label_stays_one_token(self):
        """Cryptic Command's real fourth mode."""
        assert _texts(
            _script_tokenizer(),
            "SV1: DB$ Draw | Defined$ You | NumCards$ 1 | SpellDescription$ Draw a card.",
        )[:6] == ["sv1", ":", "db", "$", "draw", "|"]

    def test_a_real_trigger_chain_keeps_its_labels_whole(self):
        """Goblin Trapfinder's real gen-2 trigger line, descriptions trimmed."""
        text = (
            "Destination$ Graveyard | Execute$ SV1 | Mode$ ChangesZone | "
            "ValidCard$ Card.Self [SEG] SV1: DB$ Seek | RememberFound$ True | "
            "SubAbility$ SV2 | Type$ Creature.cmcLE3+YouOwn [SEG] SV2: DB$ Animate "
            "| Keywords$ Haste"
        )
        texts = _texts(_script_tokenizer(), text)
        assert texts.count("[SEG]") == 2
        assert texts.count("sv1") == 2 and texts.count("sv2") == 2
        assert ["cmc", "le", "3"] == texts[texts.index("cmc"):texts.index("cmc") + 3]

    def test_an_amount_svar_is_not_kept_whole(self):
        """Only the chain positions FR-002a renames stay whole."""
        assert _texts(_script_tokenizer(), "NumDmg$ X") == ["num", "dmg", "$", "x"]

    def test_offsets_still_index_the_source(self):
        text = "ValidTgts$ Creature.nonDragon+YouCtrl | SubAbility$ SV2"
        for token in _script_tokenizer().tokenize(text):
            assert text[token.start:token.end].lower() == token.text

    def test_multi_word_keywords_still_merge(self):
        assert "first_strike" in _texts(_script_tokenizer(), "AddKeyword$ First Strike")

    def test_the_prose_surface_applies_no_camel_split(self, tokenizer):
        texts = _texts(tokenizer, "ValidTgts$ Creature.nonDragon+YouCtrl | SubAbility$ SV2")
        assert {"validtgts", "nondragon", "youctrl"} <= set(texts)
        assert "sv2" not in texts and texts[-2:] == ["sv", "2"]

    @pytest.mark.parametrize("surface", ["prose", "script"])
    def test_seg_is_one_token_on_either_surface(self, surface):
        vocab = _vocab(("[SEG]",))
        tokenizer = AbilityTokenizer(vocab, surface=surface)
        tokens = tokenizer.tokenize("draw [SEG] draw")
        assert [t.text for t in tokens] == ["draw", "[SEG]", "draw"]
        assert tokens[1].token_id == vocab["[SEG]"]

    def test_an_unknown_surface_is_rejected(self):
        with pytest.raises(ValueError, match="surface"):
            AbilityTokenizer(_vocab(), surface="oracle")

    def test_the_shared_tokenizer_is_unchanged_on_a_script_line(self):
        """FR-010: ``MtgTokenizer`` output pinned from before this feature."""
        from price_predictor.domain.tokenizer import MtgTokenizer

        shared = MtgTokenizer({"[PAD]": 0, "[UNK]": 1, "first_strike": 2})
        assert shared.tokenize(
            "ValidTgts$ Creature.nonDragon+YouCtrl | SubAbility$ SV2 | "
            "CounterType$ P1P1 | AddKeyword$ First Strike"
        ) == [
            "validtgts", "$", "creature", ".", "nondragon", "+", "youctrl", "|",
            "subability", "$", "sv", "2", "|", "countertype", "$", "p", "1", "p",
            "1", "|", "addkeyword", "$", "first_strike", "mana", "cost", ":",
            "none",
        ]


# ── gen-2: keyword lines and template filling (FR-012–FR-018) ───────────


def _gen2_definitions() -> dict[str, KeywordDefinition]:
    """Real Forge templates and formatters for the keywords under test."""
    return {
        "Ward": KeywordDefinition(
            keyword="Ward", formatter="Ward",
            reminder_template=(
                "Whenever this permanent becomes the target of a spell or ability "
                "an opponent controls, counter it unless that player %s."
            ),
        ),
        "Enchant": KeywordDefinition(
            keyword="Enchant", formatter="KeywordWithType",
            reminder_template=(
                "Target a %1$s as you cast this. This card enters attached to "
                "that %1$s."
            ),
        ),
        "Equip": KeywordDefinition(
            keyword="Equip", formatter=None,
            reminder_template=(
                "%s: Attach to target %s you control. Equip only as a sorcery."
            ),
        ),
        "Level up": KeywordDefinition(
            keyword="Level up", formatter="KeywordWithCost",
            reminder_template=(
                "%s: Put a level counter on this. Level up only as a sorcery."
            ),
        ),
        "Cumulative upkeep": KeywordDefinition(
            keyword="Cumulative upkeep", formatter="KeywordWithCost",
            reminder_template=(
                "At the beginning of your upkeep, put an age counter on this "
                "permanent, then sacrifice it unless you pay its upkeep cost for "
                "each age counter on it."
            ),
        ),
        "Flying": KeywordDefinition(
            keyword="Flying", formatter="SimpleKeyword",
            reminder_template=(
                "This creature can't be blocked except by creatures with flying "
                "or reach."
            ),
        ),
    }


def _open_tokenizer(known: tuple[str, ...] = ("ward",), surface="script"):
    """A vocabulary holding every word the expansions under test read."""
    words = (
        "whenever this permanent becomes the target of spell or ability an "
        "opponent controls counter it unless that player pays attach to you "
        "control only as sorcery put level on up at beginning your "
        "upkeep age then sacrifice pay its cost for each enters "
        "attached card cast creature can t be blocked except by creatures with "
        "reach"
    ).split()
    # "equip", "cumulative" and "flying" are left out: the lines under test
    # name them, and a name the vocabulary lacks is what forces expansion.
    vocab = _vocab(tuple(words) + ("{2}", "'") + known)
    return AbilityTokenizer(vocab, _gen2_definitions(), surface=surface)


class TestKeywordLines:
    """Spec Story 1 scenarios 6, 7, 10."""

    def test_the_display_name_is_the_text_before_the_first_colon(self):
        from effects.domain.ability_tokenizer import display_name_of

        assert display_name_of("Ward:2") == "Ward"
        assert display_name_of("TypeCycling:Basic:1 B") == "TypeCycling"
        assert display_name_of("Flying") == "Flying"

    def test_ward_expands_whole_with_its_value_formatted(self):
        tokenizer = _open_tokenizer()
        expanded = tokenizer.expand_keywords(
            tokenizer.tokenize("Ward:2"), probability=1.0,
        )
        texts = [t.text for t in expanded]
        assert texts[-5:] == ["that", "player", "pays", "{2}", "."]
        assert "%" not in texts
        assert texts.count("ward") == 0
        assert all(t.expanded_from == "ward" for t in expanded)

    def test_a_known_keyword_line_stays_tokens_at_inference(self):
        from effects.domain.ability_tokenizer import INFERENCE_KEYWORD_EXPAND_P

        tokenizer = _open_tokenizer()
        tokens = tokenizer.tokenize("Ward:2")
        kept = tokenizer.expand_keywords(
            tokens, probability=INFERENCE_KEYWORD_EXPAND_P,
        )
        assert [t.text for t in kept] == ["ward", ":", "2"]

    def test_an_unknown_keyword_line_expands_at_probability_zero(self):
        tokenizer = _open_tokenizer(known=())
        expanded = tokenizer.expand_keywords(tokenizer.tokenize("Ward:2"))
        assert expanded[0].text == "whenever"

    def test_an_explicit_index_repeats_its_value(self):
        tokenizer = _open_tokenizer(known=())
        expanded = tokenizer.expand_keywords(tokenizer.tokenize("Enchant:Creature"))
        texts = [t.text for t in expanded]
        assert texts.count("creature") == 2
        assert "%" not in texts and "$" not in texts

    def test_unfilled_specifiers_are_removed(self):
        """Equip's template takes two values; the line supplies one."""
        tokenizer = _open_tokenizer(known=())
        expanded = tokenizer.expand_keywords(tokenizer.tokenize("Equip:2"))
        texts = [t.text for t in expanded]
        assert texts[:6] == ["2", ":", "attach", "to", "target", "you"]
        assert "%" not in texts

    def test_a_multi_word_display_name_is_replaced_whole(self):
        tokenizer = _open_tokenizer(known=())
        line = "Cumulative upkeep:AddCounter<1/M1M1>:Put a -1/-1 counter on CARDNAME."
        expanded = tokenizer.expand_keywords(tokenizer.tokenize(line))
        texts = [t.text for t in expanded]
        assert texts[:4] == ["at", "the", "beginning", "of"]
        assert "cumulative" not in texts

    def test_a_host_bodied_keyword_line_never_expands(self):
        """FR-016: compared by display name, so a two-word name matches."""
        tokenizer = _open_tokenizer(known=())
        tokens = tokenizer.tokenize("Level up:1 W")
        assert tokenizer.expand_keywords(tokens, probability=1.0) == tokens

    def test_a_host_bodied_two_word_token_never_expands(self):
        tokenizer = AbilityTokenizer(
            _vocab(("level", "up")), _gen2_definitions(),
        )
        merged = Token(text="level_up", token_id=1)
        assert tokenizer.expandable(merged) is False

    def test_a_display_name_with_no_definition_stays_tokens(self):
        tokenizer = _open_tokenizer()
        tokens = tokenizer.tokenize("etbCounter:P1P1:2")
        assert tokenizer.keyword_line_of("etbCounter:P1P1:2") is None
        assert tokenizer.expand_keywords(tokens, probability=1.0) == tokens

    def test_a_keyword_word_inside_another_line_takes_the_token_path(self):
        """FR-014: an unknown keyword inside a non-keyword line still expands."""
        tokenizer = _open_tokenizer(known=())
        tokens = tokenizer.tokenize("AddKeyword$ Flying | Affected$ Creature")
        assert tokens[0].keyword_line is None
        expanded = tokenizer.expand_keywords(tokens)
        assert [t.expanded_from for t in expanded].count("flying") > 5
        assert "flying" not in [t.text for t in expanded if t.expanded_from is None]

    def test_the_match_is_case_insensitive(self):
        assert _open_tokenizer().keyword_line_of("WARD:2").name == "Ward"

    def test_a_chained_text_is_never_a_keyword_line(self):
        tokenizer = _open_tokenizer()
        assert tokenizer.keyword_line_of("SV1: DB$ Draw") is None
        assert tokenizer.keyword_line_of(
            "Execute$ SV1 | Mode$ Attacks [SEG] SV1: DB$ Pump",
        ) is None

    def test_a_typographic_apostrophe_reads_as_ascii(self, tmp_path):
        """Scenario 7: normalized at load, not in the file."""
        import json

        from effects.application.extract_keyword_definitions import (
            load_keyword_definitions,
        )

        path = tmp_path / "keyword-definitions.json"
        path.write_text(json.dumps({"Read ahead": {
            "reminder_template": "Chapter abilities can’t trigger.",
            "formatter": "SimpleKeyword",
        }}), encoding="utf-8")
        definition = load_keyword_definitions(path)["Read ahead"]
        assert definition.reminder_template == "Chapter abilities can't trigger."
        assert definition.formatter == "SimpleKeyword"

    def test_an_old_definitions_file_has_no_formatter(self, tmp_path):
        import json

        from effects.application.extract_keyword_definitions import (
            load_keyword_definitions,
        )

        path = tmp_path / "keyword-definitions.json"
        path.write_text(json.dumps({"Flying": {"reminder_template": "x"}}),
                        encoding="utf-8")
        assert load_keyword_definitions(path)["Flying"].formatter is None


# ── SC-002 over the real definitions ────────────────────────────────────

#: Real ``extract-keyword-definitions`` output, with the ``formatter`` field.
_FIXTURE_DEFINITIONS = (
    Path(__file__).parents[3] / "fixtures" / "effects" / "keyword-definitions.json"
)
#: The working copy, which may predate the field; swept too when present.
_OUTPUT_DEFINITIONS = (
    Path(__file__).parents[4] / "output" / "effects" / "keyword-definitions.json"
)
_FIXTURE_SIDECARS = (
    Path(__file__).parents[3] / "fixtures" / "effects" / "gen1-sidecars"
)


def _fixture_keyword_lines() -> list[str]:
    from effects.infrastructure.sidecar_io import read_sidecar

    lines = []
    for path in sorted(_FIXTURE_SIDECARS.rglob("*.provenance.json")):
        for line in read_sidecar(path).lines:
            if line.script_api_type == "Keyword" and line.script_text:
                lines.append(line.script_text)
    return lines


@pytest.mark.parametrize("surface", ["prose", "script"])
@pytest.mark.parametrize("source", ["fixture", "output"])
def test_no_definition_expands_to_a_percent_or_unk(surface, source):
    """SC-002 over every real definition and every real fixture keyword line,
    against a vocabulary seeded the way ``build-vocab`` seeds it."""
    from effects.application.build_vocab import SEEDED_SPECIALS, seed_tokens
    from effects.application.extract_keyword_definitions import (
        load_keyword_definitions,
    )
    from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS

    path = _FIXTURE_DEFINITIONS if source == "fixture" else _OUTPUT_DEFINITIONS
    if not path.exists():
        pytest.skip("no keyword-definition file in output/")
    definitions = load_keyword_definitions(path)
    staged = _fixture_keyword_lines() if surface == "script" else []
    seeded = seed_tokens(surface, definitions, staged)
    vocab: dict[str, int] = {}
    for token in (*SEEDED_SPECIALS, *MULTI_WORD_KEYWORDS, *seeded.all()):
        vocab.setdefault(token, len(vocab))
    tokenizer = AbilityTokenizer(vocab, definitions, surface=surface)

    texts = [
        tokenizer.expansion_text(name)
        for name, d in definitions.items() if d.reminder_template
    ]
    for line in staged:
        keyword = tokenizer.keyword_line_of(line)
        if keyword is not None and definitions[keyword.name].reminder_template:
            texts.append(tokenizer.expansion_text(keyword.name, keyword.details))
    assert len(texts) > 150
    for text in texts:
        tokens = tokenizer.tokenize(text)
        assert "%" not in [t.text for t in tokens], text
        assert all(t.token_id != tokenizer.unk_id for t in tokens), text


class TestTokenizerRulesByCheckpoint:
    """A loaded checkpoint is read with the grammar it trained under.

    Gen-1 tokenized its script vocabulary with the prose grammar; reading it
    with the gen-2 script rules turns camel-case parts it never saw into
    ``[UNK]``. The text below is a real gen-1 sidecar line.
    """

    _GEN1_TEXT = "Defined$ TriggeredCard | ValidTgts$ Creature.YouCtrl | DB$ Pump"

    def test_a_checkpoint_recording_no_rules_reads_the_prose_grammar(self):
        from effects.domain.ability_tokenizer import tokenizer_surface

        assert tokenizer_surface("script", {}) == "prose"
        assert tokenizer_surface("script", None) == "prose"

    def test_a_gen2_checkpoint_reads_its_vocabularys_surface(self):
        from effects.domain.ability_tokenizer import tokenizer_surface

        settings = {"tokenizer_rules": "gen-2"}
        assert tokenizer_surface("script", settings) == "script"
        assert tokenizer_surface("prose", settings) == "prose"

    def test_the_two_grammars_differ_on_a_gen1_line(self):
        vocab = _vocab(("validtgts", "creature", "youctrl", "defined"))
        legacy = [t.text for t in AbilityTokenizer(vocab, surface="prose").tokenize(
            self._GEN1_TEXT)]
        script = [t.text for t in AbilityTokenizer(vocab, surface="script").tokenize(
            self._GEN1_TEXT)]
        assert "youctrl" in legacy and "validtgts" in legacy
        assert "youctrl" not in script and "ctrl" in script
