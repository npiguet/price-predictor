"""gen-2: the script surface (FR-007–FR-010)."""

from __future__ import annotations

import pytest

from effects.domain.ability_tokenizer import AbilityTokenizer
from tests.unit.effects.domain import ability_tokenizer_support
from tests.unit.effects.domain.ability_tokenizer_support import _vocab

# The shared fixture: pytest collects it from this module's namespace.
tokenizer = ability_tokenizer_support.tokenizer


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
