"""The compositional script tokenizer (T137).

``Creature.nonDragon+OppCtrl`` is a creature, that is not a Dragon, that an
opponent controls. Read as one token it is a symbol the model has seen a handful
of times; read as three it is three restrictions it has seen thousands of times
each, in every combination. That decomposition is the whole reason the script
surface is worth having — the vocabulary is compositional in a way prose is not.
"""

from __future__ import annotations

import pytest

from effects.domain.ability_tokenizer import AbilityTokenizer

_WORDS = (
    "creature nondragon oppctrl youctrl artifact permanent land attacking "
    "blocking tapped untapped mode continuous addpower addkeyword flying "
    "putcounter defined self countertype p1p1 counternum numdmg valid "
    "db trig spell targetmin targetmax"
).split()


def _vocab() -> dict[str, int]:
    tokens = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]"]
    tokens += _WORDS
    tokens += ["$", "|", ".", "+", "1", "2", "3", "{R}", "{T}"]
    seen: dict[str, int] = {}
    for token in tokens:
        seen.setdefault(token, len(seen))
    return seen


@pytest.fixture
def tokenizer() -> AbilityTokenizer:
    return AbilityTokenizer(_vocab())


class TestSelectorDecomposition:
    def test_a_compound_selector_splits_into_its_restrictions(self, tokenizer):
        tokens = tokenizer.tokenize_script("Creature.nonDragon+OppCtrl")
        assert [t.text for t in tokens] == [
            "creature", "nondragon", "oppctrl",
        ]

    def test_each_part_keeps_its_own_offset(self, tokenizer):
        text = "Creature.nonDragon+OppCtrl"
        for token in tokenizer.tokenize_script(text):
            assert text[token.start:token.end].lower() == token.text

    def test_a_simple_selector_is_one_token(self, tokenizer):
        assert [t.text for t in tokenizer.tokenize_script("Creature")] == [
            "creature",
        ]

    def test_a_two_part_selector_splits_in_two(self, tokenizer):
        assert [
            t.text for t in tokenizer.tokenize_script("Permanent.YouCtrl")
        ] == ["permanent", "youctrl"]

    def test_the_parts_are_ordinary_vocabulary_tokens(self, tokenizer):
        tokens = tokenizer.tokenize_script("Creature.nonDragon+OppCtrl")
        assert all(t.token_id != tokenizer.unk_id for t in tokens)

    def test_an_unknown_restriction_is_unk_and_the_rest_still_resolve(
        self, tokenizer,
    ):
        """Decomposition is what limits the damage: one unfamiliar restriction
        costs one token rather than the whole selector."""
        tokens = tokenizer.tokenize_script("Creature.nonSliver+OppCtrl")
        ids = [t.token_id for t in tokens]
        assert ids[0] != tokenizer.unk_id
        assert ids[1] == tokenizer.unk_id
        assert ids[2] != tokenizer.unk_id


class TestScriptStructure:
    def test_the_key_value_separator_is_its_own_token(self, tokenizer):
        tokens = tokenizer.tokenize_script("Mode$ Continuous")
        assert [t.text for t in tokens] == ["mode", "$", "continuous"]

    def test_the_parameter_separator_is_its_own_token(self, tokenizer):
        tokens = tokenizer.tokenize_script("Mode$ Continuous | AddPower$ 1")
        assert "|" in [t.text for t in tokens]

    def test_a_full_script_line_tokenizes(self, tokenizer):
        tokens = tokenizer.tokenize_script(
            "DB$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1"
        )
        texts = [t.text for t in tokens]
        assert "putcounter" in texts
        assert "p1p1" in texts
        assert texts.count("|") == 3

    def test_a_numeric_parameter_carries_its_value(self, tokenizer):
        tokens = tokenizer.tokenize_script("NumDmg$ 3")
        number = next(t for t in tokens if t.number is not None)
        assert number.number == 3.0

    def test_mana_symbols_keep_their_case(self, tokenizer):
        assert "{R}" in [t.text for t in tokenizer.tokenize_script("Cost$ {R}")]

    def test_an_empty_script_yields_no_tokens(self, tokenizer):
        assert tokenizer.tokenize_script("") == []


class TestTheTwoSurfacesDiffer:
    def test_the_prose_tokenizer_does_not_split_selectors(self, tokenizer):
        """Prose has no compound selectors; splitting on '.' would break
        sentences at every full stop."""
        prose = tokenizer.tokenize("draw a card. creature")
        assert "." in [t.text for t in prose]

    def test_the_script_tokenizer_treats_a_dot_as_a_selector_join(
        self, tokenizer,
    ):
        script = tokenizer.tokenize_script("Creature.attacking")
        assert "." not in [t.text for t in script]
        assert [t.text for t in script] == ["creature", "attacking"]
