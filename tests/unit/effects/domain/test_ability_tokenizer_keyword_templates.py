"""gen-2: keyword lines and template filling (FR-012–FR-018), and SC-002
over the real definitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.ability_tokenizer import AbilityTokenizer, Token
from effects.domain.keyword_definitions import KeywordDefinition
from tests.unit.effects.domain.ability_tokenizer_support import _vocab


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
        from effects.domain.keyword_names import display_name_of

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

    def test_a_forced_keyword_line_expands_whole_at_probability_zero(self):
        tokenizer = _open_tokenizer()
        expanded = tokenizer.expand_keywords(
            tokenizer.tokenize("Ward:2"), forced=frozenset({"ward"}),
        )
        assert [t.text for t in expanded][-5:] == ["that", "player", "pays", "{2}", "."]
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
        assert tokenizer.keywords.line_of("etbCounter:P1P1:2") is None
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
        assert _open_tokenizer().keywords.line_of("WARD:2").name == "Ward"

    def test_a_chained_text_is_never_a_keyword_line(self):
        tokenizer = _open_tokenizer()
        assert tokenizer.keywords.line_of("SV1: DB$ Draw") is None
        assert tokenizer.keywords.line_of(
            "Execute$ SV1 | Mode$ Attacks [SEG] SV1: DB$ Pump",
        ) is None

    def test_a_typographic_apostrophe_reads_as_ascii(self, tmp_path):
        """Scenario 7: normalized at load, not in the file."""
        import json

        from effects.infrastructure.tokenizer_loading import (
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

        from effects.infrastructure.tokenizer_loading import (
            load_keyword_definitions,
        )

        path = tmp_path / "keyword-definitions.json"
        path.write_text(json.dumps({"Flying": {"reminder_template": "x"}}),
                        encoding="utf-8")
        assert load_keyword_definitions(path)["Flying"].formatter is None



# ── SC-002 over the real definitions ────────────────────────────────────

#: Real ``extract-keyword-definitions`` output, with the ``formatter`` field
#: (byte-identical to gen-3's Stage 0 output of 2026-11-15).
_FIXTURE_DEFINITIONS = (
    Path(__file__).parents[3] / "fixtures" / "effects" / "keyword-definitions.json"
)
_FIXTURE_SIDECARS = (
    Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-sidecars"
)


def _fixture_keyword_lines() -> list[str]:
    from effects.infrastructure.sidecar_io import read_sidecar

    lines = []
    for path in sorted(_FIXTURE_SIDECARS.rglob("*.provenance.json")):
        for line in read_sidecar(path).lines:
            if line.script_api_type == "Keyword" and line.script_text:
                lines.append(line.script_text)
    return lines


def _seeded_tokenizer(
    surface: str,
    definitions: dict[str, KeywordDefinition],
    staged: list[str],
) -> AbilityTokenizer:
    """A tokenizer over a vocabulary seeded the way ``build-vocab`` seeds it."""
    from effects.application.build_vocab import (
        SEEDED_SPECIALS,
        seed_tokens,
        staging_tokenizer,
    )
    from price_predictor.application.build_vocabulary import MULTI_WORD_KEYWORDS

    seeded = seed_tokens(staging_tokenizer(surface, definitions), definitions, staged)
    vocab: dict[str, int] = {}
    for token in (*SEEDED_SPECIALS, *MULTI_WORD_KEYWORDS, *seeded.all()):
        vocab.setdefault(token, len(vocab))
    return AbilityTokenizer(vocab, definitions, surface=surface)


def _expanded_keyword_texts(
    tokenizer: AbilityTokenizer,
    definitions: dict[str, KeywordDefinition],
    staged: list[str],
) -> list[str]:
    """The line text of every definition with a template, then of every staged
    keyword line whose keyword has one, filled with that line's details."""
    texts = [
        tokenizer.keywords.line_text(name)
        for name, d in definitions.items() if d.reminder_template
    ]
    for line in staged:
        keyword = tokenizer.keywords.line_of(line)
        if keyword is not None and definitions[keyword.name].reminder_template:
            texts.append(tokenizer.keywords.line_text(keyword.name, keyword.details))
    return texts


@pytest.mark.parametrize("surface", ["prose", "script"])
def test_no_definition_expands_to_a_percent_or_unk(surface):
    """SC-002 over every real definition and every real fixture keyword line,
    against a vocabulary seeded the way ``build-vocab`` seeds it."""
    from effects.infrastructure.tokenizer_loading import (
        load_keyword_definitions,
    )

    definitions = load_keyword_definitions(_FIXTURE_DEFINITIONS)
    staged = _fixture_keyword_lines() if surface == "script" else []
    tokenizer = _seeded_tokenizer(surface, definitions, staged)
    texts = _expanded_keyword_texts(tokenizer, definitions, staged)
    assert len(texts) > 150
    for text in texts:
        tokens = tokenizer.tokenize(text)
        assert "%" not in [t.text for t in tokens], text
        assert all(t.token_id != tokenizer.unk_id for t in tokens), text
