"""What a line with no prose reads: its script reading (finding 1.5, D4).

A variant card's ``.txt`` is a Forge source script, so its lines have no prose.
Each reads its script with every ``...Description$`` and ``TgtPrompt$``
parameter dropped, since the perturbation rewrites one ``Key$ value`` and
leaves those stating the original, through the script grammar on either
surface: the prose grammar has no camel split, so every API name and parameter
key read as ``[UNK]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from effects.domain.ability_tokenizer import AbilityTokenizer, Surface
from effects.domain.keyword_definitions import KeywordDefinition
from effects.domain.line_text import encoding_role_spans, encoding_text
from effects.domain.provenance import Role, RoleSpan
from effects.domain.script_text import SCRIPT_READING, SEG, script_reading

_SHOCK = (
    "SP$ DealDamage | ValidTgts$ Any | TgtPrompt$ Select any target for 2 damage | "
    "NumDmg$ 4 | StackDescription$ SpellDescription | "
    "SpellDescription$ CARDNAME deals 2 damage to any target."
)


@dataclass(frozen=True)
class _Line:
    script_text: str | None
    role_spans: tuple[RoleSpan, ...] = (RoleSpan(0, 5, Role.EFFECT),)


class TestScriptReading:
    def test_every_free_text_parameter_is_dropped(self):
        reading = script_reading(_SHOCK)
        assert reading == f"{SCRIPT_READING} SP$ DealDamage | ValidTgts$ Any | NumDmg$ 4", (
            f"the reading keeps a parameter stating the card's printed wording: {reading!r}"
        )

    def test_each_segment_loses_its_own_and_keeps_its_label(self):
        chained = (
            f"Mode$ ChangesZone | Execute$ SV1 | TriggerDescription$ When it enters, draw two. "
            f"{SEG} SV1: DB$ Draw | NumCards$ 3 | AILogic$ Always | SpellDescription$ Draw two."
        )
        assert script_reading(chained) == (
            f"{SCRIPT_READING} Mode$ ChangesZone | Execute$ SV1 "
            f"{SEG} SV1: DB$ Draw | NumCards$ 3 | AILogic$ Always"
        )

    def test_a_prompt_seen_stating_the_original_is_dropped_too(self):
        """``ChoiceTitle$``, ``SelectPrompt$``, ``CostDesc$`` and
        ``RepeatOptionalMessage$`` were found stating a perturbed value's
        original over the gen-2 variant tree."""
        script = ("DB$ ChangeZone | ChangeNum$ 3 | ChoiceTitle$ Choose up to two cards | "
                  "SelectPrompt$ Select two | CostDesc$ Pay 2 life: | "
                  "RepeatOptionalMessage$ Repeat two times?")
        assert script_reading(script) == f"{SCRIPT_READING} DB$ ChangeZone | ChangeNum$ 3"

    def test_a_keyword_script_is_kept_whole(self):
        assert script_reading("etbCounter:P1P1:2") == f"{SCRIPT_READING} etbCounter:P1P1:2"


class TestEncodingText:
    def test_a_line_with_no_prose_reads_its_script_on_either_surface(self):
        line = _Line(_SHOCK)
        for surface in Surface:
            assert encoding_text(line, None, surface) == script_reading(_SHOCK), (
                f"a variant line on the {surface} surface reads the raw script, "
                "descriptions and all"
            )

    def test_a_line_with_prose_reads_as_before(self):
        line = _Line("NumDmg$ 3 | SpellDescription$ deal 3 damage")
        prose = "deal 3 damage to any target"
        assert encoding_text(line, prose, Surface.PROSE) == prose
        assert encoding_text(line, prose, Surface.SCRIPT) == line.script_text

    def test_a_line_with_neither_reads_as_the_empty_string(self):
        assert encoding_text(_Line(None), None, Surface.PROSE) == ""
        assert encoding_text(None, None, Surface.SCRIPT) == ""

    def test_a_script_reading_takes_no_role_spans(self):
        """The spans index a prose the line does not have."""
        line = _Line(_SHOCK)
        text = encoding_text(line, None, Surface.PROSE)
        assert encoding_role_spans(line, text, Surface.PROSE) == ()


def _tokenizer(surface: Surface) -> AbilityTokenizer:
    words = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", SEG, SCRIPT_READING,
             "sp", "$", "deal", "damage", "|", "valid", "tgts", "any", "num", "dmg", "4",
             "to", "target"]
    return AbilityTokenizer({word: index for index, word in enumerate(words)}, surface=surface)


class TestTheProseTokenizerReadsAScriptReadingAsScript:
    def test_api_names_and_keys_split_at_camel_case(self):
        tokenizer = _tokenizer(Surface.PROSE)
        tokens = [t.text for t in tokenizer.tokenize(script_reading(_SHOCK))]
        assert tokens == [SCRIPT_READING, "sp", "$", "deal", "damage", "|", "valid",
                          "tgts", "$", "any", "|", "num", "dmg", "$", "4"], (
            f"the prose surface did not read the script reading by the script grammar: {tokens}"
        )

    def test_prose_keeps_the_prose_grammar(self):
        tokenizer = _tokenizer(Surface.PROSE)
        tokens = [t.text for t in tokenizer.tokenize("DealDamage to target")]
        assert tokens == ["dealdamage", "to", "target"]

    def test_a_variants_keyword_line_is_still_a_keyword_line(self):
        """A variant's swapped ``Protection from blue`` expands with its own
        quality, as a keyword line read after the reading's opener."""
        definitions = {"Protection": KeywordDefinition(
            keyword="Protection", reminder_template="can't be blocked by %s creatures")}
        tokenizer = AbilityTokenizer(
            {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, SCRIPT_READING: 3}, definitions,
            surface=Surface.PROSE)
        tokens = tokenizer.tokenize(script_reading("Protection from blue"))
        assert tokens[0].keyword_line is not None
        assert tokens[0].keyword_line.details == "blue"

    def test_both_surfaces_read_a_script_reading_alike(self):
        text = script_reading(_SHOCK)
        prose = [t.text for t in _tokenizer(Surface.PROSE).tokenize(text)]
        script = [t.text for t in _tokenizer(Surface.SCRIPT).tokenize(text)]
        assert prose == script
