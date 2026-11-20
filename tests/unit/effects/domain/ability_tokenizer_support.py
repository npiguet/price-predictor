"""The ability tokenizer (T063).

Three properties the shared ``MtgTokenizer`` cannot provide and the encoder
depends on: offsets that line up with the sidecar's role spans, whole-token
lookup with no subword fallback, and forced expansion of keywords the
vocabulary has never seen.

Builders the ``test_ability_tokenizer_*`` modules share: the base
vocabulary, the keyword definitions and the ``tokenizer`` fixture.
"""

from __future__ import annotations

import pytest

from effects.domain.ability_tokenizer import AbilityTokenizer
from effects.domain.keyword_definitions import KeywordDefinition

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
