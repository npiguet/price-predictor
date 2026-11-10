"""Writing a round's decks where the Forge worker can read them.

The generated-decks format (`LABEL;SET_CODE;Card1|...`) rather than a new one,
because `GeneratedDecksIndex` already parses it and the worker already knows how
to sample from it. What a coverage round needs beyond that is only that nothing
tries to resolve the set code — see the decks-only mode.

A steered variant round's deck (FR-058d) adds a fourth column, the rule the
deck is aimed at: ``LABEL;SET_CODE;Card1|...|Card40;RULE``. The worker stamps
every record of a game ``aimed_at`` with its two decks' rules. A card name
never holds ``;``, so the column cannot be read as part of the card list.
"""

from __future__ import annotations

from pathlib import Path

from sealed.infrastructure.pool_file_reader import format_generated_deck

#: Sentinel set code for decks drawn from the whole corpus rather than a set.
#: Never resolved against Forge's set table: the decks-only mode plays both
#: sides from the file and opens no pool.
COVERAGE_SET_CODE = "COVERAGE"

#: Separates a steered deck's rule column from its card list.
RULE_SEPARATOR = ";"


def write_deck_file(
    decks: list[list[str]], path: Path, *, label: str,
    rules: list[str | None] | None = None,
) -> int:
    """Write one deck per line under :data:`COVERAGE_SET_CODE`, replacing
    whatever was there. Returns the count.

    ``rules``, one per deck, gives each steered deck its rule column; a deck
    whose rule is None, and every deck without ``rules``, has none.

    Replacing rather than appending: a round's decks are built for that round's
    weights, and last round's decks would pull play back toward cards that have
    since been satisfied.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rules is not None and len(rules) != len(decks):
        raise ValueError(f"{len(rules)} rules for {len(decks)} decks")
    rows = [
        format_generated_deck(label, COVERAGE_SET_CODE, deck)
        + (f"{RULE_SEPARATOR}{rules[i]}" if rules is not None and rules[i] else "")
        for i, deck in enumerate(decks)
    ]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)
