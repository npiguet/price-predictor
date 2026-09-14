"""Writing a round's decks where the Forge worker can read them.

The generated-decks format (`LABEL;SET_CODE;Card1|...`) rather than a new one,
because `GeneratedDecksIndex` already parses it and the worker already knows how
to sample from it. What a coverage round needs beyond that is only that nothing
tries to resolve the set code — see the decks-only mode.
"""

from __future__ import annotations

from pathlib import Path

#: Sentinel set code for decks drawn from the whole corpus rather than a set.
#: Never resolved against Forge's set table: the decks-only mode plays both
#: sides from the file and opens no pool.
COVERAGE_SET_CODE = "COVERAGE"


def write_deck_file(
    decks: list[list[str]], path: Path, *, label: str, set_code: str,
) -> int:
    """Write one deck per line, replacing whatever was there. Returns the count.

    Replacing rather than appending: a round's decks are built for that round's
    weights, and last round's decks would pull play back toward cards that have
    since been satisfied.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{label};{set_code};{'|'.join(deck)}" for deck in decks]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)
