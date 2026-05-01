"""Parse sealed pool and generated-deck files.

Pool files (input to ``build-decks``) hold one random booster pool per line
in the format ``SET_CODE;Card1|Card2|...|CardN``.

Generated-deck files (output of ``build-decks``, input to ``match-outcomes``
self-play) hold one finished 40-card deck per line in the format
``LABEL;SET_CODE;Card1|Card2|...|Card40``, where ``LABEL`` is the
generation-method tag passed to ``build-decks --label``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GeneratedDeck:
    """One line of a generated-decks file.

    ``label`` is the generation-method tag (e.g. ``"gen-2"``,
    ``"gen-3-experimental"``) recorded by ``build-decks --label`` and used
    by ``match-outcomes`` as the ``method_A`` / ``method_B`` value when this
    deck is sampled into a self-play match.
    """
    label: str
    set_code: str
    cards: list[str]


def parse_pools(pools_file: Path) -> list[tuple[str, list[str]]]:
    """Parse a pools.txt file into ``(set_code, card_names)`` tuples.

    The file format is one pool per line:
    ``SET_CODE;Card1|Card2|...|CardN``. Blank lines are ignored.

    Args:
        pools_file: Path to the pools.txt file.

    Returns:
        One tuple per pool line. Order matches the source file.

    Raises:
        ValueError: If a non-blank line does not contain a ``;`` separator.
    """
    pools: list[tuple[str, list[str]]] = []
    for line in pools_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ";" not in stripped:
            raise ValueError(
                f"Pool line missing ';' set code separator: {stripped!r}"
            )
        set_code, _, names_field = stripped.partition(";")
        card_names = names_field.split("|") if names_field else []
        pools.append((set_code, card_names))
    return pools


def parse_generated_decks(decks_file: Path) -> list[GeneratedDeck]:
    """Parse a generated-decks file into ``GeneratedDeck`` objects.

    The file format is one deck per line:
    ``LABEL;SET_CODE;Card1|Card2|...|Card40``. Blank lines are ignored.

    Args:
        decks_file: Path to the generated-decks file.

    Returns:
        One ``GeneratedDeck`` per non-blank line, in source-file order.

    Raises:
        ValueError: If a non-blank line does not contain at least two ``;``
            separators (label and set code).
    """
    decks: list[GeneratedDeck] = []
    for line in decks_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(";", 2)
        if len(parts) < 3:
            raise ValueError(
                f"Generated-decks line missing label/set-code separators: "
                f"{stripped!r}"
            )
        label, set_code, names_field = parts
        cards = names_field.split("|") if names_field else []
        decks.append(GeneratedDeck(label=label, set_code=set_code, cards=cards))
    return decks
