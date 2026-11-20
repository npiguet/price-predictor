"""Reading the converted card trees and the MTGJSON printings dump."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from effects.domain.card_names import fold_card_name
from effects.domain.provenance import SourceTree
from effects.infrastructure.sidecar_io import SidecarCache, UnconvertedScript

logger = logging.getLogger(__name__)


def load_card_texts(
    card_files: Mapping[str, str], sidecars: SidecarCache,
) -> dict[str, list[str]]:
    """``card name -> the script texts of its sidecar's lines``; a card with
    no sidecar is skipped and counted."""
    out: dict[str, list[str]] = {}
    skipped = 0
    for name, script_file in card_files.items():
        try:
            sidecar = sidecars.get(script_file)
        except UnconvertedScript:
            skipped += 1
            continue
        out[name] = sidecar.script_texts()
    if skipped:
        logger.warning("%d converted card(s) have no sidecar; skipped", skipped)
    return out


@dataclass(frozen=True, slots=True)
class ConvertedCard:
    """One converted card file: its tree-prefixed script file and its text."""

    script_file: str
    text: str


def load_converted_cards(cards_folder: Path) -> dict[str, ConvertedCard]:
    """``folded card name -> its converted file`` for one converted tree.

    Names come from each file's ``name:`` line: several cards share a
    filename stem. They are folded here, once (:func:`fold_card_name`): the
    holdout lists, the credits and the castability consult all read the
    folded spelling, and Forge looks a deck line up without regard to case.
    The original case stays in the file's text.
    """
    from price_predictor.domain.tokenizer import extract_card_name

    cards_folder = Path(cards_folder)
    out: dict[str, ConvertedCard] = {}
    for path in sorted(cards_folder.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        name = extract_card_name(text)
        if not name:
            continue
        relative = path.relative_to(cards_folder).as_posix()
        out[fold_card_name(name)] = ConvertedCard(f"{SourceTree.CARDSFOLDER}/{relative}", text)
    return out


def _cards_of(set_info) -> list[dict]:
    """The cards of one entry of the dump's ``data``: a bare list, or a set's ``cards``."""
    return set_info if isinstance(set_info, list) else set_info.get("cards", [])


def _release_of(set_info) -> str:
    """A set's ``releaseDate``, ``""`` for a bare list or a set without one."""
    return set_info.get("releaseDate", "") if isinstance(set_info, dict) else ""


def _names_of(card: dict) -> set:
    """The names a card is indexed under: its own and its face's, whichever are given."""
    return {card.get("name"), card.get("faceName")} - {None, ""}


def _is_earlier(release: str, previous: str) -> bool:
    """Whether ``release`` is a date before ``previous``; an empty one is none."""
    return bool(release) and release < previous


def _note_release(earliest: dict[str, str], key: str, release: str) -> None:
    """Record ``release`` for ``key`` when it has none yet or this one is earlier and dated."""
    previous = earliest.get(key)
    if previous is None or _is_earlier(release, previous):
        earliest[key] = release


def load_first_printings(printings_path: Path) -> dict[str, str]:
    """``card name -> earliest release date`` from an MTGJSON dump, faces
    indexed under their own names as well as the combined one."""
    import json

    data = json.loads(Path(printings_path).read_text(encoding="utf-8"))
    earliest: dict[str, str] = {}
    for set_info in data.get("data", {}).values():
        release = _release_of(set_info)
        for card in _cards_of(set_info):
            for key in _names_of(card):
                _note_release(earliest, key, release)
    return earliest
