"""Reading the converted card trees and the MTGJSON printings dump."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
        out[name] = [
            line.script_text for line in sidecar.lines if line.script_text
        ]
    if skipped:
        logger.warning("%d converted card(s) have no sidecar; skipped", skipped)
    return out


@dataclass(frozen=True, slots=True)
class ConvertedCard:
    """One converted card file: its tree-prefixed script file and its text."""

    script_file: str
    text: str


def load_converted_cards(cards_folder: Path) -> dict[str, ConvertedCard]:
    """``card name -> its converted file`` for one converted tree.

    Names come from each file's ``name:`` line: several cards share a
    filename stem.
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
        out[name] = ConvertedCard(f"{SourceTree.CARDSFOLDER}/{relative}", text)
    return out


def load_card_files(cards_folder: Path) -> dict[str, str]:
    """``card name -> tree-prefixed script file`` for one converted tree."""
    return {
        name: card.script_file
        for name, card in load_converted_cards(cards_folder).items()
    }


def load_first_printings(printings_path: Path) -> dict[str, str]:
    """``card name -> earliest release date`` from an MTGJSON dump, faces
    indexed under their own names as well as the combined one."""
    import json

    data = json.loads(Path(printings_path).read_text(encoding="utf-8"))
    earliest: dict[str, str] = {}
    for set_info in data.get("data", {}).values():
        cards = set_info if isinstance(set_info, list) else set_info.get("cards", [])
        release = (
            set_info.get("releaseDate", "") if isinstance(set_info, dict) else ""
        )
        for card in cards:
            for key in {card.get("name"), card.get("faceName")} - {None, ""}:
                previous = earliest.get(key)
                if previous is None or (release and release < previous):
                    earliest[key] = release
    return earliest
