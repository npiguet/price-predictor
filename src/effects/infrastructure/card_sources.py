"""Reading the converted card trees and the MTGJSON printings dump.

File I/O shared by ``build-corpus``, ``collect-coverage``, ``collect-variants``,
``train-effect-model`` and the CLI.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from effects.infrastructure.sidecar_io import SidecarCache


def load_card_texts(
    card_files: Mapping[str, str], sidecars: SidecarCache,
) -> dict[str, list[str]]:
    """``card name -> the script texts of its sidecar's lines``.

    The I/O half of the holdout, kept apart from the selection so the rule can
    be exercised without a converted tree on disk.
    """
    out: dict[str, list[str]] = {}
    for name, script_file in card_files.items():
        sidecar = sidecars.get(script_file)
        if sidecar is None:
            continue
        out[name] = [
            line.script_text for line in sidecar.lines if line.script_text
        ]
    return out


def load_card_files(cards_folder: Path) -> dict[str, str]:
    """``card name -> tree-prefixed script file`` for one converted tree.

    Names come from each file's ``name:`` line, which is authoritative — the
    filename is a sanitized derivative and several cards share a stem.
    """
    from price_predictor.domain.tokenizer import extract_card_name

    cards_folder = Path(cards_folder)
    out: dict[str, str] = {}
    for path in sorted(cards_folder.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        name = extract_card_name(text)
        if not name:
            continue
        relative = path.relative_to(cards_folder).as_posix()
        out[name] = f"cardsfolder/{relative}"
    return out


def load_first_printings(printings_path: Path) -> dict[str, str]:
    """``card name -> earliest release date`` from an MTGJSON dump.

    Faces are indexed under their own names as well as the combined one, so a
    record naming either resolves.
    """
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
