"""``holdout-cards``: the depletion list ``generate-pools --exclude-cards`` reads.

A plain file rather than an import, because `sealed` must not depend on
`effects` (FR-132). It goes through ``ConvertedTrees.holdout`` like
``build-corpus``, so the list and the split cannot silently disagree.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain import default_paths
from effects.domain.split import HoldoutRule
from effects.domain.text_holdout import HoldoutUnit

if TYPE_CHECKING:
    from effects.domain.split import HeldOutCards

logger = logging.getLogger(__name__)


class HoldoutCardsError(Exception):
    """No converted card to choose a holdout from."""


@dataclass(frozen=True, slots=True)
class HoldoutCardsConfig:
    """The ``holdout-cards`` flags."""

    out: Path
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    printings_path: Path = field(
        default=Path(default_paths.DEFAULT_PRINTINGS_PATH), metadata={"help": (
            "MTGJSON dump, read only to write names in Forge's printed spelling "
            "(default: %(default)s)"
        )},
    )
    holdout: HoldoutRule = HoldoutRule()


@dataclass(frozen=True, slots=True)
class HoldoutReport:
    """The depletion list and the counts the operator reads beside it (FR-043)."""

    names: list[str]
    unit: HoldoutUnit
    #: Held-out units: masked templates under ``template``, texts under ``text``.
    keys: int
    texts: int
    converted_cards: int

    @property
    def depleted_share(self) -> float:
        """The share of converted cards the list removes from the pools."""
        return len(self.names) / self.converted_cards if self.converted_cards else 0.0


def run(config: HoldoutCardsConfig) -> HoldoutReport:
    """Choose the holdout over the converted trees and write its cards to ``out``.

    Raises ``HoldoutCardsError`` when no tree holds a converted card.
    """
    from effects.infrastructure.card_sources import load_first_printings
    from effects.infrastructure.converted_trees import SourceTrees

    trees = SourceTrees.of(config.cards_folders).converted()
    if not trees.card_files:
        where = trees.cards_folder or ", ".join(str(f) for f in config.cards_folders)
        raise HoldoutCardsError(f"no converted cards under {where}")
    printings = Path(config.printings_path)
    canonical = load_first_printings(printings) if printings.exists() else {}
    if not canonical:
        logger.warning(
            "No printings at %s: the list keeps the converted tree's lowercase "
            "spelling. Every comparison folds case, so depletion still works.",
            printings,
        )
    report = holdout_report(
        trees.holdout(config.holdout),
        converted_cards=len(trees.card_files),
        canonical_names=canonical,
    )
    write_depletion_list(report.names, config.out)
    return report


def holdout_report(
    held: HeldOutCards,
    *,
    converted_cards: int,
    canonical_names: Mapping[str, object] | None = None,
) -> HoldoutReport:
    """The depletion list with its held-out template, text and card counts.

    ``canonical_names`` (keyed by printed name) restores Forge's printed case
    for an operator to read; an unknown name keeps the lowercase, which still
    depletes since every comparison folds case. Sorted so an unchanged holdout
    writes an unchanged file.
    """
    from effects.domain.card_names import fold_card_name
    from effects.domain.text_holdout import holdout_key

    printed = {
        fold_card_name(name): name for name in (canonical_names or {})
    }
    return HoldoutReport(
        names=sorted(printed.get(name, name) for name in held.names),
        unit=held.unit,
        keys=len({holdout_key(text, held.unit) for text in held.texts}),
        texts=len(held.texts),
        converted_cards=converted_cards,
    )


def write_depletion_list(names: Iterable[str], out: Path) -> None:
    """Write one card name per line."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(names)
    out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
