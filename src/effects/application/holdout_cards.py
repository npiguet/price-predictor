"""``holdout-cards``: the depletion list ``generate-pools --exclude-cards`` reads.

The one coupling between the two halves of a depleted collection run, and it is
a plain file of card names rather than an import: `sealed` must not depend on
`effects` (FR-132).

The list and the trainer's split have to name the same cards. They are two
programs run hours apart, and a disagreement is silent — pools depleted of one
set while the split holds out another means held-out cards reach training games
and the card-disjoint stratum holds games that are not disjoint. So this command
and ``build-corpus`` both go through ``ConvertedTrees.holdout`` rather than
reimplementing the rule.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.split import HOLDOUT_MAX_CARRIERS, HOLDOUT_PERMILLE

if TYPE_CHECKING:
    from effects.domain.split import HeldOutCards

logger = logging.getLogger(__name__)


class HoldoutCardsError(Exception):
    """No converted card to choose a holdout from."""


@dataclass(frozen=True, slots=True)
class HoldoutCardsConfig:
    """The ``holdout-cards`` flags."""

    out: Path
    cards_folders: tuple[Path, ...] = (Path("output/cardsfolder"), Path("output/tokenscripts"))
    #: Read only to write the list in Forge's printed spelling.
    printings_path: Path = Path("resources/AllPrintings.json")
    holdout_permille: int = HOLDOUT_PERMILLE
    holdout_max_carriers: int = HOLDOUT_MAX_CARRIERS
    holdout_unit: str = "template"


@dataclass(frozen=True, slots=True)
class HoldoutReport:
    """The depletion list and the counts the operator reads beside it (FR-043)."""

    names: list[str]
    unit: str
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

    Raises:
        HoldoutCardsError: If no card tree among ``cards_folders`` holds a
            converted card.
    """
    from effects.infrastructure.card_sources import load_first_printings
    from effects.infrastructure.converted_trees import ConvertedTrees

    trees = ConvertedTrees.from_folders(config.cards_folders)
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
        trees.holdout(
            permille=config.holdout_permille,
            max_carriers=config.holdout_max_carriers,
            unit=config.holdout_unit,
        ),
        unit=config.holdout_unit,
        converted_cards=len(trees.card_files),
        canonical_names=canonical,
    )
    write_depletion_list(report.names, config.out)
    return report


def holdout_report(
    held: HeldOutCards,
    *,
    unit: str,
    converted_cards: int,
    canonical_names: Mapping[str, object] | None = None,
) -> HoldoutReport:
    """The depletion list with its held-out template, text and card counts.

    Args:
        canonical_names: any mapping keyed by printed card name — the
            first-printing table serves — used to resolve the converted tree's
            lowercase spelling back to the one Forge prints. A name it does not
            know keeps the converted spelling; every comparison folds case, so
            an unresolved name still depletes.

    The list is in printed case because this file crosses into another package
    and into Forge: an operator reads it, and ``getName()`` is what it is
    matched against. Sorted because the file is compared between runs, and one
    that reordered would read as a changed holdout when nothing had changed.
    """
    from effects.domain.card_names import fold_card_name
    from effects.domain.text_holdout import holdout_key

    printed = {
        fold_card_name(name): name for name in (canonical_names or {})
    }
    return HoldoutReport(
        names=sorted(printed.get(name, name) for name in held.names),
        unit=unit,
        keys=len({holdout_key(text, unit) for text in held.texts}),
        texts=len(held.texts),
        converted_cards=converted_cards,
    )


def write_depletion_list(names: Iterable[str], out: Path) -> None:
    """Write one card name per line."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(names)
    out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
