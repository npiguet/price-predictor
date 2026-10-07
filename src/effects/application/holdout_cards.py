"""``holdout-cards``: the depletion list ``generate-pools --exclude-cards`` reads.

The one coupling between the two halves of a depleted collection run, and it is
a plain file of card names rather than an import: `sealed` must not depend on
`effects` (FR-132).

The list and the trainer's split have to name the same cards. They are two
programs run hours apart, and a disagreement is silent — pools depleted of one
set while the split holds out another means held-out cards reach training games
and the card-disjoint stratum holds games that are not disjoint. So both go
through ``text_keyed_holdout`` rather than reimplementing the rule.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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


def holdout_report(
    card_files: Mapping[str, str],
    texts_by_card: Mapping[str, Iterable[str]],
    *,
    permille: int,
    max_carriers: int,
    unit: str = "text",
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
    from effects.application.train_effect_model import (
        fold_card_name,
        text_keyed_holdout,
    )
    from effects.domain.text_holdout import holdout_key

    held = text_keyed_holdout(
        dict(card_files), texts_by_card,
        permille=permille, max_carriers=max_carriers, unit=unit,
    )
    printed = {
        fold_card_name(name): name for name in (canonical_names or {})
    }
    return HoldoutReport(
        names=sorted(printed.get(name, name) for name in held.names),
        unit=unit,
        keys=len({holdout_key(text, unit) for text in held.texts}),
        texts=len(held.texts),
        converted_cards=len(card_files),
    )


def depletion_list(
    card_files: Mapping[str, str],
    texts_by_card: Mapping[str, Iterable[str]],
    *,
    permille: int,
    max_carriers: int,
    unit: str = "text",
    canonical_names: Mapping[str, object] | None = None,
) -> list[str]:
    """Every card carrying a held-out ability text, in printed case, sorted."""
    return holdout_report(
        card_files, texts_by_card, permille=permille, max_carriers=max_carriers,
        unit=unit, canonical_names=canonical_names,
    ).names


def write_depletion_list(names: Iterable[str], out: Path) -> int:
    """Write one card name per line. Returns how many were written."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(names)
    out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)
