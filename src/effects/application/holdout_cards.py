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
from pathlib import Path

logger = logging.getLogger(__name__)


def depletion_list(
    card_files: Mapping[str, str],
    texts_by_card: Mapping[str, Iterable[str]],
    *,
    permille: int,
    max_carriers: int,
) -> list[str]:
    """Every card carrying a held-out ability text, sorted.

    Sorted because the file is compared between runs: one that reordered would
    read as a changed holdout when nothing had changed.
    """
    from effects.application.train_effect_model import text_keyed_holdout

    held = text_keyed_holdout(
        dict(card_files), texts_by_card,
        permille=permille, max_carriers=max_carriers,
    )
    return sorted(held.names)


def write_depletion_list(names: Iterable[str], out: Path) -> int:
    """Write one card name per line. Returns how many were written."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(names)
    out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)
