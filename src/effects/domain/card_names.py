"""The one spelling of a card name both sides of the Python/Java boundary share.

Kept in the domain, apart from the trainer, because the coverage counter's
process-pool workers call it once per record: importing it from the trainer
loaded torch — and its CUDA libraries — into every worker for a ``lower()``.
"""

from __future__ import annotations


def fold_card_name(name: str) -> str:
    """A card name in the one spelling both sides of the boundary agree on.

    Converted card text is lowercased, so every name read out of
    ``output/cardsfolder/`` arrives as ``soul echo``. Forge's own
    ``getName()`` — and so every record's ``EntityState.name`` and every
    booster's cards — carries printed case, ``Soul Echo``. Comparing the two
    directly never matches, and nothing says so: a collection run told to
    deplete its pools depletes nothing and writes an ordinary corpus.

    ``lower()`` rather than ``casefold()`` to agree with the Java side's
    ``toLowerCase(Locale.ROOT)``; the two differ on characters no card name
    has, and a holdout the two languages disagree about is the same bug again.
    """
    return name.lower()
