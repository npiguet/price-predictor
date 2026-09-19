"""A cache line named the way a person reads it.

The geometry checks a person reads — the ward canary and the nearest-neighbour
lists — name their lines in converted prose, because that is what a reader can
judge. The cache is keyed on the encoding surface instead, which on a script
checkpoint is Forge script, so a query has to be *resolved* from its prose to
the line the model actually encoded.

The prose is spelled as ``convert`` renders it: lowercase, ``CARDNAME`` for the
card's own name, reminder text dropped, a trailing period on a sentence
(``destroy target creature.``) and none on a bare keyword (``ward {2}``).
Matching ignores case and runs of whitespace and nothing else, so a query that
drifts from the converter's spelling fails to resolve — visibly, in the report
— rather than quietly matching some other line.

``card`` pins a query whose prose alone names more than one script: six cards
print "spells your opponents cast that target CARDNAME cost {2} more to cast.",
and they need not compile to one script, so without a card the query would
stand for whichever of them was loaded first.
"""

from __future__ import annotations

from dataclasses import dataclass


def normalize_prose(text: str) -> str:
    """Case- and whitespace-insensitive form a query is matched under."""
    return " ".join(text.lower().split())


@dataclass(frozen=True, slots=True)
class LineQuery:
    """One ability line, by its converted prose and optionally its card."""

    prose: str
    card: str | None = None

    def label(self) -> str:
        return f"{self.prose} [{self.card}]" if self.card else self.prose
