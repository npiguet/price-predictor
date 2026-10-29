"""A cache line named by its converted prose, resolved to the encoded line.

The prose is spelled as ``convert`` renders it (lowercase, ``CARDNAME``, a
trailing period on a sentence and none on a bare keyword). Matching ignores
case and whitespace runs and nothing else, so a query that drifts from the
converter's spelling fails to resolve visibly rather than matching another
line. ``card`` pins a query whose prose compiles to more than one script.
"""

from __future__ import annotations

from dataclasses import dataclass


def normalize_prose(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True, slots=True)
class LineQuery:
    """One ability line, by its converted prose and optionally its card."""

    prose: str
    card: str | None = None

    def label(self) -> str:
        return f"{self.prose} [{self.card}]" if self.card else self.prose
