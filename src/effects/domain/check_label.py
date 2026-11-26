"""How a report line is named, and the table rows a line can carry beside it.

A line's name is a head (a check, a stratum or a half) and a qualifier, one
space apart: ``card-disjoint family damage`` is the head ``card-disjoint`` and
the qualifier ``family damage``. One type builds the name and one reads it
back, so no caller assembles a name by hand.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckLabel:
    """A report line's name: its head and the qualifier narrowing it.

    The head holds no space; the qualifier may.
    """

    head: str
    qualifier: str = ""

    @classmethod
    def of(cls, head: object, *parts: object) -> CheckLabel:
        """``head`` qualified by ``parts``, joined by single spaces, an empty
        part left out."""
        return cls(str(head), " ".join(text for part in parts if (text := str(part))))

    @classmethod
    def parse(cls, name: str) -> CheckLabel:
        """The label whose :func:`str` is ``name``."""
        head, _, qualifier = name.partition(" ")
        return cls(head, qualifier)

    def then(self, *parts: object) -> CheckLabel:
        """This label with ``parts`` appended to its qualifier."""
        return CheckLabel.of(self.head, self.qualifier, *parts) if self.qualifier \
            else CheckLabel.of(self.head, *parts)

    def __str__(self) -> str:
        return f"{self.head} {self.qualifier}" if self.qualifier else self.head


@dataclass(frozen=True, slots=True)
class TableRow:
    """One figure of a line's table: ``records`` is how many records it read,
    None where it is not a mean over records."""

    field: str
    value: float
    records: int | None = None
