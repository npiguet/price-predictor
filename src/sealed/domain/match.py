"""Match-level domain concepts: the two sides and who won."""

from __future__ import annotations

from enum import Enum


class Side(Enum):
    """The two players in a match. Encoded as ``A`` / ``B`` in the data files."""

    A = "A"
    B = "B"

    @classmethod
    def parse(cls, value: str, *, label: str = "side") -> "Side":
        """Parse the single-char wire form, raising a ``label``-named error."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"{label} must be 'A' or 'B', got {value!r}"
            ) from None


def match_winner(wins_a: int, wins_b: int) -> Side | None:
    """Return the winning ``Side``, or ``None`` for a tie.

    Names the ``wins_a > wins_b`` comparison the match readers reimplement.
    Best-of-odd matches never tie, but ``None`` is returned defensively so
    callers don't have to assume that.
    """
    if wins_a > wins_b:
        return Side.A
    if wins_b > wins_a:
        return Side.B
    return None
