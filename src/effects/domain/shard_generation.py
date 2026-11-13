"""The record generations a records set can hold (FR-033, FR-018).

Decided by the wire, so it is read off a record's raw dictionary: a parsed
record cannot say whether a field was there. A gen-3 record says so in
``generation``; gen-2 and gen-1 records carry no such member and are told
apart by ``random_seat``, which every gen-2 record carries and no gen-1 one does.
A ``generation`` member naming no generation (``4``, ``"3"``, ``true``,
``null``) is refused rather than read as an earlier one.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class ShardGeneration(StrEnum):
    """The three record generations a shard can hold.

    On gen-1 ``playability`` records ``actor_player`` keeps feature 023's
    meaning rather than naming the deciding player (FR-030a); gen-3 changes
    what several fields mean again (feature 025), and no reader accepts a mix.
    """

    GEN_1 = "gen-1"
    GEN_2 = "gen-2"
    GEN_3 = "gen-3"

    @classmethod
    def of(cls, raw: Mapping) -> ShardGeneration:
        """The generation of one record, read off its raw dictionary.

        Raises :class:`UnknownGenerationError` for a ``generation`` member
        other than the integer 3.
        """
        if "generation" in raw:
            check_generation_member(raw["generation"])
            return cls.GEN_3
        return cls.GEN_2 if "random_seat" in raw else cls.GEN_1


def check_generation_member(value: object) -> None:
    """Refuse a record's ``generation`` member unless it is the integer 3,
    the only generation that writes one (a boolean is no integer here)."""
    if type(value) is not int or value != 3:
        raise UnknownGenerationError(
            f"the record's generation {value!r} names no known generation; "
            "only gen-3 writes the member, as the integer 3"
        )


#: The generations the corpus, training, evaluation and probe readers accept:
#: gen-3 alone, the one the gen-3 input layout reads (FR-020). ``rule-counts``
#: keeps its own list, which still reads gen-2.
ACCEPTED_GENERATIONS: tuple[ShardGeneration, ...] = (ShardGeneration.GEN_3,)


class MixedGenerationsError(ValueError):
    """A records set holds shards of more than one generation."""


class UnacceptedGenerationError(ValueError):
    """A records set holds a generation its reader does not accept."""


class UnknownGenerationError(UnacceptedGenerationError):
    """A record's ``generation`` member names no generation at all, which
    no reader accepts."""
