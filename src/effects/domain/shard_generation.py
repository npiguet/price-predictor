"""Every reader reads gen-3 records only (FR-033, FR-018).

Decided by the wire, so it is read off a record's raw dictionary: a parsed
record cannot say whether a field was there. A gen-3 record says so in
``generation``, the integer 3. A record with no such member is of an older
generation and is refused, named: gen-2 records carry ``random_seat`` and
gen-1 ones do not. A ``generation`` member naming no generation (``4``,
``"3"``, ``true``, ``null``) is refused too, rather than read as another.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class ShardGeneration(StrEnum):
    """The record generations by name. Gen-3 is the one any reader reads;
    gen-1 and gen-2 are named only to say what a refused record is."""

    GEN_1 = "gen-1"
    GEN_2 = "gen-2"
    GEN_3 = "gen-3"


class UnacceptedGenerationError(ValueError):
    """A record of a generation no reader accepts: every reader reads gen-3 only."""


class OlderGenerationError(UnacceptedGenerationError):
    """A gen-1 or gen-2 record, which carries no ``generation`` member."""


class UnknownGenerationError(UnacceptedGenerationError):
    """A record's ``generation`` member names no generation at all."""


def older_generation_of(raw: Mapping) -> ShardGeneration:
    """The older generation a record without a ``generation`` member was
    written in, for the refusal's message alone: gen-2 by ``random_seat``,
    which every gen-2 record carries, whatever its value, and no gen-1 one."""
    return ShardGeneration.GEN_2 if "random_seat" in raw else ShardGeneration.GEN_1


def require_gen3(raw: Mapping) -> None:
    """Refuse one raw record unless it is gen-3: one with no ``generation``
    member raises :class:`OlderGenerationError` naming its generation, one
    whose member is not the integer 3 :class:`UnknownGenerationError`."""
    if "generation" not in raw:
        raise OlderGenerationError(
            f"the record is {older_generation_of(raw)} (it carries no generation "
            "member); every reader reads gen-3 records only"
        )
    value = raw["generation"]
    # A boolean is no integer here.
    if type(value) is not int or value != 3:
        raise UnknownGenerationError(
            f"the record's generation {value!r} names no known generation; "
            "only gen-3 writes the member, as the integer 3"
        )
