"""The two record generations a records set can hold (FR-033).

Decided by one field's presence on the wire, so it is read off a record's raw
dictionary: a parsed record cannot say whether the field was there.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class ShardGeneration(StrEnum):
    """The two record generations a shard can hold.

    A gen-2 record always carries ``random_seat``; a gen-1 one never does, and
    on gen-1 ``playability`` records ``actor_player`` keeps feature 023's
    meaning rather than naming the deciding player (FR-030a).
    """

    GEN_1 = "gen-1"
    GEN_2 = "gen-2"

    @classmethod
    def of(cls, raw: Mapping) -> ShardGeneration:
        """The generation of one record, read off its raw dictionary."""
        return cls.GEN_2 if "random_seat" in raw else cls.GEN_1


class MixedGenerationsError(ValueError):
    """A records set holds both gen-1 and gen-2 shards."""
