"""Which games of one shard ended, read in the pass that counts it.

A game that ended with every end-of-game write done writes a game-end
marker (``turn_action`` / ``game_end``) as its last record; one whose shard
holds none was cut (its worker killed, its game or an end-of-game write
threw) and ``build-corpus`` drops it whole. A counter that credits a unit
toward satisfaction from a shard must leave those games out, or a run thinks
a unit satisfied by records the build will drop. ``shard_io.ended_games``
reads the markers in a pass of its own; a counter that already reads every
record notes them here instead, so each shard is still read once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from effects.domain.records import EffectRecord, is_game_end


@dataclass
class GameEnds:
    """The games of one shard and those whose marker was read."""

    games: set[str] = field(default_factory=set)
    ended: set[str] = field(default_factory=set)

    def note(self, record: EffectRecord) -> None:
        self.games.add(record.game_id)
        if is_game_end(record):
            self.ended.add(record.game_id)

    @property
    def cut(self) -> frozenset[str]:
        """The games no marker ended: once the whole shard is read, the
        cut ones (and, in a shard read up to a damage, those the damage hid)."""
        return frozenset(self.games - self.ended)
