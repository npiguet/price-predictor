"""Detector family ``sba_game_loss``: a player losing the game in the state-based check.

704.5a: a player at 0 or less life loses; 704.5c: a player with ten or more
poison counters loses. The record whose events take a player there (a
resolution's effect half, or a combat) shows the rule; its ``fires`` side is
a player whose life or poison, after the record's own changes, crosses the
line, and its near miss a player who lost life or gained poison and stayed
short of it. ``observed`` is the game ending against that player: a
``player_lost`` naming them or a ``player_won`` naming another player.

Parameter ``check``: ``life_zero`` or ``poison_ten``.

A player's life loss is the record's ``life_change`` deltas for that player,
or, where the record holds none for them, the damage dealt to them: combat
damage to a player is not always filed with its own ``life_change``.

A player the record takes over the other check's line too (a combat dealing
lethal damage and the tenth poison counter at once) is on neither side: the
game's end cannot say which rule ended it.

The collector writes a non-combat kill's game end in a record of its own
after the resolution, the game-end rescue of orphaned events, never in the
resolution's effect half (gen-3's held half takes the check's moves from the
battlefield, not the game's end), so an effect half holding no game-end event
has its outcome unreadable (``observed`` None). The rescue record itself
shows no side: its board is read at the game's end, with the loser gone. A
combat record holds its damage step's game end.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors.acting_line import is_rescued_orphans

CHECKS = frozenset({"life_zero", "poison_ten"})
_GAME_END = frozenset({EventType.PLAYER_WON, EventType.PLAYER_LOST})


def _changes(record: EffectRecord, player: str) -> tuple[int | None, int, int, bool]:
    """``(life delta from life_change or None, damage dealt, poison delta, lost)``."""
    life: int | None = None
    damage = poison = 0
    lost = False
    for event in record.payload.listed_events():
        params = event.params or {}
        if event.type is EventType.PLAYER_WON and event.subjects and player not in event.subjects:
            lost = True
        if player not in event.subjects:
            continue
        if event.type is EventType.LIFE_CHANGE:
            life = (life or 0) + int(params.get("delta") or 0)
        elif event.type is EventType.DAMAGE_DEALT:
            damage += int(params.get("amount") or 0)
        elif event.type is EventType.POISON_CHANGE:
            poison += int(params.get("delta") or 0)
        elif event.type is EventType.PLAYER_LOST:
            lost = True
    return life, damage, poison, lost


def _outcome_readable(record: EffectRecord) -> bool:
    """Whether the record can hold the game's end: any but an effect half
    without a game-end event, whose game end the collector files after it."""
    if record.kind is not RecordKind.RESOLUTION:
        return True
    return any(event.type in _GAME_END for event in record.payload.listed_events())


@dataclass(frozen=True, slots=True)
class SbaGameLoss:
    id: str = "sba_game_loss"
    params: frozenset[str] = frozenset({"check"})
    general: frozenset[str] = frozenset({"check"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        check = params.get("check")
        if check not in CHECKS:
            raise ValueError(f"sba_game_loss: unknown check {check!r}; known: {sorted(CHECKS)}")
        effect_half = record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION
        if not effect_half and record.kind is not RecordKind.COMBAT:
            return None
        if is_rescued_orphans(record):
            return None
        fires, spared, observed = [], [], []
        for player in record.state.players:
            life, damage, poison, lost = _changes(record, player.id)
            loss = -life if life is not None else damage
            dead = loss > 0 and player.life - loss <= 0
            poisoned = poison > 0 and player.poison + poison >= 10
            if check == "life_zero":
                if loss <= 0 or poisoned:
                    continue
                side = dead
            else:
                if poison <= 0 or dead:
                    continue
                side = poisoned
            (fires if side else spared).append(player.id)
            if lost:
                observed.append(player.id)
        readable = _outcome_readable(record)
        if fires:
            return FamilyMatch(Polarity.FIRES,
                               any(p in observed for p in fires) if readable else None,
                               tuple(fires))
        if spared:
            return FamilyMatch(Polarity.DOES_NOT_FIRE,
                               any(p in observed for p in spared) if readable else None,
                               tuple(spared))
        return None


FAMILY = register(SbaGameLoss())
