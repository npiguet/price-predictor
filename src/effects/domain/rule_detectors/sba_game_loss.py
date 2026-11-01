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

CHECKS = frozenset({"life_zero", "poison_ten"})


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


@dataclass(frozen=True, slots=True)
class SbaGameLoss:
    id: str = "sba_game_loss"
    params: frozenset[str] = frozenset({"check"})

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
        fires, spared, observed = [], [], []
        for player in record.state.players:
            life, damage, poison, lost = _changes(record, player.id)
            if check == "life_zero":
                loss = -life if life is not None else damage
                if loss <= 0:
                    continue
                side = player.life - loss <= 0
            else:
                if poison <= 0:
                    continue
                side = player.poison + poison >= 10
            (fires if side else spared).append(player.id)
            if lost:
                observed.append(player.id)
        if fires:
            return FamilyMatch(Polarity.FIRES, any(p in observed for p in fires), tuple(fires))
        if spared:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, any(p in observed for p in spared),
                               tuple(spared))
        return None


FAMILY = register(SbaGameLoss())
