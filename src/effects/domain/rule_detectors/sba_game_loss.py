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
and only those: damage dealt to a player is not life loss where the source
has infect (poison counters instead, 702.90b), and every other hit to a
player is filed with its own ``life_change``.

A player the record takes over the other check's line too (a combat dealing
lethal damage and the tenth poison counter at once) is on neither side: the
game's end cannot say which rule ended it.

The collector writes a non-combat kill's game end in a record of its own
after the resolution, the game-end record (the outcome alone), never in the
resolution's effect half (gen-3's held half takes the check's moves from the
battlefield, not the game's end; only an ability that ends the game outright,
a ``WinsGame`` or ``LosesGame`` line, holds it, ``emittable_events``), so an
effect half holding no game-end event (:data:`_GAME_END`) has its outcome
unreadable (``observed`` None). The game-end record itself
shows no side: its board is read at the game's end, with the loser gone. A
combat record holds its damage step's game end.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.rule_detectors import (
    EventlessFamily,
    FamilyMatch,
    Matcher,
    first_side,
    register,
)
from effects.domain.state_snapshot import PlayerState

CHECKS = frozenset({"life_zero", "poison_ten"})
_GAME_END = frozenset({EventType.PLAYER_WON, EventType.PLAYER_LOST})


def _delta(event, kind: EventType) -> int:
    """The ``delta`` of ``event`` where it is a ``kind``, else 0."""
    return int((event.params or {}).get("delta") or 0) if event.type is kind else 0


def _lost_in(event, player: str) -> bool:
    """Whether ``event`` says ``player`` lost: they lost, or another won."""
    if event.type is EventType.PLAYER_WON:
        return bool(event.subjects) and player not in event.subjects
    return event.type is EventType.PLAYER_LOST and player in event.subjects


def _changes(record: EffectRecord, player: str) -> tuple[int, int, bool]:
    """``(life delta from life_change, poison delta, lost)``."""
    life = poison = 0
    lost = False
    for event in record.payload.listed_events():
        lost = lost or _lost_in(event, player)
        if player in event.subjects:
            life += _delta(event, EventType.LIFE_CHANGE)
            poison += _delta(event, EventType.POISON_CHANGE)
    return life, poison, lost


def _outcome_readable(record: EffectRecord) -> bool:
    """Whether the record can hold the game's end: any but an effect half
    without a game-end event, whose game end the collector files after it."""
    if record.kind is not RecordKind.RESOLUTION:
        return True
    return any(event.type in _GAME_END for event in record.payload.listed_events())


@dataclass(frozen=True, slots=True)
class SbaGameLoss(EventlessFamily):
    id: str = "sba_game_loss"
    kinds: frozenset[str] = frozenset({"combat", "resolution"})
    params: frozenset[str] = frozenset({"check"})
    general: frozenset[str] = frozenset({"check"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def validate(self, params: Mapping[str, Any]) -> None:
        check = params.get("check")
        if check not in CHECKS:
            raise ValueError(f"sba_game_loss: unknown check {check!r}; known: {sorted(CHECKS)}")

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return GameLossMatcher(_CHECK_SIDES[params.get("check")])

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


@dataclass(frozen=True, slots=True)
class _Toll:
    """What a record cost one player: life lost, poison gained, and whether
    each would end the game."""

    loss: int
    poison: int
    dead: bool
    poisoned: bool


def _life_side(toll: _Toll) -> bool | None:
    """``life_zero``'s side of a player: None where no life was lost or poison
    would end the game first."""
    if toll.loss <= 0 or toll.poisoned:
        return None
    return toll.dead


def _poison_side(toll: _Toll) -> bool | None:
    """``poison_ten``'s side of a player: None where no poison was gained or
    life loss would end the game first."""
    if toll.poison <= 0 or toll.dead:
        return None
    return toll.poisoned


#: Each check's side of a player: True on the ``fires`` side, False spared, None unjudged.
_CHECK_SIDES: dict[str, Callable[[_Toll], bool | None]] = {
    "life_zero": _life_side, "poison_ten": _poison_side}


def _toll(record: EffectRecord, player: PlayerState) -> tuple[_Toll, bool]:
    """What the record cost ``player``, and whether the record says they lost."""
    life, poison, lost = _changes(record, player.id)
    loss = -life
    return _Toll(loss, poison, loss > 0 and player.life - loss <= 0,
                 poison > 0 and player.poison + poison >= 10), lost


@dataclass(frozen=True, slots=True, eq=False)
class GameLossMatcher:
    """A compiled ``sba_game_loss`` declaration."""

    side_of: Callable[[_Toll], bool | None]

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        effect_half = record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION
        if not effect_half and record.kind is not RecordKind.COMBAT:
            return None
        if record.is_game_end_record:
            return None
        fires, spared, observed = self._players(record)
        # The player's life or poison is both what the check reads and
        # where the game loss falls.
        first = first_side(fires, spared)
        if first is None:
            return None
        polarity, judged = first
        readable = _outcome_readable(record)
        return FamilyMatch(polarity, any(p in observed for p in judged) if readable else None,
                           judged, judged)

    def _players(self, record: EffectRecord) -> tuple[list[str], list[str], list[str]]:
        """The players on the ``fires`` side, the spared ones, and those the
        record shows lost."""
        fires, spared, observed = [], [], []
        for player in record.state.players:
            side, lost = self._judge(record, player)
            if side is None:
                continue
            (fires if side else spared).append(player.id)
            if lost:
                observed.append(player.id)
        return fires, spared, observed

    def _judge(self, record: EffectRecord, player: PlayerState) -> tuple[bool | None, bool]:
        """``(side, lost)`` of one player; the side None where the check leaves them out."""
        toll, lost = _toll(record, player)
        return self.side_of(toll), lost


FAMILY = register(SbaGameLoss())
