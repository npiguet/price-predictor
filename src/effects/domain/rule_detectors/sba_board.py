"""The board ``sba_zone_move`` reads its state-based checks off.

What one record's events did to each entity it touches (:class:`Touched`,
read by :func:`touched_by`), and whether the record's own changes end the
game (:func:`game_ends`), in which case Forge runs no check at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from effects.domain.event_schema import Event, EventType, damage_source
from effects.domain.records import EffectRecord
from effects.domain.state_snapshot import EntityState, PlayerState, normalize_counter_type

#: Where the check's move from the battlefield lands: the graveyard, or exile
#: in its place.
DEATH_ZONES = frozenset({"graveyard", "exile"})


@dataclass(slots=True)
class Touched:
    """What the record's events did to one entity."""

    toughness_delta: int = 0
    lowered: bool = False
    damage_dealt: int = 0
    damage_sources: list[str] = field(default_factory=list)
    counter_delta: dict[str, int] = field(default_factory=dict)
    left_creatures: bool = False
    entered: bool = False
    left: bool = False
    #: The keywords the record grants it, lower-cased (``deathtouch``,
    #: ``protection from white``): the snapshot is read before them.
    gained: set[str] = field(default_factory=set)
    #: Left the battlefield by the resolution's own clause, not the check.
    moved_by_effect: bool = False
    died: bool = False

    def type_change(self, event: Event, params: Mapping[str, Any]) -> None:
        if "creature" in (params.get("types_removed") or ()):
            self.left_creatures = True

    def keyword_change(self, event: Event, params: Mapping[str, Any]) -> None:
        if not params.get("removed"):
            self.gained.update(k.lower() for k in params.get("keywords") or ())

    def pt_change(self, event: Event, params: Mapping[str, Any]) -> None:
        delta = int(params.get("toughness_delta") or 0)
        self.toughness_delta += delta
        self.lowered = self.lowered or delta < 0

    def damage(self, event: Event, params: Mapping[str, Any]) -> None:
        self.damage_dealt += int(params.get("amount") or 0)
        source = damage_source(event)
        if source:
            self.damage_sources.append(source)

    def counter_change(self, event: Event, params: Mapping[str, Any]) -> None:
        counter = normalize_counter_type(str(params.get("counter_type", "")))
        self.counter_delta[counter] = self.counter_delta.get(counter, 0) + int(
            params.get("delta") or 0)

    def zone_change(self, event: Event, params: Mapping[str, Any]) -> None:
        if params.get("to_zone") == "battlefield":
            self.entered = True
            self.died = False  # back from where it went: a flicker, not a death
        if params.get("from_zone") == "battlefield":
            self._leave(event, params)

    def _leave(self, event: Event, params: Mapping[str, Any]) -> None:
        self.left = True
        if event.attributed_to != "unresolved":
            self.moved_by_effect = True
        elif params.get("to_zone") in DEATH_ZONES:
            self.died = True

    def untouched(self, event: Event, params: Mapping[str, Any]) -> None:
        """An event of a kind no check reads: its subject is touched, and no more."""


_Touch = Callable[[Touched, Event, Mapping[str, Any]], None]

#: What each event type does to the entities it names.
_TOUCHES: dict[EventType, _Touch] = {
    EventType.TYPE_CHANGE: Touched.type_change,
    EventType.KEYWORD_CHANGE: Touched.keyword_change,
    EventType.PT_CHANGE: Touched.pt_change,
    EventType.DAMAGE_DEALT: Touched.damage,
    EventType.COUNTER_CHANGE: Touched.counter_change,
    EventType.ZONE_CHANGE: Touched.zone_change,
}


def touched_by(record: EffectRecord) -> dict[str, Touched]:
    """Every subject of the record's events, in the order first named, with
    what the events did to it."""
    found: dict[str, Touched] = {}
    for event in record.payload.listed_events():
        touch = _TOUCHES.get(event.type, Touched.untouched)
        params = event.params or {}
        for subject in event.subjects:
            touch(found.setdefault(subject, Touched()), event, params)
    return found


_GAME_END_EVENTS = (EventType.PLAYER_WON, EventType.PLAYER_LOST)


def game_ends(record: EffectRecord) -> bool:
    """Whether the record's own changes end the game: a game-end event, or a
    player its life and poison changes take to 0 life or 10 poison."""
    events = record.payload.listed_events()
    if any(event.type in _GAME_END_EVENTS for event in events):
        return True
    life = _totals(events, EventType.LIFE_CHANGE)
    poison = _totals(events, EventType.POISON_CHANGE)
    return any(_out_of_life(player, life) or _poisoned(player, poison)
               for player in record.state.players)


def _totals(events: tuple[Event, ...], kind: EventType) -> dict[str, int]:
    """Each subject's summed ``delta`` over the events of ``kind``."""
    found: dict[str, int] = {}
    for event in events:
        if event.type is kind:
            delta = int((event.params or {}).get("delta") or 0)
            for subject in event.subjects:
                found[subject] = found.get(subject, 0) + delta
    return found


def _out_of_life(player: PlayerState, life: Mapping[str, int]) -> bool:
    return player.id in life and player.life + life[player.id] <= 0


def _poisoned(player: PlayerState, poison: Mapping[str, int]) -> bool:
    return player.id in poison and player.poison + poison[player.id] >= 10


@dataclass(frozen=True, slots=True)
class Board:
    """One record as the checks read it: its snapshot, what its events did,
    and the keywords of an entity."""

    record: EffectRecord
    touched: dict[str, Touched]
    keywords: Callable[[EntityState], set[str]]

    def entity(self, entity_id: str | None) -> EntityState | None:
        return self.record.state.entity(entity_id)

    def touch(self, entity_id: str | None) -> Touched:
        """What the record did to the entity; nothing for one it left alone."""
        return self.touched.get(entity_id, Touched())

    def source_has(self, touched: Touched, wanted: frozenset[str]) -> bool:
        """Whether a source of the damage the creature was dealt has one of
        ``wanted``, printed, granted on the board, or granted by the record itself."""
        return any(self._carries(source, wanted) for source in touched.damage_sources)

    def _carries(self, source: str, wanted: frozenset[str]) -> bool:
        if wanted & self.touch(source).gained:
            return True
        entity = self.entity(source)
        return entity is not None and bool(wanted & self.keywords(entity))
