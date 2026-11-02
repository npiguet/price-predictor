"""Detector family ``sba_zone_move``: a permanent put into the graveyard by the state-based check.

Rule 704.5 lists the state-based actions that move a permanent from the
battlefield to the graveyard. Gen-2 already files every such move made in the
check after a resolution in that resolution's effect half, as a
``zone_change`` attributed ``unresolved``, so the family runs on gen-2
records and needs no gen-3 field.

Parameter ``check`` names which action, each read off the board the record
shows (the snapshot before the resolution, plus the resolution's own events):

=====================  ======================================================
``toughness_zero``     704.5f: a creature the resolution lowered, or one a
                       spell's or activation's board already shows at
                       toughness 0, whose toughness after the P/T changes
                       is 0 or less; the near miss is one it lowered that
                       keeps 1 or more
``lethal_damage``      704.5g: a creature the resolution dealt damage to, by
                       whether the damage marked reaches its toughness
``deathtouch_damage``  704.5h: a creature dealt non-lethal damage by a source
                       with deathtouch; the near miss is non-lethal damage
                       from a source without it
``loyalty_zero``       704.5i: a planeswalker the resolution removed loyalty
                       counters from, by whether none remain
``defense_zero``       704.5v: a battle the resolution removed defense
                       counters from, by whether none remain
``legend_rule``        704.5j: two or more legendary permanents with one name
                       under one controller; the near miss is a legendary
                       permanent entering with no such twin
``world_rule``         704.5k: two or more world permanents; the near miss is
                       a world permanent entering alone
``aura_illegal``       704.5m: an Aura on the battlefield whose host left it
                       in the record, or gained protection from one of the
                       Aura's colours; the near miss is an Aura whose host the
                       record touched and which stayed legal
=====================  ======================================================

The polarity is read from the board, ``observed`` from the record's own move,
so a record where Forge disagrees with the rule is the one
:attr:`RuleTag.needs_review` flags. Left out of every check, because the
record cannot say which rule moved them:

- a permanent the record shows no change of the check's kind for: a death
  alone names no cause, since gen-2 attributes a sacrifice, a death trigger's
  damage and a static's -X/-X kill ``unresolved`` too, and writes no P/T change
  for a creature the check removed;
- a permanent the resolution moved itself (an event attributed to its chain);
- an Aura that is the record's source or target: its move is the ability's
  own (a sacrifice, a destroy), which gen-2 may attribute ``unresolved``; a
  bestowed Aura, which becomes a creature when its host leaves (702.103e);
  and an Aura whose host stops being a creature or gains a protection other
  than from a colour, against which the record does not say what the Aura
  may enchant;
- in ``toughness_zero``'s near miss, a creature whose marked damage the
  lowered toughness makes lethal: 704.5g's;
- an indestructible creature in the damage checks, which destroy, and in
  ``lethal_damage`` a creature dealt non-lethal damage by a deathtouch
  source, which is ``deathtouch_damage``'s;
- in the legend and world rules, a lone entering permanent at toughness 0,
  whose death is 704.5f's;
- a creature that stops being a creature in the record: its death is another
  rule's.

The record's own move is one from the battlefield to the graveyard, or to
exile where a replacement sends the dying permanent there instead
(Disintegrate, Rest in Peace), attributed ``unresolved``, and not followed by
its return to the battlefield (a flicker).

Two records show no check at all: the game-end rescue of orphaned events,
whose board is read after its events, and a record whose changes end the
game (a player at 0 life or 10 poison, or a game-end event): Forge stops the
check at the game's end, so nothing else of it happens.

Prior art: ``lethality_counterfactuals.qualify``, which reads the same facts
off a damage or shrink resolution; the rules are ``rule_outcomes``'.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors.acting_line import is_rescued_orphans
from effects.domain.rule_outcomes import damage_lethal
from effects.domain.state_snapshot import EntityState, normalize_counter_type

CHECKS = frozenset({
    "toughness_zero", "lethal_damage", "deathtouch_damage", "loyalty_zero", "defense_zero",
    "legend_rule", "world_rule", "aura_illegal",
})
_COUNTER_OF = {"loyalty_zero": ("LOYALTY", "planeswalker"), "defense_zero": ("DEFENSE", "battle")}
#: Where the check's move from the battlefield lands: the graveyard, or exile
#: in its place.
DEATH_ZONES = frozenset({"graveyard", "exile"})


@dataclass(slots=True)
class _Touched:
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


def _touched(record: EffectRecord) -> dict[str, _Touched]:
    found: dict[str, _Touched] = {}
    for event in record.payload.listed_events():
        params = event.params or {}
        for subject in event.subjects:
            t = found.setdefault(subject, _Touched())
            kind = event.type
            if kind is EventType.TYPE_CHANGE and "creature" in (params.get("types_removed") or ()):
                t.left_creatures = True
            elif kind is EventType.KEYWORD_CHANGE and not params.get("removed"):
                t.gained.update(k.lower() for k in params.get("keywords") or ())
            elif kind is EventType.PT_CHANGE:
                delta = int(params.get("toughness_delta") or 0)
                t.toughness_delta += delta
                t.lowered = t.lowered or delta < 0
            elif kind is EventType.DAMAGE_DEALT:
                t.damage_dealt += int(params.get("amount") or 0)
                # Gen-2 names a damage event's source as its ``cause``.
                source = params.get("source") or params.get("cause")
                if source:
                    t.damage_sources.append(source)
            elif kind is EventType.COUNTER_CHANGE:
                counter = normalize_counter_type(str(params.get("counter_type", "")))
                t.counter_delta[counter] = t.counter_delta.get(counter, 0) + int(
                    params.get("delta") or 0)
            elif kind is EventType.ZONE_CHANGE:
                if params.get("to_zone") == "battlefield":
                    t.entered = True
                    t.died = False  # back from where it went: a flicker, not a death
                if params.get("from_zone") == "battlefield":
                    t.left = True
                    if event.attributed_to != "unresolved":
                        t.moved_by_effect = True
                    elif params.get("to_zone") in DEATH_ZONES:
                        t.died = True
    return found


def game_ends(record: EffectRecord) -> bool:
    """Whether the record's own changes end the game: a game-end event, or a
    player its life and poison changes take to 0 life or 10 poison."""
    life: dict[str, int] = {}
    poison: dict[str, int] = {}
    for event in record.payload.listed_events():
        if event.type in (EventType.PLAYER_WON, EventType.PLAYER_LOST):
            return True
        delta = int((event.params or {}).get("delta") or 0)
        for subject in event.subjects:
            if event.type is EventType.LIFE_CHANGE:
                life[subject] = life.get(subject, 0) + delta
            elif event.type is EventType.POISON_CHANGE:
                poison[subject] = poison.get(subject, 0) + delta
    return any((player.id in life and player.life + life[player.id] <= 0)
               or (player.id in poison and player.poison + poison[player.id] >= 10)
               for player in record.state.players)


def _is(entity: EntityState, card_type: str) -> bool:
    return card_type in {t.lower() for t in entity.types}


def _counter(entity: EntityState, name: str) -> int:
    return sum(n for counter, n in entity.counters.items()
               if normalize_counter_type(counter) == name)


def _spell_or_activation(record: EffectRecord) -> bool:
    """Whether the acting line is a spell's or an activated ability's, whose
    board is read before it resolves; a trigger's, a keyword's (persist,
    undying) may read its dead host by last known information."""
    return bool(record.ability) and all(key.trait_kind == "spell" for key in record.ability)


#: A colour's protection keyword, as the record spells it, to the colour code.
_PROTECTION_FROM = {f"protection from {name}": code for name, code in (
    ("white", "W"), ("blue", "U"), ("black", "B"), ("red", "R"), ("green", "G"))}


def _protection_side(aura: EntityState, protections: list[str]) -> bool | None:
    """Whether a protection the host gains makes the Aura illegal on it: a
    colour's against the Aura's colours; None for any other protection, whose
    quality the record does not spell out."""
    if not protections:
        return False
    found = False
    for protection in protections:
        colour = _PROTECTION_FROM.get(protection)
        if colour is None:
            return None
        found = found or colour in aura.colors
    return found


def _deathtouched(record, touched, t: _Touched, keywords) -> bool:
    """Whether a source of the damage the creature was dealt has deathtouch,
    printed, granted on the board, or granted by the record itself."""
    for source in t.damage_sources:
        entity = record.state.entity(source)
        if "deathtouch" in touched.get(source, _Touched()).gained or (
                entity is not None and "deathtouch" in keywords(entity)):
            return True
    return False


def _zero_toughness_on_entry(record, touched, entity_id) -> bool:
    """Whether an entering creature's readable toughness is 0 or less: its
    death is 704.5f's, whatever else it shows."""
    entity = record.state.entity(entity_id)
    if entity is None or entity.pt is None or not _is(entity, "creature"):
        return False
    return entity.toughness + touched[entity_id].toughness_delta <= 0


def _creature_checks(record, touched, check, keywords) -> tuple[list, list, list]:
    fires, spared, observed = [], [], []
    for entity_id, t in touched.items():
        entity = record.state.entity(entity_id)
        if (entity is None or t.left_creatures or t.moved_by_effect or entity.pt is None
                or not _is(entity, "creature")):
            continue
        toughness = entity.toughness + t.toughness_delta
        if check == "toughness_zero":
            # The board may already stand at 0 (a cost removed the counter a
            # characteristic-defining toughness counts); else the record lowers it.
            at_zero = entity.on_battlefield and toughness <= 0 and _spell_or_activation(record)
            if not (t.lowered or at_zero) or t.damage_dealt:
                continue
            side = toughness <= 0
            if not side and 0 < toughness <= entity.damage:
                continue  # marked damage now lethal: 704.5g's, not a near miss
        elif not t.damage_dealt or "indestructible" in keywords(entity):
            continue
        elif check == "lethal_damage":
            side = damage_lethal(toughness, entity.damage, t.damage_dealt)
            if not side and _deathtouched(record, touched, t, keywords):
                continue  # 704.5h's, not a near miss of 704.5g
        else:  # deathtouch_damage: non-lethal damage, by whether a source has deathtouch
            if damage_lethal(toughness, entity.damage, t.damage_dealt):
                continue
            side = _deathtouched(record, touched, t, keywords)
        (fires if side else spared).append(entity_id)
        if t.died:
            observed.append(entity_id)
    return fires, spared, observed


def _counter_checks(record, touched, check) -> tuple[list, list, list]:
    counter, card_type = _COUNTER_OF[check]
    fires, spared, observed = [], [], []
    for entity_id, t in touched.items():
        entity = record.state.entity(entity_id)
        if entity is None or not _is(entity, card_type) or t.moved_by_effect:
            continue
        delta = t.counter_delta.get(counter, 0)
        if delta >= 0:
            continue
        (fires if _counter(entity, counter) + delta <= 0 else spared).append(entity_id)
        if t.died:
            observed.append(entity_id)
    return fires, spared, observed


def _uniqueness_checks(record, touched, check) -> tuple[list, list, list]:
    """The legend and world rules: who shares the battlefield once the record's moves are made."""
    supertype = "legendary" if check == "legend_rule" else "world"
    present = [
        e for e in record.state.entities
        if supertype in {s.lower() for s in e.supertypes}
        and (e.zone == "battlefield" or touched.get(e.id, _Touched()).entered)
        and not (touched.get(e.id, _Touched()).left and not touched[e.id].died)
    ]

    def group(entity: EntityState):
        return (entity.name, entity.controller) if check == "legend_rule" else ()

    groups: dict[tuple, list[str]] = {}
    for entity in present:
        groups.setdefault(group(entity), []).append(entity.id)
    fires, spared, observed = [], [], []
    for members in groups.values():
        died = [m for m in members if touched.get(m, _Touched()).died]
        if len(members) >= 2:
            fires += members
            observed += died
        elif any(touched.get(m, _Touched()).entered for m in members):
            # A lone legend entering at toughness 0 dies by 704.5f.
            if any(_zero_toughness_on_entry(record, touched, m) for m in died):
                continue
            spared += members
            observed += died
    return fires, spared, observed


def _aura_checks(record, touched, keywords) -> tuple[list, list, list]:
    fires, spared, observed = [], [], []
    acted_on = {record.state.refs.source, *record.state.refs.targets}
    for aura in record.state.entities:
        if ("aura" not in {s.lower() for s in aura.subtypes} or not aura.attached_to
                or not aura.on_battlefield or aura.id in acted_on
                or touched.get(aura.id, _Touched()).moved_by_effect
                or "bestow" in keywords(aura)):
            continue
        host = touched.get(aura.attached_to)
        if host is None:
            continue
        protected = _protection_side(
            aura, [k for k in host.gained if k.startswith("protection")])
        if protected is None or host.left_creatures:
            # What the Aura may enchant is not on the record (gen-2 spells
            # every enchant keyword ``enchant``), so a host changing types or
            # gaining an unnamed protection says nothing of its legality.
            continue
        (fires if host.left or protected else spared).append(aura.id)
        if touched.get(aura.id, _Touched()).died:
            observed.append(aura.id)
    return fires, spared, observed


@dataclass(frozen=True, slots=True)
class SbaZoneMove:
    id: str = "sba_zone_move"
    params: frozenset[str] = frozenset({"check"})
    general: frozenset[str] = frozenset({"check"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        check = params.get("check")
        if check not in CHECKS:
            raise ValueError(f"sba_zone_move: unknown check {check!r}; known: {sorted(CHECKS)}")
        if (record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION
                or is_rescued_orphans(record) or game_ends(record)):
            return None
        touched = _touched(record)
        if check in ("toughness_zero", "lethal_damage", "deathtouch_damage"):
            keywords = KeywordResolver(lines).keywords_of
            fires, spared, observed = _creature_checks(record, touched, check, keywords)
        elif check in _COUNTER_OF:
            fires, spared, observed = _counter_checks(record, touched, check)
        elif check == "aura_illegal":
            fires, spared, observed = _aura_checks(
                record, touched, KeywordResolver(lines).keywords_of)
        else:
            fires, spared, observed = _uniqueness_checks(record, touched, check)
        if fires:
            return FamilyMatch(Polarity.FIRES, any(e in observed for e in fires), tuple(fires))
        if spared:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, any(e in observed for e in spared),
                               tuple(spared))
        return None


FAMILY = register(SbaZoneMove())
