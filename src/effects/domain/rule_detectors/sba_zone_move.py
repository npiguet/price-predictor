"""Detector family ``sba_zone_move``: a permanent put into the graveyard by the state-based check.

Rule 704.5 lists the state-based actions that move a permanent from the
battlefield to the graveyard. Gen-2 already files every such move made in the
check after a resolution in that resolution's effect half, as a
``zone_change`` attributed ``unresolved``, so the family runs on gen-2
records and needs no gen-3 field.

Parameter ``check`` names which action, each read off the board the record
shows (the snapshot before the resolution, plus the resolution's own events):

=====================  ======================================================
``toughness_zero``     704.5f: a creature whose toughness, after the
                       resolution's P/T changes, is 0 or less; the near miss
                       is a creature the resolution lowered that keeps 1 or more
``lethal_damage``      704.5g: a creature the resolution dealt damage to, by
                       whether the damage marked reaches its toughness
``deathtouch_damage``  704.5h: a creature dealt non-lethal damage by a source
                       with deathtouch; the near miss is non-lethal damage
                       from a source without it
``loyalty_zero``       704.5i: a planeswalker whose loyalty counters, after
                       the resolution's counter changes, number 0
``defense_zero``       704.5v: a battle whose defense counters number 0
``legend_rule``        704.5j: two or more legendary permanents with one name
                       under one controller; the near miss is a legendary
                       permanent entering with no such twin
``world_rule``         704.5k: two or more world permanents; the near miss is
                       a world permanent entering alone
``aura_illegal``       704.5m: an Aura whose host left the battlefield in the
                       record; the near miss is an Aura whose host the record
                       touched and which stayed
=====================  ======================================================

The polarity is read from the board, ``observed`` from the record's own move,
so a record where Forge disagrees with the rule is the one
:attr:`RuleTag.needs_review` flags. A creature that stops being a creature in
the record is left out of the creature checks: its death is another rule's.

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
from effects.domain.rule_outcomes import damage_lethal
from effects.domain.state_snapshot import EntityState, normalize_counter_type

CHECKS = frozenset({
    "toughness_zero", "lethal_damage", "deathtouch_damage", "loyalty_zero", "defense_zero",
    "legend_rule", "world_rule", "aura_illegal",
})
_COUNTER_OF = {"loyalty_zero": ("LOYALTY", "planeswalker"), "defense_zero": ("DEFENSE", "battle")}


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
                if params.get("from_zone") == "battlefield":
                    t.left = True
                    if (event.attributed_to == "unresolved"
                            and params.get("to_zone") == "graveyard"):
                        t.died = True
    return found


def _is(entity: EntityState, card_type: str) -> bool:
    return card_type in {t.lower() for t in entity.types}


def _counter(entity: EntityState, name: str) -> int:
    return sum(n for counter, n in entity.counters.items()
               if normalize_counter_type(counter) == name)


def _creature_checks(record, touched, check, keywords) -> tuple[list, list, list]:
    fires, spared, observed = [], [], []
    for entity_id, t in touched.items():
        entity = record.state.entity(entity_id)
        if (entity is None or t.left_creatures or entity.pt is None
                or not _is(entity, "creature")):
            continue
        toughness = entity.toughness + t.toughness_delta
        if check == "toughness_zero":
            if (not t.lowered and not t.died) or t.damage_dealt:
                continue
            side = toughness <= 0
        elif check == "lethal_damage":
            if not t.damage_dealt:
                continue
            side = damage_lethal(toughness, entity.damage, t.damage_dealt)
        else:  # deathtouch_damage: non-lethal damage, by whether a source has deathtouch
            if not t.damage_dealt or damage_lethal(toughness, entity.damage, t.damage_dealt):
                continue
            sources = [record.state.entity(s) for s in t.damage_sources]
            side = any(s is not None and "deathtouch" in keywords(s) for s in sources)
        (fires if side else spared).append(entity_id)
        if t.died:
            observed.append(entity_id)
    return fires, spared, observed


def _counter_checks(record, touched, check) -> tuple[list, list, list]:
    counter, card_type = _COUNTER_OF[check]
    fires, spared, observed = [], [], []
    for entity_id, t in touched.items():
        entity = record.state.entity(entity_id)
        if entity is None or not _is(entity, card_type):
            continue
        delta = t.counter_delta.get(counter, 0)
        if delta >= 0 and not t.died:
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
            spared += members
            observed += died
    return fires, spared, observed


def _aura_checks(record, touched) -> tuple[list, list, list]:
    fires, spared, observed = [], [], []
    for aura in record.state.entities:
        if "aura" not in {s.lower() for s in aura.subtypes} or not aura.attached_to:
            continue
        host = touched.get(aura.attached_to)
        if host is None:
            continue
        (fires if host.left else spared).append(aura.id)
        if touched.get(aura.id, _Touched()).died:
            observed.append(aura.id)
    return fires, spared, observed


@dataclass(frozen=True, slots=True)
class SbaZoneMove:
    id: str = "sba_zone_move"
    params: frozenset[str] = frozenset({"check"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        check = params.get("check")
        if check not in CHECKS:
            raise ValueError(f"sba_zone_move: unknown check {check!r}; known: {sorted(CHECKS)}")
        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
            return None
        touched = _touched(record)
        if check in ("toughness_zero", "lethal_damage", "deathtouch_damage"):
            keywords = KeywordResolver(lines).keywords_of
            fires, spared, observed = _creature_checks(record, touched, check, keywords)
        elif check in _COUNTER_OF:
            fires, spared, observed = _counter_checks(record, touched, check)
        elif check == "aura_illegal":
            fires, spared, observed = _aura_checks(record, touched)
        else:
            fires, spared, observed = _uniqueness_checks(record, touched, check)
        if fires:
            return FamilyMatch(Polarity.FIRES, any(e in observed for e in fires), tuple(fires))
        if spared:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, any(e in observed for e in spared),
                               tuple(spared))
        return None


FAMILY = register(SbaZoneMove())
