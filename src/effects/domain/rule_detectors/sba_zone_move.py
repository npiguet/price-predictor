"""Detector family ``sba_zone_move``: a death in the state-based check after a resolution.

Rule 704.5: the state-based check puts a creature with toughness 0 or less
(704.5f) or with lethal damage marked (704.5g) into its owner's graveyard.
Gen-2 already files every battlefield-to-graveyard move made in that check in
the resolution's effect half, as a ``zone_change`` attributed ``unresolved``,
so the family runs on gen-2 records and needs no gen-3 field.

Parameter ``check`` names which state-based action:

- ``toughness_zero``: a creature whose toughness, after the resolution's own
  P/T changes, is 0 or less. The near miss is a creature the resolution
  lowered that keeps toughness 1 or more.
- ``lethal_damage``: a creature the resolution dealt damage to, by whether
  the damage marked reaches its toughness. Deathtouch is not read, so a
  creature killed by a small deathtouch hit reaches the review list.

The polarity is read from the board (the toughness and damage the rule
decides on) and ``observed`` from the record's own move, so a record where
Forge disagrees with the rule is the one :attr:`RuleTag.needs_review` flags.
A creature that stops being a creature in the record is left out: its death
is another rule's (an Aura with nothing to enchant is 704.5m).

Prior art: ``lethality_counterfactuals.qualify``, which reads the same three
facts off a damage or shrink resolution; the rule itself is
``rule_outcomes.damage_lethal`` and ``shrink_lethal``.
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
from effects.domain.rule_outcomes import damage_lethal

CHECKS = frozenset({"toughness_zero", "lethal_damage"})


@dataclass(frozen=True, slots=True)
class _Creature:
    toughness_before: int
    damage_before: int
    toughness_delta: int = 0
    damage_dealt: int = 0
    lowered: bool = False
    damaged: bool = False
    died: bool = False


def _creatures(record: EffectRecord) -> dict[str, _Creature]:
    """Each battlefield creature the record's events name, with what happened to it."""
    found: dict[str, dict[str, Any]] = {}
    left_creatures: set[str] = set()
    for event in record.payload.listed_events():
        params = event.params or {}
        for subject in event.subjects:
            entry = found.setdefault(subject, {})
            if event.type is EventType.TYPE_CHANGE and "creature" in (
                    params.get("types_removed") or ()):
                left_creatures.add(subject)
            elif event.type is EventType.PT_CHANGE:
                delta = int(params.get("toughness_delta") or 0)
                entry["toughness_delta"] = entry.get("toughness_delta", 0) + delta
                entry["lowered"] = entry.get("lowered", False) or delta < 0
            elif event.type is EventType.DAMAGE_DEALT:
                amount = int(params.get("amount") or 0)
                entry["damage_dealt"] = entry.get("damage_dealt", 0) + amount
                entry["damaged"] = True
            elif (event.type is EventType.ZONE_CHANGE and event.attributed_to == "unresolved"
                  and params.get("from_zone") == "battlefield"
                  and params.get("to_zone") == "graveyard"):
                entry["died"] = True
    creatures = {}
    for entity_id, entry in found.items():
        entity = record.state.entity(entity_id)
        if (entity is None or entity_id in left_creatures or entity.pt is None
                or "creature" not in {t.lower() for t in entity.types}):
            continue
        creatures[entity_id] = _Creature(entity.toughness, entity.damage, **entry)
    return creatures


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
        fires: list[str] = []
        spared: list[str] = []
        observed: list[str] = []
        for entity_id, c in _creatures(record).items():
            if check == "toughness_zero":
                if not c.lowered and not c.died:
                    continue
                if c.damaged:
                    continue  # lethal damage would decide it, not toughness
                at_zero = c.toughness_before + c.toughness_delta <= 0
            else:
                if not c.damaged:
                    continue
                at_zero = damage_lethal(
                    c.toughness_before + c.toughness_delta, c.damage_before, c.damage_dealt)
            (fires if at_zero else spared).append(entity_id)
            if c.died:
                observed.append(entity_id)
        if fires:
            return FamilyMatch(Polarity.FIRES, all(e in observed for e in fires), tuple(fires))
        if spared:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, bool(observed), tuple(spared))
        return None


FAMILY = register(SbaZoneMove())
