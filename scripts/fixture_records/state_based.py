"""Effect halves the state-based check acts on: a creature dying at
toughness 0 or less (704.5f), one left at toughness 1, a creature dying of
damage (704.5g), and the deaths the ``sba_zone_move`` detector sees."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.sba_zone_move import FAMILY as SBA_ZONE_MOVE
from effects.infrastructure.record_codec import record_from_dict
from fixture_records.shapes import (
    entities_by_id,
    events,
    is_creature,
    is_death,
    is_effect_resolution,
    params,
    subjects,
)


def toughness(entity: dict) -> int | None:
    """Net toughness: ``base_pt`` plus the temporary and counter boosts."""
    base, boosts = entity.get("base_pt"), entity.get("pt_boosts")
    if base is None or boosts is None:
        return None
    return base[1] + boosts["temporary"][1] + boosts["counters"][1]


def _removes_creature_type(event: dict) -> bool:
    return "creature" in (params(event).get("types_removed") or ())


@dataclass
class _PtChanges:
    """What a record's events did to each subject's toughness and life."""

    delta: dict[str, int] = field(default_factory=dict)
    damaged: set[str] = field(default_factory=set)
    died: set[str] = field(default_factory=set)
    #: No longer creatures, so a death is not 704.5f's (an Aura with nothing
    #: to enchant is 704.5m).
    uncreatured: set[str] = field(default_factory=set)

    @classmethod
    def of(cls, record: dict) -> _PtChanges:
        changes = cls()
        for event in events(record):
            for subject in subjects(event):
                changes._read(event, subject)
        return changes

    def _read(self, event: dict, subject: str) -> None:
        kind = event["type"]
        if kind == "type_change" and _removes_creature_type(event):
            self.uncreatured.add(subject)
        elif kind == "pt_change":
            change = int(params(event).get("toughness_delta", 0))
            self.delta[subject] = self.delta.get(subject, 0) + change
        elif kind == "damage_dealt":
            self.damaged.add(subject)
        elif is_death(event):
            self.died.add(subject)

    def toughness_after(self, entity_id: str, entity: dict | None) -> int | None:
        """The creature's toughness after the changes; None for a non-creature."""
        if entity is None or entity_id in self.uncreatured:
            return None
        if not is_creature(entity):
            return None
        base = toughness(entity)
        return base + self.delta.get(entity_id, 0) if base is not None else None


def toughness_after(record: dict) -> dict[str, tuple[int, bool, bool]]:
    """``entity id -> (toughness after the record's own P/T changes, damaged,
    died in the state-based check)`` for each creature the record names."""
    changes = _PtChanges.of(record)
    found = entities_by_id(record)
    out = {}
    for entity_id in set(changes.delta) | changes.died:
        after = changes.toughness_after(entity_id, found.get(entity_id))
        if after is not None:
            out[entity_id] = (after, entity_id in changes.damaged, entity_id in changes.died)
    return out


def toughness_zero_death(record: dict) -> bool:
    """704.5f fires: a creature dies in the check at toughness 0 or less, undamaged."""
    return is_effect_resolution(record) and any(
        after <= 0 and died and not damaged
        for after, damaged, died in toughness_after(record).values())


def _lowered(record: dict) -> set[str]:
    """The subjects whose toughness a ``pt_change`` lowers."""
    return {subject for event in events(record)
            if event["type"] == "pt_change" and int(params(event).get("toughness_delta", 0)) < 0
            for subject in subjects(event)}


def left_at_toughness_one(record: dict) -> bool:
    """704.5f does not fire: a creature's toughness is lowered to 1 and it stays."""
    if not is_effect_resolution(record):
        return False
    lowered = _lowered(record)
    return any(entity_id in lowered and after == 1 and not died
               for entity_id, (after, _, died) in toughness_after(record).items())


def lethal_damage_death(record: dict) -> bool:
    """704.5g fires: a creature dealt damage by the resolution dies in the check."""
    return is_effect_resolution(record) and any(
        damaged and died and after > 0
        for after, damaged, died in toughness_after(record).values())


def _has_unresolved_zone_change(record: dict) -> bool:
    return any(event["type"] == "zone_change" and event.get("attributed_to") == "unresolved"
               for event in events(record))


def sba_fires(check: str) -> Callable[[dict], bool]:
    """A death the ``sba_zone_move`` detector puts on the rule's fires side and
    sees happen: the detector itself picks the record, so the fixture holds
    what the family test reads. Only effect halves with an unresolved
    battlefield death are parsed, which keeps the scan cheap."""
    def test(record: dict) -> bool:
        if not is_effect_resolution(record) or not _has_unresolved_zone_change(record):
            return False
        found = SBA_ZONE_MOVE.match(record_from_dict(record), None, {"check": check})
        return found is not None and found.polarity is Polarity.FIRES and found.observed
    return test
