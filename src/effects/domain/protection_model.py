"""What a protection copy is built from: the four outcomes, the rule each shows,
the :class:`Opportunity` a record offers, and the helpers every part of the
protection copies (:mod:`effects.domain.protection_copies`) shares.

Torch-free: ``build-corpus``'s survey and write workers call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from effects.domain.event_schema import Event, EventType
from effects.domain.rule_outcomes import (
    CARD_TYPES,
    CONDITIONS,
    Characteristics,
    ProtectionQuality,
    QualityKind,
)
from effects.domain.state_snapshot import EntityState, Zone


class ProtectionOutcome(StrEnum):
    """What a granted protection changes in a copied record."""

    DAMAGE_PREVENTED = "damage_prevented"
    TARGET_ILLEGAL = "target_illegal"
    BLOCK_ILLEGAL = "block_illegal"
    AURA_TO_GRAVEYARD = "aura_to_graveyard"


#: The rule each outcome shows, from a rare quality other than everything.
OUTCOME_RULES: dict[ProtectionOutcome, str] = {
    ProtectionOutcome.DAMAGE_PREVENTED: "702.16e",
    ProtectionOutcome.TARGET_ILLEGAL: "702.16b",
    ProtectionOutcome.BLOCK_ILLEGAL: "702.16f",
    ProtectionOutcome.AURA_TO_GRAVEYARD: "702.16c",
}
#: Protection from everything, which shows any of the four.
EVERYTHING_RULE = "702.16j"
PROTECTION_RULES: frozenset[str] = frozenset({*OUTCOME_RULES.values(), EVERYTHING_RULE})


@dataclass(frozen=True, slots=True)
class Opportunity:
    """One way a record can carry a protection copy."""

    outcome: ProtectionOutcome
    #: The entity granted the protection.
    protected: str
    #: The objects the quality must cover: the damage's sources, the spell or
    #: ability's card, the blocker, the Aura.
    covered: tuple[str, ...]
    #: The rare qualities every covered object has, everything left out; empty
    #: where they share none, which leaves the record to protection from
    #: everything alone.
    qualities: tuple[ProtectionQuality, ...]
    #: Whether protection from everything shows the outcome: not where an
    #: Aura or Equipment is attached to the protected permanent (702.16c-d).
    everything: bool = True

    @property
    def rule(self) -> str:
        return OUTCOME_RULES[self.outcome]


def rare_qualities(entity: EntityState) -> tuple[ProtectionQuality, ...]:
    """The rare qualities ``entity`` has, in a fixed order: its card types, its
    creature types (on a creature or a kindred card), the conditions it meets."""
    characteristics = Characteristics.of(entity)
    found = [ProtectionQuality(QualityKind.CARD_TYPE, t)
             for t in CARD_TYPES if t in characteristics.types]
    if characteristics.types & {"creature", "kindred"}:
        found += [ProtectionQuality(QualityKind.CREATURE_TYPE, s)
                  for s in sorted(characteristics.subtypes)]
    found += [ProtectionQuality(QualityKind.CONDITION, name)
              for name, condition in CONDITIONS.items() if condition.holds(characteristics)]
    return tuple(found)


def rules_offered(opportunities: tuple[Opportunity, ...]) -> tuple[str, ...]:
    """The protection rules a record's copies can be drawn for, sorted: each
    outcome's rule where a rare quality covers its objects, and 702.16j
    wherever any outcome is open."""
    rules = {o.rule for o in opportunities if o.qualities}
    if any(o.everything for o in opportunities):
        rules.add(EVERYTHING_RULE)
    return tuple(sorted(rules))


def shared_qualities(objects) -> tuple[ProtectionQuality, ...]:
    """The rare qualities every one of ``objects`` has, in the first's order."""
    lists = [rare_qualities(entity) for entity in objects]
    if not lists:
        return ()
    rest = [set(qualities) for qualities in lists[1:]]
    return tuple(q for q in lists[0] if all(q in other for other in rest))


def permanent(entities: dict[str, EntityState], ref: str | None) -> EntityState | None:
    """The entity ``ref`` names where it is on the battlefield, else None."""
    entity = entities.get(ref) if ref is not None else None
    return entity if entity is not None and entity.on_battlefield else None


def is_aura(entity: EntityState | None) -> bool:
    return entity is not None and "aura" in {s.lower() for s in entity.subtypes}


def entering(events: tuple[Event, ...], aura: str) -> Event | None:
    """The move of ``aura`` onto the battlefield among ``events``."""
    return next((event for event in events
                 if event.type is EventType.ZONE_CHANGE and event.subjects == (aura,)
                 and str(event.params.get("to_zone", "")).lower() == Zone.BATTLEFIELD), None)
