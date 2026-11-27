"""What the damage-step tests share: a creature, a damage step's global state.

The callers build different snapshots around them (their own players, tiers
and records), so only the pieces both spell the same way live here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    GlobalState,
    GrantedTemporary,
    PowerToughness,
)
from tests.unit.effects.builders import lines_of, make_entity

#: Each damage step's phase, keyed by its substep, as a real record spells them.
DAMAGE_STEP_PHASES = {
    "first_strike": "combat_first_strike_damage",
    "regular": "combat_damage",
}


def damage_step_global(substep: str) -> GlobalState:
    """Turn 4, ``P0`` active and holding priority, in the damage step ``substep``."""
    return GlobalState(
        turn=4, phase=DAMAGE_STEP_PHASES[substep], active="P0", priority="P0",
        stack_size=0, combat_substep=substep,
    )


@dataclass(frozen=True)
class CreatureTraits:
    """What a test's creature has besides its name and size."""

    controller: str = "P0"
    damage: int = 0
    #: Keywords granted until end of turn.
    keywords: tuple[str, ...] = ()
    #: The lines it prints, one per key.
    printed: tuple[ProvenanceKey, ...] = ()
    combat: CombatStatus | None = None
    #: Whether its static lines are marked active, as gen-3 writes them.
    active_statics: bool = False


def creature(
    entity_id: str, power: int, toughness: int, traits: CreatureTraits = CreatureTraits(),
) -> EntityState:
    """A battlefield creature, as ``traits`` describe it."""
    entity = make_entity(
        entity_id, entity_id, controller=traits.controller, types=("creature",),
        pt=PowerToughness(base=(power, toughness)), damage=traits.damage,
        combat=traits.combat, lines=lines_of(traits.printed),
        granted_temporary=GrantedTemporary(keywords=traits.keywords),
    )
    if not traits.active_statics:
        return entity
    return replace(entity, lines=tuple(
        replace(line, active=True) if line.kind == "static" else line
        for line in entity.lines))
