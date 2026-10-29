"""The pre-event game state every effect record carries.

Stored as data and tensorized at training time by
:mod:`effects.domain.effect_head_input`, so a representation change is a code
change rather than a re-collection. Controllers are absolute player ids — mine
and opponent derive from the record's ``actor_player`` — and an absent
inclusion tier means uncollected, not empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from effects.domain.event_schema import Event
from effects.domain.provenance import ProvenanceKey


class InclusionTier(IntEnum):
    """What a snapshot was allowed to include, in application order.

    Ordered: a corpus never holds tier 3 without tier 2.
    """

    #: Every entity-valued ref, in whatever zone it sits, with that zone recorded.
    REFERENCED = 1
    #: Global state, battlefield entities, and command-zone effect cards.
    CORE = 2
    #: Stack objects that no ref already pulled in.
    UNREFERENCED_STACK = 3
    #: Hand and graveyard contents that no ref already pulled in.
    UNREFERENCED_HAND_GRAVEYARD = 4


#: The tiers stage-one collection writes; later stages extend this prefix.
STAGE_ONE_TIERS: frozenset[InclusionTier] = frozenset(
    {InclusionTier.REFERENCED, InclusionTier.CORE}
)

COLORS: tuple[str, ...] = ("W", "U", "B", "R", "G", "C")


@dataclass(frozen=True, slots=True)
class GlobalState:
    """Turn structure and the parts of the board that belong to nobody."""

    turn: int
    phase: str
    active: str
    priority: str | None
    stack_size: int
    combat_substep: str | None = None
    #: Command-zone emblems, as the provenance key of the line that made each.
    emblems: tuple[ProvenanceKey, ...] = ()


@dataclass(frozen=True, slots=True)
class PlayerState:
    """One player's resources.

    ``floating_mana`` and ``untapped_production`` are keyed by the symbols in
    :data:`COLORS`; ``this_turn`` holds the per-turn counters (creatures died,
    spells cast, lands played) that trigger conditions read.
    """

    id: str
    life: int
    hand: int
    library: int
    graveyard: int
    poison: int = 0
    energy: int = 0
    this_turn: dict[str, int] = field(default_factory=dict)
    floating_mana: dict[str, int] = field(default_factory=dict)
    untapped_production: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PowerToughness:
    """Power and toughness decomposed into the three sources that set it.

    Kept separate because a boost and a counter are different effects with
    the same total.
    """

    base: tuple[int, int]
    boosts: tuple[int, int] = (0, 0)
    counters: tuple[int, int] = (0, 0)

    @property
    def total(self) -> tuple[int, int]:
        return (
            self.base[0] + self.boosts[0] + self.counters[0],
            self.base[1] + self.boosts[1] + self.counters[1],
        )


@dataclass(frozen=True, slots=True)
class CombatStatus:
    """An entity's place in the current combat, if any."""

    attacking: str | None = None
    blocking: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    #: True once this attacker is blocked, even if every blocker later leaves.
    became_blocked: bool = False


@dataclass(frozen=True, slots=True)
class GrantedTemporary:
    """What the timestamped change tables report for one entity.

    Provenance keys where a grant resolves to a printed line, bare keywords
    where it does not ("gains flying until end of turn"). Separate from
    ``granted_attached`` because temporary grants ride the overlay rather than
    the ability tokens, and gate 2 strips a keyword from whichever carries it.
    """

    keywords: tuple[str, ...] = ()
    abilities: tuple[ProvenanceKey, ...] = ()


def normalize_counter_type(raw: str) -> str:
    """Forge's spelling of a counter type, in the one this package reads.

    Forge prints ``"+1/+1"`` where :data:`COUNTER_TYPES` spells ``P1P1``;
    without this the commonest counters fall through to
    ``counters_delta_other``. Forge's naming rule read backwards, so rarer
    shapes (``+1/+0``) need no table.
    """
    return (
        raw.strip()
        .replace("+", "P")
        .replace("-", "M")
        .replace("/", "")
        .upper()
    )


@dataclass(frozen=True, slots=True)
class StackExtras:
    """Fields an entity carries only while it is a stack object."""

    targets: tuple[str, ...] = ()
    #: Announced amount per target, e.g. divided damage.
    per_target_amounts: dict[str, int] = field(default_factory=dict)
    #: "Up to N" counts the caster announced.
    up_to_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntityState:
    """One card, token, stack object, or command-zone effect in the snapshot.

    Characteristics are computed (post-layer), never printed; a
    ``continuous`` record's snapshot omits the acting static's own
    contributions.
    """

    id: str
    name: str
    zone: str
    controller: str
    face: int = 0
    copy_source: str | None = None
    token_script_id: str | None = None

    types: tuple[str, ...] = ()
    subtypes: tuple[str, ...] = ()
    supertypes: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    mana_value: int = 0
    pt: PowerToughness | None = None

    tapped: bool = False
    sick: bool = False
    damage: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    combat: CombatStatus | None = None
    attached_to: str | None = None
    face_down: bool = False

    granted_attached: tuple[ProvenanceKey, ...] = ()
    granted_temporary: GrantedTemporary = field(default_factory=GrantedTemporary)

    #: The entity's own printed lines, as keys into its sidecar.
    printed: tuple[ProvenanceKey, ...] = ()

    stack_extras: StackExtras | None = None


@dataclass(frozen=True, slots=True)
class Refs:
    """What the acting ability points at, and what was chosen for it."""

    targets: tuple[str, ...] = ()
    source: str | None = None
    modes: tuple[str, ...] = ()
    x: int | None = None
    #: Resolution-time engine choices, keyed by the choice's name in the script.
    choices: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """The board as it stood immediately before the record's event.

    ``global_`` is the JSON field ``global``.
    """

    global_: GlobalState
    players: tuple[PlayerState, ...]
    entities: tuple[EntityState, ...]
    refs: Refs = field(default_factory=Refs)
    #: ``rewrite`` and ``trigger`` records only: the incoming event being handled.
    pending_event: Event | None = None
    #: Which tiers the collector wrote. Never inferred.
    tiers: frozenset[InclusionTier] = STAGE_ONE_TIERS

    def collected(self, tier: InclusionTier) -> bool:
        """Whether ``tier`` was collected; check before reading an empty list
        as an empty zone."""
        return tier in self.tiers

    def entity(self, entity_id: str) -> EntityState | None:
        for candidate in self.entities:
            if candidate.id == entity_id:
                return candidate
        return None

    def player(self, player_id: str) -> PlayerState | None:
        for candidate in self.players:
            if candidate.id == player_id:
                return candidate
        return None

    def controller_tag(self, controller: str, actor_player: str) -> str:
        """``"mine"`` or ``"opponent"`` relative to the record's actor."""
        return "mine" if controller == actor_player else "opponent"
