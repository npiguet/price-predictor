"""The pre-event game state every effect record carries.

State is stored as **data** — names plus dynamic attributes — and tensorized at
training time by :mod:`effects.domain.effect_head_input`. That is the one
collection principle the corpus rests on: the corpus is expensive to rebuild and
cheap to re-read, so every representation change has to be a code change rather
than a re-collection.

Two rules are easy to get wrong and are enforced here rather than left to a
collector's discretion:

- **Perspective is not stored.** Controllers are absolute player ids; mine and
  opponent derive at training time from the record's ``actor_player``. A
  snapshot written from one player's point of view could not be reused by a
  record whose actor is the other player.
- **An absent inclusion tier means uncollected, not empty.** A stage-one
  snapshot has no stack contents because tier 3 was not collected, not because
  the stack was empty, and a reader that confuses the two trains the model to
  predict on a board it never saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from effects.domain.event_schema import Event
from effects.domain.provenance import ProvenanceKey


class InclusionTier(IntEnum):
    """What a snapshot was allowed to include, in application order.

    Tiers 1–2 are collected from stage one, 3 from stage two, 4 from stage
    three. They are ordered because each stage adds the next one: a corpus never
    holds tier 3 without tier 2.
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

    Kept separate rather than summed because an ability that adds a boost and
    one that adds a counter are different effects with the same total, and the
    model has to predict which one happened.
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

    Provenance keys where a grant resolves to a printed line, and bare keyword
    strings where it does not — "gains flying until end of turn" names no line.
    This stays separate from ``granted_attached`` because the model reads the two
    differently: an entity's ability tokens are its printed and
    attachment-granted lines only, while temporary grants ride the overlay. Gate
    2's perturbation removes a keyword from whichever of the two carries it, so
    merging them would make the gate unable to tell which channel to strip.
    """

    keywords: tuple[str, ...] = ()
    abilities: tuple[ProvenanceKey, ...] = ()


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

    Characteristics are **computed** (post-layer), never printed. The single
    exception is a ``continuous`` record, whose snapshot has the acting static's
    own contributions removed from every layer channel it wrote — otherwise the
    model would be asked to predict a state that already contains its answer.
    """

    id: str
    name: str
    zone: str
    controller: str
    face: int = 0
    copy_source: str | None = None
    token_script_id: str | None = None

    # computed characteristics
    types: tuple[str, ...] = ()
    subtypes: tuple[str, ...] = ()
    supertypes: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    mana_value: int = 0
    pt: PowerToughness | None = None

    # board state
    tapped: bool = False
    sick: bool = False
    damage: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    combat: CombatStatus | None = None
    attached_to: str | None = None
    face_down: bool = False

    # granted abilities — the two channels stay separate; see GrantedTemporary
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

    ``global_`` is spelled with a trailing underscore because ``global`` is a
    Python keyword; the JSON field is ``global``.
    """

    global_: GlobalState
    players: tuple[PlayerState, ...]
    entities: tuple[EntityState, ...]
    refs: Refs = field(default_factory=Refs)
    #: ``rewrite`` and ``trigger`` records only: the incoming event being handled.
    pending_event: Event | None = None
    #: Which tiers the collector was configured to write. Never inferred.
    tiers: frozenset[InclusionTier] = STAGE_ONE_TIERS

    def collected(self, tier: InclusionTier) -> bool:
        """Whether ``tier`` was collected at all.

        The rule readers must apply before concluding anything from an empty
        list: a snapshot with no stack entities and without
        :attr:`InclusionTier.UNREFERENCED_STACK` says nothing about the stack.
        """
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
        """``"mine"`` or ``"opponent"`` relative to the record's actor.

        Derived here rather than stored, which is what lets one snapshot serve
        records with different actors.
        """
        return "mine" if controller == actor_player else "opponent"
