"""The canonical event vocabulary and its per-type field normalization.

Types name observable outcomes, not the script API that produced them: two
effects that both move a card to the graveyard emit ``zone_change``. The
connector's ``ApiEvents.RULES`` maps Forge ``ApiType`` names to the types each
produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar


class EventType(StrEnum):
    """Every outcome an effect record can attribute to an ability."""

    # ── existence and zones ────────────────────────────────────────────
    ZONE_CHANGE = "zone_change"
    DESTROYED = "destroyed"
    SACRIFICED = "sacrificed"
    REGENERATED = "regenerated"
    PHASED = "phased"
    TOKEN_CREATED = "token_created"
    PERMANENT_COPIED = "permanent_copied"
    CARD_MADE = "card_made"
    PLAYER_REMOVED = "player_removed"

    # ── player resources ───────────────────────────────────────────────
    LIFE_CHANGE = "life_change"
    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_PREVENTED = "damage_prevented"
    DAMAGE_HEALED = "damage_healed"
    POISON_CHANGE = "poison_change"
    ENERGY_CHANGE = "energy_change"
    RADIATION_CHANGE = "radiation_change"
    MANA_PRODUCED = "mana_produced"
    MANA_LOST = "mana_lost"

    # ── card flow ──────────────────────────────────────────────────────
    CARD_DRAWN = "card_drawn"
    CARD_DISCARDED = "card_discarded"
    CARD_MILLED = "card_milled"
    CARD_REVEALED = "card_revealed"
    CARD_LOOKED_AT = "card_looked_at"
    LIBRARY_REORDERED = "library_reordered"
    LIBRARY_SHUFFLED = "library_shuffled"

    # ── permanent characteristics ──────────────────────────────────────
    COUNTER_CHANGE = "counter_change"
    PT_CHANGE = "pt_change"
    KEYWORD_CHANGE = "keyword_change"
    TYPE_CHANGE = "type_change"
    COLOR_CHANGE = "color_change"
    TEXT_CHANGE = "text_change"
    ABILITY_CHANGE = "ability_change"
    CONTROL_CHANGE = "control_change"
    OWNERSHIP_CHANGE = "ownership_change"
    ATTACHED = "attached"
    UNATTACHED = "unattached"
    FACE_CHANGE = "face_change"
    TAPPED = "tapped"
    UNTAPPED = "untapped"
    STATE_FLAG_CHANGE = "state_flag_change"
    RESTRICTION_CHANGE = "restriction_change"

    # ── combat ─────────────────────────────────────────────────────────
    ATTACKERS_DECLARED = "attackers_declared"
    BLOCKERS_DECLARED = "blockers_declared"
    BECAME_BLOCKED = "became_blocked"
    REMOVED_FROM_COMBAT = "removed_from_combat"
    COMBAT_ENDED = "combat_ended"

    # ── stack and spells ───────────────────────────────────────────────
    SPELL_CAST = "spell_cast"
    ABILITY_ACTIVATED = "ability_activated"
    SPELL_COUNTERED = "spell_countered"
    SPELL_COPIED = "spell_copied"
    TARGETS_CHANGED = "targets_changed"
    X_CHANGED = "x_changed"
    DELAYED_TRIGGER_CREATED = "delayed_trigger_created"
    TRIGGER_FIRED = "trigger_fired"
    REPLACEMENT_APPLIED = "replacement_applied"
    CONTINUOUS_EFFECT_CREATED = "continuous_effect_created"

    # ── turn structure ─────────────────────────────────────────────────
    PHASE_ADDED = "phase_added"
    PHASE_SKIPPED = "phase_skipped"
    TURN_ADDED = "turn_added"
    TURN_SKIPPED = "turn_skipped"
    TURN_ENDED = "turn_ended"
    TURN_ORDER_REVERSED = "turn_order_reversed"
    DAY_NIGHT_CHANGED = "day_night_changed"

    # ── game status ────────────────────────────────────────────────────
    PLAYER_WON = "player_won"
    PLAYER_LOST = "player_lost"
    GAME_DRAWN = "game_drawn"
    GAME_RESTARTED = "game_restarted"
    MONARCH_CHANGED = "monarch_changed"
    INITIATIVE_TAKEN = "initiative_taken"
    RING_TEMPTS = "ring_tempts"
    DUNGEON_VENTURED = "dungeon_ventured"
    SPEED_CHANGED = "speed_changed"

    # ── choices and randomness ─────────────────────────────────────────
    CHOICE_MADE = "choice_made"
    VOTE_TAKEN = "vote_taken"
    COIN_FLIPPED = "coin_flipped"
    DICE_ROLLED = "dice_rolled"
    CLASH_RESOLVED = "clash_resolved"
    PILES_MADE = "piles_made"


# Per-type parameter keys: a collector writes exactly these and a reader may
# assume no others. A type absent from this table carries no params.
EVENT_PARAMS: dict[EventType, tuple[str, ...]] = {
    # library_position counts from the top, absent outside a library: without
    # it a `Moved` replacement putting a card second-from-top reads as an
    # identity rewrite.
    EventType.ZONE_CHANGE: (
        "from_zone", "to_zone", "cause", "library_position",
    ),
    EventType.DESTROYED: ("regenerable", "cause"),
    EventType.SACRIFICED: ("cause",),
    EventType.PHASED: ("out",),
    EventType.TOKEN_CREATED: ("token_script_id", "characteristics", "count"),
    EventType.PERMANENT_COPIED: ("copy_source", "count"),
    EventType.CARD_MADE: ("card_name", "to_zone", "count"),
    EventType.LIFE_CHANGE: ("delta",),
    EventType.DAMAGE_DEALT: ("amount", "combat", "source", "excess"),
    # Forge prevents by replacement, so this and the `rewrite` record for the
    # same replacement are two views of one engine event, not two confirming
    # signals.
    EventType.DAMAGE_PREVENTED: ("amount", "source"),
    EventType.DAMAGE_HEALED: ("amount",),
    EventType.POISON_CHANGE: ("delta",),
    EventType.ENERGY_CHANGE: ("delta",),
    EventType.RADIATION_CHANGE: ("delta",),
    EventType.MANA_PRODUCED: ("mana_by_color",),
    EventType.MANA_LOST: ("mana_by_color",),
    EventType.CARD_DRAWN: ("count",),
    EventType.CARD_DISCARDED: ("count", "random"),
    EventType.CARD_MILLED: ("count",),
    EventType.CARD_REVEALED: ("count", "from_zone"),
    EventType.CARD_LOOKED_AT: ("count", "from_zone"),
    EventType.LIBRARY_REORDERED: ("count",),
    EventType.COUNTER_CHANGE: ("counter_type", "delta"),
    EventType.PT_CHANGE: ("power_delta", "toughness_delta", "set_pt"),
    EventType.KEYWORD_CHANGE: ("keywords", "removed"),
    EventType.TYPE_CHANGE: ("types_added", "types_removed", "overwrite"),
    EventType.COLOR_CHANGE: ("colors", "colors_removed", "overwrite"),
    EventType.ABILITY_CHANGE: ("abilities", "removed"),
    EventType.CONTROL_CHANGE: ("controller",),
    EventType.OWNERSHIP_CHANGE: ("owner",),
    EventType.ATTACHED: ("attached_to",),
    EventType.UNATTACHED: ("detached_from",),
    EventType.FACE_CHANGE: ("to_state",),
    EventType.STATE_FLAG_CHANGE: ("flag", "value"),
    EventType.RESTRICTION_CHANGE: ("restriction", "value"),
    EventType.ATTACKERS_DECLARED: ("defender",),
    EventType.BLOCKERS_DECLARED: ("blocked",),
    EventType.BECAME_BLOCKED: ("blockers",),
    EventType.SPELL_CAST: ("without_paying", "from_zone"),
    EventType.SPELL_COPIED: ("count", "new_targets"),
    EventType.TARGETS_CHANGED: ("targets",),
    EventType.X_CHANGED: ("value",),
    # Emitted by the six Replace* APIs, the overriding ability of a
    # replacement, so it shares an engine event with that replacement's
    # `rewrite` record. A narrow subset: most replacements run another API or
    # none, so a `rewrite` usually has no `replacement_applied` sibling. The
    # empty row exists only to carry this note.
    EventType.REPLACEMENT_APPLIED: (),
    EventType.CONTINUOUS_EFFECT_CREATED: ("layers",),
    EventType.PHASE_ADDED: ("phase", "count"),
    EventType.PHASE_SKIPPED: ("phase",),
    EventType.TURN_ADDED: ("count",),
    EventType.TURN_SKIPPED: ("count",),
    EventType.DAY_NIGHT_CHANGED: ("to",),
    EventType.PLAYER_LOST: ("reason",),
    EventType.DUNGEON_VENTURED: ("dungeon", "room"),
    EventType.SPEED_CHANGED: ("delta",),
    EventType.CHOICE_MADE: ("choice_kind", "value"),
    EventType.VOTE_TAKEN: ("options", "tally"),
    # called is False for a NoCall$ flip, whose results are heads/tails
    # rather than win/loss; a win rate over results == "win" would silently
    # drop those.
    EventType.COIN_FLIPPED: ("results", "called"),
    EventType.DICE_ROLLED: ("sides", "results"),
    EventType.CLASH_RESOLVED: ("won",),
    EventType.PILES_MADE: ("piles", "chosen"),
}


# Retired types, never in a corpus, each recorded elsewhere: ``cost_adjusted``
# as a decision candidate's ``cost_after_adjustment``,
# ``damage_assignment_ordered`` as the combat payload's ``assignment_choices``,
# and ``name_change`` as a continuous contribution's ``name``.


#: The types whose own parameter row declares ``cause``: the set the validator
#: measures ``cause``'s population rate over, since over the whole vocabulary
#: a near-zero rate would be correct.
CAUSE_BEARING_TYPES: frozenset[EventType] = frozenset(
    event_type for event_type, params in EVENT_PARAMS.items() if "cause" in params
)


#: ``attributed_to`` when the acting line's own top-level clause produced the
#: event, rather than one of its sub-abilities.
ATTRIBUTION_ROOT = "root"

#: ``attributed_to`` when the collector looked for a producing clause and the
#: pointer named nothing on the acting chain.
ATTRIBUTION_UNRESOLVED = "unresolved"

class AttributionKind(StrEnum):
    """Which of the four things an event's ``attributed_to`` is saying.

    ``absent`` means unknown, not root: shards collected before the sentinels
    wrote ``null`` for both.
    """

    ABSENT = "absent"
    ROOT = "root"
    UNRESOLVED = "unresolved"
    SUB_ABILITY = "sub_ability"


def attribution_kind(attributed_to: str | None) -> AttributionKind:
    """The :class:`AttributionKind` an event's ``attributed_to`` is saying."""
    if attributed_to is None:
        return AttributionKind.ABSENT
    if attributed_to == ATTRIBUTION_ROOT:
        return AttributionKind.ROOT
    if attributed_to == ATTRIBUTION_UNRESOLVED:
        return AttributionKind.UNRESOLVED
    return AttributionKind.SUB_ABILITY


@dataclass(frozen=True, slots=True)
class Event:
    """One attributed outcome inside an effect payload.

    ``subjects`` are entity or player refs, never names — the snapshot carries
    the identities. ``duration`` is ``None`` for an immediate outcome and names
    the end condition (``end_of_turn``, ``your_next_turn``, ``permanent``) for a
    continuous one.

    ``attributed_to`` is tri-state, and the three states are what
    :func:`attribution_kind` reads back:

    ======================  ==================================================
    a chain index (``"2"``) the sub-ability at that position produced it
    ``root``                the acting line's own clause produced it
    ``unresolved``          a producing clause was sought and the pointer
                            named nothing on this chain
    ======================  ==================================================

    ``None`` is a writer that predates the sentinels: unknown.
    """

    type: EventType
    subjects: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    duration: str | None = None
    attributed_to: str | None = None

    #: Params any event may carry, saying where it came from: ``mode`` is the
    #: engine's name for the producing hook (too many to map into the closed
    #: vocabulary), ``cause`` the ref of the object that caused it, absent
    #: where the hook names none.
    PROVENANCE_PARAMS: ClassVar[frozenset[str]] = frozenset({"mode", "cause"})

    def __post_init__(self) -> None:
        allowed = set(EVENT_PARAMS.get(self.type, ())) | self.PROVENANCE_PARAMS
        unknown = set(self.params) - allowed
        if unknown:
            raise ValueError(
                f"event {self.type.value} carries params {sorted(unknown)} that "
                f"the schema does not normalize; allowed: {sorted(allowed)}"
            )

    def as_dict(self) -> dict:
        return {
            "type": self.type.value,
            "subjects": list(self.subjects),
            "params": dict(self.params),
            "duration": self.duration,
            "attributed_to": self.attributed_to,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Event:
        return cls(
            type=EventType(data["type"]),
            subjects=tuple(data.get("subjects", ())),
            params=dict(data.get("params", {})),
            duration=data.get("duration"),
            attributed_to=data.get("attributed_to"),
        )
