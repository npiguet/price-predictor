"""The canonical event vocabulary and its per-type field normalization.

An effect payload carries a list of ``{type, subjects, params, duration,
attributed_to}``. The type vocabulary is the union of Forge's trigger types, its
bus events, and the diffs a collection bracket computes; this module is its
single owner, so a collector and the model's target derivation cannot disagree
about what an event is called or which parameters it carries.

Types name **observable outcomes**, not the script API that produced them. Two
effects that both move a card to the graveyard emit ``zone_change``, and the
model learns one field group for it. That is the whole reason the vocabulary is
smaller than Forge's own effect-API list.

The collector's ``ApiEvents.RULES`` (Java, in the connector) maps those APIs --
keyed by the ``ApiType`` enum member name, the name Forge looks an effect up by
at runtime, not the effect class that implements it -- to the types each
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


# Per-type parameter keys. A collector writes exactly these keys for a type and a
# reader may assume no others, which is what "per-type field normalization" buys:
# the same outcome from two different script APIs lands in the same slots.
# A type absent from this table carries no params beyond its subjects.
EVENT_PARAMS: dict[EventType, tuple[str, ...]] = {
    # library_position is where in the library the card landed, counted from
    # the top: a `Moved` replacement that puts a card second-from-top instead
    # of into the graveyard changes only that, and without the slot the two
    # halves of the rewrite read alike and the record is dropped as an
    # identity. Absent where the destination is not a library.
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
    # Prevention is implemented entirely as replacement effects in Forge, so
    # this and the `rewrite` record for the same replacement are two views of
    # ONE engine event, one call frame apart: the `rewrite` record is written
    # from inside executeReplacement, before the engine computes the prevented
    # amount this row carries. They are not independent evidence of the same
    # prevention -- a reader that treats a `damage_prevented` row and its
    # sibling `rewrite` row as two confirming signals will double-count it.
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
    # colors_removed is additive: the head has a colors_lost field and the type
    # carried no way to say a colour went away, so nothing ever reached it.
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
    # The six Replace* APIs are exactly the getOverridingAbility() of a
    # replacement effect, so this and the `rewrite` record for the same
    # replacement are two views of ONE engine event, one call frame apart: the
    # `rewrite` record is written by RewriteChannel.record, from the
    # replacement hook around the same executeReplacement call this row's
    # ability runs from. They are not
    # independent evidence of the same replacement -- a reader that treats a
    # `replacement_applied` row and its sibling `rewrite` row as two
    # confirming signals will double-count it. (`rewrite` is not written
    # unconditionally either, for what it is worth: PatchedCollectors drops a
    # fork's own replacements and deduplicates by digest, so a
    # `replacement_applied` can legitimately have no visible `rewrite` sibling
    # in a given shard -- the "two views of one event" claim is about what the
    # engine fires, not a promise that a collector always keeps both halves.)
    #
    # It is also a NARROWER view than `rewrite`, and not by a little: measured
    # across all 2,459 replacement definitions in the sealed/draft pool, only
    # 14.4% (353) name one of these six APIs as the overriding ability -- 55.7%
    # (1,369) name some other API (802 of them `DB$ Tap`, the enters-tapped
    # class Guardian Idol belongs to and the single largest replacement shape
    # in the game), and 30.0% (737) have no `ReplaceWith$` at all (Prevent /
    # Skip / CantHappen). By card, 289 files can produce `replacement_applied`
    # against 1,235 that cannot. So a `rewrite` with no `replacement_applied`
    # sibling is the MODAL case -- roughly six shots in seven -- not a rare
    # corner this row happens to miss; "strict subset" is true but undersells
    # how narrow the six APIs are against everything `ReplaceWith$` can name.
    # No Replace* API is scripted anywhere in the pool outside a
    # `ReplaceWith$`/`ReplacementEffects$` chain -- the 103 cardsfolder
    # occurrences that do not look like one on a first grep all resolve to a
    # `ReplacementEffects$` SVar on an `Effect` -- so the subset holds in the
    # direction that actually matters: nothing this row's RULES entries claim
    # ever resolves outside a real replacement.
    #
    # The empty tuple is deliberate, not a placeholder: the event
    # carries subjects and provenance and nothing else, so unlike
    # DAMAGE_PREVENTED's row this one exists only to hold the comment above.
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
    # called distinguishes the two genuinely different result vocabularies
    # coin_flipped carries: heads/tails (NoCall$ True -- no caller, so
    # nothing to win or lose against) versus win/loss (every other shape).
    # The token sets are disjoint, so the dialect is inferable from the
    # values alone -- but a reader computing a win rate from results ==
    # "win" would silently drop every NoCall flip and have no signal that
    # it did. False for NoCall, true otherwise.
    EventType.COIN_FLIPPED: ("results", "called"),
    EventType.DICE_ROLLED: ("sides", "results"),
    EventType.CLASH_RESOLVED: ("won",),
    EventType.PILES_MADE: ("piles", "chosen"),
}


#: Event types the vocabulary once declared, and where the fact they named is
#: actually recorded. Retired rather than wired: each would have given one fact
#: two spellings, and a reader comparing a corpus against the vocabulary would
#: have read their absence as a collection failure.
#:
#: Removing a declared type is safe in exactly one direction: no corpus has ever
#: contained one, because nothing could emit them.
SUPERSEDED_EVENT_TYPES: dict[str, str] = {
    "cost_adjusted":
        "playability/decision payload, candidates[].cost_after_adjustment",
    "damage_assignment_ordered":
        "combat payload, assignment_choices",
    "name_change":
        "continuous payload, contributions[].name",
}


#: The types whose own parameter row declares ``cause``, so a collector that
#: leaves it empty on one of them is silent about something the schema asked
#: for. Every type may carry a ``cause`` — it is a provenance param — but only
#: these were judged to have one worth naming, and the validator measures the
#: population rate over exactly this set rather than over the whole vocabulary,
#: where a near-zero rate would be correct and mean nothing.
CAUSE_BEARING_TYPES: frozenset[EventType] = frozenset(
    event_type for event_type, params in EVENT_PARAMS.items() if "cause" in params
)


#: ``attributed_to`` when the acting line's own top-level clause produced the
#: event, rather than one of its sub-abilities.
ATTRIBUTION_ROOT = "root"

#: ``attributed_to`` when the collector looked for a producing clause and the
#: pointer named nothing on the acting chain. Distinct from :data:`ATTRIBUTION_ROOT`
#: because the two are different facts and were spelled the same for a whole
#: collection run: a null meant "the root acted" and "the pointer did not land"
#: at once, so 96.8% of resolution events said nothing at all and there was no
#: way to tell a working attribution channel from a dead one.
ATTRIBUTION_UNRESOLVED = "unresolved"

#: The sentinels, apart from a sub-ability's chain index.
ATTRIBUTION_SENTINELS: frozenset[str] = frozenset(
    {ATTRIBUTION_ROOT, ATTRIBUTION_UNRESOLVED}
)


class AttributionKind(StrEnum):
    """Which of the four things an event's ``attributed_to`` is saying.

    ``absent`` is the pre-tri-state spelling and means *unknown*, not *root*:
    shards collected before the sentinels existed wrote ``null`` for both the
    root line and a pointer that did not land, and the corpus is append-only, so
    the ambiguity cannot be resolved after the fact — only reported apart from
    the records that do say.
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

    ``None`` is none of the three: it is what a writer that predates the
    sentinels left behind, and it means *unknown*. Widening the value set
    rather than redefining the field is compatibility rule 1 — a reader that
    predates the sentinels still parses them as the strings they are.
    """

    type: EventType
    subjects: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    duration: str | None = None
    attributed_to: str | None = None

    #: Params any event may carry, because they say where it came from rather
    #: than what it was. ``mode`` is the engine's own name for the hook that
    #: produced it, kept because Forge has some two hundred trigger modes and
    #: this vocabulary is a closed set of outcomes: mapping every one would be
    #: a table nobody could keep true, and carrying the name loses nothing.
    #: ``cause`` is the entity or player ref of the object that caused the
    #: event, or absent where the hook names none. It is kept here as well as
    #: in ``EVENT_PARAMS`` so a type that has no ``cause`` row may still carry
    #: one; the two spellings the schema once had — a ref here, a comma-joined
    #: list of Forge run-parameter *key names* in the collector — could not both
    #: be true, and the key list was the accident: it was near-constant per
    #: event type and named no cause at all.
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
