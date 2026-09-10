"""The canonical event vocabulary and its per-type field normalization.

An effect payload carries a list of ``{type, subjects, params, duration,
attributed_to}``. The type vocabulary is the union of Forge's trigger types, its
bus events, and the diffs a collection bracket computes; this module is its
single owner, so a collector and the model's target derivation cannot disagree
about what an event is called or which parameters it carries.

Types name **observable outcomes**, not the script API that produced them. Two
effects that both move a card to the graveyard emit ``zone_change``, and the
model learns one field group for it. That is the whole reason the vocabulary is
smaller than Forge's own effect-API list (``EFFECT_APIS`` in
:mod:`effects.domain.forge_effect_apis`, a test asserts the size relationship
rather than a count copied here to go stale).

:data:`EFFECT_API_EVENTS` maps every one of those APIs -- keyed by its
``ApiType`` enum member name, the name Forge looks an effect up by at runtime,
not the effect class that implements it -- to the types it can produce, and
:data:`EXCLUDED_EFFECT_APIS` records the ones that produce none, each with its
reason. ``test_event_schema_completeness`` asserts the two together cover the
checked-in API list, so a Forge upgrade that adds an effect API fails the fast
suite rather than silently going unrecorded.
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
    EventType.COIN_FLIPPED: ("results",),
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


def attribution_kind(attributed_to: str | None) -> str:
    """Which of the four things an event's ``attributed_to`` is saying.

    ``absent`` is the pre-tri-state spelling and means *unknown*, not *root*:
    shards collected before the sentinels existed wrote ``null`` for both the
    root line and a pointer that did not land, and the corpus is append-only, so
    the ambiguity cannot be resolved after the fact — only reported apart from
    the records that do say.
    """
    if attributed_to is None:
        return "absent"
    if attributed_to == ATTRIBUTION_ROOT:
        return "root"
    if attributed_to == ATTRIBUTION_UNRESOLVED:
        return "unresolved"
    return "sub_ability"


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


# Shorthands used only to keep the mapping table below readable.
_ANIMATE = (
    EventType.TYPE_CHANGE, EventType.PT_CHANGE, EventType.KEYWORD_CHANGE,
    EventType.COLOR_CHANGE, EventType.ABILITY_CHANGE,
    EventType.CONTINUOUS_EFFECT_CREATED,
)
_PUMP = (
    EventType.PT_CHANGE, EventType.KEYWORD_CHANGE,
    EventType.CONTINUOUS_EFFECT_CREATED,
)
_CHOICE = (EventType.CHOICE_MADE,)

# Every Forge effect API and the outcomes it can produce. A composite mechanic
# lists what it actually does — Amass both puts counters and may create a token,
# so it maps to both, and a record from either sub-ability finds its type here.
EFFECT_API_EVENTS: dict[str, tuple[EventType, ...]] = {
    "ActivateAbility": (EventType.ABILITY_ACTIVATED,),
    "AddPhase": (EventType.PHASE_ADDED,),
    "AddTurn": (EventType.TURN_ADDED,),
    "Airbend": (EventType.ZONE_CHANGE,),
    "AlterAttribute": (EventType.STATE_FLAG_CHANGE,),
    "Amass": (EventType.COUNTER_CHANGE, EventType.TOKEN_CREATED),
    "Animate": _ANIMATE,
    "AnimateAll": _ANIMATE,
    "AssignGroup": _CHOICE,
    "Attach": (EventType.ATTACHED,),
    "Balance": (
        EventType.SACRIFICED, EventType.CARD_DISCARDED, EventType.ZONE_CHANGE,
    ),
    "BecomeMonarch": (EventType.MONARCH_CHANGED,),
    "BecomesBlocked": (EventType.BECAME_BLOCKED,),
    "BidLife": (EventType.LIFE_CHANGE, EventType.CHOICE_MADE),
    "Blight": (EventType.COUNTER_CHANGE,),
    "Block": (EventType.BLOCKERS_DECLARED,),
    "Bond": (EventType.STATE_FLAG_CHANGE,),
    "Camouflage": (EventType.LIBRARY_SHUFFLED, EventType.CHOICE_MADE),
    "ChangeCombatants": (
        EventType.ATTACKERS_DECLARED, EventType.BLOCKERS_DECLARED,
        EventType.REMOVED_FROM_COMBAT,
    ),
    "ChangeSpeed": (EventType.SPEED_CHANGED,),
    "ChangeTargets": (EventType.TARGETS_CHANGED,),
    "ChangeText": (
        EventType.TEXT_CHANGE, EventType.COLOR_CHANGE, EventType.TYPE_CHANGE,
    ),
    "ChangeX": (EventType.X_CHANGED,),
    "ChangeZone": (EventType.ZONE_CHANGE,),
    "ChangeZoneAll": (EventType.ZONE_CHANGE,),
    "ChangeZoneResolve": (EventType.ZONE_CHANGE,),
    "ChooseCard": _CHOICE,
    "NameCard": _CHOICE,  # was ChooseCardName, the effect class name
    "ChooseColor": _CHOICE,
    "ChooseDirection": _CHOICE,
    "ChooseEvenOdd": _CHOICE,
    "GenericChoice": _CHOICE,  # was ChooseGeneric, the effect class name
    "ChooseNumber": _CHOICE,
    "ChoosePlayer": _CHOICE,
    "ChooseSource": _CHOICE,
    "ChooseType": _CHOICE,
    "Clash": (EventType.CLASH_RESOLVED, EventType.CARD_REVEALED),
    "ClassLevelUp": (
        EventType.STATE_FLAG_CHANGE, EventType.ABILITY_CHANGE,
        EventType.CONTINUOUS_EFFECT_CREATED,
    ),
    "Cloak": (
        EventType.FACE_CHANGE, EventType.ZONE_CHANGE, EventType.TAPPED,
    ),
    "Clone": (EventType.PERMANENT_COPIED, EventType.CONTINUOUS_EFFECT_CREATED),
    "Connive": (
        EventType.CARD_DRAWN, EventType.CARD_DISCARDED, EventType.COUNTER_CHANGE,
    ),
    "ExchangeControl": (EventType.CONTROL_CHANGE,),  # was ControlExchange
    "ExchangeControlVariant": (EventType.CONTROL_CHANGE,),  # was ControlExchangeVariant
    "GainControl": (EventType.CONTROL_CHANGE,),  # was ControlGain
    "GainControlVariant": (EventType.CONTROL_CHANGE,),  # was ControlGainVariant
    "ControlSpell": (EventType.CONTROL_CHANGE,),
    "CopyPermanent": (EventType.PERMANENT_COPIED, EventType.TOKEN_CREATED),
    "CopySpellAbility": (EventType.SPELL_COPIED,),
    "Counter": (EventType.SPELL_COUNTERED, EventType.ZONE_CHANGE),
    "MoveCounter": (EventType.COUNTER_CHANGE,),  # was CountersMove
    "MultiplyCounter": (EventType.COUNTER_CHANGE,),  # was CountersMultiply
    "Proliferate": (EventType.COUNTER_CHANGE,),  # was CountersProliferate
    "PutCounter": (EventType.COUNTER_CHANGE,),  # was CountersPut
    "PutCounterAll": (EventType.COUNTER_CHANGE,),  # was CountersPutAll
    "AddOrRemoveCounter": (EventType.COUNTER_CHANGE,),  # was CountersPutOrRemove
    "RemoveCounter": (EventType.COUNTER_CHANGE,),  # was CountersRemove
    "RemoveCounterAll": (EventType.COUNTER_CHANGE,),  # was CountersRemoveAll
    "DamageAll": (EventType.DAMAGE_DEALT,),
    # DamageBase (DamageBaseEffect) is dropped, not renamed: it is an abstract
    # superclass of DamageAll/DealDamage/EachDamage/Fight with no ApiType
    # member of its own, so `sa.getApi().name()` can never equal "DamageBase".
    "DealDamage": (EventType.DAMAGE_DEALT,),  # was DamageDeal
    "EachDamage": (EventType.DAMAGE_DEALT,),  # was DamageEach
    "PreventDamage": (  # was DamagePrevent
        EventType.DAMAGE_PREVENTED, EventType.CONTINUOUS_EFFECT_CREATED,
    ),
    "DamageResolve": (EventType.DAMAGE_DEALT,),
    "DayTime": (EventType.DAY_NIGHT_CHANGED,),
    "Debuff": (EventType.KEYWORD_CHANGE, EventType.CONTINUOUS_EFFECT_CREATED),
    "DelayedTrigger": (EventType.DELAYED_TRIGGER_CREATED,),
    "Destroy": (EventType.DESTROYED, EventType.ZONE_CHANGE),
    "DestroyAll": (EventType.DESTROYED, EventType.ZONE_CHANGE),
    "Detain": (EventType.RESTRICTION_CHANGE,),
    "Dig": (
        EventType.CARD_LOOKED_AT, EventType.ZONE_CHANGE, EventType.CARD_REVEALED,
    ),
    "DigMultiple": (
        EventType.CARD_LOOKED_AT, EventType.ZONE_CHANGE, EventType.CARD_REVEALED,
    ),
    "DigUntil": (
        EventType.CARD_LOOKED_AT, EventType.ZONE_CHANGE, EventType.CARD_REVEALED,
    ),
    "Discard": (EventType.CARD_DISCARDED,),
    "Discover": (
        EventType.CARD_REVEALED, EventType.ZONE_CHANGE, EventType.SPELL_CAST,
        EventType.LIBRARY_REORDERED,
    ),
    "DrainMana": (EventType.MANA_LOST,),
    "Draw": (EventType.CARD_DRAWN,),
    "Earthbend": (EventType.COUNTER_CHANGE, EventType.KEYWORD_CHANGE),
    "Effect": (EventType.CONTINUOUS_EFFECT_CREATED,),
    "Encode": (EventType.STATE_FLAG_CHANGE, EventType.ZONE_CHANGE),
    "EndCombatPhase": (EventType.COMBAT_ENDED,),
    "EndTurn": (EventType.TURN_ENDED,),
    "Endure": (EventType.COUNTER_CHANGE, EventType.TOKEN_CREATED),
    "Explore": (
        EventType.CARD_REVEALED, EventType.COUNTER_CHANGE, EventType.ZONE_CHANGE,
    ),
    "Fight": (EventType.DAMAGE_DEALT,),
    "FlipCoin": (EventType.COIN_FLIPPED,),
    "FlipOntoBattlefield": (EventType.ZONE_CHANGE, EventType.FACE_CHANGE),
    "Fog": (EventType.DAMAGE_PREVENTED, EventType.CONTINUOUS_EFFECT_CREATED),
    "GameDrawn": (EventType.GAME_DRAWN,),  # was GameDraw
    "LosesGame": (EventType.PLAYER_LOST,),  # was GameLoss
    "WinsGame": (EventType.PLAYER_WON,),  # was GameWin
    "Goad": (EventType.RESTRICTION_CHANGE,),
    "Haunt": (EventType.ATTACHED, EventType.ZONE_CHANGE),
    "HealDamage": (EventType.DAMAGE_HEALED,),
    "Heist": (
        EventType.ZONE_CHANGE, EventType.FACE_CHANGE, EventType.CARD_LOOKED_AT,
    ),
    "ImmediateTrigger": (EventType.DELAYED_TRIGGER_CREATED,),
    "Incubate": (EventType.TOKEN_CREATED, EventType.COUNTER_CHANGE),
    "Intensify": (EventType.COUNTER_CHANGE,),
    # Grants a real activated ability (cost-gated, resolve() mutates state) that
    # makes a static ability's continuous effect stop applying to a player or
    # card until end of turn -- the same shape of fact as Detain/Goad/MustBlock,
    # just granting an exemption instead of adding a restriction. Reachable in
    # sealed/draft: Damping Engine, Leonin Arbiter, Lost in Thought and
    # Volrath's Curse all grant it (StaticAbilityContinuous.IgnoreEffectCost).
    "InternalIgnoreEffect": (EventType.RESTRICTION_CHANGE,),
    "InternalRadiation": (
        EventType.CARD_MILLED, EventType.LIFE_CHANGE, EventType.RADIATION_CHANGE,
    ),
    "Investigate": (EventType.TOKEN_CREATED,),
    "Learn": (EventType.ZONE_CHANGE, EventType.CARD_DISCARDED, EventType.CARD_DRAWN),
    "ExchangeLife": (EventType.LIFE_CHANGE,),  # was LifeExchange
    "ExchangeLifeVariant": (EventType.LIFE_CHANGE,),  # was LifeExchangeVariant
    "GainLife": (EventType.LIFE_CHANGE,),  # was LifeGain
    "LoseLife": (EventType.LIFE_CHANGE,),  # was LifeLose
    "SetLife": (EventType.LIFE_CHANGE,),  # was LifeSet
    "LookAt": (EventType.CARD_LOOKED_AT,),
    "LosePerpetual": (
        EventType.KEYWORD_CHANGE, EventType.PT_CHANGE, EventType.ABILITY_CHANGE,
    ),
    "MakeCard": (EventType.CARD_MADE,),
    "Mana": (EventType.MANA_PRODUCED,),
    "ManaReflected": (EventType.MANA_PRODUCED,),
    "Manifest": (EventType.FACE_CHANGE, EventType.ZONE_CHANGE),
    # ManifestBase (ManifestBaseEffect) is dropped, not renamed: it is an
    # abstract superclass of Cloak/Manifest/ManifestDread with no ApiType
    # member of its own.
    "ManifestDread": (
        EventType.CARD_LOOKED_AT, EventType.FACE_CHANGE, EventType.ZONE_CHANGE,
    ),
    "Meld": (EventType.FACE_CHANGE, EventType.ZONE_CHANGE),
    "Mill": (EventType.CARD_MILLED,),
    "MultiplePiles": (EventType.PILES_MADE,),
    "MustBlock": (EventType.RESTRICTION_CHANGE,),
    "Mutate": (
        EventType.ZONE_CHANGE, EventType.ABILITY_CHANGE, EventType.PT_CHANGE,
    ),
    "GainOwnership": (EventType.OWNERSHIP_CHANGE,),  # was OwnershipGain
    "PeekAndReveal": (EventType.CARD_LOOKED_AT, EventType.CARD_REVEALED),
    # Permanent (PermanentEffect) is dropped, not renamed: PermanentCreature
    # and PermanentNoncreature both extend it, but no ApiType member points at
    # it directly.
    "PermanentCreature": (EventType.ZONE_CHANGE,),
    "PermanentNoncreature": (EventType.ZONE_CHANGE,),
    "Phases": (EventType.PHASED,),
    "Play": (EventType.SPELL_CAST, EventType.ZONE_CHANGE),
    "PlayLandVariant": (EventType.SPELL_CAST, EventType.ZONE_CHANGE),
    "Poison": (EventType.POISON_CHANGE,),
    # was PowerExchange; PowerExchangeEffect.resolve() schedules an
    # addUntilCommand that reverts the swap at end of turn unless the duration
    # is Permanent/Perpetual, so CONTINUOUS_EFFECT_CREATED is a real second
    # outcome here, not carried over by rename alone.
    "ExchangePower": (EventType.PT_CHANGE, EventType.CONTINUOUS_EFFECT_CREATED),
    "Protection": _PUMP,  # was Protect
    "ProtectionAll": _PUMP,  # was ProtectAll
    "Pump": _PUMP,
    "PumpAll": _PUMP,
    "Radiation": (EventType.RADIATION_CHANGE,),
    "RearrangeTopOfLibrary": (EventType.LIBRARY_REORDERED,),
    "Recruit": (EventType.CARD_DRAWN,),
    "Regenerate": (EventType.REGENERATED, EventType.CONTINUOUS_EFFECT_CREATED),
    "Regeneration": (EventType.REGENERATED, EventType.CONTINUOUS_EFFECT_CREATED),
    "RemoveFromCombat": (EventType.REMOVED_FROM_COMBAT,),
    "RemoveFromGame": (EventType.ZONE_CHANGE,),
    "RemoveFromMatch": (EventType.PLAYER_REMOVED,),
    "ReorderZone": (EventType.LIBRARY_REORDERED, EventType.LIBRARY_SHUFFLED),
    "ReplaceEffect": (EventType.REPLACEMENT_APPLIED,),  # was Replace
    "ReplaceCounter": (EventType.REPLACEMENT_APPLIED, EventType.COUNTER_CHANGE),
    "ReplaceDamage": (EventType.REPLACEMENT_APPLIED, EventType.DAMAGE_DEALT),
    "ReplaceMana": (EventType.REPLACEMENT_APPLIED, EventType.MANA_PRODUCED),
    "ReplaceSplitDamage": (EventType.REPLACEMENT_APPLIED, EventType.DAMAGE_DEALT),
    "ReplaceToken": (EventType.REPLACEMENT_APPLIED, EventType.TOKEN_CREATED),
    "RestartGame": (EventType.GAME_RESTARTED,),
    "Reveal": (EventType.CARD_REVEALED,),
    "RevealHand": (EventType.CARD_REVEALED,),
    "ReverseTurnOrder": (EventType.TURN_ORDER_REVERSED,),
    "RingTemptsYou": (EventType.RING_TEMPTS, EventType.CHOICE_MADE),
    "RollDice": (EventType.DICE_ROLLED,),
    "Sacrifice": (EventType.SACRIFICED, EventType.ZONE_CHANGE),
    "SacrificeAll": (EventType.SACRIFICED, EventType.ZONE_CHANGE),
    "Scry": (EventType.CARD_LOOKED_AT, EventType.LIBRARY_REORDERED),
    "Seek": (EventType.ZONE_CHANGE,),
    "SetState": (EventType.FACE_CHANGE,),
    "Shuffle": (EventType.LIBRARY_SHUFFLED,),
    "SkipPhase": (EventType.PHASE_SKIPPED,),
    "SkipTurn": (EventType.TURN_SKIPPED,),
    "Surveil": (EventType.CARD_LOOKED_AT, EventType.ZONE_CHANGE),
    "SwitchBlock": (EventType.BLOCKERS_DECLARED, EventType.BECAME_BLOCKED),
    "TakeInitiative": (EventType.INITIATIVE_TAKEN,),
    "Tap": (EventType.TAPPED,),
    "TapAll": (EventType.TAPPED,),
    "TapOrUntap": (EventType.TAPPED, EventType.UNTAPPED),
    "TapOrUntapAll": (EventType.TAPPED, EventType.UNTAPPED),
    # was TextBoxExchange; TextBoxExchangeEffect swaps SpellAbilities,
    # Triggers, ReplacementEffects, StaticAbilities and Keywords (ABILITY_CHANGE)
    # and calls updateChangedText() on both cards (TEXT_CHANGE) -- both are
    # real outcomes of the swap, not carried over by rename alone.
    "ExchangeTextBox": (EventType.TEXT_CHANGE, EventType.ABILITY_CHANGE),
    "TimeTravel": (EventType.COUNTER_CHANGE,),
    "Token": (EventType.TOKEN_CREATED,),
    "TwoPiles": (EventType.PILES_MADE,),
    "Unattach": (EventType.UNATTACHED,),
    "UnlockDoor": (EventType.STATE_FLAG_CHANGE, EventType.ABILITY_CHANGE),
    "Untap": (EventType.UNTAPPED,),
    "UntapAll": (EventType.UNTAPPED,),
    "Venture": (EventType.DUNGEON_VENTURED,),
    "VillainousChoice": (EventType.CHOICE_MADE,),
    "Vote": (EventType.VOTE_TAKEN,),
    "ExchangeZone": (EventType.ZONE_CHANGE,),  # was ZoneExchange
}

# APIs that produce no event of their own. Each reason is one of three kinds:
# script plumbing that only dispatches other abilities, a variant-format
# mechanic the sealed and draft corpora can never contain, or a UI-only action.
EXCLUDED_EFFECT_APIS: dict[str, str] = {
    # ── script plumbing: the dispatched sub-ability carries the outcome ──
    "BlankLine": "formatting no-op in a card script; changes no game state",
    "Branch": "control flow; the branch's chosen sub-ability produces the events",
    "Charm": (
        "modal selection; the chosen option line is the record's acting ability "
        "and its own sub-ability produces the events (FR-017)"
    ),
    "Cleanup": "removes the effect's own bookkeeping when its duration ends",  # was CleanUp
    "CompanionChoose": (
        "wraps the pregame companion-selection prompt in a no-op ability "
        "(SpellAbility.EmptySa, whose resolve() is empty) purely so the choice "
        "has an sa to record against; the choice has no in-game effect of its own"
    ),
    "InternalLegendaryRule": (
        "wraps the legend-rule which-one-to-keep prompt in the same no-op "
        "EmptySa as CompanionChoose; the state-based action moves the rest to "
        "the graveyard on its own path, not through this ability's resolution"
    ),
    # DetachedCardEffect is dropped, not excluded under its old name: it
    # extends Card, not SpellAbilityEffect, and has no ApiType member at all --
    # `sa.getApi()` can never be this class regardless of formatting.
    "Repeat": "control flow; the repeated sub-ability produces the events",
    "RepeatEach": "control flow; the repeated sub-ability produces the events",
    "StoreSVar": "writes a script variable; changes no game state",
    "Subgame": (
        "runs a nested game (Shahrazad); the subgame's own records would need "
        "their own game_id, and no sealed- or draft-legal card has it"
    ),
    # ── variant formats this corpus cannot contain ──
    "Abandon": "Archenemy scheme; not reachable in sealed or draft",
    "AdvanceCrank": "Unstable contraption; not reachable in sealed or draft",
    "AssembleContraption": "Unstable contraption; not reachable in sealed or draft",
    "ChaosEnsues": "Planechase; not reachable in sealed or draft",
    "ChooseSector": "Unfinity attraction; not reachable in sealed or draft",
    "ClaimThePrize": "Unfinity attraction; not reachable in sealed or draft",
    "Draft": "in-game drafting (Conspiracy); not reachable in sealed or draft",
    "OpenAttraction": "Unfinity attraction; not reachable in sealed or draft",
    "Planeswalk": "Planechase; not reachable in sealed or draft",
    "RollPlanarDice": "Planechase; not reachable in sealed or draft",
    "RunChaos": "Planechase; not reachable in sealed or draft",
    "SetInMotion": "Archenemy scheme; not reachable in sealed or draft",
    "ControlPlayer": (
        "controls another player's turn (Mindslaver); the controlled player's "
        "own actions are recorded under their own acting ability"
    ),
    # CountersNoteEffect is dropped, not excluded under its old name: its
    # ApiType entry is commented out in Forge (`//NoteCounters (...)`), so it
    # is not a live member -- the class survives only as a helper other
    # effects (ChangeZoneEffect, EffectEffect) call directly.
}


def event_types_for(api: str) -> tuple[EventType, ...]:
    """The outcomes a Forge effect API can produce.

    Raises for an API that is neither mapped nor excluded — a Forge upgrade
    added an effect and the vocabulary has not caught up.
    """
    if api in EFFECT_API_EVENTS:
        return EFFECT_API_EVENTS[api]
    if api in EXCLUDED_EFFECT_APIS:
        return ()
    raise KeyError(
        f"Forge effect API {api!r} is neither mapped in EFFECT_API_EVENTS nor "
        "listed in EXCLUDED_EFFECT_APIS; regenerate forge_effect_apis.py and "
        "extend the vocabulary"
    )
