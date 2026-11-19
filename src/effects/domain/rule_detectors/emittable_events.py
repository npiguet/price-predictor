"""Which event types each record kind can carry, as the Java collector writes them.

A detector declaration that asks a record kind for an event the collector
never writes there reads every record of that kind as the event's absence: an
``outcome_event`` near miss for each resolution of the declared API, an
``absent`` declaration's ``fires`` side for each, polarity and ``observed``
agreeing so no review list catches it (V5 #14). Such a declaration is
*untaggable*: :class:`DetectorIndex` sets it aside with its reason, and
``rule-counts`` lists it, rather than letting it tag.

The table is built by the event's source in the connector
(``forge-connector/.../effects/``), so a source gaining an event is one line,
and ``test_emittable_events`` reads the sources to keep it so:

- :data:`BUS`: built from Forge's event bus (``BusEvents``, ``BusEventReader``,
  ``StatDiffer``, ``ControlCorrection``) and filed wherever the bus event
  lands: a resolution's effect half, the open combat damage step, or the
  turn-based action under way. The game's end (:data:`GAME_END`) is filed in
  the game-end record: a combat, the game-end rescue record after a
  state-based check, or the effect half of the ability that ended the game
  outright (a ``WinsGame`` or ``LosesGame`` line);
- :data:`CLAUSE_HOOKS`: built by a resolving clause's hooks (``ApiEvents``,
  ``ClauseOutcomeRouter``'s outcome hook, ``ManaChannel``) into the effect
  half of the resolution it belongs to. A clause a replacement runs while no
  resolution is open (a regeneration shield used by lethal combat damage, a
  replacement run in the cleanup's discard) is filed as any consequence is
  (``BusBracketCollector.route``): on the open combat damage step's record,
  else in the turn-based action's window. So a combat or turn-action record
  can list a clause-hook event too;
- :data:`CHOKE_POINTS`: the two outcome-hook events Forge reports at an
  engine choke point with no resolving ability, routed as a bus event is;
- :data:`RUN_PARAMS`: built by ``RunParamNormalizer`` through ``RunParamRows`` from a trigger's or a
  replacement's run parameters, the trigger record's ``event`` and the rewrite
  record's ``incoming``/``outgoing``;
- :data:`TAPPED`: what the bracket routes (``BusBracketCollector.route``:
  the bus's events and the choke points'), which three taps take before it
  files them (``EventTap``). ``ForkPremise`` takes, in a fork, everything the
  state-based check after the fork's edit publishes, into a ``turn_action`` /
  ``fork_edit`` record, whose kind lists them already. ``ManaChannel`` takes a
  mana ability's own:
  what its payment did beyond its taps (a creature sacrificed for mana) and
  what its sub-abilities did (a pain land's damage), into its mana record, a
  resolution, whose kind lists them already. ``RewriteChannel`` takes a
  set-aside ability's run (Swans of Bryn Argoll's draw) into the waiting
  rewrite record's ``buffered_events`` (:data:`BUFFERED`).

A rewrite's ``buffered_events`` are their own place, not the rewrite's events:
no detector reading a rewrite record (``rewrite_event``: the event that came
in, what became of it) reads them, and ``listed_events`` leaves them out. A
declaration that wants them reads ``RewritePayload.buffered_events`` and
asks :func:`carries_buffered` whether the place can hold its event.

A second fact decides whether an event's *absence* is readable on one
record: the hook-derived events (:data:`HOOK_DERIVED`) need the engine
hooks, which a degraded worker lacks, and which a forced resolution (an
``interventional`` fork's lone effect half) does not hear apart from its
resolution start, its modes, its conditions and its mana
(``InterventionFork``'s ``ForkHooks``). On such a record a hook-derived event
that is missing may have happened (:func:`absence_readable`). A protection
fork's records, played forward with the hooks routed to them, are read as
any record is.

Torch-free.
"""

from __future__ import annotations

from effects.domain.event_schema import EventType as E
from effects.domain.records import CollectionMode, EffectRecord, RecordKind

#: Built from the event bus (``BusEvents``, ``BusEventReader``, ``StatDiffer``,
#: ``ControlCorrection``).
BUS: frozenset[E] = frozenset({
    E.ZONE_CHANGE, E.DAMAGE_DEALT, E.LIFE_CHANGE, E.COUNTER_CHANGE, E.POISON_CHANGE,
    E.TAPPED, E.UNTAPPED, E.CARD_DRAWN, E.CARD_DISCARDED, E.CARD_MILLED, E.PHASED,
    E.DAY_NIGHT_CHANGED, E.ATTACHED, E.UNATTACHED, E.CARD_LOOKED_AT, E.LIBRARY_SHUFFLED,
    E.LIBRARY_REORDERED, E.PLAYER_WON, E.PLAYER_LOST, E.ENERGY_CHANGE, E.RADIATION_CHANGE,
    E.SPEED_CHANGED, E.REGENERATED, E.PT_CHANGE, E.TYPE_CHANGE, E.COLOR_CHANGE,
    E.CONTROL_CHANGE, E.KEYWORD_CHANGE, E.ABILITY_CHANGE, E.NAME_CHANGE, E.MANA_COST_CHANGE,
    E.STATE_FLAG_CHANGE, E.FACE_CHANGE, E.DESTROYED, E.SACRIFICED, E.REMOVED_FROM_COMBAT,
})
#: Built by a resolving clause's hooks into its resolution's effect half.
CLAUSE_HOOKS: frozenset[E] = frozenset({
    # ApiEvents
    E.TURN_ADDED, E.TURN_SKIPPED, E.PHASE_ADDED, E.PHASE_SKIPPED, E.X_CHANGED,
    E.OWNERSHIP_CHANGE, E.CHOICE_MADE, E.DAMAGE_HEALED, E.ABILITY_CHANGE,
    E.CONTINUOUS_EFFECT_CREATED, E.TARGETS_CHANGED, E.DELAYED_TRIGGER_CREATED,
    E.REPLACEMENT_APPLIED, E.MONARCH_CHANGED, E.RING_TEMPTS, E.INITIATIVE_TAKEN,
    E.REMOVED_FROM_COMBAT, E.COMBAT_ENDED, E.TEXT_CHANGE, E.BECAME_BLOCKED,
    E.STATE_FLAG_CHANGE, E.CARD_LOOKED_AT,
    # the outcome hook (ClauseOutcomeRouter.KNOWN_OUTCOME_TYPES)
    E.COIN_FLIPPED, E.CLASH_RESOLVED, E.VOTE_TAKEN, E.PILES_MADE, E.DUNGEON_VENTURED,
    E.SPELL_COPIED, E.PERMANENT_COPIED, E.CARD_MADE, E.RESTRICTION_CHANGE, E.CARD_REVEALED,
    E.DAMAGE_PREVENTED, E.SPELL_COUNTERED, E.TOKEN_CREATED, E.DICE_ROLLED, E.TURN_ENDED,
    # ManaChannel: a resolving spell's or trigger's mana, a mana ability's record
    E.MANA_PRODUCED,
})
#: Outcome-hook events Forge reports with no resolving ability, routed as bus events are.
CHOKE_POINTS: frozenset[E] = frozenset({E.CARD_REVEALED, E.DAMAGE_PREVENTED})
#: Built by ``RunParamNormalizer`` from the mode rows of ``RunParamRows`` (a
#: mode with no row is a ``state_flag_change``).
RUN_PARAMS: frozenset[E] = frozenset({
    E.ZONE_CHANGE, E.DAMAGE_DEALT, E.DAMAGE_PREVENTED, E.CARD_DRAWN, E.CARD_DISCARDED,
    E.CARD_MILLED, E.LIFE_CHANGE, E.COUNTER_CHANGE, E.TAPPED, E.UNTAPPED,
    E.MANA_PRODUCED, E.ATTACHED, E.UNATTACHED, E.SPELL_CAST, E.SPELL_COUNTERED,
    E.SPELL_COPIED, E.TOKEN_CREATED, E.ATTACKERS_DECLARED, E.BLOCKERS_DECLARED,
    E.BECAME_BLOCKED, E.LIBRARY_SHUFFLED, E.CARD_LOOKED_AT, E.PHASED, E.MANA_LOST,
    E.DICE_ROLLED, E.PLAYER_WON, E.FACE_CHANGE, E.DESTROYED, E.SACRIFICED, E.PLAYER_LOST,
    E.STATE_FLAG_CHANGE,
})
#: What the bracket routes and a channel's ``EventTap`` may take first: the
#: bus's events and the choke points' (``BusBracketCollector.route``).
TAPPED: frozenset[E] = BUS | CHOKE_POINTS
#: The events a rewrite record's ``buffered_events`` can hold: the set-aside
#: ability's run, taken by ``RewriteChannel``'s tap.
BUFFERED: frozenset[E] = TAPPED
#: The game's end, filed in the game-end record (module docstring).
GAME_END: frozenset[E] = frozenset({E.PLAYER_WON, E.PLAYER_LOST})
#: The APIs whose own resolution ends the game, so their effect half holds its end.
GAME_ENDING_APIS: frozenset[str] = frozenset({"WinsGame", "LosesGame"})

#: The event types a record of each kind can list. A resolution's cost half
#: lists its spell's move to the stack alone, and no detector reads it as an
#: outcome; playability and continuous records list none.
EMITTABLE: dict[RecordKind, frozenset[E]] = {
    RecordKind.RESOLUTION: BUS | CLAUSE_HOOKS,
    RecordKind.COMBAT: BUS | CHOKE_POINTS | CLAUSE_HOOKS,
    RecordKind.TURN_ACTION: BUS | CHOKE_POINTS | CLAUSE_HOOKS,
    RecordKind.TRIGGER: RUN_PARAMS,
    RecordKind.REWRITE: RUN_PARAMS,
    RecordKind.CONTINUOUS: frozenset(),
    RecordKind.PLAYABILITY: frozenset(),
}

#: Events only the engine hooks build on a resolution: missing on a degraded
#: worker's records whatever happened. A forced resolution and a probe hear
#: them as a live record does (``ForkHooks.clauseEvent``, ``ForkHooks.outcome``).
HOOK_DERIVED: frozenset[E] = CLAUSE_HOOKS - BUS


def carries(kind: RecordKind | str, event: E | str, *, acting_line: bool = False,
            api: str | None = None) -> bool:
    """Whether a record of ``kind`` can list an event of type ``event``; with
    ``acting_line``, a record acting through a line (of the API ``api`` where
    one is named), which holds the game's end only when its own API ends the
    game."""
    kind, event = RecordKind(kind), E(event)
    if (acting_line and kind is RecordKind.RESOLUTION and event in GAME_END
            and api not in GAME_ENDING_APIS):
        return False
    return event in EMITTABLE.get(kind, frozenset())


def carries_buffered(event: E | str) -> bool:
    """Whether a rewrite record's ``buffered_events`` can hold ``event``."""
    return E(event) in BUFFERED


def absence_readable(record: EffectRecord, event: E) -> bool:
    """Whether ``record`` lacking an ``event`` means it did not happen: false
    for a hook-derived event on a degraded worker's record, which no hook
    told of it. A patched worker's forced resolution hears the hooks as a
    live record does."""
    if record.mode is CollectionMode.DEGRADED:
        return event not in HOOK_DERIVED
    return True
