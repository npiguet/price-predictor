"""The effect-record schema: one observed game event or decision point.

The schema is frozen: later collection widens the corpus but never redefines a
field. The compatibility rules are in
``specs/023-ability-effect-model/contracts/record-schema.md`` and, with no
version field, ``tests/unit/effects/domain/test_schema_compatibility.py`` is
their enforcement. Pure dataclasses; serialization lives in
``infrastructure/record_codec.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from effects.domain.event_schema import Event
from effects.domain.fork_operation import ForkOperation
from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import StateSnapshot


class RecordKind(StrEnum):
    RESOLUTION = "resolution"
    REWRITE = "rewrite"
    CONTINUOUS = "continuous"
    COMBAT = "combat"
    TRIGGER = "trigger"
    PLAYABILITY = "playability"
    #: A turn-based action: a draw, an untap, a cleanup discard (gen-3). Never
    #: written to a curated stratum.
    TURN_ACTION = "turn_action"


class Moment(StrEnum):
    """Which half of a resolution pair a ``resolution`` record is."""

    ACTIVATION = "activation"
    RESOLUTION = "resolution"


class PlayabilitySubkind(StrEnum):
    DECISION = "decision"
    ATTACKERS = "attackers"
    BLOCKERS = "blockers"


class TurnActionSubkind(StrEnum):
    """Which turn-based action a ``turn_action`` record holds."""

    DRAW = "draw"
    CLEANUP_DISCARD = "cleanup_discard"
    UNTAP = "untap"
    PHASING = "phasing"
    DAY_NIGHT = "day_night"
    LORE_COUNTER = "lore_counter"
    #: A fork's premise, on fork records only: the state-based check a
    #: protection fork's grant or a directed fork's edit makes, run before
    #: the fork plays on (contract ``record-schema-delta.md`` § Kinds).
    FORK_EDIT = "fork_edit"
    #: The game-end marker: a game's last record, written only for a game that
    #: ended with every end-of-game write done. A game with none in its shard
    #: was cut (contract ``record-schema-delta.md`` § Kinds).
    GAME_END = "game_end"


#: The kinds whose ``subkind`` is set, each with its own subkind vocabulary.
SUBKINDS_OF: dict[RecordKind, type[StrEnum]] = {
    RecordKind.PLAYABILITY: PlayabilitySubkind,
    RecordKind.TURN_ACTION: TurnActionSubkind,
}


class CollectionMode(StrEnum):
    """Whether the worker found the engine patch hooks at startup."""

    PATCHED = "patched"
    DEGRADED = "degraded"


class ResolutionOutcome(StrEnum):
    """What became of a stack item.

    Gen-3 writes it on the effect half (root spec § 8.14, FR-042b), never ``countered``: a countered
    item writes no effect half, its fate being the countering clause's
    ``spell_countered``, or ``ability_countered`` for an ability.
    """

    RESOLVED = "resolved"
    FIZZLED = "fizzled"
    PARTIALLY_FIZZLED = "partially_fizzled"
    DECLINED = "declined"
    COUNTERED = "countered"


class RewriteResult(StrEnum):
    """What a replacement effect's handler reported, as Forge names it.

    Mirrors ``forge.game.replacement.ReplacementResult`` member for member,
    lower_snake_case for the wire; ``test_records.py`` pins the two
    together, since a value the reader raises on loses append-only records.
    Most replacements never edit the event's parameter map, so without this
    "enters tapped" and "prevent that damage" would read as no change at all.
    """

    REPLACED = "replaced"
    NOT_REPLACED = "not_replaced"
    PREVENTED = "prevented"
    UPDATED = "updated"
    SKIPPED = "skipped"


#: Results the handler returns before running any ability, so one carrying an
#: ``outgoing`` or a ``replaced_by`` is a collector defect. ``NotReplaced`` also
#: returns early but its prevention branch writes ``PreventedAmount`` into the
#: map, so it may legitimately carry an ``outgoing``.
REWRITE_RESULTS_WITHOUT_ABILITY: frozenset[RewriteResult] = frozenset({
    RewriteResult.PREVENTED,
    RewriteResult.SKIPPED,
})

#: Results usually reached by running the ``ReplaceWith$`` ability, the source
#: of ``replaced_by``. Watched, not enforced: a ``ReplacementResult$`` script
#: returns one of these with no ability at all.
REWRITE_RESULTS_RUNNING_AN_ABILITY: frozenset[RewriteResult] = frozenset({
    RewriteResult.REPLACED,
    RewriteResult.UPDATED,
})

#: The legality subkinds; gen-3 writes ``what_if`` (FR-031) on ``blockers``
#: alone. A ``decision`` is always a real evaluation.
LEGALITY_SUBKINDS: frozenset[PlayabilitySubkind] = frozenset({
    PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS,
})


class LegalityDecision(StrEnum):
    """Whether a legality record is the declaration actually made (FR-033)."""

    REAL = "real"
    WHAT_IF = "what-if"

    @classmethod
    def of(cls, record) -> LegalityDecision:
        """``record`` is anything carrying ``what_if``: a record or a result row.

        An attackers record carries no ``what_if``: the collector writes one
        per creature of the declaration actually made, and none for a query,
        so every one is real.
        """
        return cls.WHAT_IF if record.what_if else cls.REAL


class SeatPolicy(StrEnum):
    """Whether the acting seat was Forge's AI or the random seat (FR-030)."""

    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy"

    @classmethod
    def of(cls, record) -> SeatPolicy:
        return cls.OFF_POLICY if record.random_seat else cls.ON_POLICY


class UnresolvedReason(StrEnum):
    """Why a record that does name an acting line carries no key for it.

    Separates "no printed line exists" (an engine-built card such as the
    Monarch) from "the resolver failed": ``unindexable`` is a resolver bug and
    never appears in a healthy run.
    """

    ENGINE_EFFECT = "engine_effect"
    NO_CARD_STATE = "no_card_state"
    UNKNOWN_KIND = "unknown_kind"
    GRANTED_KEYWORD = "granted_keyword"
    GRANTED_TRAIT = "granted_trait"
    UNINDEXABLE = "unindexable"


#: Kinds whose envelope carries no ``ability``: no single line acts in a combat
#: or a legality declaration, and a decision acts through its candidate's line
#: (:attr:`EffectRecord.has_act_line`), which the payload names.
KINDS_WITHOUT_ACTING_ABILITY: frozenset[RecordKind] = frozenset({
    RecordKind.COMBAT, RecordKind.PLAYABILITY, RecordKind.TURN_ACTION,
})

#: Envelope fields that say how a record was collected, never what the board
#: was: none of them may reach the model (compatibility rule 4).
COLLECTION_METADATA_FIELDS: frozenset[str] = frozenset({
    "mode", "interventional", "fork", "synthetic", "ability_unresolved", "random_seat",
    "what_if", "generation", "aimed_at", "directed", "copy_dropped", "fork_kind",
})


class ForkKind(StrEnum):
    """The kind of fork that wrote a record: its envelope's ``fork_kind``,
    absent on a live record.

    Mirrors the connector's ``ForkKind`` enum; ``test_wire_closed_lists.py``
    reads its source and ``WireVocabularyTest.forkKinds`` spells it out, so
    one is changed on both sides or neither.
    """

    #: A forced resolution's effect half.
    INTERVENTION = "intervention"
    #: A damage-step probe's branch.
    PROBE = "probe"
    #: A protection fork played to the end of the turn.
    PROTECTION = "protection"
    #: A directed fork played to the end of the turn.
    DIRECTED = "directed"


class RecordOrigin(StrEnum):
    """What produced a record: the live game, one of the four forks, or
    ``build-corpus`` (:attr:`EffectRecord.origin`)."""

    LIVE = "live"
    PROBE = "probe"
    FORCED = "forced"
    PROTECTION_FORK = "protection_fork"
    DIRECTED_FORK = "directed_fork"
    SYNTHETIC = "synthetic"


#: The origin of each fork kind's records.
_FORK_ORIGINS: dict[ForkKind, RecordOrigin] = {
    ForkKind.INTERVENTION: RecordOrigin.FORCED,
    ForkKind.PROBE: RecordOrigin.PROBE,
    ForkKind.PROTECTION: RecordOrigin.PROTECTION_FORK,
    ForkKind.DIRECTED: RecordOrigin.DIRECTED_FORK,
}

#: Each fork kind's ``(interventional, directed set)`` flags, the contract's
#: flag-signature table (``record-schema-delta.md`` (025) § Envelope).
_FORK_SIGNATURES: dict[ForkKind, tuple[bool, bool]] = {
    ForkKind.INTERVENTION: (True, False),
    ForkKind.PROBE: (False, False),
    ForkKind.PROTECTION: (True, False),
    ForkKind.DIRECTED: (False, True),
}


@dataclass(frozen=True, slots=True)
class DirectedTag:
    """What a directed fork changed: the scenario variant and the operation that applied it."""

    variant: str
    operation: ForkOperation


@dataclass(frozen=True, slots=True)
class Curation:
    """How ``build-corpus`` wrote one copy of a record (research T1): curated
    records only, never a collected shard. Curation metadata: it says how the
    record was copied, never what the board was, so it never reaches the model.

    ``natural_share`` (n) is the copies the budgets give the record, 1 in a
    validation stratum, and ``copies`` (d) every copy written, natural and
    added. ``drawn_rule`` is set on an added copy only, ``drawn_rules`` lists
    the rule of every added copy of the record, and ``condition_entities`` and
    ``outcome_entities`` hold the drawn rule's tag entities in their two roles
    (research R1). ``class_weight`` is what a reader weighing a row of a
    validation stratum multiplies by: the stratum's multiplier for the
    record's class, which gives the stratum the training stratum's class mix
    (FR-059a); 1 on a training row.
    """

    natural_share: int
    copies: int
    drawn_rule: str | None = None
    drawn_rules: tuple[str, ...] = ()
    condition_entities: tuple[str, ...] = ()
    outcome_entities: tuple[str, ...] = ()
    class_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.drawn_rule is not None and self.drawn_rule not in self.drawn_rules:
            raise ValueError(
                f"drawn rule {self.drawn_rule} is not among the record's drawn "
                f"rules {self.drawn_rules}"
            )

    @property
    def is_added(self) -> bool:
        return self.drawn_rule is not None

    @property
    def rule_entities(self) -> tuple[str, ...]:
        """Both roles, condition first, each entity once: where the hard
        rule mask trains the affected gate (FR-061)."""
        return tuple(dict.fromkeys((*self.condition_entities, *self.outcome_entities)))


class _PayloadQueries:
    """What every payload answers about its events and keys; each default is
    the answer of a payload carrying none of it."""

    __slots__ = ()

    def happened_events(self) -> tuple[Event, ...]:
        """The events that happened, which on most kinds are the record's outcome."""
        return ()

    def listed_events(self) -> tuple[Event, ...]:
        """The payload's own ``events`` list, on the kinds that write one."""
        return ()

    def all_events(self) -> tuple[Event, ...]:
        """Every event anywhere in the payload, proposed and evaluated ones too."""
        return ()

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
        """Every printed-line key the payload carries, in payload order."""
        return iter(())

    def acting_keys(
        self, ability: tuple[ProvenanceKey, ...],
    ) -> tuple[ProvenanceKey, ...]:
        """The keys ``[ACT]`` reads, given the record's acting ``ability``."""
        return ability


@dataclass(frozen=True, slots=True)
class Costs:
    """What the actor actually paid, as paid rather than as printed.

    ``life`` is the sum of the life payments; the entity channels name the
    permanents or cards by ref, as ``tapped`` and ``sacrificed`` do.
    """

    mana_by_color: dict[str, int] = field(default_factory=dict)
    tapped: tuple[str, ...] = ()
    life: int = 0
    sacrificed: tuple[str, ...] = ()
    discarded: tuple[str, ...] = ()
    exiled: tuple[str, ...] = ()
    #: Net counters per type, signed: a loyalty ``-2`` is ``{"LOYALTY": -2}``
    #: and a ``+1`` is ``{"LOYALTY": 1}``.
    counters: dict[str, int] = field(default_factory=dict)
    returned: tuple[str, ...] = ()
    revealed: tuple[str, ...] = ()
    exerted: tuple[str, ...] = ()
    untapped: tuple[str, ...] = ()
    #: What a convoke cost tapped, apart from ``tapped``.
    convoked: tuple[str, ...] = ()
    #: What a delve cost exiled, apart from ``exiled``.
    delved: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActivationPayload(_PayloadQueries):
    """``resolution`` / ``activation``: the cost half of a resolution pair.

    ``{events, costs}`` (root spec § 8.14, FR-042a): what casting did — a
    spell card's ``zone_change`` to the stack, ``[]`` for an activated ability
    — and what was paid. It carries no ``outcome``, which is the effect
    half's: written at the cast, before the item's fate, it always links. The
    cast's events are listed, not happened: they describe the cast itself,
    which no per-entity output supervises, so :func:`events_of` gives none.
    """

    costs: Costs
    #: What casting did.
    events: tuple[Event, ...] = ()

    def listed_events(self) -> tuple[Event, ...]:
        return self.events

    def all_events(self) -> tuple[Event, ...]:
        return self.events


@dataclass(frozen=True, slots=True)
class ConditionMet:
    """Whether one conditional clause of a resolution met its condition."""

    #: The clause, spelled as its events' ``attributed_to``: ``root`` or the
    #: chain index as a string.
    segment: str
    met: bool


@dataclass(frozen=True, slots=True)
class ResolutionPayload(_PayloadQueries):
    """``resolution`` / ``resolution``: the effect half.

    Attribution granularity is the sub-ability; an event attributed to a link
    the sidecar does not map falls back to the root line rather than being
    dropped, so the list is never lossy.
    """

    events: tuple[Event, ...] = ()
    #: Gen-3: one entry per conditional clause that resolved in the half, in
    #: resolution order; empty where none did.
    conditions_met: tuple[ConditionMet, ...] | None = None
    #: Gen-3 (FR-042b): what became of the stack item — resolved, fizzled,
    #: partially fizzled or declined. None where the half resolves no stack
    #: item (a mana record, the game-end rescue record) and on records collected
    #: before the record moments.
    outcome: ResolutionOutcome | None = None
    #: Whether the record wrote an ``outcome`` key at all, null included, so a
    #: record read and written back keeps its bytes. Not part of equality.
    outcome_written: bool = field(default=False, compare=False)

    def happened_events(self) -> tuple[Event, ...]:
        return self.events

    def listed_events(self) -> tuple[Event, ...]:
        return self.events

    def all_events(self) -> tuple[Event, ...]:
        return self.events


@dataclass(frozen=True, slots=True)
class RewritePayload(_PayloadQueries):
    """``rewrite``: what a replacement effect received, and what it did.

    The parameter maps are deep-copied at the hook: replacements mutate nested
    structures in place, so a shallow copy would record the outgoing event in
    both slots. Forge mostly substitutes by running a different ability rather
    than editing the event, which ``result`` and ``replaced_by`` record.

    Gen-3: ``incoming_unknown`` says Forge could not copy the event as it
    arrived; ``incoming`` then holds the event as it left and ``outgoing`` is
    None, which says nothing about an in-place edit (:attr:`arrived`,
    :attr:`departed`, :attr:`rewritten`). A damage replacement that set an
    ability aside (Swans of Bryn Argoll's draw, a ``Prevent$ True`` rider)
    names it in ``buffered``, says in ``buffered_ran`` whether Forge ran it,
    and holds its run's events in ``buffered_events``: the outcome of this
    record, which no resolution half holds. Several records waiting on one
    run: the oldest holds the events, the others ran with none. A
    ``ReplaceDamage`` run in place whose sub-ability is set aside (Forcefield)
    names one line in both ``replaced_by`` and ``buffered``.
    """

    incoming: Event
    #: The rewritten event, or ``None`` when the parameter map was not edited.
    outgoing: Event | None = None
    #: ``None`` only on a record written before the hook passed one — "no
    #: opinion", which must not be read as ``not_replaced``.
    result: RewriteResult | None = None
    #: The ability that ran instead, keyed like the record's ``ability``.
    replaced_by: tuple[ProvenanceKey, ...] = field(default_factory=tuple)
    #: Gen-3: ``incoming`` is the event as it left, what came in unknown.
    incoming_unknown: bool = False
    #: Gen-3: the provenance keys of the ability the replacement set aside
    #: (``()`` where it keys to no printed line); None where it set none aside.
    buffered: tuple[ProvenanceKey, ...] | None = None
    #: Whether Forge ran the set-aside ability; None where none was set aside.
    buffered_ran: bool | None = None
    #: The events of the set-aside ability's run this record took, attributed
    #: on that ability's chain.
    buffered_events: tuple[Event, ...] = ()

    @property
    def arrived(self) -> Event | None:
        """The event as it arrived at the replacement; None when unknown."""
        return None if self.incoming_unknown else self.incoming

    @property
    def departed(self) -> Event:
        """The event as it left: the edited one, else the one that arrived
        (or, when that is unknown, the one ``incoming`` holds)."""
        return self.outgoing if self.outgoing is not None else self.incoming

    @property
    def rewritten(self) -> bool | None:
        """Whether the event's parameter map was edited in place; None when
        what arrived is unknown, so no edit can be told."""
        if self.incoming_unknown:
            return None
        return self.outgoing is not None

    @property
    def holds_buffered_run(self) -> bool:
        """Whether this record holds events of its set-aside ability's run
        (the one record of a shared run that took them)."""
        return bool(self.buffered_ran and self.buffered_events)

    def happened_events(self) -> tuple[Event, ...]:
        # The incoming event is what was proposed, not what happened; returning
        # it would train the head on the replaced event. The set-aside
        # ability's run did happen, and no other record holds it.
        edited = (self.outgoing,) if self.outgoing is not None else ()
        return (*edited, *self.buffered_events)

    def all_events(self) -> tuple[Event, ...]:
        pair = (self.incoming,) if self.outgoing is None else (self.incoming, self.outgoing)
        return (*pair, *self.buffered_events)

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
        yield from self.replaced_by
        yield from self.buffered or ()


@dataclass(frozen=True, slots=True)
class ChannelChange:
    """One contribution channel, read through the token grammar
    :class:`Contribution` describes."""

    gained: tuple[str, ...] = ()
    #: Each removed value, its leading ``-`` dropped.
    lost: tuple[str, ...] = ()
    #: A lone ``=``: the static sets the line outright. The record names nothing
    #: it displaced; on gen-3, targets read it off the board without the static.
    sets_outright: bool = False

    @classmethod
    def parse(cls, tokens: tuple[str, ...]) -> ChannelChange:
        gained: list[str] = []
        lost: list[str] = []
        for token in tokens:
            if token == "=":
                continue
            if token.startswith("-"):
                lost.append(token[1:])
            else:
                gained.append(token)
        return cls(gained=tuple(gained), lost=tuple(lost), sets_outright="=" in tokens)


@dataclass(frozen=True, slots=True)
class Contribution:
    """One entity's share of a continuous effect, per layer channel.

    ``types`` and ``colors`` are each one flat list because the record schema
    fixes the field, so the operation rides on the token:

    ==================  ====================================================
    ``creature``        the static adds this type (``W``: this colour)
    ``-creature``       the static removes it
    ``=``               the tokens after it are a line the static *sets*
                        rather than adds to; ``=`` alone with ``C`` is what
                        turns a permanent colourless
    ``all-creature-types``  and its ``-all-*`` siblings: a whole class at once
    ==================  ====================================================

    Types are spelled as the snapshot spells an entity's own, subtypes and
    supertypes included.

    ``set_pt`` is the base power and toughness the static sets (layer 7a or
    7b), a half ``None`` where it sets only the other; ``None`` for a static
    that sets no base. ``controller`` is the player the static gives control
    to, ``None`` for one that changes no controller. The snapshot beside them
    is the board without the static, so its base and controller are the ones
    the static replaces.
    """

    entity: str
    pt_boost: tuple[int, int] = (0, 0)
    set_pt: tuple[int | None, int | None] | None = None
    keywords: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    #: Gen-3: a token per layer-6 change, a granted line's key as
    #: ``ProvenanceKey.describe`` spells it (``runtime`` for none), ``-`` before
    #: one taken away; None on a record written before the member existed.
    abilities: tuple[str, ...] | None = None
    #: Gen-3: the hidden keywords the static grants, as Forge writes them.
    hidden_keywords: tuple[str, ...] | None = None
    name: str | None = None
    controller: str | None = None

    @property
    def keyword_changes(self) -> ChannelChange:
        """The keywords granted and removed; the channel takes the ``-`` too."""
        return ChannelChange.parse(self.keywords)

    @property
    def type_changes(self) -> ChannelChange:
        return ChannelChange.parse(self.types)

    @property
    def color_changes(self) -> ChannelChange:
        return ChannelChange.parse(self.colors)


@dataclass(frozen=True, slots=True)
class ContinuousPayload(_PayloadQueries):
    """``continuous``: one record per ``(game, static, board_hash)``."""

    contributions: tuple[Contribution, ...] = ()
    board_hash: str = ""


@dataclass(frozen=True, slots=True)
class CombatPayload(_PayloadQueries):
    """``combat``: one record per damage step, so a first-strike combat yields two."""

    attackers: tuple[str, ...] = ()
    blocks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    assignment_choices: dict[str, dict[str, int]] = field(default_factory=dict)
    events: tuple[Event, ...] = ()
    #: Damage-step probes only: the stripped keyword and its carrier, both
    #: needed to reproduce the perturbation on a board with two tramplers.
    probed_keyword: str | None = None
    probed_entity: str | None = None

    def happened_events(self) -> tuple[Event, ...]:
        return self.events

    def listed_events(self) -> tuple[Event, ...]:
        return self.events

    def all_events(self) -> tuple[Event, ...]:
        return self.events


@dataclass(frozen=True, slots=True)
class TriggerPayload(_PayloadQueries):
    """``trigger``: an evaluated trigger condition and whether it fired: on
    gen-3, ``fired`` is a trigger Forge ran (its condition held and it went on
    the stack), and an evaluation that held and never ran writes no record.

    Non-fired negatives are drawn from same-event-type evaluations at roughly
    1:1, so the model learns the condition rather than the base rate.
    """

    event: Event
    fired: bool

    def happened_events(self) -> tuple[Event, ...]:
        # An unfired trigger's event is the one it evaluated, not one it caused.
        return (self.event,) if self.fired else ()

    def all_events(self) -> tuple[Event, ...]:
        return (self.event,)


class PlayabilityCheck(StrEnum):
    """The checks a gen-3 candidate can fail, in the order the collector lists
    them (``FE/PlayabilityCheck.java``, pinned by ``test_records.py``)."""

    SPLIT_SECOND = "split_second"
    CASTING_SPEED = "casting_speed"
    WHOSE_TURN = "whose_turn"
    PHASE = "phase"
    COMBAT = "combat"
    ACTIVATION_LIMIT = "activation_limit"
    LAND_PLAY_LIMIT = "land_play_limit"
    SCRIPT_CONDITION = "script_condition"
    ZONE = "zone"
    STATIC_PROHIBITION = "static_prohibition"
    COST = "cost"
    SCRIPT_RESTRICTION = "script_restriction"


@dataclass(frozen=True, slots=True)
class FailedCheck:
    """One check a gen-3 candidate failed; the check keeps the string the
    record carries (:class:`PlayabilityCheck`)."""

    check: str
    #: One key where a static decided the check (``static_prohibition``),
    #: the candidate's own ``responsible_static``; else empty.
    responsible_static: tuple[ProvenanceKey, ...] = ()


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ability the engine evaluated for legality.

    The verdict is rules-level only: the AI's policy judgments ("life in
    danger") are never recorded, or the model would learn Forge's play style.
    """

    ability: tuple[ProvenanceKey, ...]
    can_play: bool
    affordable: bool
    has_legal_target: bool
    legal_targets: tuple[str, ...] = ()
    cost_after_adjustment: dict[str, int] = field(default_factory=dict)
    responsible_static: tuple[ProvenanceKey, ...] = ()
    #: Gen-3: whether the play's timing allows it now (``can_play`` is
    #: ``canPlay ∧ timing ∧`` no static prohibition).
    timing: bool | None = None
    #: Gen-3: every check the candidate failed, in :class:`PlayabilityCheck`
    #: order; empty for a playable one.
    failed_checks: tuple[FailedCheck, ...] | None = None
    #: Gen-3: whether the candidate is a land play (FR-033a), judged by
    #: ``Player.canPlayLand``. None on a record collected before land plays
    #: were candidates.
    land_play: bool | None = None


@dataclass(frozen=True, slots=True)
class PlayabilityDecisionPayload(_PayloadQueries):
    """``playability`` / ``decision``: every candidate at one decision point."""

    candidates: tuple[Candidate, ...] = ()

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
        # A failed check's key is its candidate's responsible_static again,
        # so it is not yielded twice.
        for candidate in self.candidates:
            yield from candidate.ability
            yield from candidate.responsible_static

    def acting_keys(
        self, ability: tuple[ProvenanceKey, ...],
    ) -> tuple[ProvenanceKey, ...]:
        # A decision names no acting line of its own; its ``[ACT]`` carries the
        # candidate's ``e`` (FR-060a), and Java writes one candidate per record.
        return self.candidates[0].ability if self.candidates else ()


@dataclass(frozen=True, slots=True)
class ForbiddenEntity:
    """An entity a static effect keeps out of a legal set, and which static.

    Gen-3 blockers: ``responsible_static`` the key of the line keeping the
    creature out: the static keeping it from blocking the anchor, else the
    one keeping it from blocking at all (Pacifism), else its own keyword line
    Forge reads as a keyword (Grakk's "can't attack or block", the "alone"
    forms), else empty (a hidden keyword with no line, Panic's "can't
    block", or the rules alone). ``responsible_own`` when that line is the
    creature's own (a static it hosts, a keyword it prints) rather than
    another card's; ``responsible_unresolved`` the reason, in the
    ``ability_unresolved`` vocabulary, when a responsible line exists and has
    no key (``responsible_static`` then empty). ``required_elsewhere`` when
    the restrictions alone would let it block the anchor and Forge refuses
    the block for a requirement to block another attacker (a lure, provoke,
    an attacker that must be blocked if able, by exactly one or by two or
    more creatures, while that is unmet; 509.1c). Gen-3 attackers carry the
    first three the same way.
    """

    entity: str
    responsible_static: tuple[ProvenanceKey, ...] = ()
    required_elsewhere: bool = False
    responsible_own: bool = False
    responsible_unresolved: str | None = None

    def __post_init__(self) -> None:
        if self.responsible_unresolved is None:
            return
        if self.responsible_static:
            raise ValueError(
                f"forbidden {self.entity}: responsible_unresolved="
                f"{self.responsible_unresolved!r} beside a responsible_static key"
            )
        if self.responsible_unresolved not in UnresolvedReason:
            raise ValueError(
                f"forbidden {self.entity}: responsible_unresolved="
                f"{self.responsible_unresolved!r} is outside the closed vocabulary"
            )
        object.__setattr__(
            self, "responsible_unresolved", UnresolvedReason(self.responsible_unresolved),
        )

    @property
    def explained(self) -> bool:
        """Whether the record names what keeps the creature out: a line,
        keyed or not, or a requirement elsewhere. A hidden keyword is read
        off the entity's ``granted_temporary`` instead."""
        return bool(self.responsible_static) or (
            self.responsible_unresolved is not None or self.required_elsewhere
        )


@dataclass(frozen=True, slots=True)
class PlayabilityAttackersPayload(_PayloadQueries):
    """``playability`` / ``attackers``: who may attack, and what stops the rest.

    A gen-3 record is anchored: one per creature the attacking player
    controls, naming the creature, the defenders it may attack and whether
    it must attack. ``legal_attackers`` and ``forbidden`` are the
    declaration's, the same on each of its records. ``anchor_attacker`` is
    None where a record is not anchored.
    """

    legal_attackers: tuple[str, ...] = ()
    forbidden: tuple[ForbiddenEntity, ...] = ()
    anchor_attacker: str | None = None
    #: The refs (``P``/``E``) of the defenders the anchor may attack, in
    #: Forge's defender order; empty for a creature that may attack none.
    legal_defenders: tuple[str, ...] = ()
    #: An attack requirement applies to the anchor, whether or not it can attack.
    must_attack: bool = False
    #: A declared anchor only: whether declaring it tapped it. None otherwise.
    tapped_by_attack: bool | None = None

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
        return _forbidding_statics(self.forbidden)


@dataclass(frozen=True, slots=True)
class PlayabilityBlockersPayload(_PayloadQueries):
    """``playability`` / ``blockers``: who may block one anchored attacker."""

    anchor_attacker: str
    legal_blockers: tuple[str, ...] = ()
    forbidden: tuple[ForbiddenEntity, ...] = ()
    min_blockers: int = 0
    #: Gen-3: the listed creatures, legal or forbidden, in candidate order,
    #: that a requirement makes block the anchor (509.1c), whether or not
    #: they can obey it.
    must_block: tuple[str, ...] | None = None

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
        return _forbidding_statics(self.forbidden)


@dataclass(frozen=True, slots=True)
class TurnActionPayload(_PayloadQueries):
    """``turn_action``: the events of one turn-based action, which no ability
    caused; on the game-end marker none, and the records the cap withheld."""

    events: tuple[Event, ...] = ()
    #: The game-end marker's: the game's records ``--max-records-per-game``
    #: kept out of the shard. None on every other turn action.
    records_withheld: int | None = None

    def happened_events(self) -> tuple[Event, ...]:
        return self.events

    def listed_events(self) -> tuple[Event, ...]:
        return self.events

    def all_events(self) -> tuple[Event, ...]:
        return self.events


def _forbidding_statics(forbidden: tuple[ForbiddenEntity, ...]) -> Iterator[ProvenanceKey]:
    for entry in forbidden:
        yield from entry.responsible_static


Payload = (
    ActivationPayload
    | ResolutionPayload
    | RewritePayload
    | ContinuousPayload
    | CombatPayload
    | TriggerPayload
    | PlayabilityDecisionPayload
    | PlayabilityAttackersPayload
    | PlayabilityBlockersPayload
    | TurnActionPayload
)

#: The payload type each ``(kind, moment/subkind)`` must carry.
PAYLOAD_TYPES: dict[tuple[RecordKind, Moment | StrEnum | None], type] = {
    (RecordKind.RESOLUTION, Moment.ACTIVATION): ActivationPayload,
    (RecordKind.RESOLUTION, Moment.RESOLUTION): ResolutionPayload,
    (RecordKind.REWRITE, None): RewritePayload,
    (RecordKind.CONTINUOUS, None): ContinuousPayload,
    (RecordKind.COMBAT, None): CombatPayload,
    (RecordKind.TRIGGER, None): TriggerPayload,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.DECISION): PlayabilityDecisionPayload,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.ATTACKERS): PlayabilityAttackersPayload,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.BLOCKERS): PlayabilityBlockersPayload,
    **{(RecordKind.TURN_ACTION, subkind): TurnActionPayload for subkind in TurnActionSubkind},
}


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """One JSONL line: an observed game event or decision point.

    A fork's record names the fork that wrote it in ``fork_kind``; its flags
    follow from it, and an intervention and a protection fork share theirs,
    so only ``fork_kind`` tells them apart (:attr:`origin`):

    ==========================  ============  ===============  ======  ============
    record                      kind          interventional   fork    fork_kind
    ==========================  ============  ===============  ======  ============
    ordinary observation        any           False            False   None
    interventional resolution   resolution    True             True    intervention
    protection fork (gen-3)     any           True             True    protection
    damage-step probe           combat        False            True    probe
    directed fork (gen-3)       any           False            True    directed
    ==========================  ============  ===============  ======  ============

    A directed fork also carries ``directed`` and the rule it aimed at in
    ``aimed_at``. A forced resolution writes its effect half alone, with no
    ``link_id``; a protection fork's cost halves link to their effect halves.
    """

    record_id: str
    run_id: str
    timestamp: str
    game_id: str
    kind: RecordKind
    actor_player: str
    state: StateSnapshot
    payload: Payload
    mode: CollectionMode = CollectionMode.DEGRADED
    moment: Moment | None = None
    subkind: PlayabilitySubkind | TurnActionSubkind | None = None
    link_id: str | None = None
    mirror_of: str | None = None
    variant_of: str | None = None
    interventional: bool = False
    fork: bool = False
    synthetic: bool = False
    #: ``actor_player`` is the random seat (FR-030).
    random_seat: bool = False
    #: ``blockers`` only: False for the declaration actually made, True for an
    #: AI query about a hypothetical (FR-031).
    what_if: bool | None = None
    #: The acting line, several keys where the line merged several traits; on a
    #: modal resolution, the chosen ``option`` line's key (FR-017).
    ability: tuple[ProvenanceKey, ...] | None = None
    #: Set only where a line was looked for and none found; a reason given as
    #: its string is read as the member.
    ability_unresolved: UnresolvedReason | None = None
    #: The record generation, 3: the decode refuses every other (FR-018).
    generation: int = 3
    #: The rules the run or fork aimed at (FR-009a); None on every record of a
    #: run that aimed at none. Collection metadata.
    aimed_at: tuple[str, ...] | None = None
    #: A directed fork's variant and operation; None elsewhere. Collection metadata.
    directed: DirectedTag | None = None
    #: What the copier left out of a fork's copy for want of a counterpart in
    #: it, ``(kind, count)`` pairs in kind order (``GameCopier.NoCounterpart``
    #: lower-cased, ``wire_closed_lists.COPY_DROPPED_KINDS``); None on a live
    #: record and on a fork whose copy left nothing out. Collection metadata:
    #: it decides nothing, it is counted (``validate-corpus``).
    copy_dropped: tuple[tuple[str, int], ...] | None = None
    #: The kind of fork that wrote the record; None on a live record, which
    #: carries no ``fork_kind``, and on a synthetic one, which copies a live
    #: record. Collection metadata.
    fork_kind: ForkKind | None = None
    #: How ``build-corpus`` copied the record (research T1); None on every
    #: collected record. Curation metadata.
    curation: Curation | None = None
    #: Envelope keys a newer writer added that this code does not model, kept
    #: verbatim so a read-and-rewrite round trip is lossless (compatibility
    #: rule 1).
    extra_fields: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._check_discriminators()
        self._check_ability()
        self._check_flags()
        self._check_fork_kind()

    def _check_discriminators(self) -> None:
        if (self.moment is not None) != (self.kind is RecordKind.RESOLUTION):
            raise ValueError(
                f"moment is the resolution kind's discriminator; kind="
                f"{self.kind.value} carries moment={self.moment}"
            )
        subkinds = SUBKINDS_OF.get(self.kind)
        if (self.subkind is not None) != (subkinds is not None) or (
            subkinds is not None and not isinstance(self.subkind, subkinds)
        ):
            raise ValueError(
                f"subkind is the playability and turn-action kinds' discriminator; "
                f"kind={self.kind.value} carries subkind={self.subkind}"
            )
        discriminator = self.discriminator
        expected = PAYLOAD_TYPES[(self.kind, discriminator)]
        if not isinstance(self.payload, expected):
            raise ValueError(
                f"{self.kind.value}"
                f"{'/' + discriminator.value if discriminator else ''} takes a "
                f"{expected.__name__}, got {type(self.payload).__name__}"
            )

    def _check_ability(self) -> None:
        if self.kind in KINDS_WITHOUT_ACTING_ABILITY and self.ability is not None:
            raise ValueError(
                f"{self.kind.value} records have no single acting line, so they "
                "carry no ability"
            )
        if self.ability_unresolved is None:
            return
        if self.kind in KINDS_WITHOUT_ACTING_ABILITY:
            raise ValueError(
                f"{self.kind.value} records look for no acting line, so they "
                f"cannot report ability_unresolved={self.ability_unresolved!r}"
            )
        if self.ability:
            raise ValueError(
                "a record cannot both name an acting line and say it could not "
                f"find one (ability_unresolved={self.ability_unresolved!r})"
            )
        if self.ability_unresolved not in UnresolvedReason:
            raise ValueError(
                f"ability_unresolved={self.ability_unresolved!r} is outside the "
                f"closed vocabulary {sorted(reason.value for reason in UnresolvedReason)}"
            )
        object.__setattr__(
            self, "ability_unresolved", UnresolvedReason(self.ability_unresolved),
        )

    def _check_flags(self) -> None:
        # A forced resolution's effect half, or any record of a protection fork
        # played on to the turn's end (FR-058), cost halves linked as live ones.
        if self.interventional and not self.fork:
            raise ValueError("an interventional record is always a fork")
        if self.mirror_of is not None and not self.fork:
            raise ValueError("mirror_of names the real record a fork mirrors")
        if self.copy_dropped is not None and not self.fork:
            raise ValueError("copy_dropped says what a fork's copy left out")
        if self.copy_dropped is not None and (
                not self.copy_dropped or any(count < 1 for _kind, count in self.copy_dropped)):
            # The connector writes the member only for a lossy copy, and each
            # kind only with at least one drop (Forks.dropped).
            raise ValueError(
                f"copy_dropped names kinds with at least one drop: {self.copy_dropped}")
        if self.variant_of is not None and not self.synthetic:
            raise ValueError("variant_of names the card a synthetic script came from")
        if self.directed is not None:
            if not self.fork or self.interventional:
                raise ValueError("a directed fork is a fork that intervened in nothing")
            if self.aimed_at is None:
                raise ValueError("a directed fork names the rule it aimed at in aimed_at")
        if self.what_if is not None and (
            self.subkind not in LEGALITY_SUBKINDS
        ):
            raise ValueError(
                "what_if classes a legality record; "
                f"kind={self.kind.value} subkind={self.subkind} carries it"
            )

    def _check_fork_kind(self) -> None:
        """A fork kind is on a collected fork's record alone, with its kind's
        flags; a synthetic record copies a live one and carries none."""
        kind = self.fork_kind
        if kind is None:
            return
        if not self.fork or self.synthetic:
            raise ValueError(
                f"fork_kind={kind.value} names the fork a collected fork record came from")
        interventional, directed = _FORK_SIGNATURES[kind]
        if (self.interventional, self.directed is not None) != (interventional, directed):
            raise ValueError(
                f"a {kind.value} fork's record carries interventional={interventional} "
                f"and {'a' if directed else 'no'} directed tag")

    @property
    def origin(self) -> RecordOrigin:
        """What produced the record, read off ``synthetic`` and ``fork_kind``
        alone: a synthetic record is a ``build-corpus`` copy of a live one,
        whatever its flags; a fork's is its ``fork_kind``'s.

        A fork record with no ``fork_kind`` was written before the member
        existed and is refused: its kind is never guessed from the flags.
        """
        if self.synthetic:
            return RecordOrigin.SYNTHETIC
        if not self.fork:
            return RecordOrigin.LIVE
        if self.fork_kind is None:
            raise ValueError(
                f"fork record {self.record_id} names no fork_kind: written before the member")
        return _FORK_ORIGINS[self.fork_kind]

    @property
    def is_game_end_record(self) -> bool:
        """Whether the record is a game-end record: an effect half with
        ``ability`` the empty tuple and no ``ability_unresolved`` (it seeks no
        line), no cost half and no outcome, which holds the game's
        ``player_lost`` and ``player_won`` when no other record took them
        (contract ``record-schema-delta.md`` (025), ``player_lost``).

        Not the game-end marker (:func:`is_game_end`). An unkeyed mana record
        also has a null outcome and no link, and carries a reason for its
        empty ability.
        """
        payload = self.payload
        return (isinstance(payload, ResolutionPayload) and payload.outcome is None
                and self.link_id is None and self._seeks_no_line)

    @property
    def _seeks_no_line(self) -> bool:
        """``ability`` is the empty tuple and says nothing about why."""
        return self.ability == () and self.ability_unresolved is None

    @property
    def discriminator(self) -> Moment | PlayabilitySubkind | TurnActionSubkind | None:
        """The ``moment`` or ``subkind`` that, with ``kind``, picks the payload type."""
        return self.moment or self.subkind

    @property
    def has_act_line(self) -> bool:
        """Whether one line acts, filling the ``[ACT]`` slot and keying the
        record's rarity: not in a combat or a legality declaration, which
        sample uniformly instead (FR-087)."""
        return (self.kind not in (RecordKind.COMBAT, RecordKind.TURN_ACTION)
                and self.subkind not in LEGALITY_SUBKINDS)

    @property
    def is_probe(self) -> bool:
        """A damage-step probe: a combat fork that intervened in nothing and
        applied no directed change."""
        return (
            self.kind is RecordKind.COMBAT
            and self.fork
            and not self.interventional
            and self.directed is None
        )


def is_game_end(record: EffectRecord) -> bool:
    """Whether ``record`` is a game-end marker: no outcome, no detector reads it."""
    return record.subkind is TurnActionSubkind.GAME_END


def events_of(record: EffectRecord) -> tuple[Event, ...]:
    """The events that happened in a record, whatever its kind."""
    return record.payload.happened_events()


def key_sites(record: EffectRecord) -> Iterator[ProvenanceKey]:
    """Every printed-line key a record names, except its emblems', in record order.

    The acting ``ability``, each entity's line keys (printed and granted, a
    gen-3 face not up's included) and temporary grants, a gen-3 floating
    effect's and delayed trigger's source lines and every line a link ends
    at, and every key its payload carries. Emblems are left out: their
    engine-built traits key to a script path in no converted tree, so a
    sidecar join would report every planeswalker ultimate as a mismatch.
    """
    if record.ability:
        yield from record.ability
    state = record.state
    for entity in state.entities:
        yield from entity.line_keys
        yield from entity.granted_temporary.abilities
    for made in (*state.global_.floating_effects, *state.global_.delayed_triggers):
        yield from made.source_lines
    for link in state.links:
        yield from link.line_keys()
    yield from record.payload.provenance_keys()


def distinct_records(records: Iterable[EffectRecord]) -> Iterator[EffectRecord]:
    """Each record's first row: a curated stratum repeats a record's row for
    every copy curation wrote of it, and a reader counting records reads each
    once (FR-068a)."""
    seen: set[str] = set()
    for record in records:
        if record.record_id not in seen:
            seen.add(record.record_id)
            yield record


def acting_keys(record: EffectRecord) -> tuple[ProvenanceKey, ...]:
    """The keys ``[ACT]`` reads: the acting line, or a decision's candidate.

    Rarity, the per-text caps and gate 1's stratum all key on this, so they
    agree with what ``[ACT]`` reads.
    """
    return record.payload.acting_keys(record.ability or ())
