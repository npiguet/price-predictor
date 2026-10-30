"""The effect-record schema: one observed game event or decision point.

The schema is frozen: later collection widens the corpus but never redefines a
field. The compatibility rules are in
``specs/023-ability-effect-model/contracts/record-schema.md`` and, with no
version field, ``tests/unit/effects/domain/test_schema_compatibility.py`` is
their enforcement. Pure dataclasses; serialization lives in
``infrastructure/record_codec.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from effects.domain.event_schema import Event
from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import StateSnapshot


class RecordKind(StrEnum):
    RESOLUTION = "resolution"
    REWRITE = "rewrite"
    CONTINUOUS = "continuous"
    COMBAT = "combat"
    TRIGGER = "trigger"
    PLAYABILITY = "playability"


class Moment(StrEnum):
    """Which half of a resolution pair a ``resolution`` record is."""

    ACTIVATION = "activation"
    RESOLUTION = "resolution"


class PlayabilitySubkind(StrEnum):
    DECISION = "decision"
    ATTACKERS = "attackers"
    BLOCKERS = "blockers"


class CollectionMode(StrEnum):
    """Whether the worker found the engine patch hooks at startup."""

    PATCHED = "patched"
    DEGRADED = "degraded"


class ResolutionOutcome(StrEnum):
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

#: Outcomes whose cost half has no linked effect half, so it carries no
#: ``link_id``. Only ``resolved`` and ``partially_fizzled`` reach resolution.
PARTNERLESS_OUTCOMES: frozenset[ResolutionOutcome] = frozenset({
    ResolutionOutcome.FIZZLED,
    ResolutionOutcome.DECLINED,
    ResolutionOutcome.COUNTERED,
})

#: The legality subkinds, the ones that carry ``what_if`` (FR-031); a
#: ``decision`` is always a real evaluation.
LEGALITY_SUBKINDS: frozenset[PlayabilitySubkind] = frozenset({
    PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS,
})


class LegalityDecision(StrEnum):
    """Whether a legality record is the declaration actually made (FR-033).

    ``unknown`` is a gen-1 record, collected before ``what_if`` existed; it is
    never a real decision.
    """

    REAL = "real"
    WHAT_IF = "what-if"
    UNKNOWN = "unknown"

    @classmethod
    def of(cls, record) -> LegalityDecision:
        """``record`` is anything carrying ``what_if``: a record or a result row."""
        if record.what_if is None:
            return cls.UNKNOWN
        return cls.WHAT_IF if record.what_if else cls.REAL


class SeatPolicy(StrEnum):
    """Whether the acting seat was Forge's AI or the random seat (FR-030).

    A gen-1 record carries no ``random_seat`` and reads as on-policy.
    """

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
    RecordKind.COMBAT, RecordKind.PLAYABILITY,
})


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
    """What the actor actually paid, as paid rather than as printed."""

    mana_by_color: dict[str, int] = field(default_factory=dict)
    tapped: tuple[str, ...] = ()
    life: int = 0
    sacrificed: tuple[str, ...] = ()
    discarded: tuple[str, ...] = ()
    exiled: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActivationPayload(_PayloadQueries):
    """``resolution`` / ``activation``: the cost half of a resolution pair."""

    costs: Costs
    outcome: ResolutionOutcome

    @property
    def has_partner(self) -> bool:
        return self.outcome not in PARTNERLESS_OUTCOMES


@dataclass(frozen=True, slots=True)
class ResolutionPayload(_PayloadQueries):
    """``resolution`` / ``resolution``: the effect half.

    Attribution granularity is the sub-ability; an event attributed to a link
    the sidecar does not map falls back to the root line rather than being
    dropped, so the list is never lossy.
    """

    events: tuple[Event, ...] = ()

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
    """

    incoming: Event
    #: The rewritten event, or ``None`` when the parameter map was not edited.
    outgoing: Event | None = None
    #: ``None`` only on a record written before the hook passed one — "no
    #: opinion", which must not be read as ``not_replaced``.
    result: RewriteResult | None = None
    #: The ability that ran instead, keyed like the record's ``ability``.
    replaced_by: tuple[ProvenanceKey, ...] = field(default_factory=tuple)

    @property
    def rewritten(self) -> bool:
        """Whether the event's parameter map was edited in place."""
        return self.outgoing is not None

    def happened_events(self) -> tuple[Event, ...]:
        # The incoming event is what was proposed, not what happened; returning
        # it would train the head on the replaced event.
        return (self.outgoing,) if self.outgoing is not None else ()

    def all_events(self) -> tuple[Event, ...]:
        if self.outgoing is None:
            return (self.incoming,)
        return (self.incoming, self.outgoing)

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
        return iter(self.replaced_by)


@dataclass(frozen=True, slots=True)
class ChannelChange:
    """One contribution channel, read through the token grammar
    :class:`Contribution` describes."""

    gained: tuple[str, ...] = ()
    #: Each removed value, its leading ``-`` dropped.
    lost: tuple[str, ...] = ()
    #: A lone ``=``: the static sets the line outright, so what it displaced is
    #: not named (the snapshot is the board after the static).
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
    """

    entity: str
    pt_boost: tuple[int, int] = (0, 0)
    keywords: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    name: str | None = None

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
    """``trigger``: an evaluated trigger condition and whether it fired.

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


@dataclass(frozen=True, slots=True)
class PlayabilityDecisionPayload(_PayloadQueries):
    """``playability`` / ``decision``: every candidate at one decision point."""

    candidates: tuple[Candidate, ...] = ()

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
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
    """An entity a static effect keeps out of a legal set, and which static."""

    entity: str
    responsible_static: tuple[ProvenanceKey, ...] = ()


@dataclass(frozen=True, slots=True)
class PlayabilityAttackersPayload(_PayloadQueries):
    """``playability`` / ``attackers``: who may attack, and what stops the rest."""

    legal_attackers: tuple[str, ...] = ()
    forbidden: tuple[ForbiddenEntity, ...] = ()

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
        return _forbidding_statics(self.forbidden)


@dataclass(frozen=True, slots=True)
class PlayabilityBlockersPayload(_PayloadQueries):
    """``playability`` / ``blockers``: who may block one anchored attacker."""

    anchor_attacker: str
    legal_blockers: tuple[str, ...] = ()
    forbidden: tuple[ForbiddenEntity, ...] = ()
    min_blockers: int = 0

    def provenance_keys(self) -> Iterator[ProvenanceKey]:
        return _forbidding_statics(self.forbidden)


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
)

#: The payload type each ``(kind, moment/subkind)`` must carry.
PAYLOAD_TYPES: dict[tuple[RecordKind, Moment | PlayabilitySubkind | None], type] = {
    (RecordKind.RESOLUTION, Moment.ACTIVATION): ActivationPayload,
    (RecordKind.RESOLUTION, Moment.RESOLUTION): ResolutionPayload,
    (RecordKind.REWRITE, None): RewritePayload,
    (RecordKind.CONTINUOUS, None): ContinuousPayload,
    (RecordKind.COMBAT, None): CombatPayload,
    (RecordKind.TRIGGER, None): TriggerPayload,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.DECISION): PlayabilityDecisionPayload,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.ATTACKERS): PlayabilityAttackersPayload,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.BLOCKERS): PlayabilityBlockersPayload,
}


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """One JSONL line: an observed game event or decision point.

    Three flag signatures distinguish how a record was produced, and the probe's
    ``interventional = False`` is what separates it from an intervention by
    flags alone:

    ==========================  ============  ===============  ======
    record                      kind          interventional   fork
    ==========================  ============  ===============  ======
    ordinary observation        any           False            False
    interventional resolution   resolution    True             True
    damage-step probe           combat        False            True
    ==========================  ============  ===============  ======
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
    subkind: PlayabilitySubkind | None = None
    link_id: str | None = None
    mirror_of: str | None = None
    variant_of: str | None = None
    interventional: bool = False
    fork: bool = False
    synthetic: bool = False
    #: ``actor_player`` is the random seat (FR-030). None on a gen-1 record:
    #: falsy, so on-policy, but kept apart from False so a rewritten record
    #: stays the generation it was collected in (FR-033).
    random_seat: bool | None = False
    #: ``attackers``/``blockers`` only: False for the declaration actually made,
    #: True for an AI query about a hypothetical (FR-031); None on gen-1.
    what_if: bool | None = None
    #: The acting line, several keys where the line merged several traits; on a
    #: modal resolution, the chosen ``option`` line's key (FR-017).
    ability: tuple[ProvenanceKey, ...] | None = None
    #: Set only where a line was looked for and none found; a reason given as
    #: its string is read as the member.
    ability_unresolved: UnresolvedReason | None = None
    #: Envelope keys a newer writer added that this code does not model, kept
    #: verbatim so a read-and-rewrite round trip is lossless (compatibility
    #: rule 1).
    extra_fields: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._check_discriminators()
        self._check_ability()
        self._check_flags()
        self._check_link_id()

    def _check_discriminators(self) -> None:
        if (self.moment is not None) != (self.kind is RecordKind.RESOLUTION):
            raise ValueError(
                f"moment is the resolution kind's discriminator; kind="
                f"{self.kind.value} carries moment={self.moment}"
            )
        if (self.subkind is not None) != (self.kind is RecordKind.PLAYABILITY):
            raise ValueError(
                f"subkind is the playability kind's discriminator; kind="
                f"{self.kind.value} carries subkind={self.subkind}"
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
        if self.interventional:
            if self.kind is not RecordKind.RESOLUTION:
                raise ValueError(
                    "an interventional record is a forced resolution; kind="
                    f"{self.kind.value}"
                )
            if not self.fork:
                raise ValueError("an interventional record is always a fork")
        if self.mirror_of is not None and not self.fork:
            raise ValueError("mirror_of names the real record a fork mirrors")
        if self.variant_of is not None and not self.synthetic:
            raise ValueError("variant_of names the card a synthetic script came from")
        if self.what_if is not None and (
            self.subkind not in LEGALITY_SUBKINDS
        ):
            raise ValueError(
                "what_if classes a legality record; "
                f"kind={self.kind.value} subkind={self.subkind} carries it"
            )

    def _check_link_id(self) -> None:
        if self.link_id is None:
            return
        if self.interventional:
            raise ValueError(
                "an interventional resolution writes the effect half only, so it "
                "has no activation partner to link to"
            )
        payload = self.payload
        if isinstance(payload, ActivationPayload) and not payload.has_partner:
            raise ValueError(
                f"a {payload.outcome.value} activation never reaches resolution, "
                "so it has no linked effect half"
            )

    @property
    def discriminator(self) -> Moment | PlayabilitySubkind | None:
        """The ``moment`` or ``subkind`` that, with ``kind``, picks the payload type."""
        return self.moment or self.subkind

    @property
    def has_act_line(self) -> bool:
        """Whether one line acts, filling the ``[ACT]`` slot and keying the
        record's rarity: not in a combat or a legality declaration, which
        sample uniformly instead (FR-087)."""
        return self.kind is not RecordKind.COMBAT and self.subkind not in LEGALITY_SUBKINDS

    @property
    def is_probe(self) -> bool:
        """A damage-step probe: a combat fork that intervened in nothing."""
        return (
            self.kind is RecordKind.COMBAT
            and self.fork
            and not self.interventional
        )


def events_of(record: EffectRecord) -> tuple[Event, ...]:
    """The events that happened in a record, whatever its kind."""
    return record.payload.happened_events()


def key_sites(record: EffectRecord) -> Iterator[ProvenanceKey]:
    """Every printed-line key a record names, except its emblems', in record order.

    The acting ``ability``, each entity's printed and granted keys, and every
    key its payload carries. Emblems are left out: their engine-built traits
    key to a script path in no converted tree, so a sidecar join would report
    every planeswalker ultimate as a mismatch.
    """
    if record.ability:
        yield from record.ability
    for entity in record.state.entities:
        yield from entity.printed
        yield from entity.granted_attached
        yield from entity.granted_temporary.abilities
    yield from record.payload.provenance_keys()


def acting_keys(record: EffectRecord) -> tuple[ProvenanceKey, ...]:
    """The keys ``[ACT]`` reads: the acting line, or a decision's candidate.

    Rarity, the per-text caps and gate 1's stratum all key on this, so they
    agree with what ``[ACT]`` reads.
    """
    return record.payload.acting_keys(record.ability or ())
