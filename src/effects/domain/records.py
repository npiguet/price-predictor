"""The effect-record schema: one observed game event or decision point.

This schema is **fixed before stage-one collection begins**. Later stages widen
the corpus — new ``kind`` values become reachable, new snapshot tiers appear —
but never redefine a field, which is what lets a stage-one corpus stay trainable
alongside stage-four records. The four compatibility rules are stated in
``specs/023-ability-effect-model/contracts/record-schema.md`` and enforced by
``tests/unit/effects/domain/test_schema_compatibility.py``; there is no version
field, so the tests are the enforcement.

Pure dataclasses with no torch and no IO — serialization lives in
``infrastructure/record_io.py``, the same split
``draft/domain/draft_geometry.py`` makes against ``draft_record_io``.
"""

from __future__ import annotations

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
    """Whether the worker found the engine patch hooks at startup.

    Detected per run rather than compiled in, because the patch lives in a
    sibling checkout that is rebuilt independently and re-patched by hand after
    every Forge upgrade. A build-time switch would silently mislabel a corpus
    collected after a lapsed patch.
    """

    PATCHED = "patched"
    DEGRADED = "degraded"


class ResolutionOutcome(StrEnum):
    RESOLVED = "resolved"
    FIZZLED = "fizzled"
    PARTIALLY_FIZZLED = "partially_fizzled"
    DECLINED = "declined"
    COUNTERED = "countered"


#: Outcomes whose cost half has no linked effect half, so it carries no
#: ``link_id``. Only ``resolved`` and ``partially_fizzled`` reach resolution.
PARTNERLESS_OUTCOMES: frozenset[ResolutionOutcome] = frozenset({
    ResolutionOutcome.FIZZLED,
    ResolutionOutcome.DECLINED,
    ResolutionOutcome.COUNTERED,
})

#: Envelope fields that describe how a record was collected. They must never
#: reach the model as inputs (FR-018): a model that can see ``fork`` learns to
#: predict differently on forks, which is exactly the confound the forks exist
#: to avoid. ``ability_unresolved`` belongs here for the same reason: it says
#: why the collector could not name a line, which is a fact about the resolver
#: and not about the game.
COLLECTION_METADATA_FIELDS: frozenset[str] = frozenset({
    "mode", "interventional", "fork", "synthetic", "ability_unresolved",
})

#: Why a record that does name an acting line carries no key for it. A closed
#: vocabulary, because the point of the field is to separate "no printed line
#: exists" from "the resolver failed": an engine-built card (the Monarch, a
#: dungeon, an emblem) has no script file in any tree and never will, while
#: ``unindexable`` is the signature of a resolver bug and must not appear in a
#: healthy run at all.
UNRESOLVED_REASONS: frozenset[str] = frozenset({
    "engine_effect", "no_card_state", "unknown_kind",
    "granted_keyword", "granted_trait", "unindexable",
})

#: Kinds where no single line acts, so the envelope carries no ``ability``.
#: A combat record's participants each act; a playability record is a verdict
#: about candidates, not the resolution of one of them.
KINDS_WITHOUT_ACTING_ABILITY: frozenset[RecordKind] = frozenset({
    RecordKind.COMBAT, RecordKind.PLAYABILITY,
})


# ── per-kind payloads ───────────────────────────────────────────────────


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
class ActivationPayload:
    """``resolution`` / ``activation``: the cost half of a resolution pair."""

    costs: Costs
    outcome: ResolutionOutcome

    @property
    def has_partner(self) -> bool:
        return self.outcome not in PARTNERLESS_OUTCOMES


@dataclass(frozen=True, slots=True)
class ResolutionPayload:
    """``resolution`` / ``resolution``: the effect half.

    Attribution granularity is the sub-ability; an event attributed to a link
    the sidecar does not map falls back to the root line rather than being
    dropped, so the list is never lossy.
    """

    events: tuple[Event, ...] = ()


@dataclass(frozen=True, slots=True)
class RewritePayload:
    """``rewrite``: what a replacement effect received and what it emitted.

    The parameter maps are deep-copied at the hook. The handler's own copy is
    shallow and replacements mutate nested structures in place, so a shallow
    copy would record the outgoing event in both slots.
    """

    incoming: Event
    outgoing: Event


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

    Types are spelled the way the snapshot spells an entity's own, and carry
    subtypes and supertypes as well as the core types the model has fields for.
    """

    entity: str
    pt_boost: tuple[int, int] = (0, 0)
    keywords: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ContinuousPayload:
    """``continuous``: one record per stable board.

    Coalescing per ``(game, static, board_hash)`` is itself the cap — a static
    that applies every recompute would otherwise write a record per priority
    pass, and none of them would say anything the first did not.
    """

    contributions: tuple[Contribution, ...] = ()
    board_hash: str = ""


@dataclass(frozen=True, slots=True)
class CombatPayload:
    """``combat``: one record per damage step.

    A first-strike combat therefore yields two records, which is the only way a
    keyword that acts by splitting the step can be visible in the corpus at all.
    """

    attackers: tuple[str, ...] = ()
    blocks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    assignment_choices: dict[str, dict[str, int]] = field(default_factory=dict)
    events: tuple[Event, ...] = ()
    #: Damage-step probes only: which keyword was stripped, and from which
    #: creature. Both are needed to reproduce the perturbation model-side —
    #: a board with two tramplers gives the keyword alone two readings, and the
    #: comparison the probe exists for would be against the wrong one. Absent on
    #: an observed combat record, which perturbs nothing.
    probed_keyword: str | None = None
    probed_entity: str | None = None


@dataclass(frozen=True, slots=True)
class TriggerPayload:
    """``trigger``: an evaluated trigger condition and whether it fired.

    Non-fired negatives are drawn from same-event-type evaluations at roughly
    1:1, so the model learns the condition rather than the base rate.
    """

    event: Event
    fired: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ability the engine evaluated for legality.

    ``verdict`` is rules-level only. The AI's policy judgments ("another time",
    "life in danger") are never recorded — they are the AI's opinion, not the
    game's rules, and training on them would teach the model Forge's play style
    instead of Magic.
    """

    ability: tuple[ProvenanceKey, ...]
    can_play: bool
    affordable: bool
    has_legal_target: bool
    legal_targets: tuple[str, ...] = ()
    cost_after_adjustment: dict[str, int] = field(default_factory=dict)
    responsible_static: tuple[ProvenanceKey, ...] = ()


@dataclass(frozen=True, slots=True)
class PlayabilityDecisionPayload:
    """``playability`` / ``decision``: every candidate at one decision point."""

    candidates: tuple[Candidate, ...] = ()


@dataclass(frozen=True, slots=True)
class ForbiddenEntity:
    """An entity a static effect keeps out of a legal set, and which static."""

    entity: str
    responsible_static: tuple[ProvenanceKey, ...] = ()


@dataclass(frozen=True, slots=True)
class PlayabilityAttackersPayload:
    """``playability`` / ``attackers``: who may attack, and what stops the rest."""

    legal_attackers: tuple[str, ...] = ()
    forbidden: tuple[ForbiddenEntity, ...] = ()


@dataclass(frozen=True, slots=True)
class PlayabilityBlockersPayload:
    """``playability`` / ``blockers``: who may block one anchored attacker."""

    anchor_attacker: str
    legal_blockers: tuple[str, ...] = ()
    forbidden: tuple[ForbiddenEntity, ...] = ()
    min_blockers: int = 0


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
PAYLOAD_TYPES: dict[tuple[RecordKind, str | None], type] = {
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


# ── the envelope ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """One JSONL line: an observed game event or decision point.

    Immutable once written. ``record_id`` and ``game_id`` both carry the worker
    slot and the JVM lifetime it counted in, because workers count
    independently and each JVM counts from zero; ``game_id`` is the join key
    for a checkpoint's recorded split, so a repeated one merges unrelated
    games.

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
    #: The acting line, as several keys where the rendered line merged several
    #: traits. On a modal resolution this is the chosen ``option`` line's key,
    #: not the parent ``spell`` line's (FR-017).
    ability: tuple[ProvenanceKey, ...] | None = None
    #: Why ``ability`` came back empty on a kind that does name a line, from
    #: :data:`UNRESOLVED_REASONS`. Set only where there was a line to look for
    #: and none was found, so an empty ``ability`` alone no longer conflates
    #: "the Monarch has no printed line anywhere" with "the resolver regressed"
    #: — the ambiguity that hid a broken resolver for a whole collection run.
    #: Collection metadata; never a model input.
    ability_unresolved: str | None = None
    #: Envelope keys a newer writer added that this code does not model, kept
    #: verbatim so reading and rewriting a shard is lossless. This is the
    #: mechanism behind compatibility rule 1: an added field must survive a
    #: round trip through code that predates it. Never a model input.
    extra_fields: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._check_discriminators()
        self._check_ability()
        self._check_flags()
        self._check_link_id()

    # ── invariants ──────────────────────────────────────────────────────

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
        discriminator = self.moment or self.subkind
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
        # A reason without a failed lookup is a contradiction, and a reason on
        # a kind that names no line is a category error — both would make the
        # field unreadable as the diagnostic it exists to be.
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
        if self.ability_unresolved not in UNRESOLVED_REASONS:
            raise ValueError(
                f"ability_unresolved={self.ability_unresolved!r} is outside the "
                f"closed vocabulary {sorted(UNRESOLVED_REASONS)}"
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

    # ── derived views ───────────────────────────────────────────────────

    @property
    def is_probe(self) -> bool:
        """A damage-step probe: a combat fork that intervened in nothing."""
        return (
            self.kind is RecordKind.COMBAT
            and self.fork
            and not self.interventional
        )

    @property
    def worker(self) -> str:
        """The worker slot this record's ids were counted by."""
        return self.record_id.split(".")[-2].split("-")[0]

    @property
    def lifetime(self) -> str:
        """The worker JVM lifetime the ids were counted in.

        Empty on a shard written before ids carried one. The counters restart
        at zero in every worker JVM and a collection run is hundreds of JVMs
        per slot, so this is the segment that makes ``record_id`` and
        ``game_id`` unique rather than merely per-worker.
        """
        return self.record_id.split(".")[-2].partition("-")[2]

    def model_input_fields(self) -> dict:
        """The envelope fields the model may see.

        Collection metadata is excluded here rather than filtered downstream, so
        a new consumer cannot reach it by accident. What remains describes the
        game situation; what is dropped describes how we came to observe it.
        """
        return {
            "kind": self.kind,
            "moment": self.moment,
            "subkind": self.subkind,
            "actor_player": self.actor_player,
            "ability": self.ability,
            "state": self.state,
            "payload": self.payload,
        }
