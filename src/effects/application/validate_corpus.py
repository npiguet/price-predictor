"""Invariants a shard directory must hold, measured rather than asserted.

Run this on the **first few minutes** of a collection pass, not on the finished
corpus. The first collected corpus took eight hours and twelve defects were
found afterwards by reading it; every one of them was already visible in the
first minute of shards, and none of them raised anything. A collector that
writes a field as a literal, or reuses an id, or picks its snapshot depth per
call site produces records that parse cleanly, satisfy every contract test, and
are worthless.

So this reports numbers, not verdicts: each invariant prints what it measured,
because "the duplicate rate is 4.3%" is actionable in a way that "duplicates:
FAIL" is not, and because a run that barely clears a threshold is something an
operator needs to see before spending eight hours on it.

**Measuring and judging are separate.** A :class:`Finding` that carries no
threshold reports ``[WATCH]`` and never fails a run: some numbers — how often an
event says where a card came from, how many resolution events name a producing
clause — are worth watching every pass without being a verdict, and the ones
whose thresholds default to ``None`` become verdicts the moment an operator
passes a floor. The opposite mistake is the more expensive one: the "each
game_id names one game" check failed 168 of 177 games on a *healthy* smoke
corpus, and an invariant that fails on every good run teaches an operator to
skip the whole checklist.

Streaming, one pass, over :func:`~effects.infrastructure.record_io.read_shard`
so shard discovery, gzip-member recovery and the trailing-partial-line rule stay
in the reader where they already live. Memory is proportional to the window:
every ``record_id``, one digest per record and every distinct provenance key are
held, which is why the window is minutes and why ``--limit`` exists.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from effects.domain.event_schema import (
    CAUSE_BEARING_TYPES,
    Event,
    EventType,
    attribution_kind,
)
from effects.domain.provenance import KeyResolution, ProvenanceKey
from effects.domain.records import (
    KINDS_WITHOUT_ACTING_ABILITY,
    REWRITE_RESULTS_RUNNING_AN_ABILITY,
    REWRITE_RESULTS_WITHOUT_ABILITY,
    ActivationPayload,
    CollectionMode,
    CombatPayload,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.infrastructure.sidecar_io import UnconfiguredTree

#: Envelope fields that differ between two otherwise identical records because
#: of *when* they were written rather than *what* they say. Blanked before
#: hashing, so the duplicate rate counts "the same observation twice in one
#: game". ``mode`` joins them because it is constant within a run, and keeping
#: it would only make a degraded and a patched run incomparable.
_IDENTITY_FIELDS = (
    "record_id", "run_id", "timestamp", "game_id",
    "link_id", "mirror_of", "variant_of", "mode",
)

#: The six cost channels, in the order the schema lists them.
_COST_FIELDS = (
    "mana_by_color", "tapped", "life", "sacrificed", "discarded", "exiled",
)

#: The channels a window with enough activations in it must have exercised.
#: Not all six: sacrifice, discard and exile costs are genuinely rare in a
#: sealed limited pool and life payment nearly so, and a check that cries wolf
#: on those is a check an operator learns to skip. Mana and tapping are paid
#: many times a game by every deck, so a window that has neither is reporting
#: on a collector rather than on a format — which is exactly what the first
#: corpus did: 14.7M records, `tapped` populated zero times.
_COST_FIELDS_ALWAYS_PAID = ("mana_by_color", "tapped")

#: How many offending examples a finding's detail carries. Enough to recognise a
#: pattern, few enough that a broken run's report still fits on a screen.
_EXAMPLES = 5


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where each measured rate stops being acceptable.

    Defaults are set against the first corpus's measurements, so a run that
    reproduces any of its defects fails rather than squeaking through: it keyed
    59% of its resolution records, duplicated 4.3% of the corpus, paired 48% of
    its link ids, and held 804 ``game_id`` values for 31,662 games.

    A field typed ``float | None`` defaults to ``None``, which means *watch this
    number, do not judge it*. Those are the channels whose healthy value is not
    yet known — a ``zone_change`` out of nowhere is legitimate for a card made
    rather than moved — so pinning a floor before the collector settles would
    make the report a wolf-crier. Passing a floor turns the measurement into a
    verdict without changing what is measured.
    """

    #: Share of records that must resolve their acting line to a printed key.
    min_keyed_rate: float = 0.95
    #: Share of records that may be byte-identical to an earlier record of the
    #: same kind in the same game.
    max_duplicate_rate: float = 0.02
    #: Share of link ids that may lack exactly one activation and one
    #: resolution half. A worker killed mid-game truncates its last block, so a
    #: few unpaired halves at the tail of a window are expected.
    max_unpaired_link_rate: float = 0.02
    #: Distinct entity names one game may show. Two forty-card sealed decks
    #: plus tokens sit far below this; the first corpus averaged about 460 per
    #: ``game_id``, because dozens of unrelated games shared one.
    max_names_per_game: int = 120
    #: Activation records a window needs before a dead cost channel means
    #: anything. Under this, a channel reading zero is scarcity; over it, it is
    #: a channel nothing writes to.
    min_cost_evidence: int = 200
    #: How far a snapshot's turn may sit below the highest already seen in its
    #: game. Records are written when they are observed but carry the snapshot
    #: of the moment they describe, so a deferred half can legitimately lag by
    #: a turn; a *game boundary* shows up as a jump of many turns, not of one.
    turn_jump_tolerance: int = 1
    #: Share of a record's own events that may repeat another event in the same
    #: record. Two identical outcomes on identical subjects inside one
    #: resolution are one outcome written twice; the smoke corpus duplicated
    #: 13.68% of its events this way while its *record* duplicate rate read
    #: 0.00%, which is why the record-level check alone could not see it.
    max_duplicate_event_rate: float = 0.02
    #: Share of trigger records that may report ``fired = true``. The negatives
    #: are drawn from same-event-type evaluations at a ~1:1 target, so a window
    #: far above this is a sampler that stopped drawing them — and a corpus of
    #: positives teaches the base rate rather than the condition.
    max_trigger_fired_share: float = 0.65
    #: Share of ``zone_change`` events that must say ``from_zone``. ``None``
    #: watches the number without judging it.
    min_zone_change_from_zone_rate: float | None = None
    #: Share of resolution events that must name a producing clause — a
    #: sub-ability index, the root line, or an explicit ``unresolved``. ``None``
    #: watches the number without judging it.
    min_attributed_rate: float | None = None
    #: Share of fork records' events that must name a producing clause.
    #: Separate from :attr:`min_attributed_rate` because the fork collectors
    #: are different code that the attribution channel reached last: the
    #: aggregate rate hides a fork path writing ``null`` on every event behind
    #: the observed records' healthy majority. ``None`` watches it.
    min_fork_attributed_rate: float | None = None
    #: Share of the events whose own parameter row declares ``cause`` that
    #: must populate it. An empty ``cause`` is why two attackers dealing 1 to
    #: the same player serialize byte-identically and read as a duplicated
    #: event. ``None`` watches the number: the healthy share is below 1 —
    #: some hooks genuinely name no causing object — and is not yet known.
    min_cause_rate: float | None = None
    #: Share of ``replaced``/``updated`` rewrite records that must name the
    #: ability that ran instead. ``None`` watches the number without judging
    #: it, and it has to stay watched until someone measures the healthy share:
    #: a replacement scripted with ``ReplacementResult$`` returns one of those
    #: two outcomes having run no ability at all, so the ceiling is below 1 and
    #: nobody knows by how much.
    min_rewrite_replaced_by_rate: float | None = None
    #: Share of fork records that may carry a ``state.global.turn`` differing
    #: from the record they mirror. Judged at zero rather than watched: a fork
    #: is taken *at* the moment it mirrors, so its snapshot is that moment's,
    #: and there is no healthy rate at which the two disagree. The smoke
    #: corpus disagreed on 3.9% of probe forks, always behind, and that was
    #: the whole residue of the game_id invariant — which reported it as a
    #: backward turn jump, naming the game rather than the collector.
    max_mirror_turn_disagreement_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class Finding:
    """One invariant's result: what it measured, and whether that is allowed.

    ``watched`` is the separation the module docstring argues for: a watched
    finding has a measurement and no threshold, prints ``[WATCH]``, and cannot
    fail a run. It reports everything a judged finding reports — the number is
    the point — so promoting one to a verdict is a threshold, never a new
    measurement.
    """

    name: str
    ok: bool
    measured: str
    detail: tuple[str, ...] = ()
    watched: bool = False

    def lines(self) -> tuple[str, ...]:
        tag = "WATCH" if self.watched else ("PASS" if self.ok else "FAIL")
        head = f"[{tag}] {self.name}: {self.measured}"
        return (head, *(f"         {line}" for line in self.detail))


def validate_corpus(
    records: Iterable[EffectRecord],
    thresholds: Thresholds | None = None,
    sidecars=None,
) -> list[Finding]:
    """Every invariant, measured over one pass of ``records``.

    ``sidecars`` is a :class:`~effects.infrastructure.sidecar_io.SidecarCache`
    (anything with its ``get(script_file)``) and is what makes the keyword-join
    check possible at all. ``None`` reports that check as *unchecked* rather
    than as holding: a converted tree that is not on this machine is not
    evidence that the corpus joins.
    """
    tally = _Tally()
    for record in records:
        tally.add(record)
    return tally.findings(thresholds or Thresholds(), sidecars)


def read_window(directory: Path, limit: int = 0) -> Iterator[EffectRecord]:
    """The first ``limit`` records under ``directory``; 0 reads all of them.

    Imported lazily, like the rest of the application layer's infrastructure
    reach, so the domain tests do not pay for the reader.
    """
    from effects.infrastructure.record_io import iter_shards, read_shard

    seen = 0
    for shard in iter_shards(Path(directory)):
        for record in read_shard(shard):
            yield record
            seen += 1
            if limit and seen >= limit:
                return


# ── the pass ────────────────────────────────────────────────────────────


def _kind_label(record: EffectRecord) -> str:
    """``resolution/activation``, ``playability/decision``, or a bare kind.

    The discriminated kinds are counted apart because their collectors are
    different code: an activation half and an effect half share a ``kind`` and
    nothing else, so a defect in one would be diluted by the other.
    """
    discriminator = record.moment or record.subkind
    return (
        f"{record.kind.value}/{discriminator.value}"
        if discriminator else record.kind.value
    )


def _blake(data) -> bytes:
    return hashlib.blake2b(
        json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        digest_size=16,
    ).digest()


def _digest(record: EffectRecord) -> bytes:
    """A record's content, identity stripped, as 16 bytes.

    Rendered through the writer rather than hashed field by field, so a field
    added later is covered without being added here — the same reason
    ``field_coverage`` walks the dataclasses.
    """
    from effects.infrastructure.record_io import record_to_dict

    data = record_to_dict(record)
    for name in _IDENTITY_FIELDS:
        data.pop(name, None)
    return _blake(data)


def _own_events(record: EffectRecord) -> tuple[Event, ...]:
    """The events a record lists as its own outcomes.

    Resolution and combat payloads only. A ``trigger`` carries the event it
    evaluated rather than one it caused, and a ``rewrite`` carries the same
    event on both sides whenever the replacement edited nothing — counting
    either as a repeat would report a duplicate on every healthy record of
    those kinds. That stays true now that ``outgoing`` may be null: the old
    shards it was written for are still in the corpus.
    """
    payload = record.payload
    if isinstance(payload, (ResolutionPayload, CombatPayload)):
        return payload.events
    return ()


def _all_events(record: EffectRecord) -> tuple[Event, ...]:
    """Every event anywhere in a record, for the per-type field measurements.

    Wider than :func:`_own_events`, because "does a ``zone_change`` say where
    the card came from" is a question about how the collector fills that type's
    params, and it fills them the same way for every kind that carries one.
    """
    payload = record.payload
    if isinstance(payload, (ResolutionPayload, CombatPayload)):
        return payload.events
    if isinstance(payload, TriggerPayload):
        return (payload.event,)
    if isinstance(payload, RewritePayload):
        # `outgoing` is null when the replacement rewrote nothing in place,
        # which is most of them; there is no second event to measure there.
        return (
            (payload.incoming, payload.outgoing)
            if payload.outgoing else (payload.incoming,)
        )
    return ()


def event_type_coverage(records: Iterable[EffectRecord]) -> Finding:
    """Which of the declared event types a corpus actually contains.

    Watched, not judged: a short window legitimately misses rare types — the
    remembered-result and choice families depend on which decks were drawn,
    so their absence in one run is not a verdict on the collector — and a
    floor nobody has calibrated would fail every smoke run. What this must
    not do is stay silent: 29 of the vocabulary's types were unreachable for
    the life of the project and no report ever said so before this check
    existed.

    This measures *fires*, a narrower and strictly harder question than the
    static guard in ``test_event_schema_completeness.py`` answers. That guard
    scans the connector's Java source for a reference to each type's
    constant, so it catches a type nothing points at; it cannot catch a type
    referenced from dead code that never actually runs, which is exactly what
    ``damage_prevented`` and ``spell_copied`` were — both referenced in
    ``PatchedCollectors.java`` while firing zero times across a 10.1M-record
    corpus. A declared type this check never saw is one of those two gaps or a
    type genuinely too rare for the window; the guard closes the first gap,
    this closes the second, and telling one absence from the other is for
    whoever reads the report.

    A standalone one-pass entry point over ``records`` so a caller wanting
    only this number does not pay for every other invariant's bookkeeping.
    :class:`_Tally` accumulates the identical set during its own pass and
    reports it through :meth:`_Tally.findings`, and both routes end at
    :func:`_event_type_coverage_finding` so "declared" and "seen" are never
    computed two different ways.
    """
    seen: set[str] = set()
    for record in records:
        for event in _all_events(record):
            seen.add(event.type.value)
    return _event_type_coverage_finding(seen)


def _event_type_coverage_finding(seen: set[str]) -> Finding:
    declared = {member.value for member in EventType}
    unseen = tuple(sorted(declared - seen))
    return Finding(
        name="declared event types are observed in the window",
        ok=True,
        watched=True,
        measured=(
            f"{len(seen)}/{len(declared)} declared event types appeared at "
            "least once"
        ),
        detail=unseen,
    )


def _provenance_keys(record: EffectRecord) -> Iterator[ProvenanceKey]:
    """Every printed-line key a record names, except emblems.

    Emblems are excluded deliberately rather than by oversight: an emblem's
    traits are built by the engine and key to a script path that exists in no
    converted tree, so joining them would report a mismatch on every healthy
    corpus that happens to resolve a planeswalker ultimate.
    """
    if record.ability:
        yield from record.ability
    for entity in record.state.entities:
        yield from entity.printed
        yield from entity.granted_attached
        yield from entity.granted_temporary.abilities
    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload):
        for candidate in payload.candidates:
            yield from candidate.ability
            yield from candidate.responsible_static
    elif isinstance(
        payload, (PlayabilityAttackersPayload, PlayabilityBlockersPayload)
    ):
        for forbidden in payload.forbidden:
            yield from forbidden.responsible_static


def _is_deferred_mana_record(record: EffectRecord) -> bool:
    """A resolution half the mana reservoir held back until the game ended.

    Mana activations are reservoir-sampled (Algorithm R) and flushed at
    ``close()``, because which one survives is not known until the game is
    over. The snapshot is taken when the mana was *made*, so a flushed record
    lands after records from every later turn and reads as a backward turn jump
    — on 168 of 177 games in a healthy smoke corpus. Exempting it is what keeps
    the game-boundary check worth reading; not exempting it taught an operator
    to ignore the whole report.

    Recognised by the payload rather than by the acting card. The reservoir
    holds mana abilities and a basic land is merely the commonest one, so a
    resolution half whose every event is ``mana_produced`` is the case, whoever
    printed the line. Narrow on purpose: a record that also does something else
    is not one of these, and an activation half is not exempt at all — nothing
    defers those, so a backward jump on one is still a game boundary.
    """
    payload = record.payload
    return (
        record.moment is Moment.RESOLUTION
        and isinstance(payload, ResolutionPayload)
        and bool(payload.events)
        and all(event.type is EventType.MANA_PRODUCED for event in payload.events)
    )


#: :func:`_join_result` for a key whose source tree this run was not given —
#: a stage-four corpus names ``variant-scripts`` keys, and a validate run
#: without ``--variant-scripts`` has nothing to join them against. Unchecked,
#: never a mismatch: absence of a tree is not evidence about the corpus.
_UNCHECKED = object()

#: :func:`_join_result` for a key outside every ``(face, trait_kind)`` range
#: its sidecar declares. Not a mismatch — Forge attaches such traits to a live
#: card — but not a join either, and reported on its own so an operator sees
#: how much of the window resolves that way.
_RUNTIME_ONLY = object()


def _join_result(sidecars, key: ProvenanceKey):
    """``None`` where ``key`` joins, a sentinel, or why it does not.

    ``resolution_of`` rather than ``row_for``, because ``row_for`` answers
    ``None`` to a line the sidecar dropped and to a key outside every range it
    declared alike, and only the first of those is a join. Sorting the rest is
    this function's whole job: a tree this run was not given is something it
    cannot say anything about, and a key inside a declared range that names no
    line is the one an operator has to act on.
    """
    try:
        sidecars.path_for(key.script_file)
    except UnconfiguredTree:
        return _UNCHECKED
    try:
        resolution = sidecars.resolution_of(key)
    except KeyError as exc:
        return f"{key.script_file} {key.trait_kind}[{key.index_within_kind}]: {exc}"
    if resolution is KeyResolution.UNCONVERTED:
        return f"{key.script_file}: no sidecar beside the converted card"
    if resolution is KeyResolution.RUNTIME_ONLY:
        return _RUNTIME_ONLY
    return None


@dataclass(slots=True)
class _Game:
    """What one ``game_id`` has shown so far."""

    max_turn: int = 0
    backward_jump: int = 0
    names: set[str] = field(default_factory=set)
    digests: set[bytes] = field(default_factory=set)


class _Tally:
    """Running counts for every invariant, filled by one pass."""

    def __init__(self) -> None:
        self.total = 0
        #: Every record id seen, and the turn its snapshot claims. One dict
        #: rather than a set plus a parallel map: the id membership test and
        #: the mirror comparison want the same keys, and the window is held in
        #: memory precisely once either way.
        self.record_turns: dict[str, int] = {}
        self.duplicate_ids = 0
        self.duplicate_id_examples: list[str] = []
        self.games: dict[str, _Game] = defaultdict(_Game)
        self.by_kind: Counter[str] = Counter()
        self.duplicates_by_kind: Counter[str] = Counter()
        self.tiers_by_kind: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
        self.links: dict[str, list[str]] = defaultdict(list)
        self.outcomes: Counter[str] = Counter()
        self.activations = 0
        self.cost_fields: Counter[str] = Counter()
        self.costs_populated = 0
        #: label -> [records, ``ability`` present, ``ability`` non-empty]
        self.acting: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        self.unresolved: Counter[str] = Counter()
        #: Records whose ``ability`` is empty and which say nothing about why.
        self.silent_empty_ability = 0
        self.silent_empty_examples: list[str] = []
        self.modes: Counter[str] = Counter()
        self.mode_examples: dict[str, str] = {}
        self.mana_flush_exempt = 0
        #: Every distinct printed-line key the window named, joined once.
        self.keys: set[ProvenanceKey] = set()
        #: Every distinct event type the window named, anywhere in a record —
        #: what event_type_coverage measures against the full vocabulary.
        self.event_types_seen: set[str] = set()
        self.events_seen = 0
        self.duplicate_events = 0
        self.duplicate_events_by_type: Counter[str] = Counter()
        self.zone_changes = 0
        self.zone_changes_with_from = 0
        self.zone_change_to: Counter[str] = Counter()
        self.resolution_events = 0
        self.attribution: Counter[str] = Counter()
        self.triggers: Counter[bool] = Counter()
        #: evaluated trigger mode -> [fired, evaluated]
        self.trigger_modes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.probe_forks = 0
        self.probed_keywords: Counter[str] = Counter()
        self.interventions = 0
        #: (fork record id, the record it mirrors, the fork's own turn).
        #: Compared after the pass, not during it, because a shard may carry
        #: the fork before the record it mirrors.
        self.mirrors: list[tuple[str, str, int]] = []
        self.fork_events = 0
        self.fork_attribution: Counter[str] = Counter()
        #: event type -> [carries a cause, seen]
        self.cause_bearing: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.rewrites = 0
        #: result value -> [carries an outgoing, names a replaced_by, seen].
        #: Keyed by the wire spelling, plus ``absent`` for a record written
        #: before the hook passed one — the three-way split the reader keeps.
        self.rewrite_results: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        #: Rewrites whose ``outgoing`` is byte-identical to their ``incoming``.
        #: Harmless on a pre-contract shard and a defect on a record that
        #: carries a ``result``, so the two are counted apart.
        self.rewrite_identity = 0
        self.rewrite_identity_examples: list[str] = []
        #: ``prevented``/``skipped`` records carrying something they cannot
        #: have: those outcomes return above the ability call.
        self.rewrite_impossible = 0
        self.rewrite_impossible_examples: list[str] = []

    def add(self, record: EffectRecord) -> None:
        self.total += 1
        label = _kind_label(record)
        self.by_kind[label] += 1
        self.modes[record.mode.value] += 1
        self.mode_examples.setdefault(record.mode.value, record.record_id)

        turn = record.state.global_.turn
        if record.record_id in self.record_turns:
            self.duplicate_ids += 1
            if len(self.duplicate_id_examples) < _EXAMPLES:
                self.duplicate_id_examples.append(record.record_id)
        else:
            self.record_turns[record.record_id] = turn
        if record.mirror_of is not None:
            self.mirrors.append((record.record_id, record.mirror_of, turn))

        game = self.games[record.game_id]
        game.names.update(entity.name for entity in record.state.entities)
        if _is_deferred_mana_record(record):
            # Its snapshot is older than the game it was written into, in both
            # directions: it must neither report a jump nor raise the high-water
            # mark a later record is measured against.
            self.mana_flush_exempt += 1
        else:
            if turn < game.max_turn:
                game.backward_jump = max(game.backward_jump, game.max_turn - turn)
            game.max_turn = max(game.max_turn, turn)

        digest = _digest(record)
        if digest in game.digests:
            self.duplicates_by_kind[label] += 1
        else:
            game.digests.add(digest)

        self.tiers_by_kind[label][
            tuple(sorted(int(tier) for tier in record.state.tiers))
        ] += 1

        if record.link_id is not None:
            self.links[record.link_id].append(
                record.moment.value if record.moment else "?"
            )

        if record.kind not in KINDS_WITHOUT_ACTING_ABILITY:
            counts = self.acting[label]
            counts[0] += 1
            counts[1] += record.ability is not None
            counts[2] += bool(record.ability)
            if record.ability_unresolved:
                self.unresolved[record.ability_unresolved] += 1
            elif record.ability is not None and not record.ability:
                self.silent_empty_ability += 1
                if len(self.silent_empty_examples) < _EXAMPLES:
                    self.silent_empty_examples.append(f"{record.record_id} ({label})")

        self.keys.update(_provenance_keys(record))
        self._add_events(record)

        payload = record.payload
        if isinstance(payload, ActivationPayload):
            self.activations += 1
            self.outcomes[payload.outcome.value] += 1
            self._add_costs(payload)
        elif isinstance(payload, TriggerPayload):
            self.triggers[payload.fired] += 1
            # The engine's own name for the hook, where it wrote one: the
            # vocabulary is a closed set of outcomes and Forge has some two
            # hundred trigger modes, so this is the only per-mode axis there is.
            mode = payload.event.params.get("mode") or payload.event.type.value
            counts = self.trigger_modes[str(mode)]
            counts[0] += payload.fired
            counts[1] += 1
        elif isinstance(payload, RewritePayload):
            self._add_rewrite(record, payload)
        elif isinstance(payload, CombatPayload) and record.is_probe:
            self.probe_forks += 1
            if payload.probed_keyword:
                self.probed_keywords[payload.probed_keyword] += 1
        if record.interventional:
            self.interventions += 1

    def _add_events(self, record: EffectRecord) -> None:
        seen: set[bytes] = set()
        for event in _own_events(record):
            digest = _blake(event.as_dict())
            self.events_seen += 1
            if digest in seen:
                self.duplicate_events += 1
                self.duplicate_events_by_type[event.type.value] += 1
            else:
                seen.add(digest)
        for event in _all_events(record):
            self.event_types_seen.add(event.type.value)
            if event.type is EventType.ZONE_CHANGE:
                self.zone_changes += 1
                self.zone_changes_with_from += bool(event.params.get("from_zone"))
                self.zone_change_to[str(event.params.get("to_zone"))] += 1
            if event.type in CAUSE_BEARING_TYPES:
                counts = self.cause_bearing[event.type.value]
                counts[0] += bool(event.params.get("cause"))
                counts[1] += 1
            if record.fork:
                self.fork_events += 1
                self.fork_attribution[attribution_kind(event.attributed_to)] += 1
        if isinstance(record.payload, ResolutionPayload):
            for event in record.payload.events:
                self.resolution_events += 1
                self.attribution[attribution_kind(event.attributed_to)] += 1

    def _add_rewrite(self, record: EffectRecord, payload: RewritePayload) -> None:
        self.rewrites += 1
        counts = self.rewrite_results[
            payload.result.value if payload.result else "absent"
        ]
        counts[0] += payload.outgoing is not None
        counts[1] += bool(payload.replaced_by)
        counts[2] += 1
        if payload.result is None:
            # Pre-contract shard: an identity `outgoing` is what that writer
            # always produced, and counting it as a defect would fail every
            # window that reaches back past the change.
            return
        if payload.outgoing is not None and (
            payload.outgoing.as_dict() == payload.incoming.as_dict()
        ):
            self.rewrite_identity += 1
            if len(self.rewrite_identity_examples) < _EXAMPLES:
                self.rewrite_identity_examples.append(
                    f"{record.record_id} ({payload.result.value})"
                )
        if payload.result in REWRITE_RESULTS_WITHOUT_ABILITY and (
            payload.outgoing is not None or payload.replaced_by
        ):
            self.rewrite_impossible += 1
            if len(self.rewrite_impossible_examples) < _EXAMPLES:
                self.rewrite_impossible_examples.append(
                    f"{record.record_id} ({payload.result.value})"
                )

    def _add_costs(self, payload: ActivationPayload) -> None:
        costs = payload.costs
        paid = {
            "mana_by_color": bool(costs.mana_by_color),
            "tapped": bool(costs.tapped),
            "life": costs.life != 0,
            "sacrificed": bool(costs.sacrificed),
            "discarded": bool(costs.discarded),
            "exiled": bool(costs.exiled),
        }
        for name, filled in paid.items():
            self.cost_fields[name] += filled
        self.costs_populated += any(paid.values())

    # ── the report ──────────────────────────────────────────────────────

    def findings(self, limits: Thresholds, sidecars=None) -> list[Finding]:
        return [
            self._unique_record_ids(),
            self._one_game_per_game_id(limits),
            self._links_pair(limits),
            *self._acting_lines(limits),
            self._empty_ability_says_why(),
            *self._keys_join_their_sidecar(sidecars),
            self._collected_patched(),
            self._outcome_varies(),
            self._costs_carry_something(limits),
            self._uniform_tiers(),
            self._duplicates(limits),
            self._duplicate_events(limits),
            self._trigger_negatives(limits),
            self._rewrites_say_what_happened(),
            self._rewrites_do_not_copy_their_event(),
            self._rewrite_outcomes_agree_with_their_payload(),
            self._rewrites_name_the_substituted_ability(limits),
            self._zone_changes_say_where_from(limits),
            self._resolution_events_name_a_clause(limits),
            self._fork_events_name_a_clause(limits),
            self._cause_bearing_events_name_a_cause(limits),
            self._forks_share_their_mirrors_turn(limits),
            self._probe_forks_were_taken(),
            self._event_type_coverage(),
        ]

    def _unique_record_ids(self) -> Finding:
        rate = self.duplicate_ids / self.total if self.total else 0.0
        return Finding(
            name="record_id is unique across the run's shards",
            ok=self.duplicate_ids == 0,
            measured=(
                f"{self.duplicate_ids} repeats of {len(self.record_turns)} "
                f"distinct ids over {self.total} records ({rate:.1%})"
            ),
            detail=tuple(f"repeated: {rid}" for rid in self.duplicate_id_examples),
        )

    def _one_game_per_game_id(self, limits: Thresholds) -> Finding:
        jumped = {
            gid: game.backward_jump for gid, game in self.games.items()
            if game.backward_jump > limits.turn_jump_tolerance
        }
        crowded = {
            gid: len(game.names) for gid, game in self.games.items()
            if len(game.names) > limits.max_names_per_game
        }
        widest = max((len(g.names) for g in self.games.values()), default=0)
        detail = [
            f"{gid} steps back {jump} turns"
            for gid, jump in sorted(jumped.items())[:_EXAMPLES]
        ] + [
            f"{gid} shows {count} distinct card names"
            for gid, count in sorted(crowded.items())[:_EXAMPLES]
        ]
        return Finding(
            name="each game_id names one game",
            ok=not jumped and not crowded,
            measured=(
                f"{len(self.games)} game ids; {len(jumped)} span a backward "
                f"turn jump, {len(crowded)} exceed "
                f"{limits.max_names_per_game} distinct card names "
                f"(widest {widest}); {self.mana_flush_exempt} records exempted "
                "as deferred mana-reservoir flushes"
            ),
            detail=tuple(detail),
        )

    def _links_pair(self, limits: Thresholds) -> Finding:
        want = sorted((Moment.ACTIVATION.value, Moment.RESOLUTION.value))
        paired = sum(
            1 for halves in self.links.values() if sorted(halves) == want
        )
        total = len(self.links)
        unpaired = total - paired
        rate = unpaired / total if total else 0.0
        sizes = Counter(len(halves) for halves in self.links.values())
        return Finding(
            name="every link_id joins one activation to one resolution",
            ok=rate <= limits.max_unpaired_link_rate,
            measured=(
                f"{paired}/{total} link ids paired, {unpaired} unpaired "
                f"({rate:.1%}, limit {limits.max_unpaired_link_rate:.1%})"
            ),
            detail=tuple(
                f"{count} link ids carry {size} half(s)"
                for size, count in sorted(sizes.items())
            ),
        )

    def _acting_lines(self, limits: Thresholds) -> list[Finding]:
        """One finding per group of kinds whose records do name a line.

        Split into "the field is there at all" and "a key resolved", because
        the two failures need different fixes and look identical in a corpus: a
        collector that never calls ``.ability(...)`` writes ``null`` on every
        record of its kind, while a resolver that cannot reach a printed line
        writes an empty list. The first corpus had both, on different kinds.
        """
        groups = {
            "trigger and rewrite": ("trigger", "rewrite"),
            "resolution": ("resolution/activation", "resolution/resolution"),
            "continuous": ("continuous",),
        }
        out: list[Finding] = []
        for name, labels in groups.items():
            seen = sum(self.acting[label][0] for label in labels)
            present = sum(self.acting[label][1] for label in labels)
            keyed = sum(self.acting[label][2] for label in labels)
            title = f"{name} records name an acting line"
            if not seen:
                out.append(Finding(
                    name=title, ok=True,
                    measured="no records of this kind in the window",
                ))
                continue
            keyed_rate = keyed / seen
            out.append(Finding(
                name=title,
                ok=present == seen and keyed_rate >= limits.min_keyed_rate,
                measured=(
                    f"{present}/{seen} carry the field, {keyed}/{seen} resolve "
                    f"to a key ({keyed_rate:.1%}, floor "
                    f"{limits.min_keyed_rate:.1%})"
                ),
                detail=tuple(
                    f"unresolved: {reason} x {count}"
                    for reason, count in sorted(self.unresolved.items())
                ),
            ))
        return out

    def _empty_ability_says_why(self) -> Finding:
        """An empty ``ability`` that carries no reason is unreadable evidence.

        "The Monarch has no printed line in any tree" and "the resolver
        regressed" both write ``ability: []``, and nothing tells them apart
        after the fact — that ambiguity hid a broken resolver for a whole
        collection run. ``ability_unresolved`` is the field that separates
        them, so an empty ability without one says nothing at all.
        """
        empty = self.silent_empty_ability + sum(self.unresolved.values())
        return Finding(
            name="every empty ability says why it is empty",
            ok=self.silent_empty_ability == 0,
            measured=(
                f"{self.silent_empty_ability} of {empty} empty abilities carry "
                "no ability_unresolved reason"
            ),
            detail=tuple(
                f"silent: {example}" for example in self.silent_empty_examples
            ) or tuple(
                f"reason {reason}: {count}"
                for reason, count in sorted(self.unresolved.items())
            ),
        )

    def _keys_join_their_sidecar(self, sidecars) -> list[Finding]:
        """Every provenance key the window names resolves in its own sidecar.

        This is the check that would have caught the launch blocker: the
        converter and the collector numbered a card's keywords differently, so
        records joined *silently* to the wrong printed line. Every key resolved
        to some row — just not the row the card prints — and nothing downstream
        could notice, because a wrong row is indistinguishable from a right one
        until a human reads the text beside a record.

        Two findings over one join, because the two kinds of key fail for
        different reasons and only one of them has a calibrated threshold.
        ``keyword`` keys carry the verdict: they are the kind that disagreed,
        and the only kind whose ordinal is derived rather than positional.
        Every other kind — ``spell``, ``static``, ``trigger``, ``replacement``
        — is reported as its own watched number rather than folded into the
        keyword line's tail, because a mismatch there has a cause a keyword
        mismatch does not (a reconversion between collection and training, or
        a line the converter never emits at all, like an ``SVar``-borne static
        on an Effect card) and nobody has yet measured what its healthy value
        is. Watched *first*: the number is what turns it into a verdict later,
        and failing runs on an uncalibrated check is how an operator learns to
        skip the whole report.

        A third finding counts the keys that resolve outside every range their
        sidecar declares. Those are not mismatches and must not fail the run,
        but they are not joins either, and a sidecar that declares no keyword
        line at all puts *every* keyword key of that face there — the launch
        blocker wearing the one disguise this check would otherwise read as
        healthy. Tallied beside the unjoinable count, and named in the keyword
        finding as well, so the share is visible where the verdict is.
        """
        keyword_title = "keyword provenance keys join their sidecar"
        other_title = "non-keyword provenance keys join their sidecar"
        outside_title = "provenance keys outside every declared range"
        if sidecars is None:
            unchecked = Finding(
                name=keyword_title, ok=True, watched=True,
                measured=(
                    f"{len(self.keys)} distinct keys, none checked: no converted "
                    "tree was readable (pass --cards-folder)"
                ),
            )
            return [
                unchecked,
                replace(unchecked, name=other_title),
                replace(unchecked, name=outside_title),
            ]
        checked: Counter[str] = Counter()
        unjoinable: Counter[str] = Counter()
        runtime_only: Counter[str] = Counter()
        unchecked_by_tree: Counter[str] = Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        ordered = sorted(
            self.keys,
            key=lambda k: (k.script_file, k.trait_kind, k.index_within_kind),
        )
        for key in ordered:
            result = _join_result(sidecars, key)
            if result is _UNCHECKED:
                unchecked_by_tree[key.tree] += 1
                continue
            checked[key.trait_kind] += 1
            if result is _RUNTIME_ONLY:
                runtime_only[key.trait_kind] += 1
                continue
            if result is None:
                continue
            unjoinable[key.trait_kind] += 1
            bucket = examples["keyword" if key.trait_kind == "keyword" else "other"]
            if len(bucket) < _EXAMPLES:
                bucket.append(result)
        keywords = checked.get("keyword", 0)
        broken = unjoinable.get("keyword", 0)
        keyword_outside = runtime_only.get("keyword", 0)
        skipped = sum(unchecked_by_tree.values())
        total_checked = sum(checked.values())
        other_checked = total_checked - keywords
        other_broken = sum(unjoinable.values()) - broken
        other_rate = other_broken / other_checked if other_checked else 0.0
        outside = sum(runtime_only.values())
        outside_rate = outside / total_checked if total_checked else 0.0
        trees = tuple(
            f"{count} keys in {tree}, a tree this run was not given"
            for tree, count in sorted(unchecked_by_tree.items())
        )
        return [
            Finding(
                name=keyword_title,
                ok=broken == 0,
                measured=(
                    f"{broken}/{keywords} keyword keys fail to join, "
                    f"{keyword_outside} outside every declared range, "
                    f"{skipped} unchecked"
                ),
                detail=trees + tuple(examples["keyword"]),
            ),
            Finding(
                name=other_title,
                ok=True,
                watched=True,
                measured=(
                    f"{other_broken}/{other_checked} keys of every other trait "
                    f"kind fail to join ({other_rate:.1%}, watched, no ceiling)"
                ),
                detail=tuple(
                    f"{kind}: {unjoinable[kind]}/{count} unjoinable"
                    for kind, count in sorted(checked.items())
                    if kind != "keyword"
                ) + tuple(examples["other"]),
            ),
            Finding(
                name=outside_title,
                ok=True,
                watched=True,
                measured=(
                    f"{outside}/{total_checked} keys resolve outside every "
                    f"(face, trait_kind) range their sidecar declares "
                    f"({outside_rate:.1%}, watched, no ceiling)"
                ),
                detail=tuple(
                    f"{kind}: {runtime_only[kind]}/{checked[kind]} outside "
                    "every declared range"
                    for kind in sorted(runtime_only)
                ),
            ),
        ]

    def _collected_patched(self) -> Finding:
        """A degraded run silently loses four record kinds.

        ``mode`` is detected per run by probing for the patch hooks, so one
        stock checkout in the pool writes a corpus that parses, validates and
        trains — with no mana records, no per-clause attribution, no trigger
        cause channel and no replacement hook. The mixed case is the dangerous
        one: the missing kinds read as scarcity.
        """
        shown = ", ".join(
            f"{name} {count}" for name, count in sorted(self.modes.items())
        ) or "none"
        degraded = self.total - self.modes.get(CollectionMode.PATCHED.value, 0)
        return Finding(
            name="every record was collected in patched mode",
            ok=degraded == 0,
            measured=f"{degraded}/{self.total} not patched ({shown})",
            detail=tuple(
                f"first {name} record: {self.mode_examples[name]}"
                for name in sorted(self.modes)
                if name != CollectionMode.PATCHED.value
            ),
        )

    def _outcome_varies(self) -> Finding:
        shown = ", ".join(
            f"{name} {count}" for name, count in sorted(self.outcomes.items())
        ) or "none"
        return Finding(
            name="outcome takes more than one value",
            # An empty window cannot answer this. Reported as holding rather
            # than as broken, the same way an absent kind is: a window with no
            # activation records has nothing to say about their outcomes, and a
            # check that fails on silence teaches an operator to ignore it.
            ok=len(self.outcomes) > 1 or self.activations == 0,
            measured=(
                f"{len(self.outcomes)} distinct over {self.activations} "
                f"activation records: {shown}"
            ),
            detail=() if len(self.outcomes) > 1 else (
                "a real game counters or fizzles something, so one value means "
                "the collector is writing a literal",
            ),
        )

    def _costs_carry_something(self, limits: Thresholds) -> Finding:
        shown = ", ".join(
            f"{name} {self.cost_fields[name]}" for name in _COST_FIELDS
        )
        dead = [
            name for name in _COST_FIELDS_ALWAYS_PAID
            if not self.cost_fields[name]
        ]
        enough = self.activations >= limits.min_cost_evidence
        # Nothing paid anywhere is a defect at any size; a single dead channel
        # only means something once the window is big enough for its absence to
        # be about the collector rather than about the pool.
        broken = self.activations and (
            not self.costs_populated or (enough and dead)
        )
        detail: tuple[str, ...] = ()
        if broken:
            detail = (
                f"never populated over {self.activations} activation records: "
                + ", ".join(dead or _COST_FIELDS),
                "reading a Cost object whose parts were never the ones paid "
                "looks exactly like this",
            )
        elif dead and not enough:
            detail = (
                f"{', '.join(dead)} unexercised, but {self.activations} "
                f"activation records is under the {limits.min_cost_evidence} "
                "this would need to mean anything",
            )
        return Finding(
            name="cost fields are not all empty",
            ok=not broken,
            measured=(
                f"{self.costs_populated}/{self.activations} activation records "
                f"paid something ({shown})"
            ),
            detail=detail,
        )

    def _uniform_tiers(self) -> Finding:
        """Snapshot depth is a run-level property, so one vector or none.

        A depth chosen per collector makes ``state.tiers`` a proxy for how the
        record was collected — in the first corpus tier 4 appeared on the
        interventional records and nowhere else, which is a perfect predictor
        of a field the schema forbids the model to see.
        """
        seen: Counter[tuple[int, ...]] = Counter()
        for counts in self.tiers_by_kind.values():
            seen.update(counts)
        detail = [
            f"{label}: " + ", ".join(
                f"{list(tiers)} x {count}"
                for tiers, count in sorted(counts.items())
            )
            for label, counts in sorted(self.tiers_by_kind.items())
        ]
        return Finding(
            name="snapshot tier depth is uniform across kinds",
            ok=len(seen) <= 1,
            measured=(
                f"{len(seen)} distinct tier vectors over "
                f"{len(self.tiers_by_kind)} kind groups"
            ),
            detail=tuple(detail) if len(seen) > 1 else (),
        )

    def _duplicates(self, limits: Thresholds) -> Finding:
        worst = 0.0
        detail = []
        for label, count in sorted(self.by_kind.items()):
            duplicates = self.duplicates_by_kind[label]
            rate = duplicates / count if count else 0.0
            worst = max(worst, rate)
            detail.append(f"{label}: {duplicates}/{count} ({rate:.1%})")
        total_dupes = sum(self.duplicates_by_kind.values())
        overall = total_dupes / self.total if self.total else 0.0
        return Finding(
            name="exact duplicates within a game stay rare, per kind",
            ok=worst <= limits.max_duplicate_rate,
            measured=(
                f"{total_dupes}/{self.total} overall ({overall:.1%}); worst "
                f"kind {worst:.1%}, limit {limits.max_duplicate_rate:.1%}"
            ),
            detail=tuple(detail),
        )

    def _duplicate_events(self, limits: Thresholds) -> Finding:
        """The duplication the record-level check cannot see.

        Two byte-identical events inside one record are one outcome written
        twice — same subjects, same params, same clause — and the record around
        them is unique, so the per-kind duplicate rate reads 0.00% while 13.68%
        of the corpus's events are repeats. A model trained on that learns that
        abilities deal their damage twice.
        """
        rate = self.duplicate_events / self.events_seen if self.events_seen else 0.0
        return Finding(
            name="no record repeats an event inside itself",
            ok=rate <= limits.max_duplicate_event_rate,
            measured=(
                f"{self.duplicate_events}/{self.events_seen} events repeat "
                f"another in the same record ({rate:.2%}, limit "
                f"{limits.max_duplicate_event_rate:.2%})"
            ),
            detail=tuple(
                f"{event_type}: {count}"
                for event_type, count in
                self.duplicate_events_by_type.most_common(_EXAMPLES)
            ),
        )

    def _trigger_negatives(self, limits: Thresholds) -> Finding:
        """Fired against not-fired, aggregate and per evaluated mode.

        Per mode as well, because the aggregate hides the failure this exists to
        catch: a sampler that draws no negatives for one trigger mode is
        invisible behind every other mode's balance, and the model then learns
        that mode's base rate instead of its condition.
        """
        fired = self.triggers[True]
        total = fired + self.triggers[False]
        share = fired / total if total else 0.0
        skewed = sorted(
            (
                (counts[0] / counts[1], mode, counts)
                for mode, counts in self.trigger_modes.items()
            ),
            reverse=True,
        )
        return Finding(
            name="trigger records draw negatives against positives",
            ok=not total or share <= limits.max_trigger_fired_share,
            measured=(
                f"{fired}:{total - fired} fired:not-fired over {total} trigger "
                f"records ({share:.1%} fired, ceiling "
                f"{limits.max_trigger_fired_share:.1%})"
            ),
            detail=tuple(
                f"{mode}: {counts[0]}:{counts[1] - counts[0]} ({rate:.1%} fired)"
                for rate, mode, counts in skewed[:_EXAMPLES]
            ),
        )

    def _rewrites_without_a_result(self) -> int:
        """Rewrites written before the hook passed one.

        Read through ``.get``: ``rewrite_results`` is a ``defaultdict`` and
        indexing it here would invent an ``absent`` row on a healthy window,
        which every rewrite finding then prints.
        """
        counts = self.rewrite_results.get("absent")
        return counts[2] if counts else 0

    def _rewrite_shown(self) -> str:
        """The result histogram, for whichever rewrite finding is printing."""
        return ", ".join(
            f"{name} {counts[2]}"
            for name, counts in sorted(self.rewrite_results.items())
        ) or "none"

    def _rewrites_say_what_happened(self) -> Finding:
        """Which of the five replacement outcomes the handler reported.

        The whole ``rewrite`` channel hangs off this field. Without it the
        payload models an edit-the-event mechanism Forge does not have, and
        "enters tapped", "exile it instead" and "prevent that damage" all record
        as an event that came out as it went in — which is why 87% of the hook's
        calls were dropped as identity rewrites and the channel collected 34
        usable records in 1.88M against a 7% share of the trainer's mix.

        Judged at zero rather than watched: a rewrite record with no result is
        either a pre-contract shard in the window, which validate-corpus is not
        meant to be pointed at, or a hook that lost the argument. Both are worth
        stopping for, and the detail below says which.
        """
        absent = self._rewrites_without_a_result()
        named = self.rewrites - absent
        return Finding(
            name="rewrite records say which replacement result happened",
            ok=absent == 0,
            measured=(
                f"{named}/{self.rewrites} rewrite records carry a result "
                f"({self._rewrite_shown()})"
            ),
            detail=() if absent == 0 else (
                f"{absent} carry none: either the window reaches back to shards "
                "collected before the hook passed one, or the listener is "
                "reading the wrong argument slot",
            ),
        )

    def _rewrites_do_not_copy_their_event(self) -> Finding:
        """``outgoing`` written as a copy of ``incoming`` says nothing.

        A record whose two halves are byte-identical is what made this channel
        unreadable: it cannot be told from a replacement that genuinely changed
        one parameter back to its own value, and there is no such thing. Null
        says "not rewritten in place" honestly, and every outcome except an
        in-place parameter edit should be writing it.

        Counted only over records that carry a ``result``. A pre-contract writer
        had no null to write, so failing on its shards would report a defect
        that was fixed rather than one that is present.
        """
        judged = self.rewrites - self._rewrites_without_a_result()
        rate = self.rewrite_identity / judged if judged else 0.0
        return Finding(
            name="a rewrite's outgoing event is not a copy of its incoming one",
            ok=self.rewrite_identity == 0,
            measured=(
                f"{self.rewrite_identity}/{judged} result-carrying rewrites "
                f"repeat their incoming event as outgoing ({rate:.1%})"
            ),
            detail=tuple(self.rewrite_identity_examples),
        )

    def _rewrite_outcomes_agree_with_their_payload(self) -> Finding:
        """``prevented`` and ``skipped`` cannot carry a payload.

        Both are bare ``return`` statements above the
        ``playSpellAbilityNoStack`` call in ``executeReplacementInternal``: no
        ability ran, and nothing was written to the parameter map on the way
        out. A record of one of those carrying an ``outgoing`` or a
        ``replaced_by`` is therefore a listener reading a stale argument, not a
        rare game state — which is the failure a reflective ``InvocationHandler``
        makes silently, because nothing compiles against the signature.

        ``not_replaced`` is deliberately not judged here. Its prevention branch
        writes ``PreventedAmount`` into the map before returning, so it may
        legitimately carry an ``outgoing``.
        """
        ran = ", ".join(
            f"{name} {counts[0]}/{counts[2]} with outgoing, "
            f"{counts[1]}/{counts[2]} with replaced_by"
            for name, counts in sorted(self.rewrite_results.items())
        ) or "none"
        cannot = sorted(r.value for r in REWRITE_RESULTS_WITHOUT_ABILITY)
        return Finding(
            name="outgoing is null on the results that rewrote nothing",
            ok=self.rewrite_impossible == 0,
            measured=(
                f"{self.rewrite_impossible} {'/'.join(cannot)} records carry an "
                f"outgoing or a replaced_by, of {self.rewrites} rewrites ({ran})"
            ),
            detail=tuple(self.rewrite_impossible_examples),
        )

    def _rewrites_name_the_substituted_ability(self, limits: Thresholds) -> Finding:
        """Which ability ran instead, on the outcomes that ran one.

        Watched by default, and it has to stay watched until the healthy share
        is measured: ``Replaced`` and ``Updated`` are also reachable from a
        replacement scripted with ``ReplacementResult$``, which returns the
        outcome having run no ability at all, so the ceiling is genuinely below
        1 and nobody knows by how much. The number still matters — a resolver
        that keys nothing reads as zero here — which is exactly what a watched
        finding is for.
        """
        floor = limits.min_rewrite_replaced_by_rate
        seen = named = 0
        for result in REWRITE_RESULTS_RUNNING_AN_ABILITY:
            counts = self.rewrite_results.get(result.value)
            if counts:
                named += counts[1]
                seen += counts[2]
        rate = named / seen if seen else 0.0
        limit = "watched, no floor" if floor is None else f"floor {floor:.1%}"
        return Finding(
            name="rewrites name the ability that ran instead",
            ok=floor is None or not seen or rate >= floor,
            watched=floor is None,
            measured=(
                f"{named}/{seen} replaced/updated records name a replaced_by "
                f"key ({rate:.1%}, {limit})"
            ),
            detail=tuple(
                f"{name}: {counts[1]}/{counts[2]} name one"
                for name, counts in sorted(self.rewrite_results.items())
            ),
        )

    def _zone_changes_say_where_from(self, limits: Thresholds) -> Finding:
        """Where a card came from, and how often "the stack" is where it went.

        A ``zone_change`` without ``from_zone`` says a card arrived somewhere
        and not what left: nothing separates a graveyard recursion from a token
        entering. Watched rather than judged by default, because some arrivals
        genuinely have no origin — a card *made* rather than moved — so the
        healthy share is not yet known. The ``to_zone=stack`` share rides along
        because a channel that is mostly casts is reporting the stack rather
        than the board.
        """
        floor = limits.min_zone_change_from_zone_rate
        rate = (
            self.zone_changes_with_from / self.zone_changes
            if self.zone_changes else 0.0
        )
        to_stack = self.zone_change_to.get("stack", 0)
        stack_rate = to_stack / self.zone_changes if self.zone_changes else 0.0
        limit = "watched, no floor" if floor is None else f"floor {floor:.1%}"
        return Finding(
            name="zone_change events say where the card came from",
            ok=floor is None or not self.zone_changes or rate >= floor,
            watched=floor is None,
            measured=(
                f"{self.zone_changes_with_from}/{self.zone_changes} carry "
                f"from_zone ({rate:.1%}, {limit}); {to_stack} say "
                f"to_zone=stack ({stack_rate:.1%})"
            ),
            detail=tuple(
                f"to_zone={zone}: {count}"
                for zone, count in self.zone_change_to.most_common(_EXAMPLES)
            ),
        )

    def _resolution_events_name_a_clause(self, limits: Thresholds) -> Finding:
        """The tri-state attribution pointer, broken out by what it says.

        ``absent`` is the state that says nothing: a writer that predates the
        sentinels wrote ``null`` for "the root line acted" and for "the pointer
        did not land" alike, and the corpus is append-only, so the two can never
        be separated afterwards. The rate below is the share saying *something*
        — a sub-ability index, the root, or an explicit unresolved — which is
        the number that moves when the attribution channel is wired.
        """
        floor = limits.min_attributed_rate
        named = self.resolution_events - self.attribution.get("absent", 0)
        rate = named / self.resolution_events if self.resolution_events else 0.0
        limit = "watched, no floor" if floor is None else f"floor {floor:.1%}"
        return Finding(
            name="resolution events name what produced them",
            ok=floor is None or not self.resolution_events or rate >= floor,
            watched=floor is None,
            measured=(
                f"{named}/{self.resolution_events} resolution events name a "
                f"clause ({rate:.1%}, {limit})"
            ),
            detail=tuple(
                f"{state}: {count}"
                for state, count in sorted(self.attribution.items())
            ),
        )

    def _fork_events_name_a_clause(self, limits: Thresholds) -> Finding:
        """The attribution pointer on the records the fork collectors write.

        Broken out from :meth:`_resolution_events_name_a_clause` because the
        forks are written by different code, and the aggregate hid exactly
        that: the observed records reached 84.6% naming a clause while every
        one of the fork collectors' events said ``null``, and the two numbers
        averaged to something that looked like a channel merely warming up.
        A fork whose events name nothing cannot be compared clause by clause
        with the record it mirrors, which is the entire point of taking it.
        """
        floor = limits.min_fork_attributed_rate
        named = self.fork_events - self.fork_attribution.get("absent", 0)
        rate = named / self.fork_events if self.fork_events else 0.0
        limit = "watched, no floor" if floor is None else f"floor {floor:.1%}"
        return Finding(
            name="fork records' events name what produced them",
            ok=floor is None or not self.fork_events or rate >= floor,
            watched=floor is None,
            measured=(
                f"{named}/{self.fork_events} events on fork records name a "
                f"clause ({rate:.1%}, {limit})"
            ),
            detail=tuple(
                f"{state}: {count}"
                for state, count in sorted(self.fork_attribution.items())
            ),
        )

    def _cause_bearing_events_name_a_cause(self, limits: Thresholds) -> Finding:
        """Whether the events that declare a ``cause`` populate it.

        Measured over :data:`~effects.domain.event_schema.CAUSE_BEARING_TYPES`
        alone — the types whose own parameter row asks for one — because every
        type *may* carry a cause and most correctly never do, so a rate over
        the whole vocabulary would read near zero on a healthy corpus and mean
        nothing.

        It is also the identity channel the duplicate-event check depends on:
        two attackers dealing the same damage to the same player differ in
        nothing but their cause, so an empty ``cause`` turns two real outcomes
        into one outcome written twice, and the two checks disagree with each
        other rather than with the collector.
        """
        floor = limits.min_cause_rate
        with_cause = sum(counts[0] for counts in self.cause_bearing.values())
        seen = sum(counts[1] for counts in self.cause_bearing.values())
        rate = with_cause / seen if seen else 0.0
        limit = "watched, no floor" if floor is None else f"floor {floor:.1%}"
        return Finding(
            name="events that declare a cause name one",
            ok=floor is None or not seen or rate >= floor,
            watched=floor is None,
            measured=(
                f"{with_cause}/{seen} events of the {len(CAUSE_BEARING_TYPES)} "
                f"cause-bearing types carry a cause ({rate:.1%}, {limit})"
            ),
            detail=tuple(
                f"{event_type}: {counts[0]}/{counts[1]}"
                for event_type, counts in sorted(self.cause_bearing.items())
            ),
        )

    def _forks_share_their_mirrors_turn(self, limits: Thresholds) -> Finding:
        """A fork's snapshot is of the moment it forked, which is its mirror's.

        The probe re-runs one damage step from the state the real combat was
        in, so the two records describe one moment and must agree about which
        turn it is. When they do not, the fork's ``turn`` is unusable for
        time-ordering — and the failure surfaced somewhere much less
        actionable: as a *backward turn jump* in "each game_id names one
        game", which named the game and blamed a merge that had not happened.
        15 of 384 probe forks in the smoke corpus sat a turn behind their
        mirror, never ahead, and were the whole residue of that invariant.

        Judged at zero rather than watched, unlike the other new checks: there
        is no rate at which two records of one moment may disagree about it.
        Forks whose mirror fell outside the window are counted apart and
        judged not at all — ``--limit`` cuts the stream mid-game, and the
        missing half is the reader's doing, not the collector's.
        """
        disagreed: list[str] = []
        uncomparable = 0
        for record_id, mirror_of, turn in self.mirrors:
            mirror_turn = self.record_turns.get(mirror_of)
            if mirror_turn is None:
                uncomparable += 1
            elif mirror_turn != turn:
                disagreed.append(
                    f"{record_id} at turn {turn} mirrors {mirror_of} at turn "
                    f"{mirror_turn}"
                )
        comparable = len(self.mirrors) - uncomparable
        rate = len(disagreed) / comparable if comparable else 0.0
        return Finding(
            name="a fork's turn agrees with the record it mirrors",
            ok=rate <= limits.max_mirror_turn_disagreement_rate,
            measured=(
                f"{len(disagreed)}/{comparable} forks disagree with their "
                f"mirror about the turn ({rate:.1%}, limit "
                f"{limits.max_mirror_turn_disagreement_rate:.1%}); "
                f"{uncomparable} mirrors fell outside the window"
            ),
            detail=tuple(disagreed[:_EXAMPLES]),
        )

    def _probe_forks_were_taken(self) -> Finding:
        """Whether the probe path ran at all, and on which keywords.

        ``--probe-keywords`` defaults to empty and empty takes no fork, so the
        whole path can have unit tests and no runtime evidence whatever — which
        is what the smoke corpus showed: zero probe forks in 55,296 records.
        Watched rather than judged, because a run that deliberately names no
        keyword is not a broken run; the number is here so an operator who
        *meant* to probe finds out in minutes rather than at evaluation.
        """
        shown = ", ".join(
            f"{keyword} {count}"
            for keyword, count in self.probed_keywords.most_common()
        ) or "none"
        return Finding(
            name="probe forks were taken",
            ok=True,
            watched=True,
            measured=(
                f"{self.probe_forks} damage-step probe forks, "
                f"{self.interventions} interventional forks; keywords probed: "
                f"{shown}"
            ),
            detail=() if self.probe_forks else (
                "zero probes means --probe-keywords was empty; that flag, not "
                "--probes-per-game, is what turns the path on",
            ),
        )

    def _event_type_coverage(self) -> Finding:
        """The check-list registration of :func:`event_type_coverage`.

        Reads the set :meth:`add` already built while it walked every event
        for the zone-change, cause-bearing and fork-attribution checks, so
        registering this costs one more tuple in :meth:`findings` and no
        second pass over the window.
        """
        return _event_type_coverage_finding(self.event_types_seen)
