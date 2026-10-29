"""Invariants a shard directory must hold, measured rather than asserted.

Meant for the first few minutes of a collection pass: a collector that writes a
literal, reuses an id or varies its snapshot depth produces records that parse
and pass every contract test, so each invariant prints the number it measured,
pass or fail.

**Measuring and judging are separate.** A :class:`Finding` with no threshold is
watched: it prints ``[WATCH]`` and never fails a run, and passing a floor makes
it a verdict without changing what is measured. A number stays watched until
its healthy value is known, because an invariant that fails on every good run
teaches an operator to skip the whole report.

One streaming pass, holding every ``record_id``, one digest per record and every
distinct provenance key, which is why ``--limit`` exists.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Protocol

from effects.domain.canonical_digest import canonical_digest
from effects.domain.event_schema import (
    CAUSE_BEARING_TYPES,
    AttributionKind,
    Event,
    EventType,
    attribution_kind,
)
from effects.domain.provenance import KeyResolution, ProvenanceKey
from effects.domain.records import (
    KINDS_WITHOUT_ACTING_ABILITY,
    LEGALITY_SUBKINDS,
    PAYLOAD_TYPES,
    REWRITE_RESULTS_RUNNING_AN_ABILITY,
    REWRITE_RESULTS_WITHOUT_ABILITY,
    ActivationPayload,
    CollectionMode,
    CombatPayload,
    Costs,
    EffectRecord,
    Moment,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
    UnresolvedReason,
    key_sites,
)
from effects.infrastructure.sidecar_io import SidecarCache, UnconfiguredTree

#: Envelope fields that say when a record was written rather than what it says,
#: blanked before hashing; ``mode`` is constant within a run.
_IDENTITY_FIELDS = (
    "record_id", "run_id", "timestamp", "game_id",
    "link_id", "mirror_of", "variant_of", "mode",
)

#: The six cost channels, in the order the schema lists them.
_COST_FIELDS = tuple(spec.name for spec in fields(Costs))

#: The channels a big enough window must have exercised: every deck pays mana
#: and taps many times a game, while sacrifice, discard, exile and life costs
#: are genuinely rare in a sealed pool.
_COST_FIELDS_ALWAYS_PAID = ("mana_by_color", "tapped")

#: How many offending examples a finding's detail carries.
_EXAMPLES = 5

#: The ``rewrite_results`` row for a rewrite written before the hook passed a
#: ``result``.
_NO_RESULT = "absent"



@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where each measured rate stops being acceptable.

    A ``float | None`` field defaulting to ``None`` is watched, not judged. Each
    field is a ``validate-corpus`` flag, its ``help`` metadata the flag's help.
    """

    min_keyed_rate: float = field(default=0.95, metadata={"help": (
        "Share of records naming a line that must resolve it to a printed key "
        "(default: %(default)s)"
    )})
    max_duplicate_rate: float = field(default=0.02, metadata={"help": (
        "Share of a kind's records that may repeat an earlier record of the "
        "same kind in the same game (default: %(default)s)"
    )})
    #: A worker killed mid-game truncates its last block, leaving a few
    #: unpaired halves.
    max_unpaired_link_rate: float = field(default=0.02, metadata={"help": (
        "Share of link ids that may lack one activation and one resolution "
        "half (default: %(default)s)"
    )})
    #: Two forty-card sealed decks plus tokens sit far below this.
    max_names_per_game: int = field(default=120, metadata={"help": (
        "Distinct card names one game_id may show before it reads as more "
        "than one game (default: %(default)s)"
    )})
    min_cost_evidence: int = field(default=200, metadata={"help": (
        "Activation records needed before a cost channel reading zero means "
        "the collector rather than the pool (default: %(default)s)"
    )})
    #: A deferred half can legitimately lag by a turn; a game boundary jumps
    #: many.
    turn_jump_tolerance: int = field(default=1, metadata={"help": (
        "Turns a snapshot may sit below the highest already seen in its game "
        "before that reads as a game boundary; deferred mana-reservoir flushes "
        "are exempt (default: %(default)s)"
    )})
    max_duplicate_event_rate: float = field(default=0.02, metadata={"help": (
        "Share of events that may repeat another event in the same record, "
        "which --max-duplicate-rate's whole-record comparison cannot see "
        "(default: %(default)s)"
    )})
    #: Far above ~1:1 the sampler stopped drawing negatives, and positives
    #: alone teach the base rate.
    max_trigger_fired_share: float = field(default=0.65, metadata={"help": (
        "Share of trigger records that may report fired=true; the negative "
        "sampler targets ~1:1 (default: %(default)s)"
    )})
    min_zone_change_from_zone_rate: float | None = field(default=None, metadata={
        "help": (
            "Share of zone_change events that must carry from_zone; unset, it "
            "is reported as [WATCH] without failing the run"
        ),
    })
    min_attributed_rate: float | None = field(default=None, metadata={"help": (
        "Share of resolution events that must name a producing clause; unset, "
        "it is reported as [WATCH] without failing the run"
    )})
    min_fork_attributed_rate: float | None = field(default=None, metadata={"help": (
        "Share of events on fork records that must name a producing clause, "
        "apart from --min-attributed-rate; unset, it is reported as [WATCH]"
    )})
    #: Some hooks genuinely name no causing object, so the healthy share is
    #: below 1 by an unknown amount.
    min_cause_rate: float | None = field(default=None, metadata={"help": (
        "Share of the events whose schema row declares a cause that must "
        "populate it; unset, it is reported as [WATCH]"
    )})
    #: A replacement scripted with ``ReplacementResult$`` returns
    #: ``replaced``/``updated`` having run no ability, so the healthy share is
    #: below 1 by an unknown amount.
    min_rewrite_replaced_by_rate: float | None = field(default=None, metadata={
        "help": (
            "Share of the rewrite records whose result ran an ability that must "
            "name it; unset, it is reported as [WATCH]"
        ),
    })
    #: A fork is taken at the moment it mirrors, so no disagreement is healthy.
    max_mirror_turn_disagreement_rate: float = field(default=0.0, metadata={"help": (
        "Share of fork records whose turn may differ from the record they "
        "mirror, a stale snapshot (default: %(default)s)"
    )})


@dataclass(frozen=True, slots=True)
class Finding:
    """One invariant's result: what it measured, and whether that is allowed.
    A ``watched`` finding has no threshold and cannot fail a run."""

    name: str
    ok: bool
    measured: str
    detail: tuple[str, ...] = ()
    watched: bool = False

    def lines(self) -> tuple[str, ...]:
        tag = "WATCH" if self.watched else ("PASS" if self.ok else "FAIL")
        head = f"[{tag}] {self.name}: {self.measured}"
        return (head, *(f"         {line}" for line in self.detail))


class Findings(list[Finding]):
    """Every finding of one pass, in report order."""

    def broken(self) -> list[Finding]:
        return [finding for finding in self if not finding.ok]

    def summary(self) -> list[str]:
        """The closing lines: what broke, or that every judged invariant held."""
        broken = self.broken()
        watched = sum(1 for finding in self if finding.watched)
        judged = len(self) - watched
        if broken:
            return [
                f"{len(broken)} of {judged} judged invariants broken:",
                *(f"  {finding.name}" for finding in broken),
            ]
        return [f"all {judged} judged invariants hold ({watched} watched, not judged)"]


def validate_corpus(
    records: Iterable[EffectRecord],
    thresholds: Thresholds | None = None,
    sidecars: SidecarCache | None = None,
    presence: FieldPresence | None = None,
) -> Findings:
    """Every invariant, measured over one pass of ``records``.

    ``sidecars`` of ``None`` reports the key-join checks as unchecked rather
    than as holding. ``presence`` is the per-shard envelope tally
    :func:`read_window` fills from the raw lines (FR-033), since a parsed
    record cannot say whether a field was on the wire.
    """
    checks = _checks(sidecars)
    for record in records:
        for check in checks:
            check.add(record)
    limits = thresholds or Thresholds()
    findings = Findings(
        finding for check in checks for finding in check.findings(limits)
    )
    if presence is not None:
        findings.append(presence.finding())
    return findings


@dataclass(slots=True)
class _ShardPresence:
    """One shard's raw lines, and how many carry each gen-2 envelope field."""

    records: int = 0
    with_random_seat: int = 0
    #: The legality subkinds, which must carry ``what_if`` on a gen-2 shard.
    legality: int = 0
    legality_with_what_if: int = 0
    others_with_what_if: int = 0


@dataclass
class FieldPresence:
    """Which gen-2 envelope fields each shard's raw lines carry (FR-033).

    Per shard, because a shard is one collector build: one where only some
    records carry ``random_seat`` came from a collector stamping it on some
    paths only.
    """

    shards: dict[str, _ShardPresence] = field(default_factory=dict)

    def observe(self, shard: str, data: dict) -> None:
        counts = self.shards.setdefault(shard, _ShardPresence())
        counts.records += 1
        counts.with_random_seat += "random_seat" in data
        legality = (
            data.get("kind") == RecordKind.PLAYABILITY.value
            and data.get("subkind") in LEGALITY_SUBKINDS
        )
        if legality:
            counts.legality += 1
            counts.legality_with_what_if += "what_if" in data
        else:
            counts.others_with_what_if += "what_if" in data

    def finding(self) -> Finding:
        partial_seat, partial_what_if, stray = [], [], []
        for shard, counts in sorted(self.shards.items()):
            records, seats = counts.records, counts.with_random_seat
            legality, what_ifs = counts.legality, counts.legality_with_what_if
            if 0 < seats < records:
                partial_seat.append(f"{shard}: random_seat on {seats} of {records} records")
            # A gen-2 shard owes what_if on every legality record; a gen-1 one
            # on none, so only a partial set is wrong there.
            if what_ifs < legality and (seats or what_ifs):
                partial_what_if.append(
                    f"{shard}: what_if on {what_ifs} of {legality} legality records"
                )
            if counts.others_with_what_if:
                stray.append(
                    f"{shard}: what_if on {counts.others_with_what_if} "
                    "non-legality record(s)"
                )
        gen2 = sum(1 for counts in self.shards.values() if counts.with_random_seat)
        bad = partial_seat + partial_what_if + stray
        return Finding(
            name="gen-2 envelope fields are on every record of the kinds that carry them",
            ok=not bad,
            measured=(
                f"{len(self.shards)} shard(s), {gen2} carrying random_seat; "
                f"{len(partial_seat)} with random_seat on some records only, "
                f"{len(partial_what_if)} with what_if on some legality records "
                f"only, {len(stray)} with what_if on another kind"
            ),
            detail=tuple(bad[:_EXAMPLES]),
        )


def _joined(halves: list[tuple[str, tuple[ProvenanceKey, ...]]]) -> bool:
    """Whether one link's halves form a valid resolution (FR-029d): one cost
    half and one effect half, or several effect halves all acting through mode
    keys (``option`` set) of one charm root."""
    costs = [keys for moment, keys in halves if moment == Moment.ACTIVATION.value]
    effects = [keys for moment, keys in halves if moment == Moment.RESOLUTION.value]
    if len(costs) != 1 or not effects or len(costs) + len(effects) != len(halves):
        return False
    if len(effects) == 1:
        return True
    roots = set()
    for keys in effects:
        if not keys or any(key.option is None for key in keys):
            return False
        roots |= {key.root for key in keys}
    return len(roots) == 1


def read_window(
    directory: Path, limit: int = 0, presence: FieldPresence | None = None,
) -> Iterator[EffectRecord]:
    """The first ``limit`` records under ``directory``; 0 reads all of them.
    ``presence``, when given, is filled from each raw line before it is parsed.
    """
    from effects.infrastructure.record_io import (
        iter_record_dicts,
        iter_shards,
        record_from_dict,
    )

    root = Path(directory)
    seen = 0
    for shard in iter_shards(root):
        name = shard.relative_to(root).as_posix() if shard.is_relative_to(root) else str(shard)
        for data in iter_record_dicts(shard):
            if presence is not None:
                presence.observe(name, data)
            yield record_from_dict(data)
            seen += 1
            if limit and seen >= limit:
                return


def _kind_label(record: EffectRecord) -> str:
    """``resolution/activation``, ``playability/decision``, or a bare kind:
    discriminated kinds are counted apart because their collectors are
    different code."""
    return _label(record.kind, record.discriminator)


def _label(kind: RecordKind, discriminator: Moment | PlayabilitySubkind | None) -> str:
    return f"{kind.value}/{discriminator.value}" if discriminator else kind.value


def _labels_of(*kinds: RecordKind) -> tuple[str, ...]:
    """Every kind label a record of one of ``kinds`` can carry."""
    return tuple(
        _label(kind, discriminator)
        for kind, discriminator in PAYLOAD_TYPES if kind in kinds
    )


def _digest(record: EffectRecord) -> bytes:
    """A record's content, identity stripped, as 16 bytes. Rendered through the
    writer, so a field added later is covered without being added here."""
    from effects.infrastructure.record_io import record_to_dict

    data = record_to_dict(record)
    for name in _IDENTITY_FIELDS:
        data.pop(name, None)
    return canonical_digest(data)


def _is_deferred_mana_record(record: EffectRecord) -> bool:
    """A resolution half the mana reservoir held back until the game ended.

    Its snapshot is from when the mana was made, so it lands after later turns
    and would read as a backward turn jump. Recognised by the payload (every
    event ``mana_produced``) rather than the acting card; nothing defers an
    activation half, so a jump on one is still a game boundary.
    """
    payload = record.payload
    return (
        record.moment is Moment.RESOLUTION
        and isinstance(payload, ResolutionPayload)
        and bool(payload.events)
        and all(event.type is EventType.MANA_PRODUCED for event in payload.events)
    )


#: :func:`_join_result` for a key in a tree this run was not given (such as
#: ``variant-scripts`` without ``--variant-scripts``): unchecked, never a
#: mismatch.
_UNCHECKED = object()

#: :func:`_join_result` for a key outside every ``(face, trait_kind)`` range
#: its sidecar declares: a trait Forge attached at runtime, neither a join nor
#: a mismatch.
_RUNTIME_ONLY = object()


def _join_result(sidecars: SidecarCache, key: ProvenanceKey):
    """``None`` where ``key`` joins, a sentinel, or why it does not.

    ``resolution_of`` rather than ``row_for``, which answers ``None`` both to a
    dropped line (a join) and to a key outside every range (not one).
    """
    try:
        sidecars.path_for(key.script_file)
    except UnconfiguredTree:
        return _UNCHECKED
    try:
        resolution = sidecars.resolution_of(key)
    except KeyError as exc:
        return f"{key.label}: {exc}"
    if resolution is KeyResolution.UNCONVERTED:
        return f"{key.script_file}: no sidecar beside the converted card"
    if resolution is KeyResolution.RUNTIME_ONLY:
        return _RUNTIME_ONLY
    return None


def _rate(part: int, whole: int) -> float:
    """``part`` as a share of ``whole``, and 0.0 of an empty one."""
    return part / whole if whole else 0.0


@dataclass(slots=True)
class _Share:
    """How many of the cases ``seen`` were a hit."""

    hit: int = 0
    seen: int = 0

    def note(self, hit: bool) -> None:
        self.hit += hit
        self.seen += 1

    @property
    def rate(self) -> float:
        return _rate(self.hit, self.seen)


def _histogram(counts: Iterable[tuple[str, int]]) -> str:
    """``name count, name count``, in the order given, or ``none`` when empty."""
    return ", ".join(f"{name} {count}" for name, count in counts) or "none"


def _keep_example(examples: list[str], example: str) -> None:
    """Record ``example`` unless ``examples`` already holds :data:`_EXAMPLES`."""
    if len(examples) < _EXAMPLES:
        examples.append(example)


def _floor_finding(
    name: str,
    floor: float | None,
    seen: int,
    rate: float,
    measured: Callable[[str], str],
    detail: Iterable[str],
) -> Finding:
    """A rate judged against a floor, watched when the floor is ``None``.

    ``measured`` is handed the floor's label to place in its own sentence. An
    empty window holds whatever the floor.
    """
    limit = "watched, no floor" if floor is None else f"floor {floor:.1%}"
    return Finding(
        name=name,
        ok=floor is None or not seen or rate >= floor,
        watched=floor is None,
        measured=measured(limit),
        detail=tuple(detail),
    )


class _Check(Protocol):
    """One invariant: running counts filled by the pass, then its findings
    (several where they read the same counts)."""

    def add(self, record: EffectRecord) -> None: ...

    def findings(self, limits: Thresholds) -> Iterator[Finding]: ...


class _RecordIds:
    """Every record id seen, and the turn its snapshot claims, which
    :class:`_MirrorTurns` reads."""

    def __init__(self) -> None:
        self.total = 0
        self.turns: dict[str, int] = {}
        self.duplicates = 0
        self.examples: list[str] = []

    def add(self, record: EffectRecord) -> None:
        self.total += 1
        if record.record_id in self.turns:
            self.duplicates += 1
            _keep_example(self.examples, record.record_id)
        else:
            self.turns[record.record_id] = record.state.global_.turn

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        rate = _rate(self.duplicates, self.total)
        yield Finding(
            name="record_id is unique across the run's shards",
            ok=self.duplicates == 0,
            measured=(
                f"{self.duplicates} repeats of {len(self.turns)} "
                f"distinct ids over {self.total} records ({rate:.1%})"
            ),
            detail=tuple(f"repeated: {rid}" for rid in self.examples),
        )


@dataclass(slots=True)
class _Game:
    """What one ``game_id`` has shown so far."""

    max_turn: int = 0
    backward_jump: int = 0
    names: set[str] = field(default_factory=set)


class _GameBoundaries:
    """Whether each ``game_id`` reads as one game: its turns and its cast."""

    def __init__(self) -> None:
        self.games: dict[str, _Game] = defaultdict(_Game)
        self.mana_flush_exempt = 0

    def add(self, record: EffectRecord) -> None:
        game = self.games[record.game_id]
        game.names.update(entity.name for entity in record.state.entities)
        if _is_deferred_mana_record(record):
            # Neither a jump nor a raise of the high-water mark later records
            # are measured against.
            self.mana_flush_exempt += 1
            return
        turn = record.state.global_.turn
        if turn < game.max_turn:
            game.backward_jump = max(game.backward_jump, game.max_turn - turn)
        game.max_turn = max(game.max_turn, turn)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
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
        yield Finding(
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


class _Links:
    """Every half carrying each ``link_id``, joined after the pass."""

    def __init__(self) -> None:
        #: link_id -> [(moment, acting keys)] for every half carrying it.
        self.links: dict[str, list[tuple[str, tuple[ProvenanceKey, ...]]]] = defaultdict(list)

    def add(self, record: EffectRecord) -> None:
        if record.link_id is not None:
            self.links[record.link_id].append((
                record.moment.value if record.moment else "?",
                tuple(record.ability or ()),
            ))

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        paired = sum(1 for halves in self.links.values() if _joined(halves))
        total = len(self.links)
        unpaired = total - paired
        rate = _rate(unpaired, total)
        sizes = Counter(len(halves) for halves in self.links.values())
        yield Finding(
            name=(
                "every link_id joins one activation to its resolution(s): one, "
                "or one per chosen mode of a charm"
            ),
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


@dataclass(slots=True)
class _ActingCounts:
    """One kind label's records, against how many name their acting line."""

    records: int = 0
    #: ``ability`` is on the record at all, even as an empty list.
    present: int = 0
    #: ``ability`` resolved to at least one key.
    keyed: int = 0


class _ActingLines:
    """Whether the kinds that act through a line name it, and why when not."""

    #: Finding name -> the kind labels it groups.
    GROUPS = {
        "trigger and rewrite": _labels_of(RecordKind.TRIGGER, RecordKind.REWRITE),
        "resolution": _labels_of(RecordKind.RESOLUTION),
        "continuous": _labels_of(RecordKind.CONTINUOUS),
    }

    def __init__(self) -> None:
        self.acting: dict[str, _ActingCounts] = defaultdict(_ActingCounts)
        self.unresolved: Counter[UnresolvedReason] = Counter()
        #: Records whose ``ability`` is empty and which say nothing about why.
        self.silent_empty = 0
        self.silent_empty_examples: list[str] = []

    def add(self, record: EffectRecord) -> None:
        if record.kind in KINDS_WITHOUT_ACTING_ABILITY:
            return
        label = _kind_label(record)
        counts = self.acting[label]
        counts.records += 1
        counts.present += record.ability is not None
        counts.keyed += bool(record.ability)
        if record.ability_unresolved:
            self.unresolved[record.ability_unresolved] += 1
        elif record.ability is not None and not record.ability:
            self.silent_empty += 1
            _keep_example(self.silent_empty_examples, f"{record.record_id} ({label})")

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        yield from self._acting_lines(limits)
        yield self._empty_ability_says_why()

    def _acting_lines(self, limits: Thresholds) -> Iterator[Finding]:
        """One finding per group of kinds that name a line, split into "the
        field is there" and "a key resolved": a collector that never sets
        ``ability`` writes ``null``, a resolver that cannot reach a line ``[]``.
        """
        for name, labels in self.GROUPS.items():
            groups = [self.acting.get(label, _ActingCounts()) for label in labels]
            seen = sum(counts.records for counts in groups)
            present = sum(counts.present for counts in groups)
            keyed = sum(counts.keyed for counts in groups)
            title = f"{name} records name an acting line"
            if not seen:
                yield Finding(
                    name=title, ok=True,
                    measured="no records of this kind in the window",
                )
                continue
            keyed_rate = keyed / seen
            yield Finding(
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
            )

    def _empty_ability_says_why(self) -> Finding:
        """An empty ``ability`` needs an ``ability_unresolved`` reason: "the
        Monarch has no printed line" and "the resolver regressed" both write
        ``ability: []``."""
        empty = self.silent_empty + sum(self.unresolved.values())
        return Finding(
            name="every empty ability says why it is empty",
            ok=self.silent_empty == 0,
            measured=(
                f"{self.silent_empty} of {empty} empty abilities carry "
                "no ability_unresolved reason"
            ),
            detail=tuple(
                f"silent: {example}" for example in self.silent_empty_examples
            ) or tuple(
                f"reason {reason}: {count}"
                for reason, count in sorted(self.unresolved.items())
            ),
        )


_KEYWORD_JOIN = "keyword provenance keys join their sidecar"
_OTHER_JOIN = "non-keyword provenance keys join their sidecar"
_OUTSIDE_JOIN = "provenance keys outside every declared range"


class _KeyJoins:
    """Every provenance key the window names resolves in its own sidecar.

    A key joined to the wrong printed line is invisible downstream. ``keyword``
    keys are judged: their ordinal is derived rather than positional, and a
    converter and collector numbering keywords differently is the known
    failure. Every other trait kind is watched, its healthy rate unmeasured. A
    third, watched finding counts keys outside every declared range: not a
    mismatch, but a sidecar declaring no keyword line puts every keyword key
    of its face there, so the keyword finding names that count too.
    """

    def __init__(self, sidecars: SidecarCache | None) -> None:
        self.sidecars = sidecars
        #: Every distinct printed-line key the window named, joined once.
        self.keys: set[ProvenanceKey] = set()

    def add(self, record: EffectRecord) -> None:
        self.keys.update(key_sites(record))

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        if self.sidecars is None:
            unchecked = Finding(
                name=_KEYWORD_JOIN, ok=True, watched=True,
                measured=(
                    f"{len(self.keys)} distinct keys, none checked: no converted "
                    "tree was readable (pass --cards-folder)"
                ),
            )
            yield unchecked
            yield replace(unchecked, name=_OTHER_JOIN)
            yield replace(unchecked, name=_OUTSIDE_JOIN)
            return
        tally = _JoinTally()
        ordered = sorted(
            self.keys,
            key=lambda k: (k.script_file, k.trait_kind, k.index_within_kind),
        )
        for key in ordered:
            tally.add(key, _join_result(self.sidecars, key))
        yield tally.keyword_finding()
        yield tally.other_finding()
        yield tally.outside_finding()


@dataclass(slots=True)
class _JoinTally:
    """One join of a window's distinct keys, by trait kind."""

    checked: Counter[str] = field(default_factory=Counter)
    unjoinable: Counter[str] = field(default_factory=Counter)
    runtime_only: Counter[str] = field(default_factory=Counter)
    unchecked_by_tree: Counter[str] = field(default_factory=Counter)
    #: ``keyword`` or ``other`` -> why the first few unjoinable keys failed.
    examples: dict[str, list[str]] = field(
        default_factory=lambda: {"keyword": [], "other": []},
    )

    def add(self, key: ProvenanceKey, result) -> None:
        """Count one key by its :func:`_join_result`."""
        if result is _UNCHECKED:
            self.unchecked_by_tree[key.tree] += 1
            return
        self.checked[key.trait_kind] += 1
        if result is _RUNTIME_ONLY:
            self.runtime_only[key.trait_kind] += 1
            return
        if result is None:
            return
        self.unjoinable[key.trait_kind] += 1
        bucket = "keyword" if key.trait_kind == "keyword" else "other"
        _keep_example(self.examples[bucket], result)

    def keyword_finding(self) -> Finding:
        broken = self.unjoinable.get("keyword", 0)
        trees = tuple(
            f"{count} keys in {tree}, a tree this run was not given"
            for tree, count in sorted(self.unchecked_by_tree.items())
        )
        return Finding(
            name=_KEYWORD_JOIN,
            ok=broken == 0,
            measured=(
                f"{broken}/{self.checked.get('keyword', 0)} keyword keys fail to "
                f"join, {self.runtime_only.get('keyword', 0)} outside every "
                f"declared range, {sum(self.unchecked_by_tree.values())} unchecked"
            ),
            detail=trees + tuple(self.examples["keyword"]),
        )

    def other_finding(self) -> Finding:
        other_checked = sum(self.checked.values()) - self.checked.get("keyword", 0)
        other_broken = (
            sum(self.unjoinable.values()) - self.unjoinable.get("keyword", 0)
        )
        other_rate = _rate(other_broken, other_checked)
        return Finding(
            name=_OTHER_JOIN,
            ok=True,
            watched=True,
            measured=(
                f"{other_broken}/{other_checked} keys of every other trait "
                f"kind fail to join ({other_rate:.1%}, watched, no ceiling)"
            ),
            detail=tuple(
                f"{kind}: {self.unjoinable[kind]}/{count} unjoinable"
                for kind, count in sorted(self.checked.items())
                if kind != "keyword"
            ) + tuple(self.examples["other"]),
        )

    def outside_finding(self) -> Finding:
        total_checked = sum(self.checked.values())
        outside = sum(self.runtime_only.values())
        outside_rate = _rate(outside, total_checked)
        return Finding(
            name=_OUTSIDE_JOIN,
            ok=True,
            watched=True,
            measured=(
                f"{outside}/{total_checked} keys resolve outside every "
                f"(face, trait_kind) range their sidecar declares "
                f"({outside_rate:.1%}, watched, no ceiling)"
            ),
            detail=tuple(
                f"{kind}: {self.runtime_only[kind]}/{self.checked[kind]} outside "
                "every declared range"
                for kind in sorted(self.runtime_only)
            ),
        )


class _Modes:
    """A degraded run writes no mana records, per-clause attribution, trigger
    cause or replacement records; mixed into a patched run, the missing kinds
    read as scarcity."""

    def __init__(self) -> None:
        self.modes: Counter[str] = Counter()
        self.examples: dict[str, str] = {}

    def add(self, record: EffectRecord) -> None:
        self.modes[record.mode.value] += 1
        self.examples.setdefault(record.mode.value, record.record_id)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        total = sum(self.modes.values())
        shown = _histogram(sorted(self.modes.items()))
        degraded = total - self.modes.get(CollectionMode.PATCHED.value, 0)
        yield Finding(
            name="every record was collected in patched mode",
            ok=degraded == 0,
            measured=f"{degraded}/{total} not patched ({shown})",
            detail=tuple(
                f"first {name} record: {self.examples[name]}"
                for name in sorted(self.modes)
                if name != CollectionMode.PATCHED.value
            ),
        )


class _Outcomes:
    """Whether activation outcomes vary, as a real game's do."""

    def __init__(self) -> None:
        self.activations = 0
        self.outcomes: Counter[str] = Counter()

    def add(self, record: EffectRecord) -> None:
        if isinstance(record.payload, ActivationPayload):
            self.activations += 1
            self.outcomes[record.payload.outcome.value] += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        shown = _histogram(sorted(self.outcomes.items()))
        yield Finding(
            name="outcome takes more than one value",
            # A window with no activation records holds, as an absent kind does.
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


class _Costs:
    """Which cost channels the activation records ever populate."""

    def __init__(self) -> None:
        self.activations = 0
        self.cost_fields: Counter[str] = Counter()
        self.populated = 0

    def add(self, record: EffectRecord) -> None:
        if not isinstance(record.payload, ActivationPayload):
            return
        self.activations += 1
        paid = {
            name: bool(getattr(record.payload.costs, name)) for name in _COST_FIELDS
        }
        for name, filled in paid.items():
            self.cost_fields[name] += filled
        self.populated += any(paid.values())

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        shown = ", ".join(
            f"{name} {self.cost_fields[name]}" for name in _COST_FIELDS
        )
        dead = [
            name for name in _COST_FIELDS_ALWAYS_PAID
            if not self.cost_fields[name]
        ]
        enough = self.activations >= limits.min_cost_evidence
        # Nothing paid anywhere is a defect at any size; one dead channel only
        # once the window is big enough to blame the collector, not the pool.
        broken = bool(self.activations and (
            not self.populated or (enough and dead)
        ))
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
        yield Finding(
            name="cost fields are not all empty",
            ok=not broken,
            measured=(
                f"{self.populated}/{self.activations} activation records "
                f"paid something ({shown})"
            ),
            detail=detail,
        )


class _Tiers:
    """Every record collects all four tiers, so one vector or none: a depth
    varying by collector makes ``state.tiers`` a proxy for collection metadata
    the model must not see."""

    def __init__(self) -> None:
        self.tiers_by_kind: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)

    def add(self, record: EffectRecord) -> None:
        self.tiers_by_kind[_kind_label(record)][
            tuple(sorted(int(tier) for tier in record.state.tiers))
        ] += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
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
        yield Finding(
            name="snapshot tier depth is uniform across kinds",
            ok=len(seen) <= 1,
            measured=(
                f"{len(seen)} distinct tier vectors over "
                f"{len(self.tiers_by_kind)} kind groups"
            ),
            detail=tuple(detail) if len(seen) > 1 else (),
        )


class _Duplicates:
    """Records byte-identical to an earlier one in the same game, per kind."""

    def __init__(self) -> None:
        self.by_kind: Counter[str] = Counter()
        self.duplicates_by_kind: Counter[str] = Counter()
        #: game_id -> the digest of every record it has shown.
        self.digests: dict[str, set[bytes]] = defaultdict(set)

    def add(self, record: EffectRecord) -> None:
        label = _kind_label(record)
        self.by_kind[label] += 1
        digest = _digest(record)
        game = self.digests[record.game_id]
        if digest in game:
            self.duplicates_by_kind[label] += 1
        else:
            game.add(digest)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        worst = 0.0
        detail = []
        for label, count in sorted(self.by_kind.items()):
            duplicates = self.duplicates_by_kind[label]
            rate = _rate(duplicates, count)
            worst = max(worst, rate)
            detail.append(f"{label}: {duplicates}/{count} ({rate:.1%})")
        total = sum(self.by_kind.values())
        total_dupes = sum(self.duplicates_by_kind.values())
        overall = _rate(total_dupes, total)
        yield Finding(
            name="exact duplicates within a game stay rare, per kind",
            ok=worst <= limits.max_duplicate_rate,
            measured=(
                f"{total_dupes}/{total} overall ({overall:.1%}); worst "
                f"kind {worst:.1%}, limit {limits.max_duplicate_rate:.1%}"
            ),
            detail=tuple(detail),
        )


class _DuplicateEvents:
    """Byte-identical events inside one record: one outcome written twice,
    which the record-level check cannot see and which teaches the model that
    abilities deal their damage twice."""

    def __init__(self) -> None:
        self.events = 0
        self.duplicates = 0
        self.by_type: Counter[str] = Counter()

    def add(self, record: EffectRecord) -> None:
        seen: set[bytes] = set()
        # Not all_events: a rewrite's unedited two sides would read as a
        # repeat on every healthy record.
        for event in record.payload.listed_events():
            digest = canonical_digest(event.as_dict())
            self.events += 1
            if digest in seen:
                self.duplicates += 1
                self.by_type[event.type.value] += 1
            else:
                seen.add(digest)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        rate = _rate(self.duplicates, self.events)
        yield Finding(
            name="no record repeats an event inside itself",
            ok=rate <= limits.max_duplicate_event_rate,
            measured=(
                f"{self.duplicates}/{self.events} events repeat "
                f"another in the same record ({rate:.2%}, limit "
                f"{limits.max_duplicate_event_rate:.2%})"
            ),
            detail=tuple(
                f"{event_type}: {count}"
                for event_type, count in self.by_type.most_common(_EXAMPLES)
            ),
        )


class _Triggers:
    """Fired against not-fired, aggregate and per evaluated mode, because a
    mode drawing no negatives hides behind every other mode's balance."""

    def __init__(self) -> None:
        self.fired: Counter[bool] = Counter()
        #: Per evaluated mode: how often it fired, of how often it was asked.
        self.modes: dict[str, _Share] = defaultdict(_Share)

    def add(self, record: EffectRecord) -> None:
        payload = record.payload
        if not isinstance(payload, TriggerPayload):
            return
        self.fired[payload.fired] += 1
        # Forge's own mode name where written: the event-type vocabulary is
        # far coarser than Forge's trigger modes.
        mode = payload.event.params.get("mode") or payload.event.type.value
        self.modes[str(mode)].note(payload.fired)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        fired = self.fired[True]
        total = fired + self.fired[False]
        share = _rate(fired, total)
        skewed = sorted(
            ((counts.rate, mode, counts) for mode, counts in self.modes.items()),
            reverse=True,
        )
        yield Finding(
            name="trigger records draw negatives against positives",
            ok=not total or share <= limits.max_trigger_fired_share,
            measured=(
                f"{fired}:{total - fired} fired:not-fired over {total} trigger "
                f"records ({share:.1%} fired, ceiling "
                f"{limits.max_trigger_fired_share:.1%})"
            ),
            detail=tuple(
                f"{mode}: {counts.hit}:{counts.seen - counts.hit} "
                f"({rate:.1%} fired)"
                for rate, mode, counts in skewed[:_EXAMPLES]
            ),
        )


@dataclass(slots=True)
class _ResultCounts:
    """One replacement result's rewrite records, and what they carry."""

    with_outgoing: int = 0
    with_replaced_by: int = 0
    seen: int = 0


class _Rewrites:
    """The rewrite channel's four findings, all read off one result histogram."""

    def __init__(self) -> None:
        self.rewrites = 0
        #: Keyed by the wire spelling, plus :data:`_NO_RESULT`.
        self.results: dict[str, _ResultCounts] = defaultdict(_ResultCounts)
        #: Result-carrying rewrites whose ``outgoing`` equals ``incoming``.
        self.identity = 0
        self.identity_examples: list[str] = []
        #: ``prevented``/``skipped`` records carrying an ``outgoing`` or a
        #: ``replaced_by``.
        self.impossible = 0
        self.impossible_examples: list[str] = []

    def add(self, record: EffectRecord) -> None:
        payload = record.payload
        if not isinstance(payload, RewritePayload):
            return
        self.rewrites += 1
        counts = self.results[
            payload.result.value if payload.result else _NO_RESULT
        ]
        counts.with_outgoing += payload.outgoing is not None
        counts.with_replaced_by += bool(payload.replaced_by)
        counts.seen += 1
        if payload.result is None:
            # Pre-contract shard: its writer always produced an identity
            # ``outgoing``.
            return
        example = f"{record.record_id} ({payload.result.value})"
        if payload.outgoing is not None and (
            payload.outgoing.as_dict() == payload.incoming.as_dict()
        ):
            self.identity += 1
            _keep_example(self.identity_examples, example)
        if payload.result in REWRITE_RESULTS_WITHOUT_ABILITY and (
            payload.outgoing is not None or payload.replaced_by
        ):
            self.impossible += 1
            _keep_example(self.impossible_examples, example)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        yield self._say_what_happened()
        yield self._do_not_copy_their_event()
        yield self._outcomes_agree_with_their_payload()
        yield self._name_the_substituted_ability(limits)

    def _without_a_result(self) -> int:
        """Rewrites written before the hook passed a result. Read through
        ``.get`` so a healthy window grows no ``absent`` row."""
        counts = self.results.get(_NO_RESULT)
        return counts.seen if counts else 0

    def _shown(self) -> str:
        """The result histogram, for whichever rewrite finding is printing."""
        return _histogram(
            (name, counts.seen) for name, counts in sorted(self.results.items())
        )

    def _say_what_happened(self) -> Finding:
        """Which of the five replacement outcomes the handler reported.

        Judged at zero: a missing result is a pre-contract shard in the window
        or a hook reading the wrong argument, and both are worth stopping for.
        """
        absent = self._without_a_result()
        named = self.rewrites - absent
        return Finding(
            name="rewrite records say which replacement result happened",
            ok=absent == 0,
            measured=(
                f"{named}/{self.rewrites} rewrite records carry a result "
                f"({self._shown()})"
            ),
            detail=() if absent == 0 else (
                f"{absent} carry none: either the window reaches back to shards "
                "collected before the hook passed one, or the listener is "
                "reading the wrong argument slot",
            ),
        )

    def _do_not_copy_their_event(self) -> Finding:
        """``outgoing`` copying ``incoming`` says nothing; null says "not
        rewritten in place". Counted only over records carrying a ``result``,
        since a pre-contract writer had no null to write."""
        judged = self.rewrites - self._without_a_result()
        rate = _rate(self.identity, judged)
        return Finding(
            name="a rewrite's outgoing event is not a copy of its incoming one",
            ok=self.identity == 0,
            measured=(
                f"{self.identity}/{judged} result-carrying rewrites "
                f"repeat their incoming event as outgoing ({rate:.1%})"
            ),
            detail=tuple(self.identity_examples),
        )

    def _outcomes_agree_with_their_payload(self) -> Finding:
        """``prevented`` and ``skipped`` cannot carry a payload.

        Both return before ``executeReplacementUnobserved`` runs any ability, so
        an ``outgoing`` or ``replaced_by`` on one is a listener reading a stale
        argument. ``not_replaced`` is not judged: its prevention branch writes
        ``PreventedAmount`` before returning.
        """
        ran = ", ".join(
            f"{name} {counts.with_outgoing}/{counts.seen} with outgoing, "
            f"{counts.with_replaced_by}/{counts.seen} with replaced_by"
            for name, counts in sorted(self.results.items())
        ) or "none"
        cannot = sorted(r.value for r in REWRITE_RESULTS_WITHOUT_ABILITY)
        return Finding(
            name="outgoing is null on the results that rewrote nothing",
            ok=self.impossible == 0,
            measured=(
                f"{self.impossible} {'/'.join(cannot)} records carry an "
                f"outgoing or a replaced_by, of {self.rewrites} rewrites ({ran})"
            ),
            detail=tuple(self.impossible_examples),
        )

    def _name_the_substituted_ability(self, limits: Thresholds) -> Finding:
        """Which ability ran instead, on the outcomes that ran one; a resolver
        that keys nothing reads zero here."""
        seen = named = 0
        for result in REWRITE_RESULTS_RUNNING_AN_ABILITY:
            counts = self.results.get(result.value)
            if counts:
                named += counts.with_replaced_by
                seen += counts.seen
        rate = _rate(named, seen)
        return _floor_finding(
            name="rewrites name the ability that ran instead",
            floor=limits.min_rewrite_replaced_by_rate,
            seen=seen,
            rate=rate,
            measured=lambda limit: (
                f"{named}/{seen} replaced/updated records name a replaced_by "
                f"key ({rate:.1%}, {limit})"
            ),
            detail=(
                f"{name}: {counts.with_replaced_by}/{counts.seen} name one"
                for name, counts in sorted(self.results.items())
            ),
        )


class _ZoneChanges:
    """Where a card came from, and how often "the stack" is where it went.

    Without ``from_zone`` a graveyard recursion reads like a token entering.
    Watched, because a card made rather than moved has no origin. A channel
    mostly ``to_zone=stack`` is reporting casts rather than the board.
    """

    def __init__(self) -> None:
        self.zone_changes = 0
        self.with_from = 0
        self.to_zone: Counter[str] = Counter()

    def add(self, record: EffectRecord) -> None:
        for event in record.payload.all_events():
            if event.type is EventType.ZONE_CHANGE:
                self.zone_changes += 1
                self.with_from += bool(event.params.get("from_zone"))
                self.to_zone[str(event.params.get("to_zone"))] += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        rate = _rate(self.with_from, self.zone_changes)
        to_stack = self.to_zone.get("stack", 0)
        stack_rate = _rate(to_stack, self.zone_changes)
        yield _floor_finding(
            name="zone_change events say where the card came from",
            floor=limits.min_zone_change_from_zone_rate,
            seen=self.zone_changes,
            rate=rate,
            measured=lambda limit: (
                f"{self.with_from}/{self.zone_changes} carry "
                f"from_zone ({rate:.1%}, {limit}); {to_stack} say "
                f"to_zone=stack ({stack_rate:.1%})"
            ),
            detail=(
                f"to_zone={zone}: {count}"
                for zone, count in self.to_zone.most_common(_EXAMPLES)
            ),
        )


def _resolution_events(record: EffectRecord) -> tuple[Event, ...]:
    """Resolution events, rated by how many say something: a sub-ability
    index, the root, or an explicit unresolved. ``absent`` says nothing, since
    a pre-sentinel writer used it for both root and unresolved."""
    if record.kind is not RecordKind.RESOLUTION:
        return ()
    return record.payload.listed_events()


def _fork_events(record: EffectRecord) -> tuple[Event, ...]:
    """Every event on a fork record, rated apart because the fork collectors
    are different code, and a fork whose events name no clause cannot be
    compared clause by clause with the record it mirrors."""
    return record.payload.all_events() if record.fork else ()


class _Attribution:
    """Whether one population of events (``events_of``) names the clause that
    produced it."""

    def __init__(
        self,
        name: str,
        noun: str,
        events_of: Callable[[EffectRecord], tuple[Event, ...]],
        floor_of: Callable[[Thresholds], float | None],
    ) -> None:
        self.name = name
        self.noun = noun
        self.events_of = events_of
        self.floor_of = floor_of
        self.events = 0
        self.attribution: Counter[AttributionKind] = Counter()

    def add(self, record: EffectRecord) -> None:
        for event in self.events_of(record):
            self.events += 1
            self.attribution[attribution_kind(event.attributed_to)] += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        named = self.events - self.attribution.get(AttributionKind.ABSENT, 0)
        rate = _rate(named, self.events)
        yield _floor_finding(
            name=self.name,
            floor=self.floor_of(limits),
            seen=self.events,
            rate=rate,
            measured=lambda limit: (
                f"{named}/{self.events} {self.noun} name a clause "
                f"({rate:.1%}, {limit})"
            ),
            detail=(
                f"{state}: {count}"
                for state, count in sorted(self.attribution.items())
            ),
        )


class _CauseBearing:
    """Whether the events that declare a ``cause`` populate it.

    Over :data:`~effects.domain.event_schema.CAUSE_BEARING_TYPES` alone, since
    most types correctly never carry one. An empty ``cause`` also misleads the
    duplicate-event check: two attackers dealing the same damage to one player
    differ only by it.
    """

    def __init__(self) -> None:
        #: Per cause-bearing event type: how often it names a cause, of how often seen.
        self.by_type: dict[str, _Share] = defaultdict(_Share)

    def add(self, record: EffectRecord) -> None:
        for event in record.payload.all_events():
            if event.type in CAUSE_BEARING_TYPES:
                self.by_type[event.type.value].note(bool(event.params.get("cause")))

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        with_cause = sum(counts.hit for counts in self.by_type.values())
        seen = sum(counts.seen for counts in self.by_type.values())
        rate = _rate(with_cause, seen)
        yield _floor_finding(
            name="events that declare a cause name one",
            floor=limits.min_cause_rate,
            seen=seen,
            rate=rate,
            measured=lambda limit: (
                f"{with_cause}/{seen} events of the {len(CAUSE_BEARING_TYPES)} "
                f"cause-bearing types carry a cause ({rate:.1%}, {limit})"
            ),
            detail=(
                f"{event_type}: {counts.hit}/{counts.seen}"
                for event_type, counts in sorted(self.by_type.items())
            ),
        )


class _MirrorTurns:
    """A fork's snapshot is of the moment it forked, so its turn must equal
    its mirror's; otherwise it surfaces as a backward turn jump blamed on a
    game merge. Forks whose mirror fell outside the window (``--limit`` cuts
    mid-game) are counted apart and not judged."""

    def __init__(self, record_ids: _RecordIds) -> None:
        self.record_ids = record_ids
        #: (fork record id, mirrored id, fork's turn), compared after the pass
        #: because a shard may carry the fork before its mirror.
        self.mirrors: list[tuple[str, str, int]] = []

    def add(self, record: EffectRecord) -> None:
        if record.mirror_of is not None:
            self.mirrors.append(
                (record.record_id, record.mirror_of, record.state.global_.turn)
            )

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        disagreed: list[str] = []
        uncomparable = 0
        for record_id, mirror_of, turn in self.mirrors:
            mirror_turn = self.record_ids.turns.get(mirror_of)
            if mirror_turn is None:
                uncomparable += 1
            elif mirror_turn != turn:
                disagreed.append(
                    f"{record_id} at turn {turn} mirrors {mirror_of} at turn "
                    f"{mirror_turn}"
                )
        comparable = len(self.mirrors) - uncomparable
        rate = _rate(len(disagreed), comparable)
        yield Finding(
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


class _ProbeForks:
    """Whether the probe path ran at all, and on which keywords. Watched: a run
    may deliberately name no keyword, but one that meant to probe should learn
    it in minutes rather than at evaluation."""

    def __init__(self) -> None:
        self.probe_forks = 0
        self.probed_keywords: Counter[str] = Counter()
        self.interventions = 0

    def add(self, record: EffectRecord) -> None:
        payload = record.payload
        if isinstance(payload, CombatPayload) and record.is_probe:
            self.probe_forks += 1
            if payload.probed_keyword:
                self.probed_keywords[payload.probed_keyword] += 1
        if record.interventional:
            self.interventions += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        shown = _histogram(self.probed_keywords.most_common())
        yield Finding(
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


class _EventTypes:
    """Which of the declared event types the window contains.

    Watched: a short window legitimately misses rare types. It measures fires,
    where ``test_event_schema_completeness.py`` only finds types the Java
    source never references, so a type referenced from code that never runs
    shows up here alone.
    """

    def __init__(self) -> None:
        #: Every distinct event type the window named, anywhere in a record.
        self.seen: set[str] = set()

    def add(self, record: EffectRecord) -> None:
        self.seen.update(event.type.value for event in record.payload.all_events())

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        declared = {member.value for member in EventType}
        unseen = tuple(sorted(declared - self.seen))
        yield Finding(
            name="declared event types are observed in the window",
            ok=True,
            watched=True,
            measured=(
                f"{len(self.seen)}/{len(declared)} declared event types "
                "appeared at least once"
            ),
            detail=unseen,
        )


def _checks(sidecars: SidecarCache | None) -> tuple[_Check, ...]:
    """Every invariant, in the order the report prints them."""
    record_ids = _RecordIds()
    return (
        record_ids,
        _GameBoundaries(),
        _Links(),
        _ActingLines(),
        _KeyJoins(sidecars),
        _Modes(),
        _Outcomes(),
        _Costs(),
        _Tiers(),
        _Duplicates(),
        _DuplicateEvents(),
        _Triggers(),
        _Rewrites(),
        _ZoneChanges(),
        _Attribution(
            name="resolution events name what produced them",
            noun="resolution events",
            events_of=_resolution_events,
            floor_of=lambda limits: limits.min_attributed_rate,
        ),
        _Attribution(
            name="fork records' events name what produced them",
            noun="events on fork records",
            events_of=_fork_events,
            floor_of=lambda limits: limits.min_fork_attributed_rate,
        ),
        _CauseBearing(),
        _MirrorTurns(record_ids),
        _ProbeForks(),
        _EventTypes(),
    )
