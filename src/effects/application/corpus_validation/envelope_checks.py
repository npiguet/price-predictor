"""The checks that read a record's envelope or the record whole: ids, games,
link halves, acting lines, collection mode, snapshot tiers, exact duplicates
and forks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

from effects.application.corpus_validation.findings import (
    EXAMPLES,
    Finding,
    Share,
    Thresholds,
    ceiling_finding,
    histogram,
    keep_example,
)
from effects.domain.canonical_digest import canonical_digest
from effects.domain.event_schema import EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    KINDS_WITHOUT_ACTING_ABILITY,
    PAYLOAD_TYPES,
    CollectionMode,
    CombatPayload,
    EffectRecord,
    Moment,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
    UnresolvedReason,
)

#: Envelope fields that say when a record was written rather than what it says,
#: blanked before hashing; ``mode`` is constant within a run.
_IDENTITY_FIELDS = (
    "record_id", "run_id", "timestamp", "game_id",
    "link_id", "mirror_of", "variant_of", "mode",
)


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


class RecordIds:
    """Every record id seen, and the turn its snapshot claims, which
    :class:`MirrorTurns` reads."""

    def __init__(self) -> None:
        self.turns: dict[str, int] = {}
        #: Records whose id an earlier record already carried.
        self.repeats = Share()
        self.examples: list[str] = []

    def add(self, record: EffectRecord) -> None:
        repeat = record.record_id in self.turns
        self.repeats.note(repeat)
        if repeat:
            keep_example(self.examples, record.record_id)
        else:
            self.turns[record.record_id] = record.state.global_.turn

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        repeats = self.repeats
        yield Finding(
            name="record_id is unique across the run's shards",
            ok=repeats.hit == 0,
            measured=(
                f"{repeats.hit} repeats of {len(self.turns)} "
                f"distinct ids over {repeats.seen} records ({repeats.rate:.1%})"
            ),
            detail=tuple(f"repeated: {rid}" for rid in self.examples),
        )


@dataclass(slots=True)
class _Game:
    """What one ``game_id`` has shown so far."""

    max_turn: int = 0
    backward_jump: int = 0
    names: set[str] = field(default_factory=set)


class GameBoundaries:
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
            for gid, jump in sorted(jumped.items())[:EXAMPLES]
        ] + [
            f"{gid} shows {count} distinct card names"
            for gid, count in sorted(crowded.items())[:EXAMPLES]
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


class _LinkHalf(NamedTuple):
    """One record carrying a ``link_id``: which half it is, and what it acts through."""

    #: None on a record of a kind with no halves, which never pairs.
    moment: Moment | None
    keys: tuple[ProvenanceKey, ...]


def _joined(halves: list[_LinkHalf]) -> bool:
    """Whether one link's halves form a valid resolution (FR-029d): one cost
    half and one effect half, or several effect halves all acting through mode
    keys (``option`` set) of one charm root."""
    costs = [half.keys for half in halves if half.moment is Moment.ACTIVATION]
    effects = [half.keys for half in halves if half.moment is Moment.RESOLUTION]
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


class Links:
    """Every half carrying each ``link_id``, joined after the pass."""

    def __init__(self) -> None:
        #: link_id -> every half carrying it.
        self.links: dict[str, list[_LinkHalf]] = defaultdict(list)

    def add(self, record: EffectRecord) -> None:
        if record.link_id is not None:
            self.links[record.link_id].append(
                _LinkHalf(record.moment, tuple(record.ability or ())),
            )

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        unpaired = Share()
        for halves in self.links.values():
            unpaired.note(not _joined(halves))
        paired = unpaired.seen - unpaired.hit
        sizes = Counter(len(halves) for halves in self.links.values())
        yield ceiling_finding(
            name=(
                "every link_id joins one activation to its resolution(s): one, "
                "or one per chosen mode of a charm"
            ),
            ceiling=limits.max_unpaired_link_rate,
            rate=unpaired.rate,
            measured=lambda limit: (
                f"{paired}/{unpaired.seen} link ids paired, {unpaired.hit} "
                f"unpaired ({unpaired.rate:.1%}, {limit})"
            ),
            detail=(
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
    #: Why the records with an empty ``ability`` have none.
    unresolved: Counter[UnresolvedReason] = field(default_factory=Counter)


class ActingLines:
    """Whether the kinds that act through a line name it, and why when not."""

    #: Finding name -> the kind labels it groups.
    GROUPS = {
        "trigger and rewrite": _labels_of(RecordKind.TRIGGER, RecordKind.REWRITE),
        "resolution": _labels_of(RecordKind.RESOLUTION),
        "continuous": _labels_of(RecordKind.CONTINUOUS),
    }

    def __init__(self) -> None:
        self.acting: dict[str, _ActingCounts] = defaultdict(_ActingCounts)
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
            counts.unresolved[record.ability_unresolved] += 1
        elif record.ability is not None and not record.ability:
            self.silent_empty += 1
            keep_example(self.silent_empty_examples, f"{record.record_id} ({label})")

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
            unresolved = sum((counts.unresolved for counts in groups), Counter())
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
                    for reason, count in sorted(unresolved.items())
                ),
            )

    def _empty_ability_says_why(self) -> Finding:
        """An empty ``ability`` needs an ``ability_unresolved`` reason: "the
        Monarch has no printed line" and "the resolver regressed" both write
        ``ability: []``."""
        unresolved = sum((counts.unresolved for counts in self.acting.values()), Counter())
        empty = self.silent_empty + unresolved.total()
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
                for reason, count in sorted(unresolved.items())
            ),
        )


class Modes:
    """A degraded run writes no mana records, per-clause attribution, trigger
    cause or replacement records; mixed into a patched run, the missing kinds
    read as scarcity."""

    def __init__(self) -> None:
        self.modes: Counter[CollectionMode] = Counter()
        self.examples: dict[CollectionMode, str] = {}

    def add(self, record: EffectRecord) -> None:
        self.modes[record.mode] += 1
        self.examples.setdefault(record.mode, record.record_id)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        total = sum(self.modes.values())
        shown = histogram(sorted(self.modes.items()))
        degraded = total - self.modes.get(CollectionMode.PATCHED, 0)
        yield Finding(
            name="every record was collected in patched mode",
            ok=degraded == 0,
            measured=f"{degraded}/{total} not patched ({shown})",
            detail=tuple(
                f"first {name} record: {self.examples[name]}"
                for name in sorted(self.modes)
                if name is not CollectionMode.PATCHED
            ),
        )


class Tiers:
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


class Duplicates:
    """Records byte-identical to an earlier one in the same game, per kind."""

    def __init__(self) -> None:
        #: Per kind label: how many records repeat an earlier one of their game.
        self.by_kind: dict[str, Share] = defaultdict(Share)
        #: game_id -> the digest of every record it has shown.
        self.digests: dict[str, set[bytes]] = defaultdict(set)

    def add(self, record: EffectRecord) -> None:
        digest = _digest(record)
        game = self.digests[record.game_id]
        self.by_kind[_kind_label(record)].note(digest in game)
        game.add(digest)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        kinds = sorted(self.by_kind.items())
        worst = max((share.rate for _, share in kinds), default=0.0)
        overall = Share.total(self.by_kind.values())
        yield ceiling_finding(
            name="exact duplicates within a game stay rare, per kind",
            ceiling=limits.max_duplicate_rate,
            rate=worst,
            measured=lambda limit: (
                f"{overall.hit}/{overall.seen} overall ({overall.rate:.1%}); "
                f"worst kind {worst:.1%}, {limit}"
            ),
            detail=(
                f"{label}: {share.hit}/{share.seen} ({share.rate:.1%})"
                for label, share in kinds
            ),
        )


class MirrorTurns:
    """A fork's snapshot is of the moment it forked, so its turn must equal
    its mirror's; otherwise it surfaces as a backward turn jump blamed on a
    game merge. Forks whose mirror fell outside the window (``--limit`` cuts
    mid-game) are counted apart and not judged."""

    def __init__(self, record_ids: RecordIds) -> None:
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
        disagreed = Share()
        examples: list[str] = []
        for record_id, mirror_of, turn in self.mirrors:
            mirror_turn = self.record_ids.turns.get(mirror_of)
            if mirror_turn is None:
                continue
            disagreed.note(mirror_turn != turn)
            if mirror_turn != turn:
                keep_example(
                    examples,
                    f"{record_id} at turn {turn} mirrors {mirror_of} at turn "
                    f"{mirror_turn}",
                )
        uncomparable = len(self.mirrors) - disagreed.seen
        yield ceiling_finding(
            name="a fork's turn agrees with the record it mirrors",
            ceiling=limits.max_mirror_turn_disagreement_rate,
            rate=disagreed.rate,
            measured=lambda limit: (
                f"{disagreed.hit}/{disagreed.seen} forks disagree with their "
                f"mirror about the turn ({disagreed.rate:.1%}, {limit}); "
                f"{uncomparable} mirrors fell outside the window"
            ),
            detail=examples,
        )


class ProbeForks:
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
        shown = histogram(self.probed_keywords.most_common())
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
