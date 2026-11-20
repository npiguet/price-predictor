"""The checks that read a record's envelope or the record whole: ids, games,
link halves, acting lines, collection mode, snapshot tiers, exact duplicates
and forks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from effects.application.corpus_validation.findings import (
    EXAMPLES,
    Finding,
    Share,
    Thresholds,
    ceiling_finding,
    histogram,
    keep_example,
    share_of,
)
from effects.domain.canonical_digest import canonical_digest
from effects.domain.event_schema import EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    KINDS_WITHOUT_ACTING_ABILITY,
    PAYLOAD_TYPES,
    ActivationPayload,
    CollectionMode,
    CombatPayload,
    EffectRecord,
    Moment,
    PlayabilitySubkind,
    RecordKind,
    RecordOrigin,
    ResolutionPayload,
    TriggerPayload,
    TurnActionPayload,
    TurnActionSubkind,
    UnresolvedReason,
    is_game_end,
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
    from effects.infrastructure.record_codec import record_to_dict

    data = record_to_dict(record)
    for name in _IDENTITY_FIELDS:
        data.pop(name, None)
    return canonical_digest(data)


def _is_deferred_mana_record(record: EffectRecord) -> bool:
    """A resolution half the mana reservoir held back until the game ended.

    Its snapshot is from when the mana ability was activated, so it lands
    after later turns and would read as a backward turn jump. Recognised by
    the payload rather than the acting card: a ``mana_produced`` in an effect
    half that resolved no stack item (an ``outcome`` written null, which
    only a mana record and the rescue record write), whatever else the mana
    ability's payment and sub-abilities did (a pain land's damage, WP7).
    Nothing defers an activation half, so a jump on one is still a game
    boundary.
    """
    payload = record.payload
    if not (record.moment is Moment.RESOLUTION and isinstance(payload, ResolutionPayload)
            and any(event.type is EventType.MANA_PRODUCED for event in payload.events)):
        return False
    return payload.outcome is None


def _is_deferred_trigger_record(record: EffectRecord) -> bool:
    """A fired trigger record the per-line reservoir held back until the game
    ended (connector C7). Its snapshot is from when the trigger fired; an
    unfired one is written as it is drawn, so a jump on it is still a game
    boundary."""
    payload = record.payload
    return (
        record.kind is RecordKind.TRIGGER
        and isinstance(payload, TriggerPayload)
        and payload.fired
    )


def _is_deferred(record: EffectRecord) -> bool:
    """A record a reservoir writes at the game's end with an earlier board."""
    return _is_deferred_mana_record(record) or _is_deferred_trigger_record(record)


#: The events a mana record holds: what its cost tapped, then the mana.


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
        if _is_deferred(record):
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
                "as deferred reservoir flushes"
            ),
            detail=tuple(detail),
        )


class KindPresence:
    """V5 #19: whether a window holds every record kind a game writes.
    Each per-kind check reads an absent kind as holding, so a collector that
    stopped writing one kind passed every check."""

    #: Kinds every game writes: two cost-and-effect resolutions, a trigger
    #: evaluated, a combat, sampled decision candidates, a declaration's
    #: attackers and blockers, a draw step.
    EXPECTED = ("resolution/activation", "resolution/resolution", "trigger", "combat",
                "playability/decision", "playability/attackers", "playability/blockers",
                "turn_action/draw")

    def __init__(self, playability_rate: float | None = None) -> None:
        self.labels: Counter[str] = Counter()
        self.records = 0
        #: The run's ``--playability-rate`` from its manifest; decision records
        #: are judged only where it is known and above 0.
        self.playability_rate = playability_rate

    def add(self, record: EffectRecord) -> None:
        self.records += 1
        label = _kind_label(record)
        if record.kind is RecordKind.TURN_ACTION and record.subkind is not None:
            label = f"{record.kind.value}/{record.subkind.value}"
        self.labels[label] += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        expected = [label for label in self.EXPECTED if label != "playability/decision"
                    or (self.playability_rate is not None and self.playability_rate > 0)]
        missing = [label for label in expected if not self.labels[label]]
        judged = self.records >= limits.min_presence_records
        # The game-end marker has a check of its own (GameEnds).
        others = sorted(set(_all_labels()) - set(expected) - set(self.labels) - {_GAME_END})
        if not missing:
            measured = f"every kind a game writes is present over {self.records} records"
        else:
            measured = (f"{self.records} records hold no {', '.join(missing)}"
                        + ("" if judged else f" (under the {limits.min_presence_records} "
                           "records this is judged at)"))
        yield Finding(
            name="every record kind a game writes is present",
            ok=not (missing and judged),
            measured=measured,
            detail=(f"watched, absent: {', '.join(others)}",) if others else (),
        )


_GAME_END = f"{RecordKind.TURN_ACTION.value}/{TurnActionSubkind.GAME_END.value}"


class GameEnds:
    """WP7: which games ended with their ``turn_action`` / ``game_end``
    marker, and what the per-game cap withheld from them.

    A game without one was cut (a killed worker, a game or an end-of-game
    write that threw) or is still running, which a window read
    mid-collection cannot tell apart: watched, as cut games are expected
    beside a kill backstop. A JVM lifetime plays its games one after another
    (the ``game_id``'s last segment counts them), so a game a later game of
    its lifetime follows has stopped; a window holding such a game and no
    marker at all is a collection that writes none, and fails.
    """

    def __init__(self) -> None:
        self.games: set[str] = set()
        self.ended: set[str] = set()
        self.withheld = 0
        self.markers = 0

    def add(self, record: EffectRecord) -> None:
        self.games.add(record.game_id)
        if is_game_end(record):
            self.ended.add(record.game_id)
            self.markers += 1
            payload = record.payload
            if isinstance(payload, TurnActionPayload) and payload.records_withheld:
                self.withheld += payload.records_withheld

    def _closed(self) -> set[str]:
        """Games a later game of their JVM lifetime follows."""
        latest: dict[str, int] = {}
        counted: dict[str, tuple[str, int]] = {}
        for game in self.games:
            lifetime, _, counter = game.rpartition(".")
            if lifetime and counter.isdigit():
                counted[game] = (lifetime, int(counter))
                latest[lifetime] = max(latest.get(lifetime, -1), int(counter))
        return {game for game, (lifetime, counter) in counted.items()
                if counter < latest[lifetime]}

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        name = "games end with their game-end marker"
        games, without = len(self.games), self.games - self.ended
        share = len(without) / games if games else 0.0
        measured = (f"{len(without)} of {games} game(s) have no game-end marker "
                    f"({share:.1%}): cut, or still running at the window's end")
        closed = self._closed()
        if games and not self.ended and closed:
            yield Finding(name=name, ok=False, measured=measured, detail=(
                f"{len(closed)} game(s) a later game of their JVM lifetime follows ended "
                "or were cut, and no game of the window wrote a marker: the collection "
                "writes none (a connector built before the marker?)",))
        else:
            yield Finding.watch(name, measured)
        if self.ended:
            yield self._unmarked_closed_finding(limits, closed)
        yield Finding.watch(
            "records the per-game cap withheld",
            f"{self.withheld} record(s) withheld over {self.markers} game-end marker(s)")

    def _unmarked_closed_finding(self, limits: Thresholds, closed: set[str]) -> Finding:
        """Games a later game of their JVM lifetime follows cannot have been
        killed (the kill ends the JVM), so one without a marker threw, or its
        end-of-game writes did. Healthy at or near 0."""
        unmarked = closed - self.ended
        share = share_of(len(unmarked), len(closed))
        return ceiling_finding(
            name="closed games end with their game-end marker",
            ceiling=limits.max_unmarked_closed_game_rate,
            rate=share,
            measured=lambda limit: (
                f"{len(unmarked)} of {len(closed)} closed game(s) (a later game of their "
                f"JVM lifetime follows) have no game-end marker ({share:.1%}, {limit})"
            ),
            detail=sorted(unmarked)[:EXAMPLES],
        )


def _all_labels() -> tuple[str, ...]:
    """Every kind label a window can hold, turn actions by subkind."""
    from effects.domain.records import TurnActionSubkind

    labels = [label for label in _labels_of(*RecordKind) if not label.startswith("turn_action")]
    return (*labels, *(f"{RecordKind.TURN_ACTION.value}/{s.value}" for s in TurnActionSubkind))


class _LinkHalf(NamedTuple):
    """One record carrying a ``link_id``: which half it is, and what it acts through."""

    #: None on a record of a kind with no halves, which never pairs.
    moment: Moment | None
    keys: tuple[ProvenanceKey, ...]


def _joined(halves: list[_LinkHalf]) -> bool:
    """Whether one link's halves form a valid resolution (FR-029d): one cost
    half and one effect half, or several effect halves all acting through mode
    keys (``option`` set) of one charm root, or a cost half alone. A cost half
    is written at the cast, before the item's fate, so it stands alone where
    the item was countered or the game ended with it on the stack (FR-042a,
    FR-042b)."""
    costs = [half.keys for half in halves if half.moment is Moment.ACTIVATION]
    effects = [half.keys for half in halves if half.moment is Moment.RESOLUTION]
    if len(costs) == 1 and len(halves) == 1:
        return True
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
        #: link_id -> ``(game_id, source)`` of each cost half.
        self.casts: dict[str, tuple[str, str | None]] = {}
        #: ``(game_id, subject)`` of every ``spell_countered`` an effect half holds.
        self.countered: set[tuple[str, str]] = set()

    def add(self, record: EffectRecord) -> None:
        if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
            self.countered.update(
                (record.game_id, subject) for event in record.payload.listed_events()
                if event.type is EventType.SPELL_COUNTERED for subject in event.subjects)
        if record.link_id is not None:
            self.links[record.link_id].append(_LinkHalf(
                record.moment, tuple(record.ability or ())))
            if isinstance(record.payload, ActivationPayload):
                self.casts[record.link_id] = (record.game_id, record.state.refs.source)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        unpaired = Share()
        for halves in self.links.values():
            unpaired.note(not _joined(halves))
        paired = unpaired.seen - unpaired.hit
        sizes = Counter(len(halves) for halves in self.links.values())
        yield ceiling_finding(
            name=(
                "every link_id joins one activation to its resolution(s): one, "
                "or one per chosen mode of a charm, or none for an "
                "activation whose item never resolved"
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
        yield self._casts_met(limits)

    def _casts_met(self, limits: Thresholds) -> Finding:
        """V5 #18: a cost half standing alone pairs (its item was
        countered, or still on the stack at the game's end), so the pairing
        check cannot see a collector writing no effect half at all. Judged
        only at the degenerate end: at least ``min_cost_evidence`` cost halves
        and not one met by its effect half; the share alone is watched."""
        alone = [link for link in self.casts if len(self.links[link]) == 1]
        countered = sum(1 for link in alone if self.casts[link] in self.countered)
        met = len(self.casts) - len(alone)
        degenerate = len(self.casts) >= limits.min_cost_evidence and met == 0
        measured = (f"{met}/{len(self.casts)} cost halves met by an effect half; "
                    f"{len(alone)} alone, {countered} of them countered by a "
                    "spell_countered in their game")
        name = "every cost half is met by its effect half unless its item never resolved"
        if degenerate:
            return Finding(name=name, ok=False, measured=measured + (
                f" (none met over {len(self.casts)}: the collector writes no effect half)"))
        return Finding.watch(name, measured)


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
        #: Game-end records: ``ability`` empty by design, for they seek no line.
        self.game_end = 0

    def add(self, record: EffectRecord) -> None:
        if record.kind in KINDS_WITHOUT_ACTING_ABILITY:
            return
        if record.is_game_end_record:
            self.game_end += 1
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
                f"no ability_unresolved reason ({self.game_end} game-end "
                "records seek no line and are not counted)"
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
    game merge.

    A mirror no record names is the reader's doing only when ``--limit`` cut
    the read (``window_cut``): then forks whose mirror fell outside the
    window are counted apart and not judged, and a window with no comparison
    at all is watched, never passed. Over a directory read whole the share of
    forks whose mirror is nowhere is judged too (a probe held for a record a
    throw dropped, an id built from the wrong counter)."""

    def __init__(self, record_ids: RecordIds, window_cut: bool = False) -> None:
        self.record_ids = record_ids
        self.window_cut = window_cut
        #: (fork record id, mirrored id, fork's turn), compared after the pass
        #: because a shard may carry the fork before its mirror.
        self.mirrors: list[tuple[str, str, int]] = []

    def add(self, record: EffectRecord) -> None:
        if record.mirror_of is not None:
            self.mirrors.append(
                (record.record_id, record.mirror_of, record.state.global_.turn)
            )

    def _compare(self) -> tuple[Share, list[str]]:
        """How many forks disagree with the turn of a mirror the window holds."""
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
        return disagreed, examples

    def _nothing_compared(self, compared: Share) -> bool:
        return bool(self.mirrors) and not compared.seen

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        disagreed, examples = self._compare()
        uncomparable = len(self.mirrors) - disagreed.seen
        name = "a fork's turn agrees with the record it mirrors"
        if self.window_cut and self._nothing_compared(disagreed):
            yield Finding.watch(
                name,
                f"no comparison made: all {uncomparable} mirrors fell outside the "
                "window --limit cut",
            )
            return
        unfound = share_of(uncomparable, len(self.mirrors))
        tail = "" if self.window_cut else (
            f" ({unfound:.1%}, limit {limits.max_unfound_mirror_rate:.1%})")
        turns = ceiling_finding(
            name=name,
            ceiling=limits.max_mirror_turn_disagreement_rate,
            rate=disagreed.rate,
            measured=lambda limit: (
                f"{disagreed.hit}/{disagreed.seen} forks disagree with their "
                f"mirror about the turn ({disagreed.rate:.1%}, {limit}); "
                f"{uncomparable} mirrors fell outside the window{tail}"
            ),
            detail=examples,
        )
        if self.window_cut or unfound <= limits.max_unfound_mirror_rate:
            yield turns
            return
        yield replace(turns, ok=False, detail=(
            "mirrors named by forks are nowhere in the directory: a probe held "
            "for a record never written", *turns.detail,
        ))


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
        if isinstance(payload, CombatPayload) and record.origin is RecordOrigin.PROBE:
            self.probe_forks += 1
            if payload.probed_keyword:
                self.probed_keywords[payload.probed_keyword] += 1
        if record.interventional:
            self.interventions += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        shown = histogram(self.probed_keywords.most_common())
        yield Finding.watch(
            name="probe forks were taken",
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


class LossyForks:
    """What share of the fork records came from a copy that left something
    out (``copy_dropped``), by kind, and what share of the forks. Watched: a
    lossy fork is kept and trained on, so the share is what the fork fidelity
    audit reads.

    Every record of a fork carries its copy's ``copy_dropped``, so a
    protection fork's ten records count ten times among the records. The
    wire names no fork, so the forks are counted as groups of records alike
    in game, game turn and origin (``RecordOrigin``: a probe also by the record
    it mirrors, a directed fork by its variant and operation): two forks of one
    kind taken in one turn of one game count once, while an intervention and
    a protection fork there are two, so the fork count is a lower bound and
    both figures are reported.
    """

    def __init__(self) -> None:
        self.forks = 0
        self.lossy = 0
        self.by_kind: Counter[str] = Counter()
        self.groups: dict[tuple, set[str]] = {}

    @staticmethod
    def group_of(record: EffectRecord) -> tuple:
        """The fork a record belongs to, as near as the wire says."""
        origin = record.origin
        extra: tuple = ()
        if record.directed is not None:
            extra = (record.directed.variant, record.directed.operation.value)
        elif origin is RecordOrigin.PROBE:
            extra = (record.mirror_of,)
        return (record.game_id, record.state.global_.turn, origin.value, *extra)

    def add(self, record: EffectRecord) -> None:
        if not record.fork:
            return
        self.forks += 1
        dropped = self.groups.setdefault(self.group_of(record), set())
        if record.copy_dropped:
            self.lossy += 1
            for kind, _count in record.copy_dropped:
                self.by_kind[kind] += 1
                dropped.add(kind)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        shown = ", ".join(
            f"{kind} {count}/{self.forks} ({count / self.forks:.1%})"
            for kind, count in sorted(self.by_kind.items())) or "none"
        forks = len(self.groups)
        by_fork: Counter[str] = Counter(kind for kinds in self.groups.values() for kind in kinds)
        lossy_forks = sum(1 for kinds in self.groups.values() if kinds)
        shown_forks = ", ".join(
            f"{kind} {count}/{forks} ({count / forks:.1%})"
            for kind, count in sorted(by_fork.items())) or "none"
        yield Finding.watch(
            name="fork records whose copy left something out",
            measured=(f"{self.lossy}/{self.forks} fork records; by kind: {shown}; "
                      f"{lossy_forks}/{forks} forks (at least; by game, turn and kind of fork); "
                      f"by kind: {shown_forks}"),
        )
