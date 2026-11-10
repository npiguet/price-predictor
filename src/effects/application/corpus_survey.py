"""The survey pass of ``build-corpus``: what every raw shard holds (FR-135).

Keyed on the provenance key rather than the ability text: folding every key to
its text needs the whole sidecar tree, so it happens once, in the main process.
Rarity, the legality mode table, the rule counts, the closed lists and the
game sets count the whole corpus (FR-141). Every record is tagged with the
catalogue's detectors (``tag_record``) as ``rule-counts`` tags it.

Where a game goes is decided once, over the whole corpus, after the survey
(:func:`effects.domain.stratum_placement.place_games`; research R3), so a
shard's records are buffered to the end of the shard (one JVM lifetime, which
no game outlives) and then split by what placement could still do with their
game:

- a game naming no held-out card whose game-disjoint draw keeps it out of the
  candidate pool, or that has a random seat, trains whatever placement
  decides: its records fold into the :class:`TrainingSupply` here, and each
  tagged one is kept as a :class:`SurveyedRecord` for the rule budgets;
- a candidate game, a Forge-AI game of the pool or naming a held-out card,
  keeps every kept record as a :class:`SurveyedRecord` and its rule inventory,
  and folds into the supply only once placement trains it;
- a random-seat game naming a held-out card is dropped and keeps nothing.

``turn_action`` records are written to no stratum (FR-059): the survey counts
them by subkind and reads nothing else of them but their game's properties.
Both passes read through one :class:`WorkerContext`, so the refusal rule, the
held-out test, the tags and a record's cell cannot drift apart.
"""

from __future__ import annotations

import functools
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.parallel_map import run_parallel
from effects.application.rule_counts import RuleCounts
from effects.domain.closed_lists import keywords_named, subtypes_named
from effects.domain.corpus_curation import (
    CONDITION_CLASSES,
    condition_key,
    game_hash,
    record_hash,
    turn_step_condition,
)
from effects.domain.corpus_manifest import SourceShard
from effects.domain.corpus_plan import (
    BuildCorpusError,
    Cell,
    Keys,
    TrainingSupply,
)
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME, qualifying_observations
from effects.domain.effect_targets import derive_targets
from effects.domain.protection_copies import (
    PROTECTION_RULES,
    protection_copy,
    qualify,
    rules_offered,
)
from effects.domain.record_quality import QualityDefect, record_refusal
from effects.domain.records import (
    EffectRecord,
    RecordKind,
    acting_keys,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleSource, RuleTag, TagResult
from effects.domain.rule_output_carriage import carried_outputs
from effects.domain.sampling import SamplingClass, sampling_class
from effects.domain.split import HeldOutCards, record_names_held_out_card
from effects.domain.stratum_placement import PlacementRule, RuleCell, rule_test_eligible
from effects.domain.tally_merge import NOT_MERGED, merge_into

if TYPE_CHECKING:
    from effects.domain.damage_step_keywords import KeywordResolver
    from effects.domain.rule_catalogue import ScenarioCatalogue
    from effects.domain.rule_detectors import DetectorIndex
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)


# ── what both passes read with ──────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReadConfig:
    """How both passes read and judge a record; pickled into every worker."""

    records_dir: Path
    #: Roots rather than a ``SidecarCache``, which does not pickle.
    sidecar_roots: dict[str, Path]
    #: ``--max-events-per-record``.
    max_events: int
    held_out: HeldOutCards
    seed: int
    #: ``rules/scenarios/`` as the build read it once (FR-060a); None tags
    #: nothing, as a catalogue declaring no detector would.
    catalogue: ScenarioCatalogue | None = None
    #: ``rule -> outputs`` the map names (FR-060a): what the survey checks
    #: each tagged record carries (``rule_output_carriage``).
    rule_outputs: dict[str, tuple[str, ...]] = field(default_factory=dict)


def protection_hash(record_id: str, *, seed: int) -> int:
    """The hash a record's protection copies (FR-058) are planned under: a
    second, distinct entry beside the record's own in the rule budgets."""
    return record_hash(f"{record_id}~pc", seed=seed)


class WorkerContext:
    """One process's read side: its config, sidecar cache, resolver, detector
    index and memo, built lazily and kept for the context's life, so a process
    reads each sidecar once rather than once per shard. The cache is the
    process's own (:func:`~effects.infrastructure.converted_trees.process_sidecars`),
    so every context over the same roots in one process shares it."""

    def __init__(self, read: ReadConfig) -> None:
        self.read = read
        self._sidecars: SidecarCache | None = None
        self._resolver = None
        self._index: DetectorIndex | None = None
        self._held_text: dict[Keys, str | None] = {}

    @property
    def sidecars(self) -> SidecarCache:
        if self._sidecars is None:
            from effects.infrastructure.converted_trees import process_sidecars

            self._sidecars = process_sidecars(self.read.sidecar_roots)
        return self._sidecars

    @property
    def resolver(self) -> KeywordResolver:
        if self._resolver is None:
            from effects.domain.damage_step_keywords import KeywordResolver

            self._resolver = KeywordResolver(self.sidecars)
        return self._resolver

    def tags(self, record: EffectRecord) -> TagResult:
        """The rules ``record`` shows, by the catalogue's detectors."""
        if self.read.catalogue is None:
            return TagResult()
        if self._index is None:
            from effects.domain.rule_detectors import DetectorIndex

            self._index = DetectorIndex.of(self.read.catalogue)
        from effects.domain.rule_detectors import tag_record

        return tag_record(record, self._index, self.sidecars)

    def records(self, shard: SourceShard) -> Iterator[EffectRecord]:
        from effects.infrastructure.shard_io import read_shard

        return read_shard(shard.path(self.read.records_dir))

    def refusal(self, record: EffectRecord) -> QualityDefect | None:
        """The one refusal rule both passes apply."""
        return record_refusal(
            record, self.sidecars.resolution_of, max_events=self.read.max_events,
        )

    def admit(
        self, record: EffectRecord, refusals: RefusalTally,
    ) -> SamplingClass | None:
        """The record's sampling class, or None when :meth:`refusal` refuses
        it, which ``refusals`` then counts: both passes' first step after
        setting ``turn_action`` records aside, which have no class."""
        klass = sampling_class(record)
        defect = self.refusal(record)
        if defect is not None:
            refusals.note(record, klass, defect)
            return None
        return klass

    def cell_of(
        self, record: EffectRecord, mode_counts: Mapping[str, int] | None = None,
    ) -> Cell:
        return Cell.of(record, self.sidecars, self.resolver, mode_counts)

    def hash_of(self, record: EffectRecord) -> int:
        return record_hash(record.record_id, seed=self.read.seed)

    def protection_hash_of(self, record: EffectRecord) -> int:
        return protection_hash(record.record_id, seed=self.read.seed)

    def held_out_text(self, keys: Keys) -> str | None:
        """The held-out script text these acting keys resolve to, normalized,
        or None. Always the script surface, whatever ``--vocab-path`` says,
        since that is the surface the holdout is chosen on."""
        if not keys or not self.read.held_out.keys:
            return None
        if keys not in self._held_text:
            self._held_text[keys] = self.read.held_out.text_acted_through(
                keys, self.sidecars.script_text,
            )
        return self._held_text[keys]


# ── the tallies ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class RefusalTally:
    """Records the quality rules refused (FR-148), by reason, by sampling class
    (the axis ``MAX_REFUSED_SHARE`` is judged on) and, for the two join
    refusals, by script, so a stale sidecar names the cards it costs."""

    by_reason: Counter[QualityDefect] = field(default_factory=Counter)
    by_class: Counter[SamplingClass] = field(default_factory=Counter)
    textless_scripts: Counter[str] = field(default_factory=Counter)
    unjoinable_scripts: Counter[str] = field(default_factory=Counter)

    def note(
        self, record: EffectRecord, klass: SamplingClass, defect: QualityDefect,
    ) -> None:
        self.by_reason[defect] += 1
        self.by_class[klass] += 1
        if defect is QualityDefect.NO_ACTING_TEXT:
            self.textless_scripts[acting_keys(record)[0].script_file] += 1
        elif defect is QualityDefect.UNJOINABLE_KEY:
            self.unjoinable_scripts[acting_keys(record)[0].script_file] += 1


@dataclass(slots=True)
class RarityTally:
    """Records and distinct games per acting-key tuple, over the whole corpus
    rather than the training stratum (FR-141)."""

    key_records: Counter[Keys] = field(default_factory=Counter)
    key_games: dict[Keys, set[int]] = field(default_factory=dict)

    def note(self, keys: Keys, game_id: str) -> None:
        self.key_records[keys] += 1
        self.key_games.setdefault(keys, set()).add(game_hash(game_id))


@dataclass(slots=True)
class GameTally:
    """Which games the corpus holds, and what placement reads about each."""

    #: Games holding at least one record the quality rules keep.
    games: set[str] = field(default_factory=set)
    #: Refused records included: the game played the card either way.
    held_out: set[str] = field(default_factory=set)
    #: Games any of whose records carries ``random_seat``.
    random_seat: set[str] = field(default_factory=set)
    #: Games holding a qualifying combat for a ``--game-disjoint-keywords`` keyword.
    keyword_combat: set[str] = field(default_factory=set)
    #: Held-out text -> games holding a resolution record acting through it,
    #: which the card-disjoint cap counts (FR-142).
    held_out_text_games: dict[str, set[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SurveyedRecord:
    """One kept record as the rule budgets, placement and the supply fold read it.

    ``cell`` is the survey's, its legality modes not yet folded; ``tags``
    every rule the detectors tag, budgeted or not (an unbudgeted rule is still
    counted); ``protection`` the budgeted protection rules the record offers
    (FR-058), planned under ``protection_hash`` as a record of their own.
    """

    game: str
    record_hash: int
    klass: SamplingClass
    cell: Cell
    keys: Keys
    tags: tuple[tuple[str, Polarity], ...]
    source: RuleSource
    #: Acts through a held-out text: never trains, even in a training game.
    held_out_text: bool = False
    protection: tuple[str, ...] = ()
    protection_hash: int = 0


@dataclass(slots=True)
class ShardSurvey:
    """One shard's contribution, or every shard's folded into
    :attr:`Survey.totals`. Plain data, so it pickles back cheaply."""

    shard: SourceShard = field(metadata=NOT_MERGED)
    #: Records of games that train whatever placement decides.
    supply: TrainingSupply
    records: int = 0
    rarity: RarityTally = field(default_factory=RarityTally)
    games: GameTally = field(default_factory=GameTally)
    refusals: RefusalTally = field(default_factory=RefusalTally)
    #: Records per class the quality rules keep, the refusal guard's other half.
    kept: Counter[SamplingClass] = field(default_factory=Counter)
    mode_counts: Counter[str] = field(default_factory=Counter)
    #: Records per class acting through a held-out text in a game naming no
    #: held-out card: never trained, wherever the game goes.
    held_out_text_dropped: Counter[SamplingClass] = field(default_factory=Counter)
    #: Tagged or protection-offering records of games that train whatever
    #: placement decides, held-out-text ones left out.
    tagged: list[SurveyedRecord] = field(default_factory=list)
    #: Every kept record of a candidate game.
    candidates: list[SurveyedRecord] = field(default_factory=list)
    #: ``(game, {(rule, polarity): records})`` per Forge-AI candidate game,
    #: over budgeted rules and rule-test-eligible tags.
    inventories: list[tuple[str, dict[RuleCell, int]]] = field(default_factory=list)
    #: Every record read, tagged as ``rule-counts`` tags it (FR-008).
    rule_counts: RuleCounts = field(default_factory=RuleCounts)
    #: ``(rule, output) -> tagged records carrying it`` over the map's
    #: outputs (``rule_output_carriage``): what makes an output effective.
    carried: Counter[tuple[str, str]] = field(default_factory=Counter)
    #: ``turn_action`` records by subkind, written to no stratum (FR-059).
    turn_actions: Counter[str] = field(default_factory=Counter)
    #: ``corpus_curation.condition_key`` -> kept records (FR-056).
    condition_counts: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    #: The closed lists' names (FR-052, FR-053), read off every kept record.
    subtypes: set[str] = field(default_factory=set)
    keywords: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class Survey:
    """What the whole corpus holds: every shard's survey folded together, and
    what only the fold can see."""

    #: Its ``shard`` is blank; ``supply`` counts the records of games that
    #: train whatever placement decides, its legality cells folded into their
    #: families, which only the whole corpus's mode counts can decide
    #: (:func:`merge_surveys`). Candidate games fold in after placement.
    totals: ShardSurvey
    shards: tuple[SourceShard, ...]
    #: Per top-level source directory (FR-149): held-out games in a depleted
    #: directory are a collection leak nothing else would notice.
    games_by_source: dict[str, int] = field(default_factory=dict)
    held_out_games_by_source: dict[str, int] = field(default_factory=dict)
    #: Games found in more than one shard, each planned from one shard's view
    #: and so possibly counted toward training slots it never fills.
    split_games: int = 0
    #: Shard name -> the games it holds a kept record of: what the write pass
    #: hands each shard's worker of the copy map.
    shard_games: dict[str, frozenset[str]] = field(default_factory=dict)


# ── the pass ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    """What every survey worker needs, sent once to each process."""

    read: ReadConfig
    #: How many smallest hashes the supply keeps per cell and key.
    heap_cap: int
    #: The candidate pool's draw (``PlacementRule.in_pool``).
    placement: PlacementRule
    #: In gate 2's spelling (FR-045).
    game_disjoint_keywords: tuple[str, ...] = ()
    #: The rules the map marks ``trained``: what the inventories count and
    #: which protection copies can be planned (FR-054, FR-058).
    budgeted: frozenset[str] = frozenset()
    #: Forge's keywords in ``normalize_keyword``'s spelling, for the closed
    #: keyword list (``closed_lists.keywords_named``).
    known_keywords: frozenset[str] = frozenset()


class SurveyWorker:
    """One process's survey side: the config, and the read context every shard
    it surveys shares. Built once per process by :func:`run_parallel`."""

    def __init__(self, config: SurveyConfig) -> None:
        self.config = config
        self.context = WorkerContext(config.read)

    def survey(self, shard: SourceShard) -> ShardSurvey:
        """One shard's survey, its games split at the end of the shard."""
        surveyor = _ShardSurveyor(self.config, self.context, ShardSurvey(
            shard=shard, supply=TrainingSupply(self.config.heap_cap),
        ))
        for record in self.context.records(shard):
            surveyor.observe(record)
        return surveyor.finish()


class _Fate(StrEnum):
    """What placement can still do with a game, as the survey sees it."""

    TRAINS = "trains"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class _PendingRecord:
    """A kept record, split once its game's fate and source are known."""

    game: str
    klass: SamplingClass
    cell: Cell
    keys: Keys
    value: int
    #: Never trains, even in a training game.
    acts_through_held_out_text: bool
    tags: tuple[RuleTag, ...]
    #: Rule-test-eligible ``(rule, polarity)`` cells over budgeted rules.
    eligible: tuple[RuleCell, ...]
    aimed: bool
    protection: tuple[str, ...]
    protection_hash: int


class _ShardSurveyor:
    """One shard's survey in progress: every record observed, then every game split."""

    def __init__(
        self, config: SurveyConfig, context: WorkerContext, out: ShardSurvey,
    ) -> None:
        self.config = config
        self.context = context
        self.out = out
        self.keyword_rows = tuple(
            KEYWORDS_BY_NAME[name] for name in config.game_disjoint_keywords
            if name in KEYWORDS_BY_NAME
        )
        self.protection_rules = frozenset(config.budgeted) & PROTECTION_RULES
        self.pending: list[_PendingRecord] = []
        #: ``(game, aimed, tags)`` of every tagged record read, refused ones
        #: included, counted once the game's source is known.
        self.counted: list[tuple[str, bool, tuple[RuleTag, ...]]] = []

    def observe(self, record: EffectRecord) -> None:
        self.out.records += 1
        self._note_game_properties(record)
        found = self._count_tags(record)
        if record.kind is RecordKind.TURN_ACTION:
            self.out.turn_actions[str(record.subkind)] += 1
            return
        klass = self.context.admit(record, self.out.refusals)
        if klass is None:
            return
        self.out.kept[klass] += 1
        self.pending.append(self._kept(record, klass, found))

    def _note_game_properties(self, record: EffectRecord) -> None:
        """The keyword threshold, the random seat and the held-out test:
        properties of the game (FR-046), so read before the quality check
        refuses the record."""
        game, games = record.game_id, self.out.games
        if record.random_seat:
            games.random_seat.add(game)
        if (
            self.keyword_rows and record.kind is RecordKind.COMBAT
            and game not in games.keyword_combat
            and qualifying_observations(record, self.keyword_rows, self.context.resolver)
        ):
            games.keyword_combat.add(game)
        if record_names_held_out_card(record, self.config.read.held_out):
            games.held_out.add(game)

    def _count_tags(self, record: EffectRecord) -> TagResult:
        found = self.context.tags(record)
        counts = self.out.rule_counts
        counts.not_countable |= found.not_countable
        counts.review |= {(record.record_id, tag.scenario) for tag in found.review}
        if found.tags:
            self.counted.append((record.game_id, record.aimed_at is not None, found.tags))
            self._note_carried(record, found.tags)
        return found

    def _note_carried(self, record: EffectRecord, tags: tuple[RuleTag, ...]) -> None:
        """Which of each tagged rule's map outputs ``record`` carries."""
        if record.kind is RecordKind.TURN_ACTION:
            return
        outputs = self.config.read.rule_outputs
        targets = None
        for tag in tags:
            wanted = outputs.get(tag.rule)
            if not wanted:
                continue
            if targets is None:
                targets = derive_targets(record)
            for output in carried_outputs(record, tag.outcome, wanted, targets=targets):
                self.out.carried[tag.rule, output] += 1

    def _kept(
        self, record: EffectRecord, klass: SamplingClass, found: TagResult,
    ) -> _PendingRecord:
        """Count a kept record into the whole-corpus tallies; return it pending."""
        out, context, game = self.out, self.context, record.game_id
        out.games.games.add(game)
        cell = context.cell_of(record)
        out.mode_counts.update(cell.modes)
        keys = acting_keys(record)
        if keys:
            out.rarity.note(keys, game)
        held_text = context.held_out_text(keys)
        if held_text is not None and record.kind is RecordKind.RESOLUTION:
            out.games.held_out_text_games.setdefault(held_text, set()).add(game)
        self._note_lists(record, klass)
        protection = self._protection(record)
        return _PendingRecord(
            game=game, klass=klass, cell=cell, keys=keys,
            value=context.hash_of(record),
            acts_through_held_out_text=held_text is not None,
            tags=found.tags,
            eligible=tuple(
                (tag.rule, tag.polarity) for tag in found.tags
                if tag.rule in self.config.budgeted and rule_test_eligible(record, tag)
            ),
            aimed=record.aimed_at is not None,
            protection=protection,
            protection_hash=context.protection_hash_of(record) if protection else 0,
        )

    def _note_lists(self, record: EffectRecord, klass: SamplingClass) -> None:
        """The condition counts and the closed lists (FR-052, FR-053, FR-056)."""
        out = self.out
        if klass in CONDITION_CLASSES:
            condition = turn_step_condition(record, self.context.sidecars)
            if condition is not None:
                out.condition_counts[condition_key(klass, condition)] += 1
        out.subtypes |= subtypes_named(record)
        out.keywords |= keywords_named(record, self.config.known_keywords)

    def _protection(self, record: EffectRecord) -> tuple[str, ...]:
        """The budgeted protection rules ``record``'s copies can be drawn for,
        each counted toward the outputs a copy drawn for it carries."""
        if not self.protection_rules:
            return ()
        opportunities = qualify(record)
        offered = tuple(rule for rule in rules_offered(opportunities)
                        if rule in self.protection_rules)
        for rule in offered:
            self._note_protection_carried(record, opportunities, rule)
        return offered

    def _note_protection_carried(self, record: EffectRecord, opportunities, rule: str) -> None:
        """The outputs one protection copy of ``record`` drawn for ``rule``
        carries: the copies are synthetic, so no detector tags them, and
        their rule's effective outputs come from here."""
        wanted = self.config.read.rule_outputs.get(rule)
        if not wanted:
            return
        made = protection_copy(record, opportunities, rule=rule, rng=random.Random(0), index=0)
        if made is None:
            return
        for output in carried_outputs(made.record, made.outcome_entities, wanted):
            self.out.carried[rule, output] += 1

    def finish(self) -> ShardSurvey:
        seat_games = self.out.games.random_seat
        for game, aimed, tags in self.counted:
            source = RuleSource.of(aimed=aimed, random_seat_game=game in seat_games)
            for tag in tags:
                self.out.rule_counts.counts[tag.rule, str(tag.polarity), str(source)] += 1
        self.out.rule_counts.records = self.out.records
        candidates = self._candidate_games()
        inventories: dict[str, Counter[RuleCell]] = defaultdict(Counter)
        held_out = self.out.games.held_out
        for pending in self.pending:
            fate = self._fate(pending.game, candidates)
            surveyed = self._surveyed(pending, seat_games)
            if pending.acts_through_held_out_text and pending.game not in held_out:
                self.out.held_out_text_dropped[pending.klass] += 1
            if fate is _Fate.CANDIDATE:
                self.out.candidates.append(surveyed)
                inventories[pending.game].update(pending.eligible)
            elif not pending.acts_through_held_out_text:
                self.out.supply.add(pending.cell, pending.keys, pending.value)
                if surveyed.tags or surveyed.protection:
                    self.out.tagged.append(surveyed)
        self.out.inventories = [
            (game, dict(inventory)) for game, inventory in sorted(inventories.items())
            if inventory
        ]
        return self.out

    def _candidate_games(self) -> frozenset[str]:
        """Games placement may put in a validation stratum: every one naming a
        held-out card, random-seat games included (FR-059), and every other
        Forge-AI game of the candidate pool; the game-disjoint stratum keeps
        Forge-AI games only."""
        games, placement = self.out.games, self.config.placement
        return frozenset(
            game for game in games.games
            if game in games.held_out or (
                game not in games.random_seat
                and placement.in_pool(game, keyword_combat=game in games.keyword_combat)
            )
        )

    def _fate(self, game: str, candidates: frozenset[str]) -> _Fate:
        return _Fate.CANDIDATE if game in candidates else _Fate.TRAINS

    def _surveyed(self, pending: _PendingRecord, seat_games: set[str]) -> SurveyedRecord:
        return SurveyedRecord(
            game=pending.game, record_hash=pending.value, klass=pending.klass,
            cell=pending.cell, keys=pending.keys,
            tags=tuple((tag.rule, tag.polarity) for tag in pending.tags),
            source=RuleSource.of(aimed=pending.aimed,
                                 random_seat_game=pending.game in seat_games),
            held_out_text=pending.acts_through_held_out_text,
            protection=pending.protection, protection_hash=pending.protection_hash,
        )


def merge_surveys(parts: Iterable[ShardSurvey], *, heap_cap: int) -> Survey:
    """Combine shard surveys into one corpus-wide picture, the supply's
    legality cells folded once every shard's mode counts are in."""
    total = ShardSurvey(shard=SourceShard(name="", size=0), supply=TrainingSupply(heap_cap))
    shards: list[SourceShard] = []
    games_by_source: dict[str, set[str]] = defaultdict(set)
    held_by_source: dict[str, set[str]] = defaultdict(set)
    shard_games: dict[str, frozenset[str]] = {}
    split_games = 0

    for part in parts:
        shards.append(part.shard)
        split_games += len(part.games.games & total.games.games)
        merge_into(total, part)
        source = part.shard.source
        games_by_source[source] |= part.games.games
        held_by_source[source] |= part.games.held_out
        shard_games[part.shard.name] = frozenset(part.games.games)
    total.supply = total.supply.resolved(total.mode_counts)

    return Survey(
        totals=total,
        shards=tuple(sorted(shards, key=lambda s: s.name)),
        games_by_source={s: len(g) for s, g in games_by_source.items()},
        held_out_games_by_source={s: len(g) for s, g in held_by_source.items()},
        split_games=split_games,
        shard_games=shard_games,
    )


def run_survey(
    records_dir: Path,
    *,
    config: SurveyConfig,
    workers: int | None = None,
    progress_every: int = 50,
) -> Survey:
    """Survey every shard, in parallel, reporting progress as it goes."""
    from effects.infrastructure.shard_io import list_shards

    root = Path(records_dir)
    shards = list_shards(root)
    if not shards:
        raise BuildCorpusError(f"{root}: no shards to build a corpus from")
    logger.info("Surveying %d shard(s) under %s", len(shards), root)
    return merge_surveys(
        run_parallel(
            shards, SurveyWorker.survey, context=functools.partial(SurveyWorker, config),
            workers=workers, label="Surveyed", progress_every=progress_every,
        ),
        heap_cap=config.heap_cap,
    )
