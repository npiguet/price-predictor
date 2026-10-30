"""The survey pass of ``build-corpus``: what every raw shard holds (FR-135).

Keyed on the provenance key rather than the ability text: folding every key to
its text needs the whole sidecar tree, so it happens once, in the main process.
Rarity, the legality mode table and the game sets count the whole corpus
(FR-141); the :class:`TrainingSupply` the plan is sized on counts
training-bound records only, so a game's records are buffered to the end of
its shard (one JVM lifetime, which no game outlives) and counted once its
route is known. Both passes read through one :class:`WorkerContext`, so the
refusal rule, the held-out test and a record's cell cannot drift apart.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.parallel_map import run_parallel
from effects.domain.corpus_curation import (
    game_hash,
    record_hash,
)
from effects.domain.corpus_manifest import SourceShard
from effects.domain.corpus_plan import (
    BuildCorpusError,
    Cell,
    GameRoute,
    GameRouting,
    Keys,
    TrainingSupply,
)
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME, qualifying_observations
from effects.domain.record_quality import QualityDefect, record_refusal
from effects.domain.records import (
    EffectRecord,
    RecordKind,
    acting_keys,
)
from effects.domain.sampling import SamplingClass, sampling_class
from effects.domain.split import HeldOutCards, record_names_held_out_card
from effects.domain.tally_merge import NOT_MERGED, merge_into

if TYPE_CHECKING:
    from effects.domain.damage_step_keywords import KeywordResolver
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)


# ── what both passes read with ──────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReadConfig:
    """How both passes read and judge a record; pickled into every worker."""

    records_dir: str
    #: Roots rather than a ``SidecarCache``, which does not pickle.
    sidecar_roots: dict[str, str]
    #: ``--max-events-per-record``.
    max_events: int
    held_out: HeldOutCards
    seed: int


class WorkerContext:
    """One process's read side: its config, sidecar cache, resolver and memo,
    built lazily and kept for the context's life, so a process reads each
    sidecar once rather than once per shard."""

    def __init__(self, read: ReadConfig) -> None:
        self.read = read
        self._sidecars: SidecarCache | None = None
        self._resolver = None
        self._held_text: dict[Keys, str | None] = {}

    @property
    def sidecars(self) -> SidecarCache:
        if self._sidecars is None:
            from effects.infrastructure.sidecar_io import SidecarCache

            self._sidecars = SidecarCache({
                name: Path(root) for name, root in self.read.sidecar_roots.items()
            })
        return self._sidecars

    @property
    def resolver(self) -> KeywordResolver:
        if self._resolver is None:
            from effects.domain.damage_step_keywords import KeywordResolver

            self._resolver = KeywordResolver(self.sidecars)
        return self._resolver

    def records(self, relative: str) -> Iterator[EffectRecord]:
        from effects.infrastructure.shard_io import read_shard

        return read_shard(Path(self.read.records_dir) / relative)

    def refusal(self, record: EffectRecord) -> QualityDefect | None:
        """The one refusal rule both passes apply."""
        return record_refusal(
            record, self.sidecars.resolution_of, max_events=self.read.max_events,
        )

    def admit(
        self, record: EffectRecord, refusals: RefusalTally,
    ) -> SamplingClass | None:
        """The record's sampling class, or None when :meth:`refusal` refuses
        it, which ``refusals`` then counts: both passes' first step."""
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
    """Which games the corpus holds, and where the survey routed them."""

    #: Games holding at least one record the quality rules keep.
    games: set[str] = field(default_factory=set)
    #: Refused records included: the game played the card either way.
    held_out: set[str] = field(default_factory=set)
    game_disjoint: set[str] = field(default_factory=set)
    #: Held-out text -> games holding a resolution record acting through it,
    #: which the card-disjoint cap counts (FR-142).
    held_out_text_games: dict[str, set[str]] = field(default_factory=dict)


@dataclass(slots=True)
class ShardSurvey:
    """One shard's contribution, or every shard's folded into
    :attr:`Survey.totals`. Plain data, so it pickles back cheaply."""

    name: str = field(metadata=NOT_MERGED)
    size: int = field(metadata=NOT_MERGED)
    supply: TrainingSupply
    records: int = 0
    rarity: RarityTally = field(default_factory=RarityTally)
    games: GameTally = field(default_factory=GameTally)
    refusals: RefusalTally = field(default_factory=RefusalTally)
    #: Records per class the quality rules keep, the refusal guard's other half.
    kept: Counter[SamplingClass] = field(default_factory=Counter)
    mode_counts: Counter[str] = field(default_factory=Counter)
    #: Training-bound records per class acting through a held-out text.
    held_out_text_dropped: Counter[SamplingClass] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class Survey:
    """What the whole corpus holds: every shard's survey folded together, and
    what only the fold can see."""

    #: Its ``name`` and ``size`` are blank; ``supply`` counts training-bound
    #: records only, its legality cells resolved by the build.
    totals: ShardSurvey
    shards: tuple[SourceShard, ...]
    #: Per top-level source directory (FR-149): held-out games in a depleted
    #: directory are a collection leak nothing else would notice.
    games_by_source: dict[str, int] = field(default_factory=dict)
    held_out_games_by_source: dict[str, int] = field(default_factory=dict)
    #: Games found in more than one shard, each planned from one shard's view
    #: and so possibly counted toward training slots it never fills.
    split_games: int = 0


# ── the pass ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    """What every survey worker needs, sent once through the pool initializer."""

    read: ReadConfig
    #: How many smallest hashes the supply keeps per cell and key.
    heap_cap: int
    routing: GameRouting
    #: In gate 2's spelling (FR-045).
    game_disjoint_keywords: tuple[str, ...] = ()


_SURVEY: tuple[SurveyConfig, WorkerContext] | None = None


def init_survey_worker(config: SurveyConfig) -> None:
    global _SURVEY
    _SURVEY = (config, WorkerContext(config.read))


def survey_shard(relative: str) -> ShardSurvey:
    """Survey one shard. Module-level so a process pool can pickle it."""
    assert _SURVEY is not None, "init_survey_worker was not run"
    config, context = _SURVEY
    return survey_shard_with(config, context, relative)


def survey_shard_with(
    config: SurveyConfig, context: WorkerContext, relative: str,
) -> ShardSurvey:
    """One shard's survey, its games routed at the end of the shard."""
    path = Path(config.read.records_dir) / relative
    surveyor = _ShardSurveyor(config, context, ShardSurvey(
        name=relative, size=path.stat().st_size, supply=TrainingSupply(config.heap_cap),
    ))
    for record in context.records(relative):
        surveyor.observe(record)
    return surveyor.finish()


@dataclass(frozen=True, slots=True)
class _PendingRecord:
    """A kept record, counted into the supply once its game's route is known."""

    game: str
    klass: SamplingClass
    cell: Cell
    keys: Keys
    value: int
    #: Never trains, even in a training game.
    acts_through_held_out_text: bool


class _ShardSurveyor:
    """One shard's survey in progress: every record observed, then every game routed."""

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
        self.keyword_games: set[str] = set()
        self.pending: list[_PendingRecord] = []

    def observe(self, record: EffectRecord) -> None:
        self.out.records += 1
        self._note_game_properties(record)
        klass = self.context.admit(record, self.out.refusals)
        if klass is None:
            return
        self.out.kept[klass] += 1
        self.pending.append(self._kept(record, klass))

    def _note_game_properties(self, record: EffectRecord) -> None:
        """The keyword threshold and the held-out test: properties of the game
        (FR-046), so read before the quality check refuses the record."""
        game = record.game_id
        if (
            self.keyword_rows and record.kind is RecordKind.COMBAT
            and game not in self.keyword_games
            and qualifying_observations(record, self.keyword_rows, self.context.resolver)
        ):
            self.keyword_games.add(game)
        if record_names_held_out_card(record, self.config.read.held_out):
            self.out.games.held_out.add(game)

    def _kept(self, record: EffectRecord, klass: SamplingClass) -> _PendingRecord:
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
        return _PendingRecord(
            game=game, klass=klass, cell=cell, keys=keys,
            value=context.hash_of(record),
            acts_through_held_out_text=held_text is not None,
        )

    def finish(self) -> ShardSurvey:
        self._fold_training_supply(self._route_games())
        return self.out

    def _route_games(self) -> dict[str, GameRoute]:
        games = self.out.games
        routes = {
            game: self.config.routing.route(
                game, held_out=game in games.held_out,
                keyword_combat=game in self.keyword_games,
            )
            for game in games.games
        }
        games.game_disjoint = {
            game for game, route in routes.items() if route is GameRoute.GAME_DISJOINT
        }
        return routes

    def _fold_training_supply(self, routes: Mapping[str, GameRoute]) -> None:
        for pending in self.pending:
            if routes[pending.game] is not GameRoute.TRAINING:
                continue
            if pending.acts_through_held_out_text:
                self.out.held_out_text_dropped[pending.klass] += 1
                continue
            self.out.supply.add(pending.cell, pending.keys, pending.value)


def source_of(relative: str) -> str:
    """The top-level directory a raw shard sits in, ``"."`` at the root."""
    return relative.split("/", 1)[0] if "/" in relative else "."


def merge_surveys(parts: Iterable[ShardSurvey], *, heap_cap: int) -> Survey:
    """Combine shard surveys into one corpus-wide picture."""
    total = ShardSurvey(name="", size=0, supply=TrainingSupply(heap_cap))
    shards: list[SourceShard] = []
    games_by_source: dict[str, set[str]] = defaultdict(set)
    held_by_source: dict[str, set[str]] = defaultdict(set)
    split_games = 0

    for part in parts:
        shards.append(SourceShard(name=part.name, size=part.size))
        split_games += len(part.games.games & total.games.games)
        merge_into(total, part)
        source = source_of(part.name)
        games_by_source[source] |= part.games.games
        held_by_source[source] |= part.games.held_out

    return Survey(
        totals=total,
        shards=tuple(sorted(shards, key=lambda s: s.name)),
        games_by_source={s: len(g) for s, g in games_by_source.items()},
        held_out_games_by_source={s: len(g) for s, g in held_by_source.items()},
        split_games=split_games,
    )


def shard_names(records_dir: Path) -> list[str]:
    """Every shard under the corpus root, as posix paths relative to it."""
    from effects.infrastructure.shard_io import iter_shards

    root = Path(records_dir)
    return [shard.relative_to(root).as_posix() for shard in iter_shards(root)]


def run_survey(
    records_dir: Path,
    *,
    config: SurveyConfig,
    workers: int | None = None,
    progress_every: int = 50,
) -> Survey:
    """Survey every shard, in parallel, reporting progress as it goes."""
    root = Path(records_dir)
    names = shard_names(root)
    if not names:
        raise BuildCorpusError(f"{root}: no shards to build a corpus from")
    logger.info("Surveying %d shard(s) under %s", len(names), root)
    return merge_surveys(
        run_parallel(
            names, survey_shard, initializer=init_survey_worker, initargs=(config,),
            workers=workers, label="Surveyed", progress_every=progress_every,
        ),
        heap_cap=config.heap_cap,
    )
