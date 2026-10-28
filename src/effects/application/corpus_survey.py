"""The survey pass of ``build-corpus``: what every raw shard holds (FR-135).

The survey keys on the provenance key rather than the ability text on purpose.
Resolving a key to its text needs a ``SidecarCache`` over tens of thousands of
sidecar files; each worker builds one lazily for the refusal rule and the
held-out test, which touch only the scripts its shards name, while the full
fold of every key to its text happens once, in the main process.

What it counts falls in two populations, kept apart on purpose:

- **The whole corpus** -- every record the quality rules keep: rarity (FR-141
  wants effective games over the whole raw corpus), the legality mode table,
  the games and the held-out games, and the held-out texts' games.
- **The training-bound records** alone -- the :class:`TrainingSupply` every
  class budget, family share and copy plan is sized on. A game's records are
  buffered to the end of its shard and counted only once the game is known to
  route to training: a raw shard is one JVM lifetime, so a game never spans
  two, and :class:`GameRouting` decides a game from the game alone. A
  supply counted over held-out and game-disjoint games too planned training
  slots the write pass then routed away.

Both passes read their shards through :func:`read_shard_remapped`, which
repairs the old collector's token keys in the raw JSON before a record is built
(FR-151), and through one :class:`WorkerContext`, so the refusal rule, the
held-out test and a record's cell cannot drift apart between them.
"""

from __future__ import annotations

import logging
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.corpus_curation import (
    SIGNATURE_CLASSES,
    outcome_signature,
    record_hash,
    signature_label,
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
from effects.domain.record_quality import (
    NO_ACTING_TEXT,
    UNJOINABLE_KEY,
    acting_text_defect,
    quality_defect,
    unjoinable_key_defect,
)
from effects.domain.records import (
    EffectRecord,
    LegalityDecision,
    RecordKind,
    acting_keys,
)
from effects.domain.sampling import CLASS_PLAYABILITY_LEGALITY, sampling_class
from effects.domain.split import HeldOutCards, record_names_held_out_card
from effects.domain.token_key_remap import RemapCounts, TokenKeyRemapper, remap_record_dict

if TYPE_CHECKING:
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)


def game_hash(game_id: str) -> int:
    """A stable 32-bit id for a game, for counting distinct games cheaply.

    Rarity needs the *number* of distinct games per text, not their names, and
    holding tens of millions of game-id strings across worker results costs
    hundreds of megabytes for a figure that is a cardinality. ``crc32`` rather
    than the built-in ``hash()``, which is salted per process (FR-088a).
    """
    return zlib.crc32(game_id.encode("utf-8"))


def read_shard_remapped(
    path: Path, remapper: TokenKeyRemapper | None, counts: RemapCounts,
) -> Iterator[EffectRecord]:
    """Every complete record of one shard, its old token keys remapped first.

    The remap works on the raw JSON before the record is built, so both
    passes see the corrected keys and nothing downstream has to know they
    were ever wrong: the survey's rarity entry, the per-text cap it sizes and
    the written shard all name the token script. Reading through the record
    and rewriting it afterwards would not do -- ``ProvenanceKey`` is frozen
    and the remap needs the entity dicts the record drops.

    ``remapper is None`` (``--no-remap-token-keys``) is a plain read, and
    ``counts`` is then left alone.
    """
    from effects.infrastructure.record_io import iter_record_dicts, record_from_dict

    for data in iter_record_dicts(path):
        if remapper is not None:
            remap_record_dict(data, remapper, counts)
        yield record_from_dict(data)


def selection_half(record: EffectRecord, klass: str) -> str:
    """``real`` / ``what-if`` on a legality record, empty on every other.

    The two halves of the legality class (FR-049 step 3): a gen-1 record,
    whose decision is unknown, selects as a what-if.
    """
    if klass != CLASS_PLAYABILITY_LEGALITY:
        return ""
    if LegalityDecision.of(record) is LegalityDecision.REAL:
        return LegalityDecision.REAL
    return LegalityDecision.WHAT_IF


def cell_of(
    record: EffectRecord, sidecars, resolver,
    legality_mode_counts: Mapping[str, int] | None = None,
) -> Cell:
    """The cell a record is selected in. Both passes call this one function.

    ``random_seat`` is deliberately not a component (FR-050): off-policy
    records compete for the same share as on-policy ones, and the manifest
    reports how many of each were written instead.

    A legality record's family is its rarest mode, which only the whole
    corpus can say. Without ``legality_mode_counts`` -- the survey -- its cell
    carries the record's modes, which :meth:`Cell.resolved` folds once the
    counts exist; with them -- the write pass -- the cell comes back resolved.
    """
    from effects.domain.rule_families import NONE, legality_modes, rule_family

    klass = sampling_class(record)
    half = selection_half(record, klass)
    if klass == CLASS_PLAYABILITY_LEGALITY:
        modes = legality_modes(record, sidecars)
        cell = Cell(klass=klass, half=half, family="" if modes else NONE, modes=modes)
        return cell if legality_mode_counts is None else cell.resolved(legality_mode_counts)
    return Cell(
        klass=klass,
        half=half,
        family=rule_family(record, sidecars, resolver=resolver),
        signature=(
            signature_label(outcome_signature(record))
            if klass in SIGNATURE_CLASSES else ""
        ),
    )


# ── what both passes read with ──────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReadConfig:
    """How both passes read and judge a record; pickled into every worker."""

    records_dir: str
    #: Tree name -> converted root, so a worker can build its own
    #: SidecarCache: a cache holds open-ended per-card state and does not
    #: pickle.
    sidecar_roots: dict[str, str]
    #: ``--max-events-per-record``, passed through to ``quality_defect``.
    max_events: int
    held_out: HeldOutCards
    seed: int
    #: The old-token-key remap, or None when it is off (FR-151). A plain
    #: object of dicts and frozensets, so it pickles with the rest.
    remapper: TokenKeyRemapper | None = None


class WorkerContext:
    """One process's read side: its config, sidecar cache, resolver and memo.

    Built once per worker process by the pool initializers, and by the main
    process for a ``workers=1`` run and for the validation samples. The cache
    and the keyword resolver are built on first use and live as long as the
    context, so a process reads each sidecar once rather than once per shard,
    and a context built for one build never answers for another's tree.
    """

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
    def resolver(self):
        """The combat keyword resolver; it memoizes per provenance key."""
        if self._resolver is None:
            from effects.domain.damage_step_keywords import KeywordResolver

            self._resolver = KeywordResolver(self.sidecars)
        return self._resolver

    def records(self, relative: str, counts: RemapCounts) -> Iterator[EffectRecord]:
        return read_shard_remapped(
            Path(self.read.records_dir) / relative, self.read.remapper, counts,
        )

    def refusal(self, record: EffectRecord) -> str | None:
        """The one refusal rule both passes apply, so they cannot drift apart."""
        defect = quality_defect(record, max_events=self.read.max_events)
        if defect is not None:
            return defect
        resolve = self.sidecars.resolution_of
        defect = unjoinable_key_defect(record, resolve)
        if defect is not None:
            return defect
        return acting_text_defect(record, resolve)

    def cell_of(
        self, record: EffectRecord, mode_counts: Mapping[str, int] | None = None,
    ) -> Cell:
        return cell_of(record, self.sidecars, self.resolver, mode_counts)

    def hash_of(self, record: EffectRecord) -> int:
        return record_hash(record.record_id, seed=self.read.seed)

    def held_out_text(self, keys: Keys | None) -> str | None:
        """The held-out script text these acting keys resolve to, or None.

        Normalized, as the holdout keys its texts. On the script surface
        whatever ``--vocab-path`` says, because that is the surface the
        holdout is chosen on, and by the holdout's unit, so a variant or a
        token printing a text whose template is held out is held out too.
        Memoized per key tuple: a corpus repeats the same cards millions of
        times.
        """
        if not keys or not self.read.held_out.keys:
            return None
        if keys in self._held_text:
            return self._held_text[keys]
        from effects.domain.text_holdout import normalize_script_text

        found = None
        for key in keys:
            line = self.sidecars.line_or_none(key)
            script = line.script_text if line is not None else None
            if script and self.read.held_out.holds_text(script):
                found = normalize_script_text(script)
                break
        self._held_text[keys] = found
        return found


# ── the tallies ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class RefusalTally:
    """Records the quality rules refused (FR-148), as both passes count them.

    By reason, and by sampling class -- the axis ``MAX_REFUSED_SHARE`` is
    judged on, since a corpus can be a few percent junk overall and still have
    lost one whole class -- and the two join refusals by the acting key's
    script, so a converter regression or a stale sidecar names the cards it is
    costing.
    """

    by_reason: Counter[str] = field(default_factory=Counter)
    by_class: Counter[str] = field(default_factory=Counter)
    textless_scripts: Counter[str] = field(default_factory=Counter)
    unjoinable_scripts: Counter[str] = field(default_factory=Counter)

    def note(self, record: EffectRecord, klass: str, defect: str) -> None:
        self.by_reason[defect] += 1
        self.by_class[klass] += 1
        if defect == NO_ACTING_TEXT:
            self.textless_scripts[acting_keys(record)[0].script_file] += 1
        elif defect == UNJOINABLE_KEY:
            self.unjoinable_scripts[acting_keys(record)[0].script_file] += 1

    def merge(self, other: RefusalTally) -> None:
        self.by_reason.update(other.by_reason)
        self.by_class.update(other.by_class)
        self.textless_scripts.update(other.textless_scripts)
        self.unjoinable_scripts.update(other.unjoinable_scripts)


@dataclass(slots=True)
class RarityTally:
    """Records and distinct games per acting-key tuple, over the whole corpus.

    The whole corpus rather than the training stratum (FR-141): rarity is how
    often the game shows an ability at all, and the coverage and variant
    collectors exist to raise exactly the counts a training-only tally would
    understate.
    """

    key_records: Counter[Keys] = field(default_factory=Counter)
    key_games: dict[Keys, set[int]] = field(default_factory=dict)

    def note(self, keys: Keys, game_id: str) -> None:
        self.key_records[keys] += 1
        self.key_games.setdefault(keys, set()).add(game_hash(game_id))

    def merge(self, other: RarityTally) -> None:
        self.key_records.update(other.key_records)
        for keys, games in other.key_games.items():
            self.key_games.setdefault(keys, set()).update(games)


@dataclass(slots=True)
class GameTally:
    """Which games the corpus holds, and where the survey routed them."""

    #: Games holding at least one record the quality rules keep.
    games: set[str] = field(default_factory=set)
    #: Games naming a held-out card, refused records included: a game that
    #: played a held-out card played it whether or not the record saying so
    #: survives the rules.
    held_out: set[str] = field(default_factory=set)
    #: Games :class:`GameRouting` placed in the game-disjoint stratum.
    game_disjoint: set[str] = field(default_factory=set)
    #: Held-out text -> games holding a resolution record acting through it,
    #: which the card-disjoint cap counts (FR-142) and gate 1 scores.
    held_out_text_games: dict[str, set[str]] = field(default_factory=dict)

    def merge(self, other: GameTally) -> None:
        self.games |= other.games
        self.held_out |= other.held_out
        self.game_disjoint |= other.game_disjoint
        for text, games in other.held_out_text_games.items():
            self.held_out_text_games.setdefault(text, set()).update(games)


@dataclass(slots=True)
class ShardSurvey:
    """One shard's contribution. Plain data, so it pickles back cheaply."""

    name: str
    size: int
    supply: TrainingSupply
    records: int = 0
    rarity: RarityTally = field(default_factory=RarityTally)
    games: GameTally = field(default_factory=GameTally)
    refusals: RefusalTally = field(default_factory=RefusalTally)
    #: Kept records per single legality mode, over the whole corpus.
    mode_counts: Counter[str] = field(default_factory=Counter)
    #: Training-bound records per class acting through a held-out text.
    held_out_text_dropped: Counter[str] = field(default_factory=Counter)
    #: Old token keys this shard's read rewrote, and the ones it refused to
    #: guess at (FR-151).
    remap: RemapCounts = field(default_factory=RemapCounts)


@dataclass(frozen=True, slots=True)
class Survey:
    """What the whole corpus holds."""

    shards: tuple[SourceShard, ...]
    records: int
    rarity: RarityTally
    #: Training-bound records only; legality cells resolved by the build.
    supply: TrainingSupply
    games: GameTally
    refusals: RefusalTally = field(default_factory=RefusalTally)
    mode_counts: Counter[str] = field(default_factory=Counter)
    held_out_text_dropped: Counter[str] = field(default_factory=Counter)
    #: Distinct games per top-level source directory, and how many of those
    #: name a held-out card. What FR-149's report is computed from: a
    #: directory of depleted shards whose held-out count is not zero is a leak
    #: in collection, and nothing else in the build would notice.
    games_by_source: dict[str, int] = field(default_factory=dict)
    held_out_games_by_source: dict[str, int] = field(default_factory=dict)
    #: The whole corpus's remap counts, summed over **every record read** --
    #: the write pass counts written records only, so its figures are a
    #: subset of these; one exceeding the survey's means the corpus changed
    #: between the two passes.
    remap: RemapCounts = field(default_factory=RemapCounts)
    #: Games found in more than one shard. A game routes at the end of its
    #: shard, so a game split across two shards was planned from one shard's
    #: view of it: still placed correctly by the write pass, which reads the
    #: merged sets, but possibly counted toward training slots it never fills.
    split_games: int = 0


# ── the pass ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    """What every survey worker needs, sent once through the pool initializer."""

    read: ReadConfig
    #: How many smallest hashes the supply keeps per cell and key.
    heap_cap: int
    routing: GameRouting
    #: ``--game-disjoint-keywords`` in gate 2's spelling (``first_strike``):
    #: a game holding a combat record gate 2 qualifies for one of these is
    #: placed at the keyword threshold (FR-045).
    game_disjoint_keywords: tuple[str, ...] = ()


_SURVEY: tuple[SurveyConfig, WorkerContext] | None = None


def init_survey_worker(config: SurveyConfig) -> None:
    """Pool initializer: one context per worker process, built once."""
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
    from effects.domain.damage_step_keywords import (
        KEYWORDS_BY_NAME,
        qualifying_observations,
    )

    path = Path(config.read.records_dir) / relative
    out = ShardSurvey(
        name=relative, size=path.stat().st_size, supply=TrainingSupply(config.heap_cap),
    )
    rows = tuple(
        KEYWORDS_BY_NAME[name] for name in config.game_disjoint_keywords
        if name in KEYWORDS_BY_NAME
    )
    keyword_games: set[str] = set()
    #: ``(game, class, cell, keys, hash, acts through a held-out text)`` per
    #: kept record, counted into the supply once its game's route is known.
    pending: list[tuple[str, str, Cell, Keys | None, int, bool]] = []

    for record in context.records(relative, out.remap):
        out.records += 1
        game = record.game_id
        # The keyword threshold and the held-out test are properties of the
        # game (FR-046), so both are read above the quality check: a game does
        # not stop holding a combat, or a held-out card, because the record
        # saying so was refused.
        if (
            rows and record.kind is RecordKind.COMBAT and game not in keyword_games
            and qualifying_observations(record, rows, context.resolver)
        ):
            keyword_games.add(game)
        if record_names_held_out_card(record, config.read.held_out):
            out.games.held_out.add(game)
        klass = sampling_class(record)
        defect = context.refusal(record)
        if defect is not None:
            out.refusals.note(record, klass, defect)
            continue
        out.games.games.add(game)
        cell = context.cell_of(record)
        out.mode_counts.update(cell.modes)
        keys = acting_keys(record) or None
        if keys:
            out.rarity.note(keys, game)
        held_text = context.held_out_text(keys)
        if held_text is not None and record.kind is RecordKind.RESOLUTION:
            out.games.held_out_text_games.setdefault(held_text, set()).add(game)
        pending.append((
            game, klass, cell, keys, context.hash_of(record), held_text is not None,
        ))

    routes = {
        game: config.routing.route(
            game, held_out=game in out.games.held_out,
            keyword_combat=game in keyword_games,
        )
        for game in out.games.games
    }
    out.games.game_disjoint = {
        game for game, route in routes.items() if route is GameRoute.GAME_DISJOINT
    }
    for game, klass, cell, keys, value, held_text in pending:
        if routes[game] is not GameRoute.TRAINING:
            continue
        if held_text:
            out.held_out_text_dropped[klass] += 1
            continue
        out.supply.add(cell, keys, value)
    return out


def source_of(relative: str) -> str:
    """The top-level directory a raw shard sits in: ``depleted``, ``full-strength``…

    ``"."`` for a shard at the root. What FR-149 reports held-out games
    against: a collection run is a directory, and a leak is a directory that
    should hold none.
    """
    return relative.split("/", 1)[0] if "/" in relative else "."


def merge_surveys(parts: Iterable[ShardSurvey], *, heap_cap: int) -> Survey:
    """Combine shard surveys into one corpus-wide picture."""
    shards: list[SourceShard] = []
    records = 0
    rarity = RarityTally()
    supply = TrainingSupply(heap_cap)
    games = GameTally()
    refusals = RefusalTally()
    mode_counts: Counter[str] = Counter()
    held_out_text_dropped: Counter[str] = Counter()
    remap = RemapCounts()
    games_by_source: dict[str, set[str]] = defaultdict(set)
    held_by_source: dict[str, set[str]] = defaultdict(set)
    split_games = 0

    for part in parts:
        shards.append(SourceShard(name=part.name, size=part.size))
        records += part.records
        split_games += len(part.games.games & games.games)
        games.merge(part.games)
        rarity.merge(part.rarity)
        supply.merge(part.supply)
        refusals.merge(part.refusals)
        mode_counts.update(part.mode_counts)
        held_out_text_dropped.update(part.held_out_text_dropped)
        remap.merge(part.remap)
        source = source_of(part.name)
        games_by_source[source] |= part.games.games
        held_by_source[source] |= part.games.held_out

    return Survey(
        shards=tuple(sorted(shards, key=lambda s: s.name)),
        records=records,
        rarity=rarity,
        supply=supply,
        games=games,
        refusals=refusals,
        mode_counts=mode_counts,
        held_out_text_dropped=held_out_text_dropped,
        games_by_source={s: len(g) for s, g in games_by_source.items()},
        held_out_games_by_source={s: len(g) for s, g in held_by_source.items()},
        remap=remap,
        split_games=split_games,
    )


def progress_line(*, done: int, total: int, elapsed: float) -> str:
    """``done of total`` with the rate and the time left.

    A bare count says the run is alive. It does not say whether the rest is ten
    minutes away or two hours, and on a pass over thousands of shards that is
    the question actually being asked.
    """
    rate = done / elapsed if elapsed > 0 else 0.0
    left = (total - done) / rate if rate > 0 else 0.0
    if rate <= 0:
        return f"{done} of {total} shard(s)"
    return (
        f"{done} of {total} shard(s), {rate * 60:.0f}/min, "
        f"~{left / 60:.0f} min left"
    )


def shard_names(records_dir: Path) -> list[str]:
    """Every shard under the corpus root, as posix paths relative to it."""
    from effects.infrastructure.record_io import iter_shards

    root = Path(records_dir)
    return [shard.relative_to(root).as_posix() for shard in iter_shards(root)]


def run_parallel(
    names: list[str], task, *, initializer, initargs: tuple, workers: int | None,
    label: str, progress_every: int = 50,
) -> Iterator:
    """``task`` over every shard name, in a pool or in process, logging progress.

    ``workers <= 1`` runs the initializer and the task in this process, so a
    small build or a test needs no pool.
    """
    started = time.monotonic()

    def report(index: int) -> None:
        if index % progress_every == 0:
            logger.info("%s %s", label, progress_line(
                done=index, total=len(names), elapsed=time.monotonic() - started,
            ))

    if workers is not None and workers <= 1:
        initializer(*initargs)
        for index, name in enumerate(names, start=1):
            yield task(name)
            report(index)
        return
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initializer, initargs=initargs,
    ) as pool:
        for index, part in enumerate(pool.map(task, names, chunksize=1), start=1):
            yield part
            report(index)


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
