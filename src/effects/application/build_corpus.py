"""Build a fixed training corpus and two fixed validation strata (FR-135).

Two passes. The survey reads every shard and reports what the whole corpus
holds; the main process folds provenance keys to ability texts, decides the
split and the write targets, and the write pass re-reads each shard and keeps
what the decisions admit.

The survey keys on the provenance key rather than the ability text on purpose.
Resolving a key to its text needs a ``SidecarCache`` over tens of thousands of
sidecar files, and building one per worker process would cost more than the
scan it serves. Keys are in the record; the fold happens once, in the process
that already has a cache for the holdout.
"""

from __future__ import annotations

import logging
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.corpus_curation import CapHeap, record_hash
from effects.domain.corpus_manifest import SourceShard
from effects.domain.provenance import ProvenanceKey
from effects.domain.record_quality import quality_defect
from effects.domain.records import EffectRecord, RecordKind

if TYPE_CHECKING:
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)

_KEY_FIELD_SEPARATOR = "|"
_KEY_SEPARATOR = ";"


class BuildCorpusError(ValueError):
    """A build this corpus, these flags or this working directory cannot do.

    Separated from a bare ``ValueError`` so the CLI can report an operator
    condition as one logged line and a non-zero exit while letting an
    internal-invariant violation — ``decide``'s cap mismatch, ``CapHeap.merge``'s
    — keep its traceback. The two arrive at the same ``except`` clause
    otherwise, and a bug in the builder then reads exactly like a misconfigured
    run.

    A ``ValueError`` subclass rather than a fresh exception type, so a caller
    that already handles the broader class keeps working.
    """


def ability_key(record: EffectRecord) -> str | None:
    """A record's whole acting-ability key tuple, as one hashable string.

    The whole tuple rather than its first element: ``ability_text_of`` resolves
    the first key that the sidecars can read, so a survey keyed on the first
    key alone would group records the trainer separates whenever the first key
    names a script with no sidecar.
    """
    if not record.ability:
        return None
    return _KEY_SEPARATOR.join(
        _KEY_FIELD_SEPARATOR.join(
            (key.script_file, str(key.face), key.trait_kind, str(key.index_within_kind))
        )
        for key in record.ability
    )


def parse_ability_key(rendered: str) -> tuple[ProvenanceKey, ...]:
    """The inverse of :func:`ability_key`."""
    keys = []
    for part in rendered.split(_KEY_SEPARATOR):
        script_file, face, trait_kind, index = part.split(_KEY_FIELD_SEPARATOR)
        keys.append(ProvenanceKey(script_file, int(face), trait_kind, int(index)))
    return tuple(keys)


def game_hash(game_id: str) -> int:
    """A stable 32-bit id for a game, for counting distinct games cheaply.

    Rarity needs the *number* of distinct games per text, not their names, and
    holding tens of millions of game-id strings across worker results costs
    hundreds of megabytes for a figure that is a cardinality. ``crc32`` rather
    than the built-in ``hash()``, which is salted per process (FR-088a).
    """
    return zlib.crc32(game_id.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    """What every survey worker needs, sent once through the pool initializer."""

    records_dir: str
    held_out_names: frozenset[str]
    held_out_script_files: frozenset[str]
    text_cap: int
    seed: int
    #: ``--max-events-per-record``, passed through to ``quality_defect``. No
    #: default: the survey counts what the write pass will keep, so a survey
    #: run under a different rule than the write pass reports availability for
    #: records that are about to be refused.
    max_events: int


@dataclass(slots=True)
class ShardSurvey:
    """One shard's contribution. Plain data, so it pickles back cheaply."""

    name: str
    size: int
    records: int
    key_games: dict[str, set[int]] = field(default_factory=dict)
    key_records: Counter[str] = field(default_factory=Counter)
    key_hashes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    class_records: Counter[str] = field(default_factory=Counter)
    #: sampling class -> rendered provenance key -> records. What splits the
    #: per-text cap's survivors across classes: the cap is corpus-wide, so
    #: only the main process can say how many records of one text survive it,
    #: and only this says which classes those records belong to. A record with
    #: no acting key is absent from here and accounted for as
    #: ``class_records - sum(class_key_records[class])``, which is exact.
    class_key_records: dict[str, Counter[str]] = field(default_factory=dict)
    #: Each key's game ids as strings, for keys carried by a held-out game.
    #: The card-disjoint cap admits whole games (FR-142) and so needs the ids
    #: themselves, not the hashed cardinality rarity counts with.
    held_out_text_games: dict[str, set[str]] = field(default_factory=dict)
    held_out_games: set[str] = field(default_factory=set)
    games: set[str] = field(default_factory=set)
    #: Records this shard's survey refused, by reason (FR-148). Counted
    #: rather than silently skipped: a corpus that is a third junk should say
    #: so in the manifest rather than in nothing.
    quality_dropped: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class Survey:
    """What the whole corpus holds."""

    shards: tuple[SourceShard, ...]
    records: int
    key_games: dict[str, set[int]]
    key_records: Counter[str]
    key_heaps: dict[str, CapHeap]
    class_records: Counter[str]
    class_key_records: dict[str, Counter[str]]
    held_out_text_games: dict[str, set[str]]
    held_out_games: frozenset[str]
    games: frozenset[str]
    quality_dropped: Counter[str] = field(default_factory=Counter)
    #: Distinct games per top-level source directory, and how many of those
    #: name a held-out card. What FR-149's report is computed from: a
    #: directory of depleted shards whose held-out count is not zero is a leak
    #: in collection, and nothing else in the build would notice.
    games_by_source: dict[str, int] = field(default_factory=dict)
    held_out_games_by_source: dict[str, int] = field(default_factory=dict)


_CONFIG: SurveyConfig | None = None


def init_survey_worker(config: SurveyConfig) -> None:
    """Pool initializer: hand every worker the config once, not per shard."""
    global _CONFIG
    _CONFIG = config


def survey_shard(relative: str) -> ShardSurvey:
    """Survey one shard. Module-level so a process pool can pickle it."""
    from effects.application.train_effect_model import (
        HeldOutCards,
        record_names_held_out_card,
        sampling_class,
    )
    from effects.infrastructure.record_io import read_shard

    config = _CONFIG
    assert config is not None, "init_survey_worker was not run"
    held_out = HeldOutCards(
        names=config.held_out_names, script_files=config.held_out_script_files,
    )
    path = Path(config.records_dir) / relative
    out = ShardSurvey(name=relative, size=path.stat().st_size, records=0)
    heaps: dict[str, CapHeap] = {}

    for record in read_shard(path):
        out.records += 1
        defect = quality_defect(record, max_events=config.max_events)
        if defect is not None:
            out.quality_dropped[defect] += 1
            continue
        out.games.add(record.game_id)
        name = sampling_class(record)
        out.class_records[name] += 1
        held = record_names_held_out_card(record, held_out)
        if held:
            out.held_out_games.add(record.game_id)
        key = ability_key(record)
        if key is None:
            continue
        out.key_records[key] += 1
        out.class_key_records.setdefault(name, Counter())[key] += 1
        out.key_games.setdefault(key, set()).add(game_hash(record.game_id))
        if held:
            out.held_out_text_games.setdefault(key, set()).add(record.game_id)
        heap = heaps.get(key)
        if heap is None:
            heap = heaps[key] = CapHeap(config.text_cap)
        heap.offer(record_hash(record.record_id, seed=config.seed))

    out.key_hashes = {key: heap.values() for key, heap in heaps.items()}
    return out


def source_of(relative: str) -> str:
    """The top-level directory a raw shard sits in: ``depleted``, ``full-strength``…

    ``"."`` for a shard at the root. What FR-149 reports held-out games
    against: a collection run is a directory, and a leak is a directory that
    should hold none.
    """
    return relative.split("/", 1)[0] if "/" in relative else "."


def merge_surveys(parts: Iterable[ShardSurvey]) -> Survey:
    """Combine shard surveys into one corpus-wide picture."""
    config = _CONFIG
    assert config is not None, "init_survey_worker was not run"
    cap = config.text_cap
    shards: list[SourceShard] = []
    records = 0
    key_games: dict[str, set[int]] = defaultdict(set)
    key_records: Counter[str] = Counter()
    key_heaps: dict[str, CapHeap] = {}
    class_records: Counter[str] = Counter()
    held_out_games: set[str] = set()
    games: set[str] = set()

    class_key_records: dict[str, Counter[str]] = defaultdict(Counter)
    held_out_text_games: dict[str, set[str]] = defaultdict(set)
    quality_dropped: Counter[str] = Counter()
    games_by_source: dict[str, set[str]] = defaultdict(set)
    held_by_source: dict[str, set[str]] = defaultdict(set)

    for part in parts:
        shards.append(SourceShard(name=part.name, size=part.size))
        records += part.records
        key_records.update(part.key_records)
        class_records.update(part.class_records)
        for name, per_key in part.class_key_records.items():
            class_key_records[name].update(per_key)
        held_out_games |= part.held_out_games
        games |= part.games
        quality_dropped.update(part.quality_dropped)
        source = source_of(part.name)
        games_by_source[source] |= part.games
        held_by_source[source] |= part.held_out_games
        for key, ids in part.held_out_text_games.items():
            held_out_text_games[key] |= ids
        for key, hashed in part.key_games.items():
            key_games[key] |= hashed
        for key, values in part.key_hashes.items():
            heap = key_heaps.get(key)
            if heap is None:
                heap = key_heaps[key] = CapHeap(cap)
            for value in values:
                heap.offer(value)

    return Survey(
        shards=tuple(sorted(shards, key=lambda s: s.name)),
        records=records,
        key_games=dict(key_games),
        key_records=key_records,
        key_heaps=key_heaps,
        class_records=class_records,
        class_key_records=dict(class_key_records),
        held_out_text_games=dict(held_out_text_games),
        held_out_games=frozenset(held_out_games),
        games=frozenset(games),
        quality_dropped=quality_dropped,
        games_by_source={s: len(g) for s, g in games_by_source.items()},
        held_out_games_by_source={s: len(g) for s, g in held_by_source.items()},
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

    started = time.monotonic()
    init_survey_worker(config)          # so a workers=1 run needs no pool
    if workers is not None and workers <= 1:
        parts = []
        for index, name in enumerate(names, start=1):
            parts.append(survey_shard(name))
            if index % progress_every == 0:
                logger.info("Surveyed %s", progress_line(
                    done=index, total=len(names),
                    elapsed=time.monotonic() - started,
                ))
        return merge_surveys(parts)

    parts = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_survey_worker, initargs=(config,),
    ) as pool:
        for index, part in enumerate(pool.map(survey_shard, names, chunksize=1), 1):
            parts.append(part)
            if index % progress_every == 0:
                logger.info("Surveyed %s", progress_line(
                    done=index, total=len(names),
                    elapsed=time.monotonic() - started,
                ))
    return merge_surveys(parts)


@dataclass(frozen=True, slots=True)
class BuildCorpusConfig:
    """The `build-corpus` flags (root spec § Curated corpus)."""

    records_dir: Path
    output: Path = Path("output/effects/corpus")
    cards_folders: tuple[str, ...] = ("output/cardsfolder", "output/tokenscripts")
    vocab_path: str = "models/effects/vocab.txt"
    variant_scripts: str | None = None
    holdout_permille: int = 20
    holdout_max_carriers: int = 8
    text_cap: int = 200
    class_mix: dict[str, float] | None = None
    training_records: int = 0
    game_disjoint_target: int = 1000
    card_disjoint_text_cap: int = 50
    shard_records: int = 2000
    max_events: int = 64
    validation_sample: int = 2048
    seed: int = 42
    workers: int | None = None
    verify: bool = False

    def __post_init__(self) -> None:
        """Refuse a negative count, and say what zero means for each.

        Zero is a real setting for all four and means something different in
        each, which is why it is spelled out rather than rejected along with
        the negatives:

        - ``--text-cap 0`` — no per-text cap; every record of every text is a
          training candidate (``CapHeap`` reads a non-positive cap this way).
        - ``--card-disjoint-text-cap 0`` — no cap on the card-disjoint
          stratum; every held-out game carrying a held-out text is admitted.
        - ``--game-disjoint-games 0`` — no game-disjoint stratum at all.
        - ``--training-records 0`` — no ceiling beyond ``--text-cap``.

        A negative is none of those and has no reading at all: it would make
        ``min(cap, total)`` negative and ``random.sample`` raise several
        thousand records into the build.
        """
        for flag, value in (
            ("--text-cap", self.text_cap),
            ("--card-disjoint-text-cap", self.card_disjoint_text_cap),
            ("--game-disjoint-games", self.game_disjoint_target),
            ("--training-records", self.training_records),
            ("--shard-records", self.shard_records),
            ("--max-events-per-record", self.max_events),
            ("--validation-sample", self.validation_sample),
        ):
            if value < 0:
                raise BuildCorpusError(
                    f"{flag} is {value}; it counts things and cannot be "
                    "negative. Zero is allowed and means no cap, no ceiling, "
                    "or an empty stratum depending on the flag — see --help."
                )

    def mix(self) -> dict[str, float]:
        from effects.application.train_effect_model import DEFAULT_KIND_MIX

        return dict(self.class_mix or DEFAULT_KIND_MIX)


@dataclass(frozen=True, slots=True)
class Decisions:
    """Everything the write pass needs, and nothing it has to look up.

    ``thresholds`` is keyed by rendered provenance key rather than by ability
    text, so a worker applies it without building a ``SidecarCache``; the fold
    from key to text happened here, once.
    """

    thresholds: dict[str, int | None]
    class_targets: dict[str, int]
    card_disjoint: frozenset[str]
    game_disjoint: frozenset[str]
    rarity: dict[str, int]
    shortfall: dict[str, int]
    capped_class_records: dict[str, int]
    key_text: dict[str, str]
    #: Rendered survey keys whose ability text is held out. What routes the
    #: gate-one slice: a card-disjoint resolution record acting on one of
    #: these is the unique-text stratum gate 1's three margins are measured
    #: on, and picking it out at write time is what saves the evaluator a
    #: second pass over the whole stratum.
    gate_one_keys: frozenset[str]


def _text_of_rendered_key(
    rendered: str, sidecars: SidecarCache, surface: str,
) -> str | None:
    """The text a survey key folds to, through the trainer's own definition."""
    from effects.application.train_effect_model import text_for_key

    for key in parse_ability_key(rendered):
        text = text_for_key(key, sidecars, surface)
        if text is not None:
            return text
    return None


def _held_out_text_of_key(
    survey: Survey, sidecars: SidecarCache, held_out_texts: frozenset[str],
) -> dict[str, str]:
    """Each survey key that resolves to a held-out ability text, and the text.

    Resolved on the **script** surface whatever ``--vocab-path`` says, because
    that is the surface the holdout itself is keyed on: ``select_holdout``
    hashes a sidecar line's ``script_text``, so a prose-surface fold would
    compare a prose string against a set of script strings and match nothing —
    silently, since an empty match looks exactly like a corpus with no
    held-out ability in it. Normalized the same way for the same reason.
    """
    from effects.domain.ability_encoder import SURFACE_SCRIPT
    from effects.domain.text_holdout import normalize_script_text

    out: dict[str, str] = {}
    for key in survey.held_out_text_games:
        script = _text_of_rendered_key(key, sidecars, SURFACE_SCRIPT)
        if script is None:
            continue
        normalized = normalize_script_text(script)
        if normalized in held_out_texts:
            out[key] = normalized
    return out


def _capped_class_records(
    survey: Survey, key_text: dict[str, str], *, cap: int,
) -> dict[str, int]:
    """What each sampling class can supply once the per-text cap has bitten.

    The number ``class_targets`` divides the mixture over, and the number
    ``build()`` divides a class's target by to get its admission rate — so an
    over-estimate here delivers a class short of the share the manifest goes
    on to record, and an under-estimate over-delivers it.

    Exact per text, because the cap's survivor count is not a guess: the
    threshold admits the ``cap`` smallest hashes of a text, or all of them
    where the text has fewer, so ``min(cap, text_records)`` survive. Splitting
    that across classes is a proportion — ``class_key_records[c][key] *
    min(cap, text_total) / text_total`` — and it is unbiased because the hash
    the cap selects on is computed from the record id alone and says nothing
    about the record's class.

    Two populations are never capped and count in full: a record with no
    acting key (``combat``, ``playability-legality``), and a key no sidecar
    can read, which ``decide`` leaves out of ``key_text`` and out of the cap
    with it. ``cap <= 0`` means no cap at all, matching ``CapHeap``.
    """
    text_records: Counter[str] = Counter()
    for key, total in survey.key_records.items():
        text = key_text.get(key)
        if text is not None:
            text_records[text] += total

    capped: dict[str, float] = {}
    for name, per_key in survey.class_key_records.items():
        survivors = 0.0
        for key, count in per_key.items():
            text = key_text.get(key)
            if text is None or cap <= 0:
                survivors += count
                continue
            total = text_records[text]
            survivors += count * min(cap, total) / total
        capped[name] = survivors

    out: dict[str, int] = {}
    for name, total in survey.class_records.items():
        keyed = sum(survey.class_key_records.get(name, {}).values())
        out[name] = round(capped.get(name, 0.0) + (total - keyed))
    return out


def _card_disjoint_games(
    survey: Survey, held_out_text_of_key: dict[str, str], *, cap: int, seed: int,
) -> frozenset[str]:
    """Held-out games admitted while every held-out text they carry is under cap.

    Games rather than records (FR-142, FR-136): a stratum built by dropping
    records inside a game would separate a probe from the combat record its
    ``mirror_of`` names, and the evaluator would score that keyword on nothing
    while reporting it as merely under-sampled.

    Two rules make the cap bind, and both are FR-142's "per **held-out**
    ability text" read literally.

    Only a held-out text is tallied. A real game carries tens of distinct
    ability texts, nearly all of them ordinary, so a tally over every text a
    game carries counted mostly things the cap is not about — and the
    held-out texts are the only ones gate 1 averages over.

    And a game is admitted only while **every** held-out text it carries is
    still under the cap, not merely one of them. Admitting on one was the
    defect: a game's own private texts sit at zero forever, so there was
    always some text under cap and the cap never declined a game. The cost of
    the strict rule is that a game pairing a saturated text with an unsaturated
    one is declined, which can leave the second short; the cap is a ceiling on
    how far any one held-out text may dominate the stratum, and a rule that
    can be pushed past its ceiling is not one.

    A game carrying no held-out text is not admitted at all. It holds nothing
    gate 1 measures, and it is withheld from training either way — a held-out
    game the cap declines is dropped, never trained on.

    Walked in a seeded order so two builds of one corpus admit the same games.
    ``cap <= 0`` means no cap, matching ``--text-cap``'s own zero.
    """
    import random

    text_games: dict[str, set[str]] = defaultdict(set)
    for key, text in held_out_text_of_key.items():
        text_games[text] |= survey.held_out_text_games[key]

    order = sorted(survey.held_out_games)
    random.Random(seed).shuffle(order)
    games_text: dict[str, set[str]] = defaultdict(set)
    for text, games in text_games.items():
        for game in games:
            games_text[game].add(text)

    taken: Counter[str] = Counter()
    admitted: set[str] = set()
    for game in order:
        texts = games_text.get(game, set())
        if not texts:
            continue
        if cap > 0 and any(taken[text] >= cap for text in texts):
            continue
        admitted.add(game)
        for text in texts:
            taken[text] += 1
    return frozenset(admitted)


def decide(
    survey: Survey,
    *,
    sidecars: SidecarCache,
    surface: str,
    held_out_texts: frozenset[str],
    config: BuildCorpusConfig,
) -> Decisions:
    """Fold keys to texts, cap, split the games, and set the class targets.

    Args:
        held_out_texts: the normalized script texts the holdout names, which
            is the unit ``--card-disjoint-text-cap`` counts (FR-142).
    """
    import random

    from effects.domain.corpus_curation import class_targets

    for key, heap in survey.key_heaps.items():
        if heap.cap != config.text_cap:
            raise ValueError(
                f"survey key {key!r} was surveyed with a per-text cap of "
                f"{heap.cap}, but config.text_cap is {config.text_cap}. "
                "decide() must be run against the Survey that this same "
                "--text-cap produced — this Survey looks like it came from a "
                "SurveyConfig built with a different text_cap, and its "
                "thresholds would silently disagree with what the manifest "
                "is about to record."
            )

    key_text: dict[str, str] = {}
    text_games: dict[str, set[int]] = defaultdict(set)
    text_heaps: dict[str, CapHeap] = {}
    for key in survey.key_records:
        text = _text_of_rendered_key(key, sidecars, surface)
        if text is None:
            # No sidecar can read this key, so it is not a text and cannot be
            # capped against one. Its records stay, uncapped: dropping them
            # would remove a card the converted corpus never held rather than
            # trimming a head.
            continue
        key_text[key] = text
        text_games[text] |= survey.key_games.get(key, set())
        heap = text_heaps.get(text)
        if heap is None:
            heap = text_heaps[text] = CapHeap(config.text_cap)
        heap.merge(survey.key_heaps[key])

    rarity = {text: len(games) for text, games in text_games.items()}
    thresholds = {key: text_heaps[text].threshold() for key, text in key_text.items()}

    held_out_text_of_key = _held_out_text_of_key(survey, sidecars, held_out_texts)
    # Only ``survey.held_out_text_games`` is walked, which is keyed on records
    # of held-out games -- exactly the population the gate-one slice draws
    # from, since the slice is a subset of the card-disjoint stratum.
    gate_one_keys = frozenset(held_out_text_of_key)
    card_disjoint = _card_disjoint_games(
        survey,
        held_out_text_of_key,
        cap=config.card_disjoint_text_cap,
        seed=config.seed,
    )
    remaining = sorted(survey.games - survey.held_out_games)
    take = min(config.game_disjoint_target, len(remaining))
    game_disjoint = frozenset(random.Random(config.seed).sample(remaining, take))

    capped = _capped_class_records(survey, key_text, cap=config.text_cap)
    try:
        targets, shortfall = class_targets(
            capped, config.mix(), ceiling=config.training_records,
        )
    except ValueError as exc:
        # `--class-mix` names classes this corpus has no records of, which is
        # the operator's flag against the operator's corpus, not a broken
        # invariant.
        raise BuildCorpusError(str(exc)) from exc
    return Decisions(
        thresholds=thresholds,
        class_targets=targets,
        card_disjoint=card_disjoint,
        game_disjoint=game_disjoint,
        rarity=rarity,
        shortfall=shortfall,
        capped_class_records=capped,
        key_text=key_text,
        gate_one_keys=gate_one_keys,
    )


#: Training admission is two independent thresholds on the record id: the
#: per-text cap, and the class's share of the mixture. Both are hashes rather
#: than counters so a worker needs no coordination with any other worker, and
#: the second is keyed differently from the first so a record is not judged
#: twice by the same number.
_CLASS_SEED_OFFSET = 0x9E3779B9

#: Every write-pass stratum, in report order. All but the last are the real
#: outputs -- shard directories a downstream reader loads, and what
#: FR-143/FR-145 mean by "each output" -- and ``OUTPUTS`` is just that
#: prefix; "dropped-held-out" is accounting only, since nothing is written
#: for it, so it is counted in ``per_stratum`` but never carries a
#: unique-text figure. "gate-one" is a *slice* of "card-disjoint" rather
#: than a fourth destination: its records are written twice on purpose, so
#: the evaluator reads gate 1's unique-text stratum without re-filtering the
#: whole card-disjoint output, and its per-stratum count therefore overlaps
#: card-disjoint's rather than partitioning with it.
STRATA: tuple[str, ...] = (
    "training", "card-disjoint", "game-disjoint", "gate-one", "dropped-held-out",
)
OUTPUTS: tuple[str, ...] = STRATA[:-1]


@dataclass(frozen=True, slots=True)
class WriteConfig:
    """Where one shard's records go, and by what rule.

    The four output directories name **parts** directories rather than the
    strata themselves: a worker writes one part per source shard, and
    ``build()`` repacks the parts into uniform shards afterwards.
    """

    records_dir: str
    training_dir: str
    card_disjoint_dir: str
    game_disjoint_dir: str
    gate_one_dir: str
    gate_one_keys: frozenset[str]
    max_events: int
    thresholds: dict[str, int | None]
    class_admit: dict[str, float]
    card_disjoint: frozenset[str]
    game_disjoint: frozenset[str]
    held_out_games: frozenset[str]
    seed: int


@dataclass(slots=True)
class WriteResult:
    kept: Counter[str] = field(default_factory=Counter)
    dropped_by_cap: Counter[str] = field(default_factory=Counter)
    read: Counter[str] = field(default_factory=Counter)
    kept_keys: dict[str, set[str]] = field(default_factory=dict)
    #: Records written to each output, keyed "training" / "card-disjoint" /
    #: "game-disjoint", plus "dropped-held-out" for held-out games the
    #: card-disjoint cap declined. FR-145 wants the report per stratum as well
    #: as per class, and a stratum nobody counted is a stratum nobody notices
    #: is empty.
    stratum: Counter[str] = field(default_factory=Counter)
    #: Ability keys admitted to each of the three real outputs -- "training"
    #: / "card-disjoint" / "game-disjoint", never "dropped-held-out", which
    #: writes nothing -- the same role ``kept_keys`` plays per class, but per
    #: stratum instead: ``build()`` folds these through ``decisions.key_text``
    #: to get each output's unique-ability-text count (FR-143, FR-145).
    stratum_keys: dict[str, set[str]] = field(default_factory=dict)
    #: Records refused by ``effects.domain.record_quality``, by reason
    #: (FR-148). Counted before every routing decision, so a defective record
    #: reaches no output at all and belongs to no class's read count.
    quality_dropped: Counter[str] = field(default_factory=Counter)


_WRITE: WriteConfig | None = None


def init_write_worker(config: WriteConfig) -> None:
    global _WRITE
    _WRITE = config


def _output_name(relative: str) -> str:
    """A flat, **injective** shard name for a source that may sit in a subdirectory.

    ``depleted/run.0-a.jsonl.gz`` and ``full-strength/run.0-a.jsonl.gz`` are
    different shards and must not write to one file — and neither may any other
    pair of distinct relative paths, because ``write_shard`` opens in truncate
    mode, so a collision is a silent overwrite in all three outputs with no
    stable answer, under ``workers > 1``, as to which source survives.

    A plain ``"/" -> "__"`` substitution is not injective: ``"a_/b"`` and
    ``"a/_b"`` both become ``"a___b"``, because an original ``_`` and a
    doubled-up ``/`` separator are indistinguishable in the output. The fix is
    the standard one for building an injective string encoding out of a small
    alphabet of specials: escape the escape character *first*. Every ``_`` in
    the result is then the first character of a two-character escape --
    ``_u`` for an original ``_``, ``_s`` for an original ``/`` -- and never an
    untouched original character (every real ``_`` was already rewritten to
    ``_u`` before ``/`` is touched), so the two cases can never be confused
    and the mapping is one-to-one.
    """
    return relative.replace("_", "_u").replace("/", "_s")


def write_shard_pass(relative: str) -> WriteResult:
    """Filter one shard into its output parts. Module-level so it pickles."""
    from effects.application.train_effect_model import sampling_class
    from effects.domain.corpus_curation import keeps, record_hash
    from effects.infrastructure.record_io import read_shard, write_shard

    config = _WRITE
    assert config is not None, "init_write_worker was not run"
    out = WriteResult()
    training: list = []
    card_disjoint: list = []
    game_disjoint: list = []
    gate_one: list = []

    for record in read_shard(Path(config.records_dir) / relative):
        name = sampling_class(record)
        # Computed once, up front: every branch below -- including the two
        # validation strata, which never used to look at it at all -- needs
        # it to track which ability texts that output actually holds.
        key = ability_key(record)
        # Before the read count, so a refused record is not counted as read
        # against a class whose availability the survey computed without it.
        defect = quality_defect(record, max_events=config.max_events)
        if defect is not None:
            out.quality_dropped[defect] += 1
            continue
        out.read[name] += 1
        if record.game_id in config.card_disjoint:
            card_disjoint.append(record)
            out.stratum["card-disjoint"] += 1
            if key is not None:
                out.stratum_keys.setdefault("card-disjoint", set()).add(key)
            if (
                record.kind is RecordKind.RESOLUTION
                and key is not None and key in config.gate_one_keys
            ):
                gate_one.append(record)
                out.stratum["gate-one"] += 1
                out.stratum_keys.setdefault("gate-one", set()).add(key)
            continue
        if record.game_id in config.game_disjoint:
            game_disjoint.append(record)
            out.stratum["game-disjoint"] += 1
            if key is not None:
                out.stratum_keys.setdefault("game-disjoint", set()).add(key)
            continue
        if record.game_id in config.held_out_games:
            # Held out but not admitted to the stratum: dropped, never trained
            # on (FR-088). Nothing is written for it, so it earns no entry in
            # stratum_keys -- only the three real outputs do.
            out.stratum["dropped-held-out"] += 1
            continue
        value = record_hash(record.record_id, seed=config.seed)
        if key is not None and not keeps(value, config.thresholds.get(key)):
            out.dropped_by_cap[name] += 1
            continue
        share = config.class_admit.get(name, 0.0)
        if share < 1.0:
            draw = record_hash(record.record_id, seed=config.seed + _CLASS_SEED_OFFSET)
            if draw >= int(share * 2**64):
                continue
        training.append(record)
        out.kept[name] += 1
        out.stratum["training"] += 1
        if key is not None:
            out.kept_keys.setdefault(name, set()).add(key)
            out.stratum_keys.setdefault("training", set()).add(key)

    shard_name = _output_name(relative)
    # Only where there is something to write. A source shard contributes to one
    # stratum far more often than to three, so writing all three unconditionally
    # produced two empty files per source shard -- 2,103 files from 701 sources.
    # An empty validation shard is read and logged before the first training
    # step; an empty training shard drawn into an epoch forfeits its share of
    # the step budget, because `_train_on_shard` returns early on one.
    for directory, records in (
        (config.training_dir, training),
        (config.card_disjoint_dir, card_disjoint),
        (config.game_disjoint_dir, game_disjoint),
        (config.gate_one_dir, gate_one),
    ):
        if records:
            write_shard(Path(directory) / shard_name, records)
    return out


def run_write_pass(
    names: list[str], *, config: WriteConfig, workers: int | None,
    progress_every: int = 50,
) -> WriteResult:
    """Write every shard's output parts, in parallel, reporting progress."""
    started = time.monotonic()
    total = WriteResult()

    def absorb(part: WriteResult) -> None:
        total.kept.update(part.kept)
        total.dropped_by_cap.update(part.dropped_by_cap)
        total.read.update(part.read)
        total.stratum.update(part.stratum)
        total.quality_dropped.update(part.quality_dropped)
        for name, keys in part.kept_keys.items():
            total.kept_keys.setdefault(name, set()).update(keys)
        for name, keys in part.stratum_keys.items():
            total.stratum_keys.setdefault(name, set()).update(keys)

    if workers is not None and workers <= 1:
        init_write_worker(config)
        for index, name in enumerate(names, start=1):
            absorb(write_shard_pass(name))
            if index % progress_every == 0:
                logger.info("Wrote %s", progress_line(
                    done=index, total=len(names),
                    elapsed=time.monotonic() - started,
                ))
        return total

    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_write_worker, initargs=(config,),
    ) as pool:
        for index, part in enumerate(
            pool.map(write_shard_pass, names, chunksize=1), start=1,
        ):
            absorb(part)
            if index % progress_every == 0:
                logger.info("Wrote %s", progress_line(
                    done=index, total=len(names),
                    elapsed=time.monotonic() - started,
                ))
    return total


def _repack_outputs(store: CorpusStore, *, shard_records: int) -> None:
    """Stream each stratum's parts into uniform shards, then drop the parts.

    The write pass writes one part per source shard, so an output's shard
    sizes mirror the raw corpus's -- hundreds of tiny files beside a handful
    of huge ones. The trainer reads a shard at a time and takes that shard's
    share of an epoch's steps, so uneven shards make the step budget uneven.
    """
    import shutil

    from effects.infrastructure.record_io import iter_shards, repack_shards

    for stratum, target in (
        ("training", store.training_dir),
        ("card-disjoint", store.card_disjoint_dir),
        ("game-disjoint", store.game_disjoint_dir),
        ("gate-one", store.gate_one_dir),
    ):
        parts = iter_shards(store.parts_dir_for(stratum))
        paths = repack_shards(parts, target, shard_records=shard_records)
        logger.info(
            "%-22s %d part(s) repacked into %d shard(s)",
            stratum, len(parts), len(paths),
        )
    shutil.rmtree(store.parts_dir, ignore_errors=True)


def build(config: BuildCorpusConfig) -> int:
    """Build a curated dataset, or verify an existing one. Returns an exit code."""
    from effects.application.train_effect_model import (
        load_card_files,
        load_card_texts,
        text_keyed_holdout,
    )
    from effects.domain.ability_encoder import surface_of
    from effects.domain.corpus_manifest import ClassCounts, CorpusManifest
    from effects.infrastructure.corpus_store import CorpusStore, current_sources
    from effects.infrastructure.sidecar_io import SidecarCache

    store = CorpusStore(config.output)
    records_dir = Path(config.records_dir)

    if config.verify:
        manifest = store.load()
        added, removed, resized = manifest.drift(current_sources(records_dir))
        for label, names in (("added", added), ("removed", removed), ("resized", resized)):
            if names:
                logger.warning(
                    "%d shard(s) %s since this dataset was built: %s%s",
                    len(names), label, ", ".join(names[:5]),
                    ", …" if len(names) > 5 else "",
                )
        if added or removed or resized:
            logger.warning("Rebuild with `python -m effects build-corpus`.")
            return 1
        logger.info("Dataset is current against %s.", records_dir)
        return 0

    folders = {Path(f).name: Path(f) for f in config.cards_folders}
    cards_folder = folders.get("cardsfolder", Path(config.cards_folders[0]))
    roots = dict(folders)
    if config.variant_scripts:
        # Keyed "variant-scripts" because that is the tree a variant line's
        # own provenance names, and the trainer registers it under the same
        # name. Without it every variant key resolves to no text: no rarity
        # entry and no per-text cap, while every real ability has both.
        roots["variant-scripts"] = Path(config.variant_scripts)
    sidecars = SidecarCache(roots)
    card_files = load_card_files(cards_folder)
    held_out = text_keyed_holdout(
        card_files,
        load_card_texts(card_files, sidecars),
        permille=config.holdout_permille,
        max_carriers=config.holdout_max_carriers,
    )
    surface = surface_of(config.vocab_path)
    logger.info(
        "Holdout: %d ability text(s) on %d card(s); encoding surface %r.",
        len(held_out.texts), len(held_out.names), surface,
    )
    if not held_out.names:
        tried = ", ".join(
            f"{folder} ({'found' if Path(folder).is_dir() else 'missing'})"
            for folder in config.cards_folders
        )
        raise BuildCorpusError(
            "Nothing is held out, so the card-disjoint stratum would be empty "
            "and gate 1 would have nothing to measure at all. Cards folders "
            f"tried: {tried}; {len(card_files)} converted card(s) read, "
            f"{len(held_out.texts)} held-out ability text(s), no held-out card. "
            "Run from the repository root so the relative --cards-folder paths "
            "resolve, raise --holdout-permille, or check that the converted "
            "tree has sidecars with script text."
        )

    survey = run_survey(
        records_dir,
        config=SurveyConfig(
            records_dir=str(records_dir),
            held_out_names=held_out.names,
            held_out_script_files=held_out.script_files,
            text_cap=config.text_cap,
            seed=config.seed,
            max_events=config.max_events,
        ),
        workers=config.workers,
    )
    logger.info(
        "Surveyed %d record(s) in %d game(s); %d game(s) name a held-out card.",
        survey.records, len(survey.games), len(survey.held_out_games),
    )

    decisions = decide(
        survey, sidecars=sidecars, surface=surface,
        held_out_texts=held_out.texts, config=config,
    )
    if not decisions.card_disjoint:
        raise BuildCorpusError(
            "No game is admitted to the card-disjoint stratum, so the dataset "
            "cannot score gate 1 and the trainer would refuse it in its first "
            f"minute. {records_dir} holds {len(survey.games)} game(s), "
            f"{len(survey.held_out_games)} of which name one of the "
            f"{len(held_out.names)} held-out card(s). Depleted and "
            "full-strength shards live in sibling subdirectories of the raw "
            "corpus root: point --records-dir at the parent of both, and check "
            "that a full-strength run — pools generated without "
            "--exclude-cards — has actually been collected. A held-out game is "
            "admitted only while every held-out ability text it carries is "
            "under --card-disjoint-text-cap, and one that carries none is "
            "never admitted."
        )
    admit = {
        name: min(1.0, target / decisions.capped_class_records[name])
        for name, target in decisions.class_targets.items()
        if decisions.capped_class_records.get(name)
    }
    # Before the first output shard, and only on a real build: a rebuild
    # replaces the dataset rather than layering over it (FR-144).
    store.clear_outputs()
    written = run_write_pass(
        [shard.name for shard in survey.shards],
        config=WriteConfig(
            records_dir=str(records_dir),
            training_dir=str(store.parts_dir_for("training")),
            card_disjoint_dir=str(store.parts_dir_for("card-disjoint")),
            game_disjoint_dir=str(store.parts_dir_for("game-disjoint")),
            gate_one_dir=str(store.parts_dir_for("gate-one")),
            gate_one_keys=decisions.gate_one_keys,
            max_events=config.max_events,
            thresholds=decisions.thresholds,
            class_admit=admit,
            card_disjoint=decisions.card_disjoint,
            game_disjoint=decisions.game_disjoint,
            held_out_games=survey.held_out_games,
            seed=config.seed,
        ),
        workers=config.workers,
    )
    _repack_outputs(store, shard_records=config.shard_records)

    per_class = {
        name: ClassCounts(
            read=written.read.get(name, 0),
            kept=written.kept.get(name, 0),
            dropped_by_cap=written.dropped_by_cap.get(name, 0),
            unique_texts=len({
                decisions.key_text.get(key, key)
                for key in written.kept_keys.get(name, ())
            }),
        )
        for name in sorted(written.read)
    }
    # The same fold as per_class's unique_texts, one axis over: by output
    # (stratum) rather than by sampling class (FR-143, FR-145). per_stratum
    # covers all four strata written.stratum can hold; unique_texts only the
    # three that are actual outputs -- dropped-held-out writes nothing, so it
    # has no ability texts of its own to count.
    per_stratum = {name: written.stratum.get(name, 0) for name in STRATA}
    trained = sum(written.kept.values())
    delivered_mix = {
        name: count / trained for name, count in sorted(written.kept.items())
    } if trained else {}
    unique_texts = {
        name: len({
            decisions.key_text.get(key, key)
            for key in written.stratum_keys.get(name, ())
        })
        for name in OUTPUTS
    }
    manifest = CorpusManifest(
        seed=config.seed,
        surface=surface,
        vocab_path=config.vocab_path,
        variant_scripts=config.variant_scripts or "",
        holdout_permille=config.holdout_permille,
        holdout_max_carriers=config.holdout_max_carriers,
        text_cap=config.text_cap,
        card_disjoint_text_cap=config.card_disjoint_text_cap,
        game_disjoint_target=config.game_disjoint_target,
        training_records=config.training_records,
        class_mix=config.mix(),
        delivered_mix=delivered_mix,
        held_out_cards=tuple(sorted(held_out.names)),
        held_out_texts=tuple(sorted(held_out.texts)),
        card_disjoint_games=tuple(sorted(decisions.card_disjoint)),
        game_disjoint_games=tuple(sorted(decisions.game_disjoint)),
        rarity=decisions.rarity,
        sources=survey.shards,
        per_class=per_class,
        per_stratum=per_stratum,
        unique_texts=unique_texts,
        shortfall=decisions.shortfall,
        quality_dropped=dict(written.quality_dropped),
        games_by_source=survey.games_by_source,
        held_out_games_by_source=survey.held_out_games_by_source,
        shard_records=config.shard_records,
        validation_sample=config.validation_sample,
        max_events_per_record=config.max_events,
    )
    store.save(manifest)

    for name, counts in per_class.items():
        logger.info(
            "%-22s read %9d  kept %9d  cap dropped %8d  unique texts %7d",
            name, counts.read, counts.kept, counts.dropped_by_cap, counts.unique_texts,
        )
    for name in OUTPUTS:
        logger.info(
            "%-22s %9d record(s)  %7d unique text(s)",
            name, per_stratum[name], unique_texts[name],
        )
    logger.info("%-22s %9d record(s)", "dropped-held-out", per_stratum["dropped-held-out"])
    for source in sorted(survey.games_by_source):
        games = survey.games_by_source[source]
        held = survey.held_out_games_by_source.get(source, 0)
        share = held / games if games else 0.0
        report = logger.warning if 0 < share < 0.05 else logger.info
        report(
            "%-22s %7d game(s), %6d name a held-out card (%.1f%%)%s",
            source, games, held, 100.0 * share,
            " — a few held-out games in a directory that should hold none is "
            "a leak in collection, not a full-strength source" if 0 < share < 0.05 else "",
        )
    for reason, count in sorted(written.quality_dropped.items()):
        logger.info("%-22s %9d record(s) refused", reason, count)
    # Requested against delivered, side by side: the mixture in `class_mix` is
    # what the build was asked for, and only this says whether it got it.
    requested = config.mix()
    for name in sorted(set(requested) | set(delivered_mix)):
        asked, got = requested.get(name, 0.0), delivered_mix.get(name, 0.0)
        logger.info(
            "%-22s requested %6.2f%%  delivered %6.2f%%  (%.2fx)",
            name, 100.0 * asked, 100.0 * got, (got / asked) if asked else 0.0,
        )
    for name, missing in sorted(decisions.shortfall.items()):
        logger.warning(
            "%s is short %d record(s) of its share: --training-records asks for "
            "more than the corpus can supply at this mixture.", name, missing,
        )
    logger.info(
        "Wrote %d training record(s) to %s (digest %s).",
        sum(written.kept.values()), store.directory, manifest.digest(),
    )
    return 0
