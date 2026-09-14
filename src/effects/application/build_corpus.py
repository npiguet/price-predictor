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
from effects.domain.records import EffectRecord

if TYPE_CHECKING:
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)

_KEY_FIELD_SEPARATOR = "|"
_KEY_SEPARATOR = ";"


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
    #: Records this shard's own per-key heap admitted, per class. An upper
    #: bound on what the text-level cap will keep — several keys can fold to
    #: one text, and only the main process knows which — and what
    #: ``class_targets`` divides the mixture over. The manifest's ``kept`` is
    #: the exact figure and comes from the write pass.
    class_capped_records: Counter[str] = field(default_factory=Counter)
    #: Each key's game ids as strings, for keys carried by a held-out game.
    #: The card-disjoint cap admits whole games (FR-142) and so needs the ids
    #: themselves, not the hashed cardinality rarity counts with.
    held_out_text_games: dict[str, set[str]] = field(default_factory=dict)
    held_out_games: set[str] = field(default_factory=set)
    games: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class Survey:
    """What the whole corpus holds."""

    shards: tuple[SourceShard, ...]
    records: int
    key_games: dict[str, set[int]]
    key_records: Counter[str]
    key_heaps: dict[str, CapHeap]
    class_records: Counter[str]
    class_capped_records: Counter[str]
    held_out_text_games: dict[str, set[str]]
    held_out_games: frozenset[str]
    games: frozenset[str]


_CONFIG: SurveyConfig | None = None


def init_survey_worker(config: SurveyConfig) -> None:
    """Pool initializer: hand every worker the config once, not per shard."""
    global _CONFIG
    _CONFIG = config


def survey_shard(relative: str) -> ShardSurvey:
    """Survey one shard. Module-level so a process pool can pickle it."""
    from effects.application.train_effect_model import (
        HeldOutCards, record_names_held_out_card, sampling_class,
    )
    from effects.domain.corpus_curation import keeps
    from effects.infrastructure.record_io import read_shard

    config = _CONFIG
    assert config is not None, "init_survey_worker was not run"
    held_out = HeldOutCards(
        names=config.held_out_names, script_files=config.held_out_script_files,
    )
    path = Path(config.records_dir) / relative
    out = ShardSurvey(name=relative, size=path.stat().st_size, records=0)
    heaps: dict[str, CapHeap] = {}

    # Held for the second walk: a record's own heap admission is only known
    # once the shard's heaps are final, so the class tally cannot be inline.
    seen: list[tuple[str, str | None, int]] = []

    for record in read_shard(path):
        out.records += 1
        out.games.add(record.game_id)
        name = sampling_class(record)
        out.class_records[name] += 1
        held = record_names_held_out_card(record, held_out)
        if held:
            out.held_out_games.add(record.game_id)
        key = ability_key(record)
        value = record_hash(record.record_id, seed=config.seed)
        seen.append((name, key, value))
        if key is None:
            continue
        out.key_records[key] += 1
        out.key_games.setdefault(key, set()).add(game_hash(record.game_id))
        if held:
            out.held_out_text_games.setdefault(key, set()).add(record.game_id)
        heap = heaps.get(key)
        if heap is None:
            heap = heaps[key] = CapHeap(config.text_cap)
        heap.offer(value)

    out.key_hashes = {key: heap.values() for key, heap in heaps.items()}
    thresholds = {key: heap.threshold() for key, heap in heaps.items()}
    for name, key, value in seen:
        # A record with no acting line has no text and so no cap; it counts
        # toward its class in full (FR-087).
        if key is None or keeps(value, thresholds[key]):
            out.class_capped_records[name] += 1
    return out


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

    class_capped: Counter[str] = Counter()
    held_out_text_games: dict[str, set[str]] = defaultdict(set)

    for part in parts:
        shards.append(SourceShard(name=part.name, size=part.size))
        records += part.records
        key_records.update(part.key_records)
        class_records.update(part.class_records)
        class_capped.update(part.class_capped_records)
        held_out_games |= part.held_out_games
        games |= part.games
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
        class_capped_records=class_capped,
        held_out_text_games=dict(held_out_text_games),
        held_out_games=frozenset(held_out_games),
        games=frozenset(games),
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
        raise ValueError(f"{root}: no shards to build a corpus from")
    logger.info("Surveying %d shard(s) under %s", len(names), root)

    init_survey_worker(config)          # so a workers=1 run needs no pool
    if workers is not None and workers <= 1:
        parts = []
        for index, name in enumerate(names, start=1):
            parts.append(survey_shard(name))
            if index % progress_every == 0:
                logger.info("Surveyed %d of %d shard(s)", index, len(names))
        return merge_surveys(parts)

    parts = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_survey_worker, initargs=(config,),
    ) as pool:
        for index, part in enumerate(pool.map(survey_shard, names, chunksize=1), 1):
            parts.append(part)
            if index % progress_every == 0:
                logger.info("Surveyed %d of %d shard(s)", index, len(names))
    return merge_surveys(parts)


@dataclass(frozen=True, slots=True)
class BuildCorpusConfig:
    """The `build-corpus` flags (root spec § Curated corpus)."""

    records_dir: Path
    output: Path = Path("output/effects/corpus")
    cards_folders: tuple[str, ...] = ("output/cardsfolder", "output/tokenscripts")
    vocab_path: str = "models/effects/vocab.txt"
    holdout_permille: int = 20
    holdout_max_carriers: int = 8
    text_cap: int = 200
    class_mix: dict[str, float] | None = None
    training_records: int = 0
    game_disjoint_target: int = 1000
    card_disjoint_text_cap: int = 50
    seed: int = 42
    workers: int | None = None
    verify: bool = False

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


def _text_of_rendered_key(rendered: str, sidecars, surface: str) -> str | None:
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

    card_disjoint = _card_disjoint_games(
        survey,
        _held_out_text_of_key(survey, sidecars, held_out_texts),
        cap=config.card_disjoint_text_cap,
        seed=config.seed,
    )
    remaining = sorted(survey.games - survey.held_out_games)
    take = min(config.game_disjoint_target, len(remaining))
    game_disjoint = frozenset(random.Random(config.seed).sample(remaining, take))

    capped = dict(survey.class_capped_records)
    targets, shortfall = class_targets(
        capped, config.mix(), ceiling=config.training_records,
    )
    return Decisions(
        thresholds=thresholds,
        class_targets=targets,
        card_disjoint=card_disjoint,
        game_disjoint=game_disjoint,
        rarity=rarity,
        shortfall=shortfall,
        capped_class_records=capped,
        key_text=key_text,
    )


#: Training admission is two independent thresholds on the record id: the
#: per-text cap, and the class's share of the mixture. Both are hashes rather
#: than counters so a worker needs no coordination with any other worker, and
#: the second is keyed differently from the first so a record is not judged
#: twice by the same number.
_CLASS_SEED_OFFSET = 0x9E3779B9

#: Every write-pass stratum, in report order. The first three are the real
#: outputs -- shard directories a downstream reader loads, and what
#: FR-143/FR-145 mean by "each output" -- and ``OUTPUTS`` is just that
#: prefix; "dropped-held-out" is accounting only, since nothing is written
#: for it, so it is counted in ``per_stratum`` but never carries a
#: unique-text figure.
STRATA: tuple[str, ...] = ("training", "card-disjoint", "game-disjoint", "dropped-held-out")
OUTPUTS: tuple[str, ...] = STRATA[:-1]


@dataclass(frozen=True, slots=True)
class WriteConfig:
    records_dir: str
    training_dir: str
    card_disjoint_dir: str
    game_disjoint_dir: str
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
    """Filter one shard into the three outputs. Module-level so it pickles."""
    from effects.application.train_effect_model import sampling_class
    from effects.domain.corpus_curation import keeps, record_hash
    from effects.infrastructure.record_io import read_shard, write_shard

    config = _WRITE
    assert config is not None, "init_write_worker was not run"
    out = WriteResult()
    training: list = []
    card_disjoint: list = []
    game_disjoint: list = []

    for record in read_shard(Path(config.records_dir) / relative):
        name = sampling_class(record)
        # Computed once, up front: every branch below -- including the two
        # validation strata, which never used to look at it at all -- needs
        # it to track which ability texts that output actually holds.
        key = ability_key(record)
        out.read[name] += 1
        if record.game_id in config.card_disjoint:
            card_disjoint.append(record)
            out.stratum["card-disjoint"] += 1
            if key is not None:
                out.stratum_keys.setdefault("card-disjoint", set()).add(key)
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
    write_shard(Path(config.training_dir) / shard_name, training)
    write_shard(Path(config.card_disjoint_dir) / shard_name, card_disjoint)
    write_shard(Path(config.game_disjoint_dir) / shard_name, game_disjoint)
    return out


def run_write_pass(
    names: list[str], *, config: WriteConfig, workers: int | None,
    progress_every: int = 50,
) -> WriteResult:
    """Write every shard's three outputs, in parallel, reporting progress."""
    total = WriteResult()

    def absorb(part: WriteResult) -> None:
        total.kept.update(part.kept)
        total.dropped_by_cap.update(part.dropped_by_cap)
        total.read.update(part.read)
        total.stratum.update(part.stratum)
        for name, keys in part.kept_keys.items():
            total.kept_keys.setdefault(name, set()).update(keys)
        for name, keys in part.stratum_keys.items():
            total.stratum_keys.setdefault(name, set()).update(keys)

    if workers is not None and workers <= 1:
        init_write_worker(config)
        for index, name in enumerate(names, start=1):
            absorb(write_shard_pass(name))
            if index % progress_every == 0:
                logger.info("Wrote %d of %d shard(s)", index, len(names))
        return total

    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_write_worker, initargs=(config,),
    ) as pool:
        for index, part in enumerate(
            pool.map(write_shard_pass, names, chunksize=1), start=1,
        ):
            absorb(part)
            if index % progress_every == 0:
                logger.info("Wrote %d of %d shard(s)", index, len(names))
    return total


def build(config: BuildCorpusConfig) -> int:
    """Build a curated dataset, or verify an existing one. Returns an exit code."""
    from effects.application.train_effect_model import (
        load_card_files, load_card_texts, text_keyed_holdout,
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
    sidecars = SidecarCache(folders)
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

    survey = run_survey(
        records_dir,
        config=SurveyConfig(
            records_dir=str(records_dir),
            held_out_names=held_out.names,
            held_out_script_files=held_out.script_files,
            text_cap=config.text_cap,
            seed=config.seed,
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
    admit = {
        name: min(1.0, target / decisions.capped_class_records[name])
        for name, target in decisions.class_targets.items()
        if decisions.capped_class_records.get(name)
    }
    written = run_write_pass(
        [shard.name for shard in survey.shards],
        config=WriteConfig(
            records_dir=str(records_dir),
            training_dir=str(store.training_dir),
            card_disjoint_dir=str(store.card_disjoint_dir),
            game_disjoint_dir=str(store.game_disjoint_dir),
            thresholds=decisions.thresholds,
            class_admit=admit,
            card_disjoint=decisions.card_disjoint,
            game_disjoint=decisions.game_disjoint,
            held_out_games=survey.held_out_games,
            seed=config.seed,
        ),
        workers=config.workers,
    )

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
        holdout_permille=config.holdout_permille,
        holdout_max_carriers=config.holdout_max_carriers,
        text_cap=config.text_cap,
        card_disjoint_text_cap=config.card_disjoint_text_cap,
        game_disjoint_target=config.game_disjoint_target,
        training_records=config.training_records,
        class_mix=config.mix(),
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
