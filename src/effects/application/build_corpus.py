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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain.corpus_curation import CapHeap, record_hash
from effects.domain.corpus_manifest import SourceShard
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import EffectRecord

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


def merge_surveys(parts) -> Survey:
    """Combine shard surveys into one corpus-wide picture."""
    config = _CONFIG
    cap = config.text_cap if config is not None else 0
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
    records_dir,
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
