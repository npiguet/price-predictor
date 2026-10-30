"""JSONL shard IO for the effect-record corpus.

The Java collector writes raw shards; Python reads them and writes the
curated shards ``build-corpus`` produces. A line's JSON form is
``record_codec``'s.
"""

from __future__ import annotations

import gzip
import logging
import math
import random
import re
import shutil
import zlib
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import IO, NamedTuple

from effects.domain.corpus_curation import record_hash
from effects.domain.records import EffectRecord
from effects.infrastructure.record_codec import (
    format_record_line,
    parse_record_line,
    record_from_dict,
)
from price_predictor.infrastructure.append_only import (
    count_complete_lines,
    iter_complete_lines,
)

logger = logging.getLogger(__name__)


def _write_line(handle: IO[str], record: EffectRecord) -> None:
    handle.write(format_record_line(record))
    handle.write("\n")


def write_shard(path: Path, records: Iterable[EffectRecord]) -> int:
    """Write a whole shard at once, gzipped; return the record count.

    One gzip member: per-block members guard a worker killed mid-run, and a
    curated shard is written by a process that finishes or leaves nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            _write_line(handle, record)
            written += 1
    return written


#: How many group files the first deal holds open: a large stratum has more
#: shards than one process may hold files open.
_GROUPS = 256


def repack_shards(
    parts: Sequence[Path], out_dir: Path, *, shard_records: int,
    seed: int = 0, records: int | None = None, prefix: str = "shard",
) -> list[Path]:
    """Deal ``parts``' games into shards of about ``shard_records`` records each.

    Whole games are dealt by a seeded hash of ``game_id``, so a game never
    spans two shards and each shard mixes games from across the stratum
    rather than one run's consecutive lifetimes (the trainer shuffles only
    inside the shard it holds). Each shard is then shuffled under ``seed``.

    Streamed through at most :data:`_GROUPS` compressed group files beside
    ``out_dir``, since a stratum does not fit in memory. ``records`` is the
    parts' record count when the caller knows it. ``shard_records <= 0``
    writes one shard per part, unshuffled. Parts are not deleted.
    """
    parts = [Path(p) for p in parts]
    if not parts:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shard_records <= 0:
        return _copy_parts_unshuffled(parts, out_dir, prefix)

    if records is None:
        records = sum(1 for part in parts for _ in iter_shard_lines(part))
    count = max(1, math.ceil(records / shard_records))
    scratch = out_dir.with_name(f".{out_dir.name}.buckets")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        filled = _deal_into_buckets(parts, scratch, seed=seed, count=count)
        # Named before writing: groups write shards out of index order.
        names = {
            index: out_dir / f"{prefix}-{number:05d}.jsonl.gz"
            for number, index in enumerate(sorted(filled), start=1)
        }
        _write_shuffled_buckets(scratch, names, seed=seed, count=count)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return list(names.values())


def _copy_parts_unshuffled(parts: list[Path], out_dir: Path, prefix: str) -> list[Path]:
    written = []
    for number, part in enumerate(parts, start=1):
        path = out_dir / f"{prefix}-{number:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for line in iter_shard_lines(part):
                handle.write(line)
                handle.write("\n")
        written.append(path)
    return written


def _bucket_of(line: str, *, seed: int, count: int) -> int:
    """The index of the output shard a line's game is dealt to."""
    return record_hash(_game_of_line(line), seed=seed) % count


def _group_path(scratch: Path, group: int) -> Path:
    return scratch / f"{group:05d}.gz"


def _deal_into_buckets(
    parts: list[Path], scratch: Path, *, seed: int, count: int,
) -> set[int]:
    """Deal every line into the group of its shard's index; return the indices.

    A group keeps the parts' order, so each shuffle sees a fixed input order.
    """
    groups = min(count, _GROUPS)
    filled: set[int] = set()
    handles = [
        gzip.open(_group_path(scratch, group), "wt", encoding="utf-8", compresslevel=1)
        for group in range(groups)
    ]
    try:
        for part in parts:
            for line in iter_shard_lines(part):
                index = _bucket_of(line, seed=seed, count=count)
                filled.add(index)
                group = handles[index % groups]
                group.write(line)
                group.write("\n")
    finally:
        for handle in handles:
            handle.close()
    return filled


def _write_shuffled_buckets(
    scratch: Path, names: dict[int, Path], *, seed: int, count: int,
) -> None:
    """Split each group into its shards, shuffle each under ``seed``, write it.

    A group is held whole, which is about ``count / _GROUPS`` shards' lines.
    """
    for group in range(min(count, _GROUPS)):
        buckets: dict[int, list[str]] = defaultdict(list)
        for line in iter_shard_lines(_group_path(scratch, group)):
            buckets[_bucket_of(line, seed=seed, count=count)].append(line)
        for index, lines in buckets.items():
            random.Random(f"{seed}:{index}").shuffle(lines)
            with gzip.open(names[index], "wt", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(line)
                    handle.write("\n")


def _game_of_line(line: str) -> str:
    """A shard line's ``game_id``, read off the envelope head without a parse."""
    head = peek_envelope(line)
    if head is not None:
        return head.game_id
    return parse_record_line(line)["game_id"]


#: Both shard spellings: plain `.jsonl` predates compression.
SHARD_GLOBS: tuple[str, ...] = ("*.jsonl.gz", "*.jsonl")


def iter_shard_lines(path: Path) -> Iterator[str]:
    """Every complete, non-blank line of a shard, its terminator stripped.

    One shape for both spellings, so a caller that writes lines back adds the
    newline itself whichever spelling it read. Blank lines are skipped: every
    reader ignores them, and a copy that kept them would be the one place
    they survived. A truncated final gzip member, or a trailing partial line,
    ends the stream.
    """
    path = Path(path)
    if path.suffix != ".gz":
        for line in iter_complete_lines(path):
            if line.strip():
                yield line
        return

    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                # A line without its newline can only be a member cut short.
                if line.endswith("\n") and line.strip():
                    yield line[:-1]
    except (EOFError, gzip.BadGzipFile, zlib.error):
        return


def iter_record_dicts(path: Path) -> Iterator[dict]:
    """Every complete record of one shard, as the dict its line parses to.

    For a caller that needs the raw JSON, such as a generation check.
    """
    for line in iter_shard_lines(Path(path)):
        yield parse_record_line(line)


def read_window(
    directory: Path, limit: int = 0,
    observe: Callable[[str, dict], None] | None = None,
) -> Iterator[EffectRecord]:
    """The first ``limit`` records under ``directory``; 0 reads all of them.

    ``observe``, when given, sees each record's raw dict, with its shard's
    path relative to ``directory``, before it is parsed: a parsed record
    cannot say whether a field was on the wire.
    """
    root = Path(directory)
    seen = 0
    for shard in iter_shards(root):
        name = shard.relative_to(root).as_posix()
        for data in iter_record_dicts(shard):
            if observe is not None:
                observe(name, data)
            yield record_from_dict(data)
            seen += 1
            if limit and seen >= limit:
                return


def read_shard(path: Path) -> Iterator[EffectRecord]:
    """Yield every complete record in one shard; skip a trailing partial line."""
    for data in iter_record_dicts(path):
        yield record_from_dict(data)


def load_shard(path: Path) -> list[EffectRecord]:
    """One shard's records, the largest unit the trainer ever holds."""
    return list(read_shard(Path(path)))


def iter_shards(directory: Path) -> list[Path]:
    """Every shard at or under ``directory``, in path order, either spelling.

    Recursive, so collection runs kept in subdirectories read as one corpus.
    Path order rather than name order, so every reader walks a corpus in the
    same order on every run.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        path for pattern in SHARD_GLOBS for path in directory.rglob(pattern)
    )


def read_records(directory: Path) -> Iterator[EffectRecord]:
    """Stream every record under ``directory``, shard by shard in path order.

    A missing directory yields nothing: an empty corpus, not an error.
    """
    for shard in iter_shards(directory):
        yield from read_shard(shard)


#: The envelope's head as both writers lay it out: the Java
#: ``EffectRecord.toJson`` and ``record_codec.record_to_dict`` emit ``record_id``,
#: ``run_id``, ``timestamp``, ``game_id`` and ``kind`` first, in that order.
_ENVELOPE_HEAD = re.compile(
    r'"record_id":"([^"]*)",.*?"game_id":"([^"]*)","kind":"([a-z_]+)"'
)

#: How far into a line the head is looked for. The five leading fields are a
#: few hundred characters; the state snapshot that follows is the bulk.
_ENVELOPE_HEAD_SPAN = 512


class EnvelopeHead(NamedTuple):
    """A shard line's leading envelope fields, read without parsing it."""

    record_id: str
    game_id: str
    kind: str


def peek_envelope(line: str) -> EnvelopeHead | None:
    """The head of one shard line, or None where it has another shape, which
    the caller then parses: a reordered envelope costs speed, never records."""
    head = _ENVELOPE_HEAD.search(line, 0, _ENVELOPE_HEAD_SPAN)
    return None if head is None else EnvelopeHead(*head.groups())


def read_records_matching(
    directory: Path,
    *,
    game_ids: Collection[str] | None = None,
    kinds: Collection[str] | None = None,
) -> Iterator[EffectRecord]:
    """The records under ``directory`` of the named games and kinds.

    The same records a filter over :func:`read_records` keeps, without
    parsing the rest; a line :func:`peek_envelope` cannot read is parsed and
    filtered the slow way.
    """
    shards = iter_shards(directory)
    for done, shard in enumerate(shards):
        if done and done % 250 == 0:
            logger.info("scanned %d of %d shards under %s", done, len(shards), directory)
        yield from iter_shard_matching(shard, game_ids=game_ids, kinds=kinds)


def iter_shard_matching(
    shard: Path,
    *,
    game_ids: Collection[str] | None = None,
    kinds: Collection[str] | None = None,
) -> Iterator[EffectRecord]:
    """The records of one shard of the named games and kinds, the rest
    skipped unparsed; :func:`read_records_matching` for one shard."""
    for line in iter_shard_lines(shard):
        head = peek_envelope(line)
        if head is not None and not _wanted(head.game_id, head.kind, game_ids, kinds):
            continue
        record = record_from_dict(parse_record_line(line))
        if head is None and not _wanted(record.game_id, record.kind.value, game_ids, kinds):
            continue
        yield record


def _wanted(
    game_id: str, kind: str, game_ids: Collection[str] | None, kinds: Collection[str] | None,
) -> bool:
    """Whether a record of ``game_id`` and ``kind`` passes both filters."""
    return (game_ids is None or game_id in game_ids) and (kinds is None or kind in kinds)


def count_records(directory: Path, *, ceiling: int = 0) -> int:
    """Complete records under ``directory``, for progress and resume.

    Counts complete lines without parsing them. A non-zero ``ceiling`` stops
    the count there and returns it, since decompressing a full corpus takes
    minutes.
    """
    total = 0
    for shard in iter_shards(directory):
        total += count_shard_records(shard)
        if ceiling and total >= ceiling:
            return ceiling
    return total


def count_shard_records(shard: Path) -> int:
    """Complete records in one shard, counted as :func:`count_records` does:
    blank lines are not records, in either spelling."""
    if shard.suffix == ".gz":
        return sum(1 for _ in iter_shard_lines(shard))
    return count_complete_lines(shard, skip_blank=True)


# ── shard generation (FR-033) ───────────────────────────────────────────

class ShardGeneration(StrEnum):
    """The two record generations a shard can hold.

    A gen-2 record always carries ``random_seat``; a gen-1 one never does, and
    on gen-1 ``playability`` records ``actor_player`` keeps feature 023's
    meaning rather than naming the deciding player (FR-030a).
    """

    GEN_1 = "gen-1"
    GEN_2 = "gen-2"


class MixedGenerationsError(ValueError):
    """A records set holds both gen-1 and gen-2 shards."""


def shard_generation(path: Path) -> ShardGeneration | None:
    """The shard's generation by its first complete record; None if empty.

    One record speaks for its shard: ``validate-corpus`` fails a shard whose
    records disagree.
    """
    for data in iter_record_dicts(path):
        return ShardGeneration.GEN_2 if "random_seat" in data else ShardGeneration.GEN_1
    return None


def refuse_mixed_generations(paths: Iterable[Path]) -> None:
    """Raise when ``paths`` hold both generations.

    Every reader of a records set calls this first (FR-033): the redefined
    ``actor_player`` has no version field of its own.
    """
    seen: dict[ShardGeneration, Path] = {}
    for path in paths:
        generation = shard_generation(path)
        if generation is not None:
            seen.setdefault(generation, Path(path))
    if len(seen) > 1:
        raise MixedGenerationsError(
            "this records set mixes gen-1 shards (no random_seat field) with "
            f"gen-2 shards, e.g. {seen[ShardGeneration.GEN_1]} and "
            f"{seen[ShardGeneration.GEN_2]}. The two read "
            "actor_player on playability records differently (FR-030a); keep "
            "them in separate directories and read one at a time."
        )
