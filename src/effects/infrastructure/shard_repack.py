"""Repacking a stratum's parts into the curated shards.

Curation (``application/build_corpus``) writes one part per source shard;
:func:`repack_shards` deals their games into shards of uniform size, mixed
across the stratum, so the trainer's per-shard step budget is even.
"""

from __future__ import annotations

import gzip
import math
import random
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from effects.domain.corpus_curation import record_hash
from effects.infrastructure.record_codec import parse_record_line
from effects.infrastructure.shard_io import iter_shard_lines, peek_envelope

#: How many group files the first deal holds open: a large stratum has more
#: shards than one process may hold files open.
_GROUPS = 256

#: Between a group line's shard index and its record: no digit, so the first
#: one on the line is always this.
_INDEX_SEPARATOR = ":"


@dataclass(frozen=True, slots=True)
class RepackSettings:
    """How :func:`repack_shards` cuts the parts into shards.

    ``shard_records`` is about how many records a shard holds; ``<= 0``
    writes one shard per part, unshuffled. ``seed`` fixes the deal and each
    shuffle. ``records`` is the parts' record count when the caller knows it.
    ``prefix`` names the shards (``<prefix>-00001.jsonl.gz``).
    """

    shard_records: int
    seed: int = 0
    records: int | None = None
    prefix: str = "shard"


def repack_shards(
    parts: Sequence[Path], out_dir: Path, settings: RepackSettings,
) -> dict[Path, int]:
    """Deal ``parts``' games into shards of about ``settings.shard_records``
    records each; every shard written, in name order, with the records written
    into it.

    Whole games are dealt by a seeded hash of ``game_id``, so a game never
    spans two shards and each shard mixes games from across the stratum
    rather than one run's consecutive lifetimes (the trainer shuffles only
    inside the shard it holds). Each shard is then shuffled under the seed.

    Streamed through at most :data:`_GROUPS` compressed group files beside
    ``out_dir``, since a stratum does not fit in memory. Parts are not deleted.
    """
    parts = [Path(p) for p in parts]
    if not parts:
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if settings.shard_records <= 0:
        return _copy_parts_unshuffled(parts, out_dir, settings.prefix)
    return _deal_and_shuffle(parts, out_dir, settings)


def _copy_parts_unshuffled(parts: list[Path], out_dir: Path, prefix: str) -> dict[Path, int]:
    written = {}
    for number, part in enumerate(parts, start=1):
        path = out_dir / f"{prefix}-{number:05d}.jsonl.gz"
        written[path] = _write_lines(path, iter_shard_lines(part))
    return written


def _deal_and_shuffle(
    parts: list[Path], out_dir: Path, settings: RepackSettings,
) -> dict[Path, int]:
    """The shuffled shards of ``parts``, dealt through a scratch directory
    beside ``out_dir`` that is gone afterwards."""
    records = settings.records
    if records is None:
        records = sum(1 for part in parts for _ in iter_shard_lines(part))
    count = max(1, math.ceil(records / settings.shard_records))
    seed = settings.seed
    scratch = out_dir.with_name(f".{out_dir.name}.buckets")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        filled = _deal_into_buckets(parts, scratch, seed=seed, count=count)
        # Named before writing: groups write shards out of index order.
        names = {
            index: out_dir / f"{settings.prefix}-{number:05d}.jsonl.gz"
            for number, index in enumerate(sorted(filled), start=1)
        }
        written = _write_shuffled_buckets(scratch, names, seed=seed, count=count)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    # Zero-padded names: path order is name order.
    return dict(sorted(written.items()))


def _write_lines(path: Path, lines: Iterable[str]) -> int:
    """Write ``lines`` as one gzipped shard, a newline after each; how many."""
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
            count += 1
    return count


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
    Each group line carries its shard's index ahead of the record (``index:``),
    so the split that follows does not hash the line a second time.
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
                group.write(f"{index}{_INDEX_SEPARATOR}{line}\n")
    finally:
        for handle in handles:
            handle.close()
    return filled


def _write_shuffled_buckets(
    scratch: Path, names: dict[int, Path], *, seed: int, count: int,
) -> dict[Path, int]:
    """Split each group into its shards by the index each line carries,
    shuffle each under ``seed``, write it; each shard's records.

    A group is held whole, which is about ``count / _GROUPS`` shards' lines.
    """
    written: dict[Path, int] = {}
    for group in range(min(count, _GROUPS)):
        buckets: dict[int, list[str]] = defaultdict(list)
        for tagged in iter_shard_lines(_group_path(scratch, group)):
            index, _, line = tagged.partition(_INDEX_SEPARATOR)
            buckets[int(index)].append(line)
        for index, lines in buckets.items():
            random.Random(f"{seed}:{index}").shuffle(lines)
            written[names[index]] = _write_lines(names[index], lines)
    return written


def _game_of_line(line: str) -> str:
    """A shard line's ``game_id``, read off the envelope head without a parse."""
    head = peek_envelope(line)
    if head is not None:
        return head.game_id
    return parse_record_line(line)["game_id"]
