"""Reading a validation stratum: every record with its copy number, as the
freeze and the control score count copies, and the control score's records
of a half's games its items name, each once, with the deviance sample's
copies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from effects.domain.records import EffectRecord


#: Reads one stratum: a fresh iterable of its shards in path order, each an
#: iterable of its records in line order (``freeze_selection_set.stratum_reader``).
StratumReader = Callable[[], Iterable[Iterable["EffectRecord"]]]

#: One shard's records in line order, each with its copy number.
CountedShard = Iterator[tuple["EffectRecord", int]]


def shard_copies(read: StratumReader) -> Iterator[CountedShard]:
    """The stratum's shards in path order, each yielding ``(record, copy)``
    for its records in line order, ``copy`` the number of the record's
    earlier occurrences in the stratum. Each shard is counted as it is read,
    so read every shard to its end before the next."""
    occurrences: dict[str, int] = defaultdict(int)
    for shard in read():
        yield _counted(shard, occurrences)


def _counted(shard: Iterable[EffectRecord], occurrences: dict[str, int]) -> CountedShard:
    for record in shard:
        copy = occurrences[record.record_id]
        occurrences[record.record_id] += 1
        yield record, copy


def stratum_copies(read: StratumReader) -> CountedShard:
    """``(record, copy)`` for every record of the stratum, in reading order
    (:func:`shard_copies`, its shards joined)."""
    for shard in shard_copies(read):
        yield from shard


def selection_records(
    read: StratumReader, *, games: frozenset[str], records: frozenset[str],
    copies: frozenset[tuple[str, int]],
) -> Iterator[tuple[EffectRecord, bool, bool]]:
    """``(record, distinct, copy)`` for every record of ``games`` that is the
    first occurrence of a record in ``records`` (``distinct``) or a copy in
    ``copies`` (``copy``), copies counted as the freeze counts them
    (:func:`stratum_copies`)."""
    for record, copy in stratum_copies(read):
        if record.game_id not in games:
            continue
        distinct = copy == 0 and record.record_id in records
        sampled = (record.record_id, copy) in copies
        if distinct or sampled:
            yield record, distinct, sampled
