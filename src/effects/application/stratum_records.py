"""Reading a validation stratum for the control score: the records of a
half's games its items name, each once, and the deviance sample's copies,
counted as the freeze counts them."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from effects.domain.records import EffectRecord


#: Reads one stratum: a fresh iterable of its shards in path order, each an
#: iterable of its records in line order (``freeze_selection_set.stratum_reader``).
StratumReader = Callable[[], Iterable[Iterable["EffectRecord"]]]


def selection_records(
    read: StratumReader, *, games: frozenset[str], records: frozenset[str],
    copies: frozenset[tuple[str, int]],
) -> Iterator[tuple[EffectRecord, bool, bool]]:
    """``(record, distinct, copy)`` for every record of ``games`` that is the
    first occurrence of a record in ``records`` (``distinct``) or a copy in
    ``copies`` (``copy``), copies counted as the freeze counts them: a
    record's earlier occurrences in the stratum, read in shard path order."""
    occurrences: dict[str, int] = defaultdict(int)
    for shard in read():
        for record in shard:
            copy = occurrences[record.record_id]
            occurrences[record.record_id] += 1
            if record.game_id not in games:
                continue
            distinct = copy == 0 and record.record_id in records
            sampled = (record.record_id, copy) in copies
            if distinct or sampled:
                yield record, distinct, sampled
