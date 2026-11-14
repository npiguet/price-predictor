"""The fixed validation samples a curated corpus ships (FR-135, FR-089).

Drawn once, by smallest seeded record hash, and written beside the strata. The
card-disjoint stratum's resolution classes draw from the gate-one slice first,
the population checkpoints are selected on, and round-robin over the held-out
texts (FR-051), so that number weighs held-out texts alike rather than by how
often Forge cast them.

A stratum writes a record once per copy, every copy under its one
``record_id``; a sample holds each record once, its natural copy where it has
one (an added copy only where none was written), so a class's quota counts
distinct records whatever its copies.
"""

from __future__ import annotations

import heapq
import logging
import random
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.corpus_curation import record_hash
from effects.domain.records import EffectRecord
from effects.domain.sampling import SamplingClass
from effects.domain.strata import VALIDATION_STRATA, Stratum

if TYPE_CHECKING:
    from effects.infrastructure.corpus_store import CorpusStore

logger = logging.getLogger(__name__)

SAMPLE_SEED_OFFSET = 0x2545F491
#: Card-disjoint classes drawn from the gate-one slice before the stratum.
GATE_ONE_CLASSES = frozenset({
    SamplingClass.RESOLUTION_EFFECT, SamplingClass.RESOLUTION_COST,
})
#: Records each held-out text gives per round of the round-robin (FR-051).
RECORDS_PER_TEXT_PER_ROUND = 4


def _sample_hash(record: EffectRecord, seed: int) -> int:
    return record_hash(record.record_id, seed=seed + SAMPLE_SEED_OFFSET)


def _is_added(record: EffectRecord) -> bool:
    return record.curation is not None and record.curation.is_added


class _DistinctSample:
    """The ``cap`` records of smallest hash, each ``record_id`` once.

    Every copy of a record shares its hash; a natural copy replaces an added
    one already held, so the sample keeps the copy that trains every output.
    """

    __slots__ = ("_cap", "_held", "_order")

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._held: dict[str, tuple[int, EffectRecord | None]] = {}
        self._order: list[tuple[int, str]] = []  # (-hash, record_id): largest hash on top

    def offer(self, value: int, record_id: str, record: EffectRecord | None = None) -> None:
        held = self._held.get(record_id)
        if held is not None:
            if record is not None and held[1] is not None and _is_added(held[1]) and not (
                    _is_added(record)):
                self._held[record_id] = (value, record)
            return
        if self._cap <= 0:
            return
        if len(self._held) >= self._cap:
            if -self._order[0][0] <= value:
                return
            _, evicted = heapq.heappop(self._order)
            del self._held[evicted]
        self._held[record_id] = (value, record)
        heapq.heappush(self._order, (-value, record_id))

    def values(self) -> list[int]:
        return [value for value, _ in self._held.values()]

    def items(self) -> list[EffectRecord]:
        """The held records, smallest hash first (ties by ``record_id``)."""
        return [record for _, _, record in sorted(
            (value, record_id, record) for record_id, (value, record) in self._held.items())]

    def __len__(self) -> int:
        return len(self._held)


def round_robin(by_text: Mapping[str, list[int]], quota: int,
                per_round: int = RECORDS_PER_TEXT_PER_ROUND) -> list[int]:
    """Values taken ``per_round`` at a time from each text in sorted order,
    smallest first, until ``quota`` or every text is exhausted."""
    ordered = {text: sorted(values) for text, values in sorted(by_text.items())}
    taken: list[int] = []
    start = 0
    while len(taken) < quota:
        progressed = False
        for values in ordered.values():
            chunk = values[start:start + per_round]
            if chunk:
                progressed = True
            for value in chunk:
                if len(taken) >= quota:
                    return taken
                taken.append(value)
        if not progressed:
            break
        start += per_round
    return taken


def _gate_one_round_robin(
    directory: Path, quota: Mapping[str, int], *, seed: int,
    held_out_text_of: Callable[[EffectRecord], str | None],
) -> dict[str, list[EffectRecord]]:
    """Each gate-one class's records, chosen round-robin over held-out texts.

    Two reads, the first keeping only hashes, so memory is bounded by the
    quota rather than by the slice.
    """
    from effects.domain.sampling import sampling_class
    from effects.infrastructure.shard_io import read_records

    per_text: dict[str, dict[str, _DistinctSample]] = {
        name: {} for name in quota if name in GATE_ONE_CLASSES
    }
    for record in read_records(directory):
        name = sampling_class(record)
        if name not in per_text:
            continue
        text = held_out_text_of(record)
        if text is None:
            continue
        heap = per_text[name].get(text)
        if heap is None:
            heap = per_text[name][text] = _DistinctSample(quota[name])
        heap.offer(_sample_hash(record, seed), record.record_id)
    chosen = {
        name: set(round_robin(
            {text: list(heap.values()) for text, heap in texts.items()}, quota[name],
        ))
        for name, texts in per_text.items()
    }
    picked: dict[str, dict[str, EffectRecord]] = {name: {} for name in per_text}
    for record in read_records(directory):
        name = sampling_class(record)
        if name not in chosen or _sample_hash(record, seed) not in chosen[name]:
            continue
        held = picked[name].get(record.record_id)
        if held is None or (_is_added(held) and not _is_added(record)):
            picked[name][record.record_id] = record
    return {name: list(records.values()) for name, records in picked.items()}


def class_quota(mix: Mapping[str, float], size: int) -> dict[str, int]:
    """Records per class in a sample of ``size``: each share rounded, never zero."""
    return {name: max(1, round(share * size)) for name, share in mix.items()}


def _collect(
    directory: Path, quota: Mapping[str, int], *, seed: int,
) -> dict[str, _DistinctSample]:
    from effects.domain.sampling import sampling_class
    from effects.infrastructure.shard_io import read_records

    heaps = {name: _DistinctSample(cap) for name, cap in quota.items()}
    for record in read_records(directory):
        name = sampling_class(record)
        if name in heaps:
            heaps[name].offer(_sample_hash(record, seed), record.record_id, record)
    return heaps


def draw_samples(
    store: CorpusStore, *, mix: Mapping[str, float], size: int, seed: int,
    held_out_text_of: Callable[[EffectRecord], str | None],
) -> dict[str, list[EffectRecord]]:
    """One mixture-matched sample per validation stratum of ``store``,
    seeded and shuffled.

    ``held_out_text_of`` names the held-out text a gate-one record acts
    through, which the round-robin spreads that slice over. ``size <= 0``
    reads nothing and returns both strata empty.
    """
    if size <= 0:
        return {stratum: [] for stratum in VALIDATION_STRATA}

    quota = class_quota(mix, size)
    out: dict[str, list[EffectRecord]] = {}

    gate_records = _gate_one_round_robin(
        store.gate_one_dir, quota, seed=seed, held_out_text_of=held_out_text_of,
    )
    for stratum in VALIDATION_STRATA:
        heaps = _collect(store.output_dir(stratum), quota, seed=seed)
        sample: list[EffectRecord] = []
        for name, cap in quota.items():
            chosen: list[EffectRecord] = []
            if stratum is Stratum.CARD_DISJOINT and name in GATE_ONE_CLASSES:
                chosen = gate_records.get(name, [])[:cap]
            taken = {r.record_id for r in chosen}
            for record in heaps[name].items():
                if len(chosen) >= cap:
                    break
                if record.record_id not in taken:
                    chosen.append(record)
            if len(chosen) < cap:
                logger.warning(
                    "%s sample: %s holds %d record(s) of the %d its share asks for",
                    stratum, name, len(chosen), cap,
                )
            sample.extend(chosen)
        random.Random(f"{seed}:{stratum}").shuffle(sample)
        out[stratum] = sample
    return out


def write_samples(
    store: CorpusStore, samples: Mapping[str, Iterable[EffectRecord]],
) -> dict[str, int]:
    from effects.infrastructure.shard_io import write_shard

    return {
        stratum: write_shard(store.sample_path(stratum), records)
        for stratum, records in samples.items()
    }
