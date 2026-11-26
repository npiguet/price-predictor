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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
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


def _natural_replaces(held: EffectRecord | None, record: EffectRecord | None) -> bool:
    """Whether ``record`` is a natural copy of the record whose added copy is ``held``."""
    if held is None or record is None:
        return False
    return _is_added(held) and not _is_added(record)


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
            if _natural_replaces(held[1], record):
                self._held[record_id] = (value, record)
            return
        if self._cap <= 0:
            return
        if len(self._held) >= self._cap and not self._evict_for(value):
            return
        self._held[record_id] = (value, record)
        heapq.heappush(self._order, (-value, record_id))

    def _evict_for(self, value: int) -> bool:
        """Drops the largest hash held if ``value`` is smaller; whether it did."""
        if -self._order[0][0] <= value:
            return False
        _, evicted = heapq.heappop(self._order)
        del self._held[evicted]
        return True

    def values(self) -> list[int]:
        return [value for value, _ in self._held.values()]

    def items(self) -> list[EffectRecord]:
        """The held records, smallest hash first (ties by ``record_id``)."""
        return [record for _, _, record in sorted(
            (value, record_id, record) for record_id, (value, record) in self._held.items())]

    def __len__(self) -> int:
        return len(self._held)


def _in_round_order(ordered: Sequence[list[int]], per_round: int) -> Iterator[int]:
    """Each text's next ``per_round`` values in turn, round after round."""
    for start in range(0, max(map(len, ordered), default=0), per_round):
        for values in ordered:
            yield from values[start:start + per_round]


def round_robin(by_text: Mapping[str, list[int]], quota: int,
                per_round: int = RECORDS_PER_TEXT_PER_ROUND) -> list[int]:
    """Values taken ``per_round`` at a time from each text in sorted order,
    smallest first, until ``quota`` or every text is exhausted."""
    ordered = [sorted(by_text[text]) for text in sorted(by_text)]
    return list(islice(_in_round_order(ordered, per_round), max(quota, 0)))


def _records_of(shards: Sequence[Path]) -> Iterator[EffectRecord]:
    """Every record of ``shards``, shard by shard in the order given: the
    shards a build wrote, never a directory's."""
    from effects.infrastructure.shard_io import read_shard

    for shard in shards:
        yield from read_shard(shard)


def _sample_of(texts: dict[str, _DistinctSample], text: str, cap: int) -> _DistinctSample:
    if text not in texts:
        texts[text] = _DistinctSample(cap)
    return texts[text]


def _hashes_per_text(
    shards: Sequence[Path], quota: Mapping[str, int], *, seed: int,
    held_out_text_of: Callable[[EffectRecord], str | None],
) -> dict[str, dict[str, _DistinctSample]]:
    """First read: each gate-one class's smallest hashes per held-out text."""
    from effects.domain.sampling import sampling_class

    per_text: dict[str, dict[str, _DistinctSample]] = {
        name: {} for name in quota if name in GATE_ONE_CLASSES
    }
    for record in _records_of(shards):
        name = sampling_class(record)
        text = held_out_text_of(record) if name in per_text else None
        if text is not None:
            _sample_of(per_text[name], text, quota[name]).offer(
                _sample_hash(record, seed), record.record_id)
    return per_text


def _keep_natural(picked: dict[str, EffectRecord], record: EffectRecord) -> None:
    held = picked.get(record.record_id)
    if held is None or _natural_replaces(held, record):
        picked[record.record_id] = record


def _records_of_hashes(
    shards: Sequence[Path], chosen: Mapping[str, set[int]], *, seed: int,
) -> dict[str, dict[str, EffectRecord]]:
    """Second read: each class's records whose hash was chosen, one per ``record_id``."""
    from effects.domain.sampling import sampling_class

    picked: dict[str, dict[str, EffectRecord]] = {name: {} for name in chosen}
    for record in _records_of(shards):
        name = sampling_class(record)
        if name in chosen and _sample_hash(record, seed) in chosen[name]:
            _keep_natural(picked[name], record)
    return picked


def _gate_one_round_robin(
    shards: Sequence[Path], quota: Mapping[str, int], *, seed: int,
    held_out_text_of: Callable[[EffectRecord], str | None],
) -> dict[str, list[EffectRecord]]:
    """Each gate-one class's records, chosen round-robin over held-out texts.

    Two reads, the first keeping only hashes, so memory is bounded by the
    quota rather than by the slice.
    """
    per_text = _hashes_per_text(shards, quota, seed=seed, held_out_text_of=held_out_text_of)
    chosen = {
        name: set(round_robin(
            {text: list(heap.values()) for text, heap in texts.items()}, quota[name],
        ))
        for name, texts in per_text.items()
    }
    picked = _records_of_hashes(shards, chosen, seed=seed)
    return {name: list(records.values()) for name, records in picked.items()}


def class_quota(mix: Mapping[str, float], size: int) -> dict[str, int]:
    """Records per class in a sample of ``size``: each share rounded, never zero."""
    return {name: max(1, round(share * size)) for name, share in mix.items()}


def _collect(
    shards: Sequence[Path], quota: Mapping[str, int], *, seed: int,
) -> dict[str, _DistinctSample]:
    from effects.domain.sampling import sampling_class

    heaps = {name: _DistinctSample(cap) for name, cap in quota.items()}
    for record in _records_of(shards):
        name = sampling_class(record)
        if name in heaps:
            heaps[name].offer(_sample_hash(record, seed), record.record_id, record)
    return heaps


@dataclass(frozen=True)
class SampleSpec:
    """What a draw is: the class ``mix`` of ``size`` records per stratum, the
    ``seed``, and ``held_out_text_of``, which names the held-out text a
    gate-one record acts through, which the round-robin spreads that slice over."""

    mix: Mapping[str, float]
    size: int
    seed: int
    held_out_text_of: Callable[[EffectRecord], str | None]


def _preselected(
    stratum: str, name: str, cap: int, gate_records: Mapping[str, list[EffectRecord]],
) -> list[EffectRecord]:
    """The records a class takes before the stratum's own: the gate-one slice's,
    in the card-disjoint stratum's resolution classes."""
    if stratum is Stratum.CARD_DISJOINT and name in GATE_ONE_CLASSES:
        return gate_records.get(name, [])[:cap]
    return []


def _top_up(chosen: list[EffectRecord], heap: _DistinctSample, cap: int) -> None:
    """Appends ``heap``'s smallest records not yet chosen until ``cap`` are."""
    taken = {r.record_id for r in chosen}
    fresh = (record for record in heap.items() if record.record_id not in taken)
    chosen.extend(islice(fresh, max(cap - len(chosen), 0)))


def _stratum_sample(
    stratum: str, heaps: Mapping[str, _DistinctSample], quota: Mapping[str, int],
    gate_records: Mapping[str, list[EffectRecord]],
) -> list[EffectRecord]:
    sample: list[EffectRecord] = []
    for name, cap in quota.items():
        chosen = _preselected(stratum, name, cap, gate_records)
        _top_up(chosen, heaps[name], cap)
        if len(chosen) < cap:
            logger.warning(
                "%s sample: %s holds %d record(s) of the %d its share asks for",
                stratum, name, len(chosen), cap,
            )
        sample.extend(chosen)
    return sample


def draw_samples(
    shards: Mapping[str, Sequence[Path]], *, spec: SampleSpec,
) -> dict[str, list[EffectRecord]]:
    """One mixture-matched sample per validation stratum, drawn from the
    ``shards`` the build wrote (each stratum's, by name), seeded and shuffled.

    ``spec.size <= 0`` reads nothing and returns both strata empty.
    """
    if spec.size <= 0:
        return {stratum: [] for stratum in VALIDATION_STRATA}

    quota = class_quota(spec.mix, spec.size)
    out: dict[str, list[EffectRecord]] = {}

    gate_records = _gate_one_round_robin(
        shards[str(Stratum.GATE_ONE)], quota, seed=spec.seed,
        held_out_text_of=spec.held_out_text_of,
    )
    for stratum in VALIDATION_STRATA:
        heaps = _collect(shards[str(stratum)], quota, seed=spec.seed)
        sample = _stratum_sample(stratum, heaps, quota, gate_records)
        random.Random(f"{spec.seed}:{stratum}").shuffle(sample)
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
