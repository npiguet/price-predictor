"""The fixed validation samples a curated corpus ships (FR-135, FR-089).

Drawn once, by smallest seeded record hash, and written beside the strata. The
card-disjoint stratum's resolution classes draw from the gate-one slice first,
the population checkpoints are selected on, and round-robin over the held-out
texts (FR-051), so that number weighs held-out texts alike rather than by how
often Forge cast them.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from effects.domain.corpus_curation import CapHeap, KeyedCapHeap, record_hash
from effects.domain.records import EffectRecord
from effects.domain.sampling import SamplingClass
from effects.domain.strata import VALIDATION_STRATA, Stratum

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
    from effects.infrastructure.record_io import read_records

    per_text: dict[str, dict[str, CapHeap]] = {
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
            heap = per_text[name][text] = CapHeap(quota[name])
        heap.offer(_sample_hash(record, seed))
    chosen = {
        name: set(round_robin(
            {text: list(heap.values()) for text, heap in texts.items()}, quota[name],
        ))
        for name, texts in per_text.items()
    }
    out: dict[str, list[EffectRecord]] = {name: [] for name in per_text}
    seen: set[str] = set()
    for record in read_records(directory):
        name = sampling_class(record)
        if name not in chosen or record.record_id in seen:
            continue
        if _sample_hash(record, seed) in chosen[name]:
            seen.add(record.record_id)
            out[name].append(record)
    return out


def class_quota(mix: Mapping[str, float], size: int) -> dict[str, int]:
    """Records per class in a sample of ``size``: each share rounded, never zero."""
    return {name: max(1, round(share * size)) for name, share in mix.items()}


def _collect(
    directory: Path, quota: Mapping[str, int], *, seed: int,
    only: frozenset[str] | None = None,
) -> dict[str, KeyedCapHeap[EffectRecord]]:
    from effects.domain.sampling import sampling_class
    from effects.infrastructure.record_io import read_records

    heaps: dict[str, KeyedCapHeap[EffectRecord]] = {
        name: KeyedCapHeap(cap) for name, cap in quota.items()
    }
    offered = 0
    for record in read_records(directory):
        name = sampling_class(record)
        if name not in heaps or (only is not None and name not in only):
            continue
        # A counter ends the tie-break: a repeated record_id would otherwise
        # compare two EffectRecords, which have no ordering.
        offered += 1
        heaps[name].offer(_sample_hash(record, seed), record, (record.record_id, offered))
    return heaps


def draw_samples(
    *, card_disjoint: Path, game_disjoint: Path, gate_one: Path,
    mix: Mapping[str, float], size: int, seed: int,
    held_out_text_of: Callable[[EffectRecord], str | None] | None = None,
) -> dict[str, list[EffectRecord]]:
    """One mixture-matched sample per stratum, seeded and shuffled.

    ``size <= 0`` reads nothing and returns both strata empty. Without
    ``held_out_text_of`` the gate-one slice is drawn by smallest hash alone
    rather than round-robin.
    """
    if size <= 0:
        return {stratum: [] for stratum in VALIDATION_STRATA}

    quota = class_quota(mix, size)
    out: dict[str, list[EffectRecord]] = {}

    if held_out_text_of is not None:
        gate_records = _gate_one_round_robin(
            gate_one, quota, seed=seed, held_out_text_of=held_out_text_of,
        )
    else:
        gate = _collect(gate_one, quota, seed=seed, only=GATE_ONE_CLASSES)
        gate_records = {name: heap.items() for name, heap in gate.items()}
    for stratum, directory in zip(VALIDATION_STRATA, (card_disjoint, game_disjoint)):
        heaps = _collect(directory, quota, seed=seed)
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


def write_samples(store, samples: Mapping[str, Iterable[EffectRecord]]) -> dict[str, int]:
    from effects.infrastructure.record_io import write_shard

    return {
        stratum: write_shard(store.sample_path(stratum), records)
        for stratum, records in samples.items()
    }
