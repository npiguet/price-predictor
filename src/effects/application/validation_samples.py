"""The fixed validation samples a curated corpus ships (FR-135, FR-089).

The trainer used to sweep every validation shard at startup to draw a
mixture-matched sample, and drew 20 gate-1 records into a 2,048-record
card-disjoint sample because it filled each class with whatever arrived
first. The sample is a property of the corpus, drawn once here by smallest
seeded record hash — the same device the per-text cap uses — and written
beside the strata so the trainer reads it and the evaluator can name it.

For the card-disjoint stratum the two resolution classes draw from the
gate-one slice first: that is the population the model ships on, and the
number that selects checkpoints should measure it. They draw from it
**round-robin over the held-out texts** (FR-051), a few records per text per
round by smallest record hash within the text, so a text with thousands of
records cannot fill the slots a text with ten needs: the number that selects
checkpoints then weighs held-out texts alike rather than by how often Forge
happened to cast them.
"""

from __future__ import annotations

import heapq
import logging
import random
from collections.abc import Iterable, Mapping
from pathlib import Path

from effects.domain.corpus_curation import record_hash
from effects.domain.effect_model import CLASS_RESOLUTION_COST, CLASS_RESOLUTION_EFFECT
from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)

SAMPLE_SEED_OFFSET = 0x2545F491
STRATA = ("card-disjoint", "game-disjoint")
#: Card-disjoint classes drawn from the gate-one slice before the stratum.
GATE_ONE_CLASSES = frozenset({CLASS_RESOLUTION_EFFECT, CLASS_RESOLUTION_COST})
#: Records each held-out text gives per round of the round-robin (FR-051).
RECORDS_PER_TEXT_PER_ROUND = 4


def round_robin(by_text: Mapping[str, list[int]], quota: int,
                per_round: int = RECORDS_PER_TEXT_PER_ROUND) -> list[int]:
    """Values taken ``per_round`` at a time from each text in turn, smallest first.

    Texts are visited in sorted order and each text's values in ascending
    order, so the result is a pure function of its inputs. Stops at ``quota``
    or when every text is exhausted.
    """
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
    text_of_key: Mapping[str, str],
) -> dict[str, list[EffectRecord]]:
    """Each gate-one class's records, chosen round-robin over held-out texts.

    Two reads of the slice: the first keeps only hashes, at most a class's
    quota per text, which bounds memory by the quota rather than by the
    slice; the second collects the chosen records.
    """
    from effects.application.build_corpus import ability_key
    from effects.application.train_effect_model import sampling_class
    from effects.infrastructure.record_io import read_records

    per_text: dict[str, dict[str, list[int]]] = {
        name: {} for name in quota if name in GATE_ONE_CLASSES
    }
    for record in read_records(directory):
        name = sampling_class(record)
        if name not in per_text:
            continue
        text = text_of_key.get(ability_key(record) or "")
        if text is None:
            continue
        value = record_hash(record.record_id, seed=seed + SAMPLE_SEED_OFFSET)
        heap = per_text[name].setdefault(text, [])
        if len(heap) < quota[name]:
            heapq.heappush(heap, -value)
        elif -value > heap[0]:
            heapq.heappushpop(heap, -value)
    chosen = {
        name: set(round_robin(
            {text: [-v for v in heap] for text, heap in texts.items()}, quota[name],
        ))
        for name, texts in per_text.items()
    }
    out: dict[str, list[EffectRecord]] = {name: [] for name in per_text}
    seen: set[str] = set()
    for record in read_records(directory):
        name = sampling_class(record)
        if name not in chosen or record.record_id in seen:
            continue
        if record_hash(record.record_id, seed=seed + SAMPLE_SEED_OFFSET) in chosen[name]:
            seen.add(record.record_id)
            out[name].append(record)
    return out


def class_quota(mix: Mapping[str, float], size: int) -> dict[str, int]:
    """Records per class in a sample of ``size``: each share rounded, never zero."""
    return {name: max(1, round(share * size)) for name, share in mix.items()}


class _Smallest:
    """The ``cap`` records with the smallest hashes seen, by heap.

    The tie-break is a monotonic counter rather than the record. ``record_id``
    is unique across a healthy corpus but the corpus is what this is reading,
    and a repeated id -- which is exactly the defect ``validate-corpus``
    exists to catch -- put two records with an equal key into one heap, where
    the comparison falls through to ``EffectRecord``. That has no ordering, so
    it raises ``TypeError`` and the build dies drawing a validation sample. A
    counter never ties, so the record is never compared.
    """

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._heap: list[tuple[int, str, int, EffectRecord]] = []
        self._offered = 0

    def offer(self, value: int, record: EffectRecord) -> None:
        self._offered += 1
        item = (-value, record.record_id, self._offered, record)
        if len(self._heap) < self.cap:
            heapq.heappush(self._heap, item)
        elif item > self._heap[0]:
            heapq.heappushpop(self._heap, item)

    def records(self) -> list[EffectRecord]:
        return [item[-1] for item in sorted(self._heap, reverse=True)]

    def __len__(self) -> int:
        return len(self._heap)


def _collect(
    directory: Path, quota: Mapping[str, int], *, seed: int,
    only: frozenset[str] | None = None,
) -> dict[str, _Smallest]:
    from effects.application.train_effect_model import sampling_class
    from effects.infrastructure.record_io import read_records

    heaps = {name: _Smallest(cap) for name, cap in quota.items()}
    for record in read_records(directory):
        name = sampling_class(record)
        if name not in heaps or (only is not None and name not in only):
            continue
        heaps[name].offer(record_hash(record.record_id, seed=seed + SAMPLE_SEED_OFFSET), record)
    return heaps


def draw_samples(
    *, card_disjoint: Path, game_disjoint: Path, gate_one: Path,
    mix: Mapping[str, float], size: int, seed: int,
    held_out_text_of_key: Mapping[str, str] | None = None,
) -> dict[str, list[EffectRecord]]:
    """One mixture-matched sample per stratum, seeded and shuffled.

    ``size <= 0`` means no fixed validation sample at all: nothing is read
    and both strata come back empty (``--validation-sample 0``).

    ``held_out_text_of_key`` (rendered provenance key → held-out text) turns
    on FR-051's round-robin for the card-disjoint resolution slots; without
    it the gate-one slice is drawn by smallest hash alone, as feature 023 did.
    """
    if size <= 0:
        return {"card-disjoint": [], "game-disjoint": []}

    quota = class_quota(mix, size)
    out: dict[str, list[EffectRecord]] = {}

    if held_out_text_of_key:
        gate_records = _gate_one_round_robin(
            gate_one, quota, seed=seed, text_of_key=held_out_text_of_key,
        )
    else:
        gate = _collect(gate_one, quota, seed=seed, only=GATE_ONE_CLASSES)
        gate_records = {name: heap.records() for name, heap in gate.items()}
    for stratum, directory in (("card-disjoint", card_disjoint), ("game-disjoint", game_disjoint)):
        heaps = _collect(directory, quota, seed=seed)
        sample: list[EffectRecord] = []
        for name, cap in quota.items():
            chosen: list[EffectRecord] = []
            if stratum == "card-disjoint" and name in GATE_ONE_CLASSES:
                chosen = gate_records.get(name, [])[:cap]
            taken = {r.record_id for r in chosen}
            for record in heaps[name].records():
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
