"""The fixed validation samples a curated corpus ships (FR-135, FR-089).

The trainer used to sweep every validation shard at startup to draw a
mixture-matched sample, and drew 20 gate-1 records into a 2,048-record
card-disjoint sample because it filled each class with whatever arrived
first. The sample is a property of the corpus, drawn once here by smallest
seeded record hash — the same device the per-text cap uses — and written
beside the strata so the trainer reads it and the evaluator can name it.

For the card-disjoint stratum the two resolution classes draw from the
gate-one slice first: that is the population the model ships on, and the
number that selects checkpoints should measure it.
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
) -> dict[str, list[EffectRecord]]:
    """One mixture-matched sample per stratum, seeded and shuffled.

    ``size <= 0`` means no fixed validation sample at all: nothing is read
    and both strata come back empty (``--validation-sample 0``).
    """
    if size <= 0:
        return {"card-disjoint": [], "game-disjoint": []}

    quota = class_quota(mix, size)
    out: dict[str, list[EffectRecord]] = {}

    gate = _collect(gate_one, quota, seed=seed, only=GATE_ONE_CLASSES)
    for stratum, directory in (("card-disjoint", card_disjoint), ("game-disjoint", game_disjoint)):
        heaps = _collect(directory, quota, seed=seed)
        sample: list[EffectRecord] = []
        for name, cap in quota.items():
            chosen: list[EffectRecord] = []
            if stratum == "card-disjoint" and name in GATE_ONE_CLASSES:
                chosen = gate[name].records()[:cap]
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
