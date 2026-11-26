"""One shard's plan: its records' rarity weights and epoch-line labels, and the
weighted batches drawn from them."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from effects.domain.rarity import TextGames
from effects.domain.sampling import (
    batches_without_replacement,
    sample_weights,
    sampling_class,
)

#: Rarity-bucket label for a record with no acting text (combat, legality).
NO_TEXT_BUCKET = "no-text"


class RecordLabels(NamedTuple):
    """Where one trained record is counted on the epoch line (FR-055)."""

    bucket: str
    family: str


@dataclass(frozen=True)
class ShardPlan:
    """One shard's records with their rarity weights and their labels: what a
    shard's steps are drawn from and counted by."""

    records: list
    #: Rarity weights, one per record.
    weights: list[float]
    #: ``record_id -> labels``, bucketed by the games the weights read.
    labels: dict[str, RecordLabels]

    @classmethod
    def of(
        cls, records: list, texts: Mapping[str, str | None], games: TextGames,
        family_of: Callable,
    ) -> ShardPlan:
        """The plan of ``records``, whose acting ``texts`` and ``games`` the
        caller read: weighted by rarity, labelled with ``family_of(record)``."""
        weights = sample_weights(
            records, lambda record: texts[record.record_id],
            games=games, class_of=sampling_class,
        )
        labels: dict[str, RecordLabels] = {}
        for record in records:
            text = texts[record.record_id]
            bucket = NO_TEXT_BUCKET if text is None else games.bucket(text)
            labels[record.record_id] = RecordLabels(bucket, family_of(record))
        return cls(records, weights, labels)

    def batches(self, batch_size: int, rng: random.Random) -> Iterator:
        """The weighted batches without replacement, drawn from ``rng``."""
        return batches_without_replacement(
            self.records, self.weights, batch_size=batch_size, rng=rng,
        )

    def shares(self, planned: Sequence) -> tuple[Counter[str], Counter[str]]:
        """``planned`` records' counts by rarity bucket and by effect family."""
        buckets: Counter[str] = Counter()
        families: Counter[str] = Counter()
        for record in planned:
            counted = self.labels[record.record_id]
            buckets[counted.bucket] += 1
            families[counted.family] += 1
        return buckets, families
