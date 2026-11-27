"""One shard's plan: its records' rarity weights and rarity buckets, and the
weighted batches drawn from them, counted on the epoch line by bucket and by
effect family (the family resolved when a batch draws the record)."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from effects.domain.rarity import TextGames
from effects.domain.sampling import (
    batches_without_replacement,
    sample_weights,
    sampling_class,
)

#: Rarity-bucket label for a record with no acting text (combat, legality).
NO_TEXT_BUCKET = "no-text"


@dataclass(frozen=True)
class ShardPlan:
    """One shard's records with their rarity weights and buckets: what a
    shard's steps are drawn from and counted by."""

    records: list
    #: Rarity weights, one per record.
    weights: list[float]
    #: ``record_id -> rarity bucket``, bucketed by the games the weights read.
    buckets: dict[str, str]
    #: A record's effect family, resolved through the sidecar join.
    family_of: Callable
    #: ``record_id -> family`` of each record a batch has drawn, so one drawn
    #: twice resolves once and one never drawn not at all.
    _families: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(
        cls, records: list, texts: Mapping[str, str | None], games: TextGames,
        family_of: Callable,
    ) -> ShardPlan:
        """The plan of ``records``, whose acting ``texts`` and ``games`` the
        caller read: weighted by rarity and bucketed by the games. A record's family
        is ``family_of(record)``, asked when a batch draws it."""
        weights = sample_weights(
            records, lambda record: texts[record.record_id],
            games=games, class_of=sampling_class,
        )
        buckets = {
            record.record_id: (
                NO_TEXT_BUCKET if texts[record.record_id] is None
                else games.bucket(texts[record.record_id])
            )
            for record in records
        }
        return cls(records, weights, buckets, family_of)

    def family(self, record) -> str:
        """``record``'s effect family, resolved on the first ask."""
        if record.record_id not in self._families:
            self._families[record.record_id] = self.family_of(record)
        return self._families[record.record_id]

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
            buckets[self.buckets[record.record_id]] += 1
            families[self.family(record)] += 1
        return buckets, families
