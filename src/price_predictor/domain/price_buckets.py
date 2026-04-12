"""Price-bucket value objects shared by training and evaluation.

Two named bucketings exist:

* ``SAMPLING_BUCKETS`` (4 buckets) drives the WeightedRandomSampler that
  oversamples expensive cards during training. Boundaries are coarse because
  fine slices would create high-variance per-bucket counts.
* ``REPORTING_BUCKETS`` (6 buckets) is used by the evaluator to break down
  per-bucket error metrics, with finer slices at the low end where most cards
  live.

Both schemes share the same ``PriceBucket`` and ``BucketScheme`` types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PriceBucket:
    """A half-open price interval ``[low, high)`` with a human label."""

    label: str
    low: float
    high: float

    def contains(self, price_eur: float) -> bool:
        return self.low <= price_eur < self.high


@dataclass(frozen=True)
class BucketScheme:
    """An ordered tuple of non-overlapping ``PriceBucket``s covering [0, ∞)."""

    buckets: tuple[PriceBucket, ...]

    def index_for(self, price_eur: float) -> int:
        """Return the index of the bucket that contains ``price_eur``.

        Falls back to the last bucket for any price beyond the highest bound;
        the schemes defined here all extend to ``inf`` so the fallback is
        only reached for negative inputs (treated as the lowest bucket).
        """
        if price_eur < 0:
            return 0
        for i, bucket in enumerate(self.buckets):
            if bucket.contains(price_eur):
                return i
        return len(self.buckets) - 1

    def __len__(self) -> int:
        return len(self.buckets)

    def __iter__(self):
        return iter(self.buckets)


SAMPLING_BUCKETS = BucketScheme((
    PriceBucket("<€2",     0.0,  2.0),
    PriceBucket("€2–10",   2.0,  10.0),
    PriceBucket("€10–50",  10.0, 50.0),
    PriceBucket(">€50",    50.0, math.inf),
))


REPORTING_BUCKETS = BucketScheme((
    PriceBucket("<€0.10",       0.0,  0.1),
    PriceBucket("€0.10–0.50",   0.1,  0.5),
    PriceBucket("€0.50–2",      0.5,  2.0),
    PriceBucket("€2–10",        2.0,  10.0),
    PriceBucket("€10–50",       10.0, 50.0),
    PriceBucket(">€50",         50.0, math.inf),
))
