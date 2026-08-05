"""Corpus-shaping helpers shared by every draft-agent trainer.

Sibling of :mod:`draft.application.draft_pick_states` (the shared per-pick state
walk): the pieces of the training pipeline that sit *between* a ``DraftRecord``
and a minibatch, and that the imitation (gen-1), offline-RL (gen-2), and online
GRPO (gen-3) trainers all need identically.

Both helpers are generic over the example type — ``length_bucketed_batches`` only
requires an ``n_tokens`` attribute — so each trainer keeps its own example
dataclass.
"""

from __future__ import annotations

import random
from typing import Protocol, TypeVar

from draft.domain.draft_geometry import DraftRecord


def leave_one_out_rewards(record: DraftRecord) -> list[float | None]:
    """Per-seat ``deck_score − mean(other non-failed deck_scores)``.

    The pod-relative (RLOO / group-baseline) reward. A seat with a failed build
    (``deck_score is None``) yields ``None`` and is excluded from every other
    seat's baseline.
    """
    scores = [s.deck_score for s in record.seats]
    rewards: list[float | None] = []
    for i, score in enumerate(scores):
        if score is None:
            rewards.append(None)
            continue
        others = [s for j, s in enumerate(scores) if j != i and s is not None]
        mean_others = sum(others) / len(others) if others else 0.0
        rewards.append(score - mean_others)
    return rewards


class _HasTokenCount(Protocol):
    @property
    def n_tokens(self) -> int: ...


E = TypeVar("E", bound=_HasTokenCount)

_BUCKET_MULTIPLIER = 50  # megabatch = bucket_multiplier * batch_size examples


def length_bucketed_batches(
    examples: list[E], batch_size: int, rng: random.Random,
) -> list[list[E]]:
    """Group similar-length examples into batches; reshuffle batch order per call.

    Examples are shuffled, partitioned into megabatches, sorted by token count
    *within* each megabatch, cut into batches, and the batch order is shuffled.
    So each batch holds near-equal-length examples (little padding wasted on
    masks) while still varying composition and order across epochs.
    """
    order = list(examples)
    rng.shuffle(order)
    mega = batch_size * _BUCKET_MULTIPLIER
    batches: list[list[E]] = []
    for start in range(0, len(order), mega):
        chunk = order[start:start + mega]
        chunk.sort(key=lambda ex: ex.n_tokens)
        for b in range(0, len(chunk), batch_size):
            batches.append(chunk[b:b + batch_size])
    rng.shuffle(batches)
    return batches
