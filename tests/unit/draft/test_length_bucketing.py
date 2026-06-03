"""Length-bucketed batching groups similar-length examples (perf, FR-036)."""

from __future__ import annotations

import random

import numpy as np

from draft.application.train_draft_agent import DraftExample, length_bucketed_batches


def _example(n_tokens: int) -> DraftExample:
    return DraftExample(
        draft_index=0,
        card_idx=np.zeros(n_tokens, dtype=np.int32),
        type_idx=np.ones(n_tokens, dtype=np.int8),  # all PACK (irrelevant here)
        packs_ago=np.zeros(n_tokens, dtype=np.int8),
        pick_ago=np.zeros(n_tokens, dtype=np.int8),
        pack_number=1,
        pick_number=1,
        imitation_active=True,
        target_token=0,
        critic_active=True,
        critic_target=0.0,
    )


def test_covers_all_examples_exactly_once() -> None:
    examples = [_example(n) for n in [5, 200, 17, 90, 3, 150, 42, 8, 230, 60]]
    batches = length_bucketed_batches(examples, batch_size=4, rng=random.Random(0))
    flat = [ex for b in batches for ex in b]
    assert len(flat) == len(examples)
    assert set(id(ex) for ex in flat) == set(id(ex) for ex in examples)
    assert all(len(b) <= 4 for b in batches)


def test_batches_are_length_homogeneous() -> None:
    # 2000 examples with a wide length range; within-batch spread should be tiny.
    rng_lengths = np.random.default_rng(1)
    examples = [_example(int(n)) for n in rng_lengths.integers(1, 275, size=2000)]
    batches = length_bucketed_batches(examples, batch_size=32, rng=random.Random(0))
    spreads = [
        max(e.n_tokens for e in b) - min(e.n_tokens for e in b)
        for b in batches if len(b) > 1
    ]
    # Random batching would average ~ (range * (n-1)/(n+1)) ≈ 250 spread;
    # bucketing must keep the median spread far smaller.
    assert float(np.median(spreads)) < 30


def test_deterministic_with_seed() -> None:
    examples = [_example(n) for n in range(1, 101)]
    a = length_bucketed_batches(examples, 8, random.Random(42))
    b = length_bucketed_batches(examples, 8, random.Random(42))
    assert [[e.n_tokens for e in batch] for batch in a] == \
           [[e.n_tokens for e in batch] for batch in b]
