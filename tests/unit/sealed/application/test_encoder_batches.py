"""Unit tests for ``train-encoder``'s batching: padding collate and the
length-bucketed sampler.
"""

from __future__ import annotations

import torch

from sealed.application.encoder_batches import _LengthBucketBatchSampler, _pad_collate
from sealed.application.encoder_heads import N_HEADS

# ── Length-bucketed batching + per-batch padding ─────────────────────


def _item(length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A dataset-shaped item: input_ids of the given length + zero label tensors."""
    return (
        torch.arange(1, length + 1, dtype=torch.long),
        torch.zeros(N_HEADS), torch.zeros(N_HEADS), torch.zeros(N_HEADS),
    )


class TestPadCollate:
    def test_pads_to_batch_max_and_builds_mask(self):
        batch = [_item(2), _item(5), _item(3)]
        ids, mask, labels, weights, head_mask = _pad_collate(batch)
        assert ids.shape == (3, 5)        # padded to the longest item
        assert mask.shape == (3, 5)
        # row 0: real positions 0,1 then PAD(=0); mask 1,1,0,0,0.
        assert ids[0].tolist() == [1, 2, 0, 0, 0]
        assert mask[0].tolist() == [1, 1, 0, 0, 0]
        assert ids[1].tolist() == [1, 2, 3, 4, 5]
        assert mask[1].tolist() == [1, 1, 1, 1, 1]
        assert labels.shape == (3, N_HEADS)
        assert weights.shape == (3, N_HEADS)
        assert head_mask.shape == (3, N_HEADS)

    def test_single_item_batch(self):
        ids, mask, *_ = _pad_collate([_item(4)])
        assert ids.shape == (1, 4)
        assert mask.tolist() == [[1, 1, 1, 1]]


class TestLengthBucketBatchSampler:
    def test_covers_every_index_once_per_epoch(self):
        lengths = [10, 1, 7, 3, 9, 2, 8]  # 7 items, batch_size 3 → 3 batches
        sampler = _LengthBucketBatchSampler(lengths, batch_size=3, shuffle=True)
        assert len(sampler) == 3
        seen = sorted(i for batch in sampler for i in batch)
        assert seen == list(range(7))

    def test_chunks_group_similar_lengths(self):
        lengths = [100, 1, 99, 2, 98, 3]  # sorted: 1,2,3 | 98,99,100
        sampler = _LengthBucketBatchSampler(lengths, batch_size=3, shuffle=False)
        batches = list(sampler)
        # not shuffled → length order; each chunk's lengths are contiguous.
        chunk_lengths = [sorted(lengths[i] for i in b) for b in batches]
        assert chunk_lengths == [[1, 2, 3], [98, 99, 100]]

    def test_shuffle_varies_batch_order_across_epochs(self):
        lengths = list(range(50))
        sampler = _LengthBucketBatchSampler(lengths, batch_size=5, shuffle=True)
        e1 = list(sampler)
        e2 = list(sampler)
        # Same coverage, but a different order (vanishingly unlikely to match).
        assert sorted(i for b in e1 for i in b) == sorted(i for b in e2 for i in b)
        assert e1 != e2

    def test_no_shuffle_is_deterministic(self):
        lengths = [5, 3, 8, 1, 9, 2]
        s = _LengthBucketBatchSampler(lengths, batch_size=2, shuffle=False)
        assert list(s) == list(s)

    def test_empty(self):
        s = _LengthBucketBatchSampler([], batch_size=4, shuffle=True)
        assert len(s) == 0
        assert list(s) == []
