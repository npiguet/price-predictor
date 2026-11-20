"""The ``train-encoder`` dataset and its batching: the corpus tokenized once,
the per-card training tuples, length-bucketed batches and per-batch padding.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from price_predictor.domain.tokenizer import MtgTokenizer
from sealed.application.encoder_heads import N_HEADS, _label_value, _per_head_weight
from sealed.application.encoder_labels import CardLabelMap
from sealed.application.encoder_training_common import RANDOM_SEED
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

_MAX_SEQ_LEN_MULTIPLE: int = 8
_MIN_MAX_SEQ_LEN: int = 64


# ── Tokenized corpus + dataset ──────────────────────────────────────────


def _tokenize_corpus(
    card_names: Iterable[str],
    tokenizer: MtgTokenizer,
    locator: ConvertedCardLocator,
) -> tuple[dict[str, list[int]], int]:
    """Tokenize every card once.

    Returns ``({card_name: token_ids}, max_seq_len)``. ``token_ids`` is
    unpadded — the per-batch :func:`_pad_collate` pads to each batch's
    own max. ``max_seq_len`` is the longest card's token count rounded
    up to a multiple of 8 (and at least ``_MIN_MAX_SEQ_LEN``); it sizes
    the encoder's position-embedding table (FR-022). The ``name:`` line
    is stripped before tokenizing, matching ``encode-cards`` (FR-014a).
    Card names have already been validated by :func:`_drop_missing_cards`,
    so a missing ``.txt`` here is a real error.
    """
    by_name: dict[str, list[int]] = {}
    max_len = 1
    for name in card_names:
        converted = locator.load_text(name)
        if converted is None:
            raise FileNotFoundError(f"No converted card text on disk for {name!r}")
        ids = tokenizer.tokenize_to_ids(converted.without_name_line())
        by_name[name] = ids
        if len(ids) > max_len:
            max_len = len(ids)
    rounded = math.ceil(max_len / _MAX_SEQ_LEN_MULTIPLE) * _MAX_SEQ_LEN_MULTIPLE
    return by_name, max(rounded, _MIN_MAX_SEQ_LEN)


class _WinnabilityDataset(Dataset):
    """Pre-built per-card training tuples for the multi-head + MLM encoder.

    Each item is ``(input_ids[L], labels[9], weights[9], head_mask[9])``,
    assembled once in ``__init__`` from the pre-tokenized corpus and the
    per-card label map — ``__getitem__`` is a pure index (no disk I/O, no
    re-tokenization per epoch). ``input_ids`` is unpadded;
    :func:`_pad_collate` pads each batch to its own max length.
    ``labels`` carries the shrunk value at each head position (0.0 at
    empty cells); ``weights`` follows FR-017a; ``head_mask`` is 1.0 at
    non-empty cells. ``lengths[i] == len(input_ids)`` for item ``i`` —
    consumed by :class:`_LengthBucketBatchSampler`.
    """

    def __init__(
        self,
        card_names: Sequence[str],
        labels: CardLabelMap,
        tokenized_by_name: dict[str, list[int]],
        shrinkage_k: float,
    ) -> None:
        self._items: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self.lengths: list[int] = []
        for name in card_names:
            ids = tokenized_by_name[name]
            lbl = labels[name]
            labels_v = torch.zeros(N_HEADS, dtype=torch.float32)
            weights_v = torch.zeros(N_HEADS, dtype=torch.float32)
            head_mask_v = torch.zeros(N_HEADS, dtype=torch.float32)
            for h in range(N_HEADS):
                v = _label_value(lbl, h)
                if v is None:
                    continue
                labels_v[h] = float(v)
                weights_v[h] = _per_head_weight(lbl, h, shrinkage_k)
                head_mask_v[h] = 1.0
            self._items.append((
                torch.tensor(ids, dtype=torch.long),
                labels_v, weights_v, head_mask_v,
            ))
            self.lengths.append(len(ids))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._items[idx]


# ── Length-bucketed batching + per-batch dynamic padding (F4) ──────────


_TrainBatch = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]  # (input_ids[B,T], attention_mask[B,T], labels[B,9], weights[B,9], head_mask[B,9])


def _pad_collate(
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> _TrainBatch:
    """Pad a batch of variable-length items to that batch's max length.

    The encoder is length-agnostic — its position embedding is sliced to
    ``seq_len`` and ``attention_mask`` marks the padding — so each batch
    only pays for its own longest card, not the global longest.
    """
    ids_list, labels, weights, head_mask = zip(*items)
    lengths = torch.tensor([t.size(0) for t in ids_list], dtype=torch.long)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        list(ids_list), batch_first=True, padding_value=MtgTokenizer.PAD_ID,
    )
    seq_len = input_ids.size(1)
    attention_mask = (
        torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
    ).long()
    return (
        input_ids,
        attention_mask,
        torch.stack(labels),
        torch.stack(weights),
        torch.stack(head_mask),
    )


class _LengthBucketBatchSampler(Sampler[list[int]]):
    """Yield batches of similar-length items so per-batch padding is small.

    ``__iter__`` sorts item indices by token length, chunks them into
    ``batch_size``-sized groups, and — when ``shuffle`` — shuffles the
    *list of chunks* (and the order within each chunk) with a generator
    that advances every epoch (deterministic from ``RANDOM_SEED``). So
    cards of similar length batch together (≈10× fewer padded positions
    on typical batches) while the SGD still sees a fresh batch order each
    epoch. The FR-017 per-batch sum-to-1 weighting is composition-robust
    by design, and token length doesn't correlate with the per-head
    sample weights, so bucketing does not bias training.
    """

    def __init__(
        self, lengths: Sequence[int], batch_size: int, *, shuffle: bool,
    ) -> None:
        self._lengths = list(lengths)
        self._batch_size = max(1, batch_size)
        self._shuffle = shuffle
        self._rng = np.random.default_rng(RANDOM_SEED)
        self._by_length = sorted(
            range(len(self._lengths)), key=lambda i: self._lengths[i],
        )

    def __len__(self) -> int:
        return math.ceil(len(self._by_length) / self._batch_size) if self._by_length else 0

    def __iter__(self):
        bs = self._batch_size
        chunks = [self._by_length[i:i + bs] for i in range(0, len(self._by_length), bs)]
        if self._shuffle:
            self._rng.shuffle(chunks)
            for chunk in chunks:
                self._rng.shuffle(chunk)
        yield from chunks
