"""PyTorch dataset for transformer training on tokenized card texts."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset

from price_predictor.application.training_sample import TrainingSample
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.infrastructure.metadata_encoder import encode_metadata


class TransformerTrainingDataset(Dataset):
    """Dataset wrapping tokenized card texts paired with shifted-log prices and metadata."""

    def __init__(
        self,
        samples: list[TrainingSample],
        max_seq_len: int,
        tokenizer: MtgTokenizer,
        log_offset: float = 2.0,
    ) -> None:
        """Construct dataset from TrainingSample objects.

        Args:
            log_offset: Offset used in log(price + log_offset) target transform.
                Must match the value stored in TransformerConfig so that inference
                applies the correct inverse transform.
        """
        all_input_ids = []
        all_attention_masks = []
        all_targets = []
        all_meta = []

        for sample in samples:
            input_ids, attention_mask = tokenizer.encode(
                sample.text, max_seq_len,
            )
            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_attention_masks.append(
                torch.tensor(attention_mask, dtype=torch.long),
            )
            all_targets.append(math.log(sample.price_eur + log_offset))
            all_meta.append(encode_metadata(sample.printing_data))

        self.input_ids = torch.stack(all_input_ids)
        self.attention_masks = torch.stack(all_attention_masks)
        self.targets = torch.tensor(all_targets, dtype=torch.float32)
        self.meta = torch.stack(all_meta)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "target": self.targets[idx],
            "meta": self.meta[idx],
        }
