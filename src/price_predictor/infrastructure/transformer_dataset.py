"""PyTorch dataset for transformer training on tokenized card texts."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset

from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.metadata_encoder import encode_metadata


class TransformerTrainingDataset(Dataset):
    """Dataset wrapping tokenized card texts paired with shifted-log prices and metadata."""

    def __init__(
        self,
        card_tuples: list[tuple[str, str, float]],
        max_seq_len: int,
        tokenizer: MtgTokenizer,
        log_offset: float = 2.0,
        printing_data_list: list[PrintingData] | None = None,
    ) -> None:
        """Construct dataset from (card_name, text_content, price_eur) tuples.

        Args:
            log_offset: Offset used in log(price + log_offset) target transform.
                Must match the value stored in TransformerConfig so that inference
                applies the correct inverse transform.
            printing_data_list: Per-card PrintingData for side-channel metadata.
                When None, every card is encoded with PrintingData.defaults().
        """
        if printing_data_list is None:
            printing_data_list = [PrintingData.defaults()] * len(card_tuples)
        elif len(printing_data_list) != len(card_tuples):
            raise ValueError(
                f"printing_data_list length ({len(printing_data_list)}) must match "
                f"card_tuples length ({len(card_tuples)})"
            )

        all_input_ids = []
        all_attention_masks = []
        all_targets = []
        all_meta = []

        for (_name, text, price), printing_data in zip(card_tuples, printing_data_list):
            input_ids, attention_mask = tokenizer.encode(text, max_seq_len)
            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
            all_targets.append(math.log(price + log_offset))
            all_meta.append(encode_metadata(printing_data))

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
