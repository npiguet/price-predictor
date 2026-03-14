"""PyTorch dataset for transformer training on tokenized card texts."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TransformerTrainingDataset(Dataset):
    """Dataset wrapping tokenized card texts paired with shifted-log prices."""

    def __init__(
        self,
        card_tuples: list[tuple[str, str, float]],
        max_seq_len: int,
    ) -> None:
        """Construct dataset from (card_name, text_content, price_eur) tuples."""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        all_input_ids = []
        all_attention_masks = []
        all_targets = []

        for _name, text, price in card_tuples:
            encoded = tokenizer(
                text,
                max_length=max_seq_len,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            all_input_ids.append(encoded["input_ids"].squeeze(0))
            all_attention_masks.append(encoded["attention_mask"].squeeze(0))
            all_targets.append(math.log(price + 2))

        self.input_ids = torch.stack(all_input_ids)
        self.attention_masks = torch.stack(all_attention_masks)
        self.targets = torch.tensor(all_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "target": self.targets[idx],
        }
