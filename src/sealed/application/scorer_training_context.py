"""What one ``train-scorer`` run's epochs read: the model, the optimizer,
the loaders, the checkpoint paths and, in Phase B, the encoder with its
tokenizer and the reference batch the embedding drift is measured on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from price_predictor.domain.tokenizer import MtgTokenizer
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.match_data_loader import EmbeddingTable


@dataclass
class ReferenceBatch:
    """The fixed set of unique cards from Phase B step 0 used for the
    ``embedding_drift`` metric (Decision §5, FR-012)."""

    card_indices: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    step0_text_vectors: torch.Tensor


@dataclass
class _TrainingContext:
    model: SetTransformerScorer
    optimizer: torch.optim.Optimizer
    embedding_table: EmbeddingTable
    train_loader: DataLoader
    val_loader: DataLoader
    latest_path: Path
    best_path: Path
    start_epoch: int
    best_val_accuracy: float
    device: torch.device
    train_config: dict[str, Any]
    encoder: torch.nn.Module | None = None
    tokenizer: MtgTokenizer | None = None
    card_token_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    locator: ConvertedCardLocator | None = None
    idx_to_name: dict[int, str] | None = None
    reference_batch: ReferenceBatch | None = None
    max_seq_len: int | None = None
    encoder_chunk_size: int = 128
    max_grad_norm: float = 100.0
    margin_weighting: str = "none"
