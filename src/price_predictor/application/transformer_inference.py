"""Shared transformer inference helpers used by predict, evaluate, and serve."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from price_predictor.domain.entities import TransformerConfig
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.metadata_encoder import encode_metadata


def shifted_log_to_eur(value: float, log_offset: float) -> float:
    """Inverse of the shifted-log target transform: max(exp(value) - log_offset, 0)."""
    return max(float(math.exp(value) - log_offset), 0.0)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def predict_shifted_log(
    model: torch.nn.Module,
    tokenizer: MtgTokenizer,
    text: str,
    printing_data: PrintingData | None,
    config: TransformerConfig,
) -> float:
    """Run a single-card forward pass and return the raw shifted-log prediction."""
    pd = printing_data if printing_data is not None else PrintingData.defaults()
    input_ids, attention_mask = tokenizer.encode(text, config.max_seq_len)
    meta = encode_metadata(pd)

    device = _model_device(model)
    input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask_t = torch.tensor([attention_mask], dtype=torch.long).to(device)
    meta_t = meta.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        return float(model(input_ids_t, attention_mask_t, meta_t).item())


def predict_batch(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference and return (predictions, targets) in shifted-log space."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            meta = batch["meta"].to(device)
            outputs = model(input_ids, attention_mask, meta)
            all_predictions.append(outputs.cpu())
            all_targets.append(batch["target"])

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    return predictions, targets
