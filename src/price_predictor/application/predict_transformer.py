"""Predict price use case: load transformer model and predict card price."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import BertTokenizer

from price_predictor.domain.entities import PriceEstimate
from price_predictor.infrastructure.transformer_store import load_model


class PredictTransformerUseCase:
    """Load a trained transformer model and predict the EUR price for a card."""

    def execute(self, card_text: str, model_dir: Path) -> PriceEstimate:
        """Predict the price from raw card text using the transformer model.

        Args:
            card_text: Raw card text to predict from.
            model_dir: Directory containing the trained transformer latest.pt.

        Returns:
            PriceEstimate with predicted EUR price and model version.
        """
        model, config = load_model(model_dir)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        encoded = tokenizer(
            card_text,
            max_length=config.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            shifted_log_pred = model(input_ids, attention_mask).item()

        predicted_price = round(float(math.exp(shifted_log_pred) - 2), 2)
        predicted_price = max(predicted_price, 0.0)

        # Extract version from model directory name
        model_version = model_dir.name or "transformer"

        return PriceEstimate(
            predicted_price_eur=predicted_price,
            model_version=model_version,
        )
