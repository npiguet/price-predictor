"""Predict price use case: load transformer model and predict card price."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from price_predictor.domain.entities import PriceEstimate
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.metadata_encoder import encode_mana_features, encode_metadata
from price_predictor.infrastructure.transformer_store import load_model


class PredictTransformerUseCase:
    """Load a trained transformer model and predict the EUR price for a card."""

    def execute(
        self,
        card_text: str,
        model_dir: Path,
        tokenizer: MtgTokenizer,
        printing_data: PrintingData | None = None,
    ) -> PriceEstimate:
        """Predict the price from raw card text using the transformer model.

        Args:
            card_text: Raw card text to predict from.
            model_dir: Directory containing the trained transformer latest.pt.
            tokenizer: MtgTokenizer to encode the card text.
            printing_data: Side-channel metadata. Defaults to PrintingData.defaults()
                if not provided.

        Returns:
            PriceEstimate with predicted EUR price and model version.
        """
        model, config = load_model(model_dir)

        if printing_data is None:
            printing_data = PrintingData.defaults()

        input_ids, attention_mask = tokenizer.encode(card_text, config.max_seq_len)
        meta = torch.cat([encode_metadata(printing_data), encode_mana_features(card_text)])

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask_t = torch.tensor([attention_mask], dtype=torch.long).to(device)
        meta_t = meta.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            shifted_log_pred = model(input_ids_t, attention_mask_t, meta_t).item()

        predicted_price = round(float(math.exp(shifted_log_pred) - config.log_offset), 2)
        predicted_price = max(predicted_price, 0.0)

        # Extract version from model directory name
        model_version = model_dir.name or "transformer"

        return PriceEstimate(
            predicted_price_eur=predicted_price,
            model_version=model_version,
        )
