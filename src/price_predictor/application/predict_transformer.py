"""Predict price use case: load transformer model and predict card price."""

from __future__ import annotations

from pathlib import Path

from price_predictor.application.transformer_inference import (
    predict_shifted_log,
    shifted_log_to_eur,
)
from price_predictor.domain.entities import PriceEstimate
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData
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
        shifted_log_pred = predict_shifted_log(
            model, tokenizer, card_text, printing_data, config
        )
        predicted_price = round(shifted_log_to_eur(shifted_log_pred, config.log_offset), 2)

        return PriceEstimate(
            predicted_price_eur=predicted_price,
            model_version=model_dir.name or "transformer",
        )
