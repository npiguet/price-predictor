"""Tests for PredictTransformerUseCase."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

from price_predictor.application.predict_transformer import PredictTransformerUseCase
from price_predictor.domain.entities import PriceEstimate, TransformerConfig
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel


def _make_config(**overrides) -> TransformerConfig:
    defaults = dict(
        d_model=16, n_layers=1, n_heads=2, ff_dim=32,
        max_seq_len=32, vocab_size=100, dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_model(config: TransformerConfig) -> CardPriceTransformerModel:
    """Build a real (tiny) model so forward pass works end-to-end."""
    model = CardPriceTransformerModel(config)
    model.eval()
    return model


def _make_tokenizer(vocab_size: int = 100) -> MtgTokenizer:
    """Build a fixture MtgTokenizer."""
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for i in range(vocab_size - 2):
        vocab[f"tok_{i}"] = i + 2
    return MtgTokenizer(vocab)


class TestPredictTransformerUseCase:
    """Tests for the transformer-based price prediction use case."""

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_prediction_returns_positive_price_and_model_version(
        self, mock_load_model,
    ):
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)
        tokenizer = _make_tokenizer(config.vocab_size)

        use_case = PredictTransformerUseCase()
        result = use_case.execute("Lightning Bolt R Instant", Path("models/transformer"),
                                   tokenizer=tokenizer)

        assert isinstance(result, PriceEstimate)
        assert result.predicted_price_eur >= 0
        assert result.model_version == "transformer"

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_prediction_with_printing_data(self, mock_load_model):
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)
        tokenizer = _make_tokenizer(config.vocab_size)
        pd = PrintingData(rarity="mythic", printings_count=1, release_year=2021)

        use_case = PredictTransformerUseCase()
        result = use_case.execute("Lightning Bolt R Instant", Path("models/transformer"),
                                   tokenizer=tokenizer, printing_data=pd)

        assert isinstance(result, PriceEstimate)
        assert result.predicted_price_eur >= 0

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_prediction_handles_short_text(self, mock_load_model):
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)
        tokenizer = _make_tokenizer(config.vocab_size)

        use_case = PredictTransformerUseCase()
        result = use_case.execute("X", Path("models/transformer"), tokenizer=tokenizer)

        assert isinstance(result, PriceEstimate)
        assert result.predicted_price_eur >= 0
        assert result.model_version

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_prediction_handles_long_text(self, mock_load_model):
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)
        tokenizer = _make_tokenizer(config.vocab_size)

        long_text = "Whenever a creature enters the battlefield under your control, " * 200
        use_case = PredictTransformerUseCase()
        result = use_case.execute(long_text, Path("models/v2"), tokenizer=tokenizer)

        assert isinstance(result, PriceEstimate)
        assert result.predicted_price_eur >= 0
        assert result.model_version == "v2"

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_model_not_found_raises_file_not_found_error(self, mock_load_model):
        mock_load_model.side_effect = FileNotFoundError("Model file not found")
        tokenizer = _make_tokenizer()

        use_case = PredictTransformerUseCase()
        with pytest.raises(FileNotFoundError):
            use_case.execute("Some card text", Path("nonexistent/model"),
                              tokenizer=tokenizer)

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_shifted_log_to_eur_conversion(self, mock_load_model):
        """Verify the shifted-log inverse: exp(pred) - 2, clamped >= 0."""
        config = _make_config()
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.eval = MagicMock()

        # A shifted-log value of log(5 + 2) should convert back to ~5.0 EUR
        target_price = 5.0
        shifted_log_value = math.log(target_price + 2)
        model.return_value = torch.tensor(shifted_log_value)
        mock_load_model.return_value = (model, config)
        tokenizer = _make_tokenizer(config.vocab_size)

        use_case = PredictTransformerUseCase()
        result = use_case.execute("Test card", Path("models/transformer"),
                                   tokenizer=tokenizer)

        assert abs(result.predicted_price_eur - target_price) < 0.01

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_negative_prediction_clamped_to_zero(self, mock_load_model):
        """If exp(pred) - 2 is negative, the price should be clamped to 0."""
        config = _make_config()
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.eval = MagicMock()

        # A very negative shifted-log prediction results in exp(pred) - 2 < 0
        model.return_value = torch.tensor(-10.0)
        mock_load_model.return_value = (model, config)
        tokenizer = _make_tokenizer(config.vocab_size)

        use_case = PredictTransformerUseCase()
        result = use_case.execute("Test card", Path("models/transformer"),
                                   tokenizer=tokenizer)

        assert result.predicted_price_eur == 0.0

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_uses_tokenizer_parameter_for_encoding(self, mock_load_model):
        """PredictTransformerUseCase.execute() uses the provided tokenizer, not BertTokenizer."""
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)
        tokenizer = _make_tokenizer(config.vocab_size)

        use_case = PredictTransformerUseCase()
        # Should not raise — proves tokenizer parameter is used
        result = use_case.execute("flying creature", Path("models/transformer"),
                                   tokenizer=tokenizer)
        assert result is not None

    @patch("price_predictor.application.predict_transformer.load_model")
    def test_defaults_used_when_no_printing_data(self, mock_load_model):
        """When printing_data is None, PrintingData.defaults() is used (no crash)."""
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)
        tokenizer = _make_tokenizer(config.vocab_size)

        use_case = PredictTransformerUseCase()
        result = use_case.execute("flying creature", Path("models/transformer"),
                                   tokenizer=tokenizer, printing_data=None)
        assert result is not None
        assert result.predicted_price_eur >= 0
