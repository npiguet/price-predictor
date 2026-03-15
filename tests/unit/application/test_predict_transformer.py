"""Tests for PredictTransformerUseCase."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

from price_predictor.application.predict_transformer import PredictTransformerUseCase
from price_predictor.domain.entities import PriceEstimate, TransformerConfig
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


def _fake_tokenizer_call(max_length: int):
    """Return a mock tokenizer __call__ that produces valid tensors."""
    def _tokenize(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"):
        input_ids = torch.ones(1, max_length, dtype=torch.long)
        attention_mask = torch.ones(1, max_length, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return _tokenize


class TestPredictTransformerUseCase:
    """Tests for the transformer-based price prediction use case."""

    @patch("price_predictor.application.predict_transformer.BertTokenizer")
    @patch("price_predictor.application.predict_transformer.load_model")
    def test_prediction_returns_positive_price_and_model_version(
        self, mock_load_model, mock_bert_tokenizer,
    ):
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = _fake_tokenizer_call(config.max_seq_len)
        mock_bert_tokenizer.from_pretrained.return_value = mock_tokenizer

        use_case = PredictTransformerUseCase()
        result = use_case.execute("Lightning Bolt R Instant", Path("models/transformer"))

        assert isinstance(result, PriceEstimate)
        assert result.predicted_price_eur >= 0
        assert result.model_version == "transformer"

    @patch("price_predictor.application.predict_transformer.BertTokenizer")
    @patch("price_predictor.application.predict_transformer.load_model")
    def test_prediction_handles_short_text(
        self, mock_load_model, mock_bert_tokenizer,
    ):
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = _fake_tokenizer_call(config.max_seq_len)
        mock_bert_tokenizer.from_pretrained.return_value = mock_tokenizer

        use_case = PredictTransformerUseCase()
        result = use_case.execute("X", Path("models/transformer"))

        assert isinstance(result, PriceEstimate)
        assert result.predicted_price_eur >= 0
        assert result.model_version

    @patch("price_predictor.application.predict_transformer.BertTokenizer")
    @patch("price_predictor.application.predict_transformer.load_model")
    def test_prediction_handles_long_text(
        self, mock_load_model, mock_bert_tokenizer,
    ):
        config = _make_config()
        model = _make_model(config)
        mock_load_model.return_value = (model, config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = _fake_tokenizer_call(config.max_seq_len)
        mock_bert_tokenizer.from_pretrained.return_value = mock_tokenizer

        long_text = "Whenever a creature enters the battlefield under your control, " * 200
        use_case = PredictTransformerUseCase()
        result = use_case.execute(long_text, Path("models/v2"))

        assert isinstance(result, PriceEstimate)
        assert result.predicted_price_eur >= 0
        assert result.model_version == "v2"

    @patch("price_predictor.application.predict_transformer.BertTokenizer")
    @patch("price_predictor.application.predict_transformer.load_model")
    def test_model_not_found_raises_file_not_found_error(
        self, mock_load_model, mock_bert_tokenizer,
    ):
        mock_load_model.side_effect = FileNotFoundError("Model file not found")

        use_case = PredictTransformerUseCase()
        with pytest.raises(FileNotFoundError):
            use_case.execute("Some card text", Path("nonexistent/model"))

    @patch("price_predictor.application.predict_transformer.BertTokenizer")
    @patch("price_predictor.application.predict_transformer.load_model")
    def test_shifted_log_to_eur_conversion(
        self, mock_load_model, mock_bert_tokenizer,
    ):
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

        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = _fake_tokenizer_call(config.max_seq_len)
        mock_bert_tokenizer.from_pretrained.return_value = mock_tokenizer

        use_case = PredictTransformerUseCase()
        result = use_case.execute("Test card", Path("models/transformer"))

        assert abs(result.predicted_price_eur - target_price) < 0.01

    @patch("price_predictor.application.predict_transformer.BertTokenizer")
    @patch("price_predictor.application.predict_transformer.load_model")
    def test_negative_prediction_clamped_to_zero(
        self, mock_load_model, mock_bert_tokenizer,
    ):
        """If exp(pred) - 2 is negative, the price should be clamped to 0."""
        config = _make_config()
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.eval = MagicMock()

        # A very negative shifted-log prediction results in exp(pred) - 2 < 0
        model.return_value = torch.tensor(-10.0)
        mock_load_model.return_value = (model, config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = _fake_tokenizer_call(config.max_seq_len)
        mock_bert_tokenizer.from_pretrained.return_value = mock_tokenizer

        use_case = PredictTransformerUseCase()
        result = use_case.execute("Test card", Path("models/transformer"))

        assert result.predicted_price_eur == 0.0
