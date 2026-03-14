"""Tests for TransformerStore (save/load .pt artifacts)."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from price_predictor.infrastructure.transformer_store import save_model, load_model


def _make_config(**overrides) -> TransformerConfig:
    defaults = dict(d_model=128, n_layers=4, n_heads=4, ff_dim=512, max_seq_len=64, vocab_size=30522, dropout=0.1)
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestTransformerStore:
    def test_save_creates_model_file(self, tmp_path: Path):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        save_model(model, config, tmp_path)
        assert (tmp_path / "model.pt").exists()

    def test_load_returns_model_and_config(self, tmp_path: Path):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        save_model(model, config, tmp_path)
        loaded_model, loaded_config = load_model(tmp_path)
        assert isinstance(loaded_model, CardPriceTransformerModel)
        assert isinstance(loaded_config, TransformerConfig)

    def test_roundtrip_preserves_config(self, tmp_path: Path):
        config = _make_config(d_model=64, n_heads=2, max_seq_len=32)
        model = CardPriceTransformerModel(config)
        save_model(model, config, tmp_path)
        _, loaded_config = load_model(tmp_path)
        assert loaded_config.d_model == 64
        assert loaded_config.n_heads == 2
        assert loaded_config.max_seq_len == 32
        assert loaded_config.n_layers == config.n_layers
        assert loaded_config.ff_dim == config.ff_dim
        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.dropout == config.dropout

    def test_roundtrip_preserves_weights(self, tmp_path: Path):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        mask = torch.ones(1, config.max_seq_len)
        with torch.no_grad():
            original_output = model(input_ids, mask)
        save_model(model, config, tmp_path)
        loaded_model, _ = load_model(tmp_path)
        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(input_ids, mask)
        assert torch.allclose(original_output, loaded_output)

    def test_load_missing_file_raises_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path)
