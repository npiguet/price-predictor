"""Save and load transformer model artifacts (.pt files)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel


def save_model(
    model: CardPriceTransformerModel,
    config: TransformerConfig,
    output_dir: Path,
) -> Path:
    """Save model state_dict and config to output_dir/model.pt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
        },
        model_path,
    )
    return model_path


def load_model(model_dir: Path) -> tuple[CardPriceTransformerModel, TransformerConfig]:
    """Load model and config from model_dir/model.pt.

    Returns (model, config) tuple.
    Raises FileNotFoundError if model.pt does not exist.
    """
    model_path = Path(model_dir) / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    config = TransformerConfig(**checkpoint["config"])
    model = CardPriceTransformerModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config
