"""Save and load transformer model artifacts (.pt files)."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.model_store import generate_model_version
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel


def save_model(
    model: CardPriceTransformerModel,
    config: TransformerConfig,
    output_dir: Path,
    version: str | None = None,
) -> tuple[str, Path]:
    """Save model state_dict and config to output_dir/<version>.pt.

    Returns (version, model_path) tuple.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = generate_model_version()

    model_path = output_dir / f"{version}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
        },
        model_path,
    )

    # Update latest copy
    latest_path = output_dir / "latest.pt"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    shutil.copy2(model_path, latest_path)

    return version, model_path


def load_model(model_dir: Path) -> tuple[CardPriceTransformerModel, TransformerConfig]:
    """Load model and config from model_dir/latest.pt.

    Returns (model, config) tuple.
    Raises FileNotFoundError if latest.pt does not exist.
    """
    model_dir = Path(model_dir)
    model_path = model_dir if model_dir.suffix == ".pt" else model_dir / "latest.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Legacy checkpoints store the config as a TransformerConfig instance;
    # newer ones store it as a plain dict via asdict(). Allowlist the dataclass
    # so weights_only=True accepts both shapes.
    with torch.serialization.safe_globals([TransformerConfig]):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    raw_config = checkpoint["config"]
    config = raw_config if isinstance(raw_config, TransformerConfig) else TransformerConfig(**raw_config)
    model = CardPriceTransformerModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config
