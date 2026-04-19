"""Save and load transformer model artifacts (.pt files)."""

from __future__ import annotations

import shutil
from pathlib import Path

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.model_store import generate_model_version
from price_predictor.infrastructure.torch_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
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
    save_checkpoint(
        model_path,
        {"state_dict": model.state_dict()},
        config,
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

    payload, config = load_checkpoint(
        model_path,
        TransformerConfig,
        weights_only=True,
        safe_globals=[TransformerConfig],
    )
    model = CardPriceTransformerModel(config)
    model.load_state_dict(payload["state_dict"])
    return model, config
