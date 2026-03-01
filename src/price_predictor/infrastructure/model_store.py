"""Model persistence for trained prediction models."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


class ModelNotFoundError(Exception):
    """Raised when a model file cannot be found."""


def generate_model_version() -> str:
    """Generate a model version string from current UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def save_model(
    model_artifact: Any,
    output_dir: str | Path,
    version: str | None = None,
) -> tuple[str, Path]:
    """Save a model artifact to disk.

    Returns (version, path) tuple.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = generate_model_version()

    model_path = output_dir / f"{version}.joblib"
    joblib.dump(model_artifact, model_path)

    # Update latest symlink/copy
    latest_path = output_dir / "latest.joblib"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    shutil.copy2(model_path, latest_path)

    return version, model_path


def load_model(model_path: str | Path) -> Any:
    """Load a model artifact from disk.

    Raises ModelNotFoundError if the file doesn't exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
