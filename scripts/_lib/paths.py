"""Repository locations every script resolves its default inputs and outputs from."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "output"
MODELS = REPO / "models"
EXPERIMENT_IMAGES = REPO / "experiments" / "images"
# The operator's sealed training corpus lives on the NAS, not under output/.
SEALED_TRAINING_DATA = Path(r"Y:\Nicolas\mtg\mtg-models-data\sealed\training-data")
