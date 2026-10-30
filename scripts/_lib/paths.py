"""Repository locations every script resolves its default inputs and outputs from."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "output"
MODELS = REPO / "models"
EXPERIMENT_IMAGES = REPO / "experiments" / "images"
# The operator's sealed training corpus lives on the NAS, not under output/.
SEALED_TRAINING_DATA = Path(r"Y:\Nicolas\mtg\mtg-models-data\sealed\training-data")
# Forge's resources in the sibling ``../forge`` checkout: beside the main
# checkout, so a worktree under .claude/worktrees/ finds it further up.
FORGE_RES = next((p / "forge" for p in REPO.parents if (p / "forge" / "forge-gui").is_dir()),
                 REPO.parent / "forge") / "forge-gui" / "res"
