"""Helpers the ``price_predictor`` CLI command modules share."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def require_path_exists(path: Path, label: str) -> bool:
    """Print an error and return False when ``path`` is missing; True otherwise."""
    if path.exists():
        return True
    logger.error("%s not found: %s", label, path)
    return False


def add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add the input/split flags shared by every train/evaluate subcommand."""
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Converted card text directory")
    parser.add_argument("--prices-path", type=str,
                        default="resources/AllPricesToday.json")
    parser.add_argument("--printings-path", type=str,
                        default="resources/AllPrintings.json")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
