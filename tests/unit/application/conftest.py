"""Shared fixture data for the application progress-logging tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def card_data_paths(fixtures_dir: Path) -> dict[str, Path]:
    """The converted-card, price and printing inputs the use cases read."""
    return {
        "output_dir": fixtures_dir / "converted_cards",
        "prices_path": fixtures_dir / "allprices_sample.json",
        "printings_path": fixtures_dir / "allprintings_sample.json",
    }
