"""Shared test fixtures for the card price predictor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FORGE_CARDS_DIR = FIXTURES_DIR / "forge_cards"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def forge_cards_dir() -> Path:
    return FORGE_CARDS_DIR


@pytest.fixture
def allprintings_path() -> Path:
    return FIXTURES_DIR / "allprintings_sample.json"


@pytest.fixture
def allprices_path() -> Path:
    return FIXTURES_DIR / "allprices_sample.json"


@pytest.fixture
def allprintings_data(allprintings_path: Path) -> dict:
    with open(allprintings_path) as f:
        return json.load(f)


@pytest.fixture
def allprices_data(allprices_path: Path) -> dict:
    with open(allprices_path) as f:
        return json.load(f)


@pytest.fixture
def sample_creature() -> Card:
    """A simple creature card (Grizzly Bears)."""
    return Card(
        name="Grizzly Bears",
        types=["Creature"],
        subtypes=["Bear"],
        mana_cost=ManaCost.parse("1 G"),
        power="2",
        toughness="2",
        layout="normal",
    )


@pytest.fixture
def sample_instant() -> Card:
    """A simple instant (Lightning Bolt)."""
    return Card(
        name="Lightning Bolt",
        types=["Instant"],
        mana_cost=ManaCost.parse("R"),
        oracle_text="Lightning Bolt deals 3 damage to any target.",
        ability_count=1,
        layout="normal",
    )


@pytest.fixture
def sample_planeswalker() -> Card:
    """A planeswalker (Jace, the Mind Sculptor)."""
    return Card(
        name="Jace, the Mind Sculptor",
        types=["Planeswalker"],
        supertypes=["Legendary"],
        subtypes=["Jace"],
        mana_cost=ManaCost.parse("2 U U"),
        loyalty="3",
        ability_count=4,
        layout="normal",
    )


@pytest.fixture
def sample_land() -> Card:
    """A basic land (Island)."""
    return Card(
        name="Island",
        types=["Land"],
        supertypes=["Basic"],
        subtypes=["Island"],
        mana_cost=None,
        oracle_text="{T}: Add {U}.",
        ability_count=1,
        layout="normal",
    )


@pytest.fixture
def sample_cards(
    sample_creature: Card,
    sample_instant: Card,
    sample_planeswalker: Card,
    sample_land: Card,
) -> list[Card]:
    """A diverse list of sample cards for testing."""
    return [sample_creature, sample_instant, sample_planeswalker, sample_land]
