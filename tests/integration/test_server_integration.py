"""Integration tests for the prediction server with a real trained model."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from price_predictor.application.predict import PredictPriceUseCase
from price_predictor.application.train import TrainModelUseCase
from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost, PrintingData
from price_predictor.infrastructure.converted_card_parser import parse_converted_text
from price_predictor.infrastructure.model_store import load_model
from price_predictor.infrastructure.server import create_app

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a real model for integration tests."""
    tmp = tmp_path_factory.mktemp("models")
    use_case = TrainModelUseCase()
    use_case.execute(
        output_dir=FIXTURES_DIR / "converted_cards",
        prices_path=FIXTURES_DIR / "allprices_sample.json",
        printings_path=FIXTURES_DIR / "allprintings_sample.json",
        output_path=tmp,
        test_split=0.2,
        random_seed=42,
    )
    return tmp


@pytest.fixture(scope="module")
def model_path(trained_model_dir: Path) -> Path:
    return trained_model_dir / "latest.joblib"


@pytest.fixture(scope="module")
def integration_client(model_path: Path) -> TestClient:
    """Create a TestClient with a real trained model."""
    artifact = load_model(model_path)
    artifact["model_version"] = model_path.stem
    app = create_app(artifact)
    return TestClient(app)


BOLT_SCRIPT = (
    "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
    "spell[1]: CARDNAME deals 3 damage to any target."
)


@pytest.mark.integration
class TestServerIntegration:
    def test_valid_card_returns_numeric_price(self, integration_client: TestClient) -> None:
        response = integration_client.post(
            "/api/v1/predict",
            content=BOLT_SCRIPT,
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sklearn"]["predicted_price_eur"], float)
        assert data["sklearn"]["predicted_price_eur"] > 0
        assert isinstance(data["sklearn"]["model_version"], str)

    def test_service_matches_standalone_prediction(
        self, integration_client: TestClient, model_path: Path
    ) -> None:
        """FR-011: Service MUST produce identical results to standalone model."""
        response = integration_client.post(
            "/api/v1/predict",
            content=BOLT_SCRIPT,
            headers={"Content-Type": "text/plain"},
        )
        service_price = response.json()["sklearn"]["predicted_price_eur"]

        # Same card through standalone use case — parse from identical text so
        # oracle_text and all fields match exactly what the server sees
        card = parse_converted_text(BOLT_SCRIPT)
        use_case = PredictPriceUseCase()
        standalone = use_case.execute(card, model_path)

        assert service_price == standalone.predicted_price_eur

    def test_single_request_under_3_seconds(self, integration_client: TestClient) -> None:
        """SC-001: Single-request response time under 3 seconds."""
        start = time.monotonic()
        response = integration_client.post(
            "/api/v1/predict",
            content=BOLT_SCRIPT,
            headers={"Content-Type": "text/plain"},
        )
        elapsed = time.monotonic() - start
        assert response.status_code == 200
        assert elapsed < 3.0


@pytest.mark.integration
class TestServerConcurrency:
    def test_10_concurrent_requests(self, integration_client: TestClient) -> None:
        """FR-009 / SC-004: 10 concurrent requests without degradation."""
        card_scripts = [
            "name: lightning bolt\nmana cost: {R}\ntypes: instant",
            "name: grizzly bears\nmana cost: {1}{G}\ntypes: creature bear\npower toughness: 2/2",
            "name: serra angel\nmana cost: {3}{W}{W}\ntypes: creature angel\npower toughness: 4/4\nspell[1]: flying\nspell[2]: vigilance",
            "name: sol ring\nmana cost: {1}\ntypes: artifact",
            "name: island\ntypes: basic land island",
            "name: ragavan\nmana cost: {R}\ntypes: legendary creature monkey pirate\npower toughness: 2/1\nspell[1]: dash",
            "name: jace\nmana cost: {2}{U}{U}\ntypes: legendary planeswalker jace\nloyalty: 3",
            "name: breeding pool\ntypes: land forest island",
            "name: made up dragon\nmana cost: {4}{R}{R}\ntypes: creature dragon\npower toughness: 6/6\nspell[1]: flying",
            "name: test enchantment\nmana cost: {1}{W}\ntypes: enchantment",
        ]

        def send_request(script: str) -> tuple[int, dict]:
            resp = integration_client.post(
                "/api/v1/predict",
                content=script,
                headers={"Content-Type": "text/plain"},
            )
            return resp.status_code, resp.json()

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_request, s) for s in card_scripts]
            results = [f.result() for f in futures]
        elapsed = time.monotonic() - start

        assert len(results) == 10
        for status_code, data in results:
            assert status_code == 200
            assert "sklearn" in data
            assert isinstance(data["sklearn"]["predicted_price_eur"], float)
        assert elapsed < 3.0


@pytest.fixture(scope="module")
def metadata_map() -> dict[str, PrintingData]:
    """A small metadata_map with a known card for enrichment tests."""
    return {
        "Lightning Bolt": PrintingData(
            is_reserved=False,
            rarity="common",
            printings_count=12,
            release_year=2018,
            legalities=["commander", "legacy", "modern", "pauper", "penny", "vintage"],
        ),
    }


@pytest.fixture(scope="module")
def enrichment_client(model_path: Path, metadata_map: dict[str, PrintingData]) -> TestClient:
    """Create a TestClient with a real trained model and a metadata_map."""
    artifact = load_model(model_path)
    artifact["model_version"] = model_path.stem
    app = create_app(artifact, metadata_map=metadata_map)
    return TestClient(app)


# Card text with no printing data lines — server should auto-fill from metadata_map
BOLT_NO_PRINTING = (
    "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
    "spell[1]: CARDNAME deals 3 damage to any target."
)

# Unknown card — name not in metadata_map
UNKNOWN_CARD = (
    "name: totally invented spell\nmana cost: {2}{B}\ntypes: sorcery\n"
    "spell[1]: target player discards a card."
)

# Unknown card with partial metadata (rarity only)
UNKNOWN_CARD_WITH_RARITY = (
    "name: totally invented spell\nmana cost: {2}{B}\ntypes: sorcery\n"
    "spell[1]: target player discards a card.\nrarity: mythic"
)


@pytest.mark.integration
class TestServerAutoFill:
    def test_known_card_without_printing_data_returns_200(
        self, enrichment_client: TestClient
    ) -> None:
        """T023: POST known card text WITHOUT printing data, server enriches internally."""
        response = enrichment_client.post(
            "/api/v1/predict",
            content=BOLT_NO_PRINTING,
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sklearn"]["predicted_price_eur"], float)
        assert data["sklearn"]["predicted_price_eur"] > 0

    def test_unknown_card_without_metadata_returns_200(
        self, enrichment_client: TestClient
    ) -> None:
        """T028: POST unknown card text (not in metadata_map), server applies defaults."""
        response = enrichment_client.post(
            "/api/v1/predict",
            content=UNKNOWN_CARD,
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sklearn"]["predicted_price_eur"], float)
        assert data["sklearn"]["predicted_price_eur"] > 0

    def test_unknown_card_with_partial_metadata_returns_200(
        self, enrichment_client: TestClient
    ) -> None:
        """T029: POST unknown card with rarity: mythic only, server fills remaining defaults."""
        response = enrichment_client.post(
            "/api/v1/predict",
            content=UNKNOWN_CARD_WITH_RARITY,
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sklearn"]["predicted_price_eur"], float)
        assert data["sklearn"]["predicted_price_eur"] > 0
