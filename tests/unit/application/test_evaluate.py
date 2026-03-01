"""Tests for EvaluateModelUseCase."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.application.evaluate import EvaluateModelUseCase
from price_predictor.application.train import TrainModelUseCase
from price_predictor.infrastructure.model_store import ModelNotFoundError


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parents[2] / "fixtures"


@pytest.fixture
def trained_model_path(fixtures_dir: Path, tmp_path: Path) -> Path:
    use_case = TrainModelUseCase()
    result = use_case.execute(
        forge_cards_path=fixtures_dir / "forge_cards",
        prices_path=fixtures_dir / "allprices_sample.json",
        printings_path=fixtures_dir / "allprintings_sample.json",
        output_path=tmp_path,
        test_split=0.2,
        random_seed=42,
    )
    return result.model_path


class TestEvaluateModelUseCase:
    def test_evaluate_returns_metrics(
        self, trained_model_path: Path, fixtures_dir: Path
    ) -> None:
        use_case = EvaluateModelUseCase()
        result = use_case.execute(
            model_path=trained_model_path,
            forge_cards_path=fixtures_dir / "forge_cards",
            prices_path=fixtures_dir / "allprices_sample.json",
            printings_path=fixtures_dir / "allprintings_sample.json",
            test_split=0.2,
            random_seed=42,
        )
        assert result.metrics.sample_count > 0
        assert result.metrics.mean_absolute_error_eur >= 0
        assert result.metrics.median_percentage_error >= 0
        assert 0 <= result.metrics.top_20_overlap <= 1.0

    def test_per_card_breakdown(
        self, trained_model_path: Path, fixtures_dir: Path
    ) -> None:
        use_case = EvaluateModelUseCase()
        result = use_case.execute(
            model_path=trained_model_path,
            forge_cards_path=fixtures_dir / "forge_cards",
            prices_path=fixtures_dir / "allprices_sample.json",
            printings_path=fixtures_dir / "allprintings_sample.json",
            test_split=0.2,
            random_seed=42,
        )
        assert result.per_card is not None
        assert len(result.per_card) == result.metrics.sample_count
        for entry in result.per_card:
            assert "name" in entry
            assert "actual_price_eur" in entry
            assert "predicted_price_eur" in entry

    def test_reproducible_evaluation(
        self, trained_model_path: Path, fixtures_dir: Path
    ) -> None:
        use_case = EvaluateModelUseCase()
        r1 = use_case.execute(
            model_path=trained_model_path,
            forge_cards_path=fixtures_dir / "forge_cards",
            prices_path=fixtures_dir / "allprices_sample.json",
            printings_path=fixtures_dir / "allprintings_sample.json",
            test_split=0.2,
            random_seed=42,
        )
        r2 = use_case.execute(
            model_path=trained_model_path,
            forge_cards_path=fixtures_dir / "forge_cards",
            prices_path=fixtures_dir / "allprices_sample.json",
            printings_path=fixtures_dir / "allprintings_sample.json",
            test_split=0.2,
            random_seed=42,
        )
        assert r1.metrics.mean_absolute_error_eur == r2.metrics.mean_absolute_error_eur
        assert r1.metrics.median_percentage_error == r2.metrics.median_percentage_error

    def test_model_not_found_raises(self, fixtures_dir: Path) -> None:
        use_case = EvaluateModelUseCase()
        with pytest.raises(ModelNotFoundError):
            use_case.execute(
                model_path=Path("nonexistent.joblib"),
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
            )
