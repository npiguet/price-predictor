"""Tests for predict CLI subcommand."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from price_predictor.application.train import TrainModelUseCase


@pytest.fixture
def trained_model_dir(tmp_path: Path) -> Path:
    fixtures = Path(__file__).parents[2] / "fixtures"
    use_case = TrainModelUseCase()
    use_case.execute(
        forge_cards_path=fixtures / "forge_cards",
        prices_path=fixtures / "allprices_sample.json",
        printings_path=fixtures / "allprintings_sample.json",
        output_path=tmp_path,
        test_split=0.2,
        random_seed=42,
    )
    return tmp_path


class TestPredictCli:
    def test_valid_args_produce_json(self, trained_model_dir: Path) -> None:
        model_path = trained_model_dir / "latest.joblib"
        result = subprocess.run(
            [sys.executable, "-m", "price_predictor", "predict",
             "--types", "Creature",
             "--mana-cost", "1 G",
             "--power", "2",
             "--toughness", "2",
             "--model-path", str(model_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "predicted_price_eur" in output
        assert "model_version" in output
        assert output["predicted_price_eur"] > 0

    def test_missing_types_exits_1(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "price_predictor", "predict",
             "--mana-cost", "R"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "types" in result.stderr.lower()

    def test_missing_model_exits_2(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "price_predictor", "predict",
             "--types", "Creature",
             "--model-path", "nonexistent/model.joblib"],
            capture_output=True, text=True,
        )
        assert result.returncode == 2

    def test_partial_args_succeed(self, trained_model_dir: Path) -> None:
        model_path = trained_model_dir / "latest.joblib"
        result = subprocess.run(
            [sys.executable, "-m", "price_predictor", "predict",
             "--types", "Creature",
             "--mana-cost", "2 W",
             "--model-path", str(model_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["predicted_price_eur"] > 0
