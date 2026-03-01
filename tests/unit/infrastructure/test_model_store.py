"""Tests for model persistence via model_store."""

import re
from pathlib import Path

import pytest

from price_predictor.infrastructure.model_store import (
    ModelNotFoundError,
    generate_model_version,
    load_model,
    save_model,
)


class TestGenerateModelVersion:
    def test_returns_string_matching_expected_pattern(self):
        version = generate_model_version()
        assert re.match(r"^\d{8}-\d{6}$", version), (
            f"Version '{version}' does not match YYYYMMDD-HHMMSS pattern"
        )


class TestSaveModel:
    def test_creates_joblib_file_and_latest_copy(self, tmp_path: Path):
        artifact = {"model": "mock", "accuracy": 0.95}
        version, model_path = save_model(artifact, tmp_path)

        assert model_path.exists()
        assert model_path.suffix == ".joblib"
        assert model_path.name == f"{version}.joblib"

        latest_path = tmp_path / "latest.joblib"
        assert latest_path.exists()

    def test_custom_version_used_in_filename(self, tmp_path: Path):
        artifact = {"model": "mock"}
        version, model_path = save_model(artifact, tmp_path, version="v1-custom")

        assert version == "v1-custom"
        assert model_path.name == "v1-custom.joblib"
        assert model_path.exists()

    def test_second_save_updates_latest(self, tmp_path: Path):
        artifact_v1 = {"version": 1}
        artifact_v2 = {"version": 2}

        save_model(artifact_v1, tmp_path, version="v1")
        save_model(artifact_v2, tmp_path, version="v2")

        latest_path = tmp_path / "latest.joblib"
        loaded = load_model(latest_path)
        assert loaded == {"version": 2}


class TestLoadModel:
    def test_loads_saved_artifact_correctly(self, tmp_path: Path):
        artifact = {"params": [1, 2, 3], "name": "test_model"}
        _, model_path = save_model(artifact, tmp_path, version="test")

        loaded = load_model(model_path)
        assert loaded == artifact

    def test_raises_model_not_found_for_missing_path(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist.joblib"
        with pytest.raises(ModelNotFoundError, match="Model file not found"):
            load_model(non_existent)
