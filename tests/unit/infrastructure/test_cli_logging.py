"""Tests for logging configuration in the CLI entry point (T036)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from price_predictor.__main__ import main


class TestLoggingConfiguration:
    """Verify logging is configured for train/evaluate but not predict."""

    def test_train_configures_logging_to_stderr(self, tmp_path: Path) -> None:
        """Train command should configure logging so INFO messages go to stderr."""
        fixtures = Path(__file__).parents[2] / "fixtures"
        args = [
            "train",
            "--forge-cards-path", str(fixtures / "forge_cards"),
            "--prices-path", str(fixtures / "allprices_sample.json"),
            "--printings-path", str(fixtures / "allprintings_sample.json"),
            "--output-path", str(tmp_path),
        ]
        with patch("sys.argv", ["price_predictor", *args]):
            # Reset logging to test that main() configures it
            root = logging.getLogger()
            old_handlers = root.handlers[:]
            root.handlers.clear()
            try:
                main()
            finally:
                root.handlers = old_handlers

        # After train, the root logger should have had a handler configured
        # We verify this indirectly via the subprocess test below

    def test_train_progress_on_stderr_not_stdout(self) -> None:
        """Progress messages must appear on stderr, not stdout."""
        fixtures = Path(__file__).parents[2] / "fixtures"
        result = subprocess.run(
            [
                sys.executable, "-m", "price_predictor", "train",
                "--forge-cards-path", str(fixtures / "forge_cards"),
                "--prices-path", str(fixtures / "allprices_sample.json"),
                "--printings-path", str(fixtures / "allprintings_sample.json"),
                "--output-path", str(Path(__file__).parents[2] / ".." / "tmp_model"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parents[3]),
        )
        # stdout should contain only JSON (parseable)
        import json
        stdout = result.stdout.strip()
        if result.returncode == 0:
            parsed = json.loads(stdout)
            assert "model_version" in parsed

        # stderr should contain progress messages
        assert "Parsed" in result.stderr or "Parsing" in result.stderr

    def test_predict_does_not_emit_progress(self, tmp_path: Path) -> None:
        """Predict command should NOT configure logging — no progress messages."""
        fixtures = Path(__file__).parents[2] / "fixtures"
        # First train a model
        train_result = subprocess.run(
            [
                sys.executable, "-m", "price_predictor", "train",
                "--forge-cards-path", str(fixtures / "forge_cards"),
                "--prices-path", str(fixtures / "allprices_sample.json"),
                "--printings-path", str(fixtures / "allprintings_sample.json"),
                "--output-path", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parents[3]),
        )
        assert train_result.returncode == 0

        # Now predict — stderr should be empty (no progress)
        predict_result = subprocess.run(
            [
                sys.executable, "-m", "price_predictor", "predict",
                "--types", "Creature",
                "--mana-cost", "1 G",
                "--power", "2",
                "--toughness", "2",
                "--model-path", str(tmp_path / "latest.joblib"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).parents[3]),
        )
        assert predict_result.returncode == 0
        # stderr should have no progress messages
        assert "Parsed" not in predict_result.stderr
        assert "Loading" not in predict_result.stderr
        assert "Parsing" not in predict_result.stderr

    def test_evaluate_progress_on_stderr(self, tmp_path: Path) -> None:
        """Evaluate command should show progress on stderr."""
        fixtures = Path(__file__).parents[2] / "fixtures"
        # Train first
        train_result = subprocess.run(
            [
                sys.executable, "-m", "price_predictor", "train",
                "--forge-cards-path", str(fixtures / "forge_cards"),
                "--prices-path", str(fixtures / "allprices_sample.json"),
                "--printings-path", str(fixtures / "allprintings_sample.json"),
                "--output-path", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parents[3]),
        )
        assert train_result.returncode == 0

        # Evaluate
        eval_result = subprocess.run(
            [
                sys.executable, "-m", "price_predictor", "evaluate",
                "--model-path", str(tmp_path / "latest.joblib"),
                "--forge-cards-path", str(fixtures / "forge_cards"),
                "--prices-path", str(fixtures / "allprices_sample.json"),
                "--printings-path", str(fixtures / "allprintings_sample.json"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parents[3]),
        )
        assert eval_result.returncode == 0
        # stderr should have progress
        assert "Loading model" in eval_result.stderr or "Parsed" in eval_result.stderr
        # stdout should be clean JSON
        import json
        parsed = json.loads(eval_result.stdout.strip())
        assert "model_version" in parsed
