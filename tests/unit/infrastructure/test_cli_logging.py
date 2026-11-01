"""Logging in the ``price_predictor`` CLI goes to stderr, never stdout.

stdout is the commands' result channel: ``predict`` prints the price and
``evaluate`` its JSON metrics there, so a progress or error line on stdout
corrupts what a caller parses. Each test runs the real entry point in a
subprocess on an input that fails before any data is loaded, which reaches the
handler's own logging without training or loading a model.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.unit.effects.builders import fresh_interpreter_env


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "price_predictor", *args],
        capture_output=True, text=True, env=fresh_interpreter_env(), timeout=120,
    )


class TestLoggingConfiguration:
    def test_train_progress_on_stderr_not_stdout(self, tmp_path: Path) -> None:
        result = _run("train", "sklearn", "--output-dir", str(tmp_path / "absent"))
        assert result.returncode == 1
        assert "Run 'convert' first" in result.stderr
        assert result.stdout == ""

    def test_predict_does_not_emit_progress(self, tmp_path: Path) -> None:
        result = _run("predict", "sklearn", "--file", str(tmp_path / "absent.txt"))
        assert result.returncode == 1
        assert "absent.txt" in result.stderr
        assert result.stdout == ""

    def test_evaluate_progress_on_stderr(self, tmp_path: Path) -> None:
        model = tmp_path / "absent.pkl"
        result = _run("evaluate", "sklearn", "--model-path", str(model))
        assert result.returncode == 1
        assert "Loading model from" in result.stderr
        assert "Model file not found" in result.stderr
        assert result.stdout == ""
