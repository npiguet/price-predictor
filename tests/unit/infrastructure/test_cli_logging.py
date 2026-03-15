"""Tests for logging configuration in the CLI entry point.

Note: These tests use subprocess to invoke the CLI and verify that
progress messages appear on stderr, not stdout. The train and evaluate
commands now require a model subcommand (sklearn/transformer).

The train sklearn and evaluate sklearn handlers are not yet wired
(they raise NotImplementedError). These tests will be fully enabled
once US1 (train) and US3 (evaluate) handlers are implemented.
"""

from __future__ import annotations

import pytest


class TestLoggingConfiguration:
    """Logging tests are deferred until US1/US2/US3 handlers are wired."""

    @pytest.mark.skip(reason="train sklearn handler not yet wired (US1)")
    def test_train_progress_on_stderr_not_stdout(self) -> None:
        pass

    @pytest.mark.skip(reason="predict sklearn handler not yet wired (US2)")
    def test_predict_does_not_emit_progress(self) -> None:
        pass

    @pytest.mark.skip(reason="evaluate sklearn handler not yet wired (US3)")
    def test_evaluate_progress_on_stderr(self) -> None:
        pass
