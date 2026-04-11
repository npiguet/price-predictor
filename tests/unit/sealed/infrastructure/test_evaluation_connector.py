"""Unit tests for the evaluation connector (worker command, file splitting)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sealed.infrastructure.evaluation_connector import EvaluationConnector


class TestWorkerCommandConstruction:
    def test_builds_java_command(self):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(
            matches_file=Path("/tmp/matches-0.txt"),
        )
        assert "java" in cmd[0]
        assert "com.pricepredictor.connector.ValidationWorkerMain" in cmd
        assert any("/tmp/matches-0.txt" in str(arg) or "\\tmp\\matches-0.txt" in str(arg) for arg in cmd)

    def test_correct_main_class(self):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(matches_file=Path("/tmp/test.txt"))
        assert "com.pricepredictor.connector.ValidationWorkerMain" in cmd


class TestMatchFileSplitting:
    def test_splits_lines_across_workers(self, tmp_path):
        matches_file = tmp_path / "matches.txt"
        lines = [f"deck_a_{i};pool_b_{i}" for i in range(10)]
        matches_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        connector = EvaluationConnector()
        split_files = connector.split_matches_file(matches_file, n_workers=3, work_dir=tmp_path)

        assert len(split_files) == 3
        total_lines = 0
        for f in split_files:
            assert f.exists()
            total_lines += sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        assert total_lines == 10

    def test_single_worker_gets_all(self, tmp_path):
        matches_file = tmp_path / "matches.txt"
        matches_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

        connector = EvaluationConnector()
        split_files = connector.split_matches_file(matches_file, n_workers=1, work_dir=tmp_path)
        assert len(split_files) == 1
        content = split_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(content) == 3


class TestOutcomeFilePath:
    def test_derives_outcome_path(self):
        connector = EvaluationConnector()
        matches = Path("/tmp/validation-matches-0.txt")
        outcome = connector.outcome_file_path(matches)
        assert str(outcome).endswith("-outcomes.txt")
        assert "validation-matches-0.txt" in str(outcome)
