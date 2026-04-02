"""T024 — Unit tests for CLI sample subcommand Stage 2 routing."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from sealed.infrastructure.cli import build_parser, run_sample


def test_sample_stage_argument_defaults_to_1():
    parser = build_parser()
    args = parser.parse_args(["sample"])
    assert args.stage == 1


def test_sample_stage_2_argument_accepted():
    parser = build_parser()
    args = parser.parse_args(["sample", "--stage", "2"])
    assert args.stage == 2


def test_sample_stage2_model_path_default(tmp_path):
    """--stage 2 without --model-path should use models/sealed/stage2/{set}/latest.pt."""
    parser = build_parser()
    args = parser.parse_args(["sample", "--stage", "2", "--set", "TEST"])

    captured_calls: list[dict] = []

    def mock_execute(**kwargs):
        captured_calls.append(kwargs)
        raise FileNotFoundError("no model")

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = mock_execute

    with patch("sealed.application.sample_stage2.SampleStage2UseCase", return_value=mock_use_case):
        result = run_sample(args)

    assert result == 2
    assert len(captured_calls) == 1
    assert "stage2" in str(captured_calls[0]["model_path"])
    assert "TEST" in str(captured_calls[0]["model_path"])


def test_sample_stage1_default_model_path(tmp_path):
    """Default --stage 1 should use models/sealed/stage1/{set}/latest.pt."""
    parser = build_parser()
    args = parser.parse_args(["sample", "--set", "RVR"])  # stage defaults to 1

    captured_calls: list[dict] = []

    def mock_execute(**kwargs):
        captured_calls.append(kwargs)
        raise FileNotFoundError("no model")

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = mock_execute

    with patch("sealed.application.sample_stage1.SampleStage1UseCase", return_value=mock_use_case):
        result = run_sample(args)

    assert result == 2
    assert len(captured_calls) == 1
    assert "stage1" in str(captured_calls[0]["model_path"])
    assert "RVR" in str(captured_calls[0]["model_path"])


def test_run_sample_stage2_dispatches_to_stage2_use_case(tmp_path):
    """--stage 2 should call SampleStage2UseCase, not SampleStage1UseCase."""
    parser = build_parser()
    args = parser.parse_args(["sample", "--stage", "2"])

    # Should return 2 (missing model), not crash
    result = run_sample(args)
    assert result == 2


def test_run_sample_stage1_doesnt_dispatch_to_stage2(tmp_path):
    """Default stage 1 should call SampleStage1UseCase."""
    parser = build_parser()
    args = parser.parse_args(["sample"])

    # missing model → return 2, not crash
    result = run_sample(args)
    assert result == 2


def test_sample_stage2_n_samples_passed(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["sample", "--stage", "2", "--n-samples", "7"])
    assert args.n_samples == 7
