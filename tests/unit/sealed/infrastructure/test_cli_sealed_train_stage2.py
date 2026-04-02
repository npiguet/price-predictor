"""T018 — Unit tests for CLI train subcommand Stage 2 routing."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from sealed.infrastructure.cli import build_parser, run_train


def test_stage_2_argument_accepted():
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2"])
    assert args.stage == 2


def test_init_from_argument_parsed():
    parser = build_parser()
    args = parser.parse_args([
        "train", "--stage", "2",
        "--init-from", "models/sealed/stage1/RVR/latest.pt",
    ])
    assert args.init_from == "models/sealed/stage1/RVR/latest.pt"


def test_init_from_default_uses_set_code():
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2", "--set", "MH3"])
    # Default init-from should use the set code template
    # (actual default checked in run_train resolution)
    assert args.set_code == "MH3"


def test_model_path_default_for_stage2():
    """When --stage 2 and no --model-path, default should be models/sealed/stage2/{set}/latest.pt."""
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2", "--set", "RVR"])
    # model_path is None → resolved in run_train
    assert args.model_path is None


def test_run_train_stage2_uses_stage2_model_path_default(tmp_path):
    """run_train with stage 2 should use models/sealed/stage2/{set}/latest.pt as default."""
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2", "--set", "TEST"])

    captured_calls: list[dict] = []

    def mock_execute(**kwargs):
        captured_calls.append(kwargs)
        raise FileNotFoundError("no pools")

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = mock_execute

    with patch("sealed.application.train_stage2.TrainStage2UseCase", return_value=mock_use_case):
        result = run_train(args)

    assert result == 2  # FileNotFoundError → code 2
    assert len(captured_calls) == 1
    assert "stage2" in str(captured_calls[0]["model_path"])
    assert "TEST" in str(captured_calls[0]["model_path"])


def test_run_train_stage2_passes_init_from(tmp_path):
    """run_train with stage 2 should pass init_from to TrainStage2UseCase."""
    parser = build_parser()
    args = parser.parse_args([
        "train", "--stage", "2", "--set", "TEST",
        "--init-from", str(tmp_path / "stage1.pt"),
    ])

    captured_calls: list[dict] = []

    def mock_execute(**kwargs):
        captured_calls.append(kwargs)
        raise ValueError("stopped")

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = mock_execute

    with patch("sealed.application.train_stage2.TrainStage2UseCase", return_value=mock_use_case):
        result = run_train(args)

    assert result == 2
    assert len(captured_calls) == 1
    assert "stage1.pt" in str(captured_calls[0]["init_from"])


def test_run_train_stage2_dispatches_to_stage2_use_case(tmp_path):
    """--stage 2 should NOT dispatch to TrainStage1UseCase."""
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2"])

    # run_train should NOT return code 1 for stage 2 (1 = unknown stage)
    result = run_train(args)
    assert result != 1, "Stage 2 should be a known stage, not exit with code 1"


def test_init_from_default_contains_stage1_path():
    """Default init_from should default to models/sealed/stage1/{set}/latest.pt."""
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2", "--set", "RVR"])
    # init_from is None (resolved in run_train) or has the default
    # Either way, check that run_train resolves it correctly
    captured_calls: list[dict] = []

    def mock_execute(**kwargs):
        captured_calls.append(kwargs)
        raise ValueError("stop")

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = mock_execute

    with patch("sealed.application.train_stage2.TrainStage2UseCase", return_value=mock_use_case):
        run_train(args)

    assert len(captured_calls) == 1
    assert "stage1" in str(captured_calls[0]["init_from"])
    assert "RVR" in str(captured_calls[0]["init_from"])
