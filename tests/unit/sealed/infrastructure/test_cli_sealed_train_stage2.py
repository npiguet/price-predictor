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


# ─── T009: New hyperparameter CLI args (feature 016) ─────────────────────────

def test_urgency_exponent_argument_parsed():
    """--urgency-exponent 3 should be parsed as float 3.0."""
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2", "--urgency-exponent", "3"])
    assert args.urgency_exponent == pytest.approx(3.0)


def test_temperature_argument_parsed():
    """--temperature 0.5 should be parsed as float 0.5."""
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2", "--temperature", "0.5"])
    assert args.temperature == pytest.approx(0.5)


def test_urgency_exponent_default_is_2():
    """--urgency-exponent default must be 2.0 when not specified."""
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2"])
    assert args.urgency_exponent == pytest.approx(2.0)


def test_temperature_default_is_1():
    """--temperature default must be 1.0 when not specified."""
    parser = build_parser()
    args = parser.parse_args(["train", "--stage", "2"])
    assert args.temperature == pytest.approx(1.0)


def test_urgency_exponent_help_includes_default():
    """--urgency-exponent help text must include '(default: 2.0)'."""
    import io
    parser = build_parser()
    help_text = io.StringIO()
    try:
        parser.parse_args(["train", "--help"])
    except SystemExit:
        pass
    # Re-capture help via formatter
    train_parser = [a for a in parser._subparsers._actions
                    if hasattr(a, '_name_parser_map')]
    if train_parser:
        sub = train_parser[0]._name_parser_map.get("train")
        if sub:
            fmt = sub.format_help()
            assert "default: 2.0" in fmt, (
                f"Expected '(default: 2.0)' in train --help. Got:\n{fmt}"
            )


def test_temperature_help_includes_default():
    """--temperature help text must include '(default: 1.0)'."""
    parser = build_parser()
    train_parser = [a for a in parser._subparsers._actions
                    if hasattr(a, '_name_parser_map')]
    if train_parser:
        sub = train_parser[0]._name_parser_map.get("train")
        if sub:
            fmt = sub.format_help()
            assert "default: 1.0" in fmt, (
                f"Expected '(default: 1.0)' in train --help. Got:\n{fmt}"
            )


def test_run_train_stage2_passes_urgency_exponent(tmp_path):
    """run_train passes urgency_exponent to TrainStage2UseCase.execute()."""
    parser = build_parser()
    args = parser.parse_args([
        "train", "--stage", "2", "--urgency-exponent", "3",
    ])
    captured_calls: list[dict] = []

    def mock_execute(**kwargs):
        captured_calls.append(kwargs)
        raise ValueError("stop")

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = mock_execute

    with patch("sealed.application.train_stage2.TrainStage2UseCase", return_value=mock_use_case):
        run_train(args)

    assert len(captured_calls) == 1
    assert captured_calls[0]["urgency_exponent"] == pytest.approx(3.0)


def test_run_train_stage2_passes_temperature(tmp_path):
    """run_train passes temperature to TrainStage2UseCase.execute()."""
    parser = build_parser()
    args = parser.parse_args([
        "train", "--stage", "2", "--temperature", "0.5",
    ])
    captured_calls: list[dict] = []

    def mock_execute(**kwargs):
        captured_calls.append(kwargs)
        raise ValueError("stop")

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = mock_execute

    with patch("sealed.application.train_stage2.TrainStage2UseCase", return_value=mock_use_case):
        run_train(args)

    assert len(captured_calls) == 1
    assert captured_calls[0]["temperature"] == pytest.approx(0.5)
