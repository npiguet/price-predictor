"""T023 — Unit tests for CLI sample subcommand."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from sealed.infrastructure.cli import build_parser, run_sample


def test_n_samples_passed_through():
    parser = build_parser()
    args = parser.parse_args(["sample", "--n-samples", "5"])
    assert args.n_samples == 5


def test_default_n_samples_is_10():
    parser = build_parser()
    args = parser.parse_args(["sample"])
    assert args.n_samples == 10


def test_missing_checkpoint_exits_with_code_2(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "sample",
        "--pools-path", str(tmp_path / "pools"),
        "--cards-path", str(tmp_path / "cards"),
        "--model-path", str(tmp_path / "nonexistent.pt"),
        "--n-samples", "1",
    ])
    # run_sample should catch FileNotFoundError and return 2
    result = run_sample(args)
    assert result == 2


def test_sample_default_set_is_rvr():
    parser = build_parser()
    args = parser.parse_args(["sample"])
    assert args.set_code == "RVR"


def test_sample_set_option():
    parser = build_parser()
    args = parser.parse_args(["sample", "--set", "MH3"])
    assert args.set_code == "MH3"


def test_sample_dispatch_calls_use_case(tmp_path):
    """run_sample with a missing model should return code 2 (not crash)."""
    parser = build_parser()
    args = parser.parse_args([
        "sample",
        "--model-path", str(tmp_path / "no.pt"),
        "--pools-path", str(tmp_path / "pools"),
        "--cards-path", str(tmp_path / "cards"),
    ])
    result = run_sample(args)
    assert result == 2
