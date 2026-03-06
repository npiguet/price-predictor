"""Integration tests for the convert CLI subcommand."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_convert_subcommand_in_help():
    """The convert subcommand should appear in CLI help output."""
    result = subprocess.run(
        [sys.executable, "-m", "price_predictor", "--help"],
        capture_output=True,
        text=True,
        cwd="src",
    )
    assert "convert" in result.stdout


def test_convert_help_shows_expected_arguments():
    """Running convert --help should show cards-path and output-path."""
    result = subprocess.run(
        [sys.executable, "-m", "price_predictor", "convert", "--help"],
        capture_output=True,
        text=True,
        cwd="src",
    )
    assert result.returncode == 0
    assert "--cards-path" in result.stdout
    assert "--output-path" in result.stdout


@pytest.mark.integration
def test_convert_produces_output(tmp_path):
    """Running convert on fixture files produces output."""
    fixture_dir = (
        tmp_path / "cardsfolder" / "t"
    )
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "test_bear.txt").write_text(
        "Name:Test Bear\nManaCost:1 G\nTypes:Creature Bear\nPT:2/2\nOracle:\n"
    )
    output_dir = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable, "-m", "price_predictor", "convert",
            "--cards-path", str(fixture_dir.parent),
            "--output-path", str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd="src",
    )
    # This test only passes if Java + Forge JARs are available
    if result.returncode == 2:
        pytest.skip("Java/JARs not available for convert command")
    assert result.returncode == 0
    assert (output_dir / "t" / "test_bear.txt").exists()
