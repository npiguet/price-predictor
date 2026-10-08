"""Integration tests for the convert CLI subcommand."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest


def _convert_environment_available() -> bool:
    """Return True iff Java and the classpath ``convert`` itself builds exist.

    Asks the same builder ``run_convert`` uses rather than naming JAR versions,
    which went stale with every Forge upgrade and silently skipped this test.
    """
    if shutil.which("java") is None:
        return False
    from price_predictor.infrastructure.forge_jvm import build_forge_classpath

    try:
        build_forge_classpath(include_full_runtime=True, include_dependency_glob=True)
    except FileNotFoundError:
        return False
    return True


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
@pytest.mark.skipif(
    not _convert_environment_available(),
    reason="Java + Forge JARs + dependency directory required",
)
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
    tokens_dir = tmp_path / "tokenscripts"
    tokens_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable, "-m", "price_predictor", "convert",
            "--cards-path", str(fixture_dir.parent),
            "--output-path", str(output_dir),
            # Both token paths too: their defaults are the real Forge token
            # scripts and the real converted tree, which this test must not
            # overwrite.
            "--tokens-path", str(tokens_dir),
            "--tokens-output-path", str(tmp_path / "tokens-output"),
        ],
        capture_output=True,
        text=True,
        cwd="src",
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "t" / "test_bear.txt").exists()
