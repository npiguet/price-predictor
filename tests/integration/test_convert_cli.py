"""Integration tests for the convert CLI subcommand."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _convert_environment_available() -> bool:
    """Return True iff Java + connector JAR + Forge dependency JARs are present."""
    if shutil.which("java") is None:
        return False
    project_root = Path(__file__).resolve().parent.parent.parent
    connector_jar = (
        project_root / "forge-connector" / "target"
        / "forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar"
    )
    forge_dir = project_root.parent / "forge"
    forge_game_jar = forge_dir / "forge-game" / "target" / "forge-game-2.0.10-SNAPSHOT.jar"
    forge_core_jar = forge_dir / "forge-core" / "target" / "forge-core-2.0.10-SNAPSHOT.jar"
    forge_deps_dir = forge_dir / "forge-game" / "target" / "dependency"
    if not connector_jar.exists() or not forge_game_jar.exists() or not forge_core_jar.exists():
        return False
    if not forge_deps_dir.is_dir() or not any(forge_deps_dir.glob("*.jar")):
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
    assert result.returncode == 0, result.stderr
    assert (output_dir / "t" / "test_bear.txt").exists()
