"""Integration tests for the convert CLI subcommand."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

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


#: Forge's own scripts of a saga and a haunt card: a Chapter or Haunt keyword
#: line expands into several triggers, and each converted line must claim the
#: triggers it printed alone.
_FORGE_CARDS = (
    "t/the_fall_of_lord_konda_fragment_of_konda.txt",
    "b/blind_hunter.txt",
)


def _forge_cardsfolder() -> Path:
    from price_predictor.infrastructure.forge_jvm import forge_dir

    return forge_dir() / "forge-gui" / "res" / "cardsfolder"


def _triggers(line: dict) -> frozenset[int]:
    return frozenset(
        key["index_within_kind"] for key in line["provenance"] if key["trait_kind"] == "trigger"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _convert_environment_available(),
    reason="Java + Forge JARs + dependency directory required",
)
def test_convert_writes_the_current_provenance_sidecars(tmp_path):
    """The sidecar beside each converted card, as the effects package reads it:
    every line carries its ``prose_offset`` (where its prose starts after the
    line's label), a saga's chapter lines each claim their own chapter's
    trigger, and a haunt card's two haunt lines claim disjoint triggers."""
    source = _forge_cardsfolder()
    for relative in _FORGE_CARDS:
        if not (source / relative).exists():
            pytest.skip(f"{relative} is not in the sibling Forge checkout")
        target = tmp_path / "cardsfolder" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    output_dir = tmp_path / "output"
    tokens_dir = tmp_path / "tokenscripts"
    tokens_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable, "-m", "price_predictor", "convert",
            "--cards-path", str(tmp_path / "cardsfolder"),
            "--output-path", str(output_dir),
            "--tokens-path", str(tokens_dir),
            "--tokens-output-path", str(tmp_path / "tokens-output"),
        ],
        capture_output=True,
        text=True,
        cwd="src",
    )
    assert result.returncode == 0, result.stderr
    # The run directory holds both converted trees: here their common parent.
    assert (tmp_path / "run.json").exists(), "convert keeps no run manifest"

    saga = json.loads(
        (output_dir / "t" / "the_fall_of_lord_konda_fragment_of_konda.provenance.json")
        .read_text(encoding="utf-8")
    )
    haunt = json.loads(
        (output_dir / "b" / "blind_hunter.provenance.json").read_text(encoding="utf-8")
    )
    for sidecar in (saga, haunt):
        assert sidecar["lines"], sidecar["card"]
        for line in sidecar["lines"]:
            assert isinstance(line.get("prose_offset"), int), line
            assert line["prose_offset"] >= 0, line

    chapters = [line for line in saga["lines"] if line["line_kind"] == "chapter"]
    assert [sorted(_triggers(line)) for line in chapters] == [[0], [1], [2]], chapters

    haunts = [line for line in haunt["lines"] if _triggers(line)]
    assert len(haunts) == 2, haunt["lines"]
    first, second = (_triggers(line) for line in haunts)
    assert first and second and not first & second, (first, second)
