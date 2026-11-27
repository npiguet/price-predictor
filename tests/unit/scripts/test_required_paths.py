"""Draft, sealed and effects commands that once defaulted a path under ``output/`` or
``models/`` take it as a required flag: started without it they
exit with a usage error naming it, and ``--help`` names no such default."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

_DRAFT_CARDS = ("--cards-path",)
_GRID = ("checkpoint.pt",)

# (command line after the interpreter, the flags its usage error must name)
COMMANDS = {
    "draft analyze-generated-decks": (["-m", "draft", "analyze-generated-decks"], _DRAFT_CARDS),
    "draft train-draft-agent-online": (["-m", "draft", "train-draft-agent-online"], _DRAFT_CARDS),
    "draft validate-builder": (["-m", "draft", "validate-builder"], _DRAFT_CARDS),
    "draft generate-draft-data": (["-m", "draft", "generate-draft-data"], _DRAFT_CARDS),
    "draft train-draft-agent": (["-m", "draft", "train-draft-agent"], _DRAFT_CARDS),
    "draft train-draft-agent-rl": (
        ["-m", "draft", "train-draft-agent-rl", "--checkpoint", "reference.pt"], _DRAFT_CARDS),
    "analyze_draft_lanes": (["scripts/analyze_draft_lanes.py"], _DRAFT_CARDS),
    "analyze_lane_discipline": (["scripts/analyze_lane_discipline.py"], _DRAFT_CARDS),
    "analyze_pick_quality": (["scripts/analyze_pick_quality.py"], _DRAFT_CARDS),
    "pick_metric_alignment": (["scripts/pick_metric_alignment.py"], _DRAFT_CARDS),
    "analyze_prefix_deck_value": (
        ["scripts/analyze_prefix_deck_value.py"], ("--cards-path", "--scorer", "--picker")),
    "analyze_winrates": (["scripts/analyze_winrates.py"], _DRAFT_CARDS),
    "compare_deck_builders": (
        ["scripts/compare_deck_builders.py"], ("--cards-path", "--output-dir")),
    "embedding_effective_rank": (
        ["scripts/embedding_effective_rank.py"],
        ("--encoder-checkpoint", "--vocab-path", "--cards-folder")),
    "split_half_reliability": (
        ["scripts/split_half_reliability.py"], ("--cards-played", "--cards-folder")),
    "effect_sweep_arm": (
        ["scripts/effect_sweep_arm.py"],
        ("--vocab-path", "--variant-scripts", "--probe-set-dir")),
    "plot_effect_model_training": (["scripts/plot_effect_model_training.py"], ("--runs-dir",)),
    "rule_entity_audit": (["scripts/rule_entity_audit.py"], ("--corpus", "--output")),
    "analyze_keyword_combat_avoidance": (
        ["scripts/analyze_keyword_combat_avoidance.py", "records"],
        ("--cards-folder", "--tokenscripts", "--variant-scripts")),
    "scan_output_vocabularies": (
        ["scripts/scan_output_vocabularies.py", "--records-dir", "records"],
        ("--keyword-definitions",)),
    "effect_embedding_probes/build_texts": (
        ["scripts/effect_embedding_probes/build_texts.py"],
        ("--checkpoint", "--abilities-root", "--reports-dir", "--variant-scripts")),
    "effect_embedding_probes/effect_profiles": (
        ["scripts/effect_embedding_probes/effect_profiles.py"],
        ("--checkpoint", "--abilities-root", "--reports-dir")),
    "effect_knowledge_probes/run": (
        ["scripts/effect_knowledge_probes/run.py"], ("--reports-dir",)),
    "effect_knowledge_probes/combat_grid": (
        ["scripts/effect_knowledge_probes/combat_grid.py", *_GRID], ("--probe-set-dir",)),
    "effect_knowledge_probes/damage_grid": (
        ["scripts/effect_knowledge_probes/damage_grid.py", *_GRID], ("--probe-set-dir",)),
    "effect_knowledge_probes/evasion_grid": (
        ["scripts/effect_knowledge_probes/evasion_grid.py", *_GRID], ("--probe-set-dir",)),
    "effect_knowledge_probes/ladder_check": (
        ["scripts/effect_knowledge_probes/ladder_check.py", *_GRID], ("--probe-set-dir",)),
}

# "(default: output/...)" or "(default: models/...)", over line wraps
# Flags the battery's shared parser refuses first, before a script's own flags are read.
REFUSED_FIRST = {
    "effect_embedding_probes/build_texts": ("--checkpoint", "--abilities-root", "--reports-dir"),
}

# Under ``output/`` or ``models/``, over line wraps
_PRE_LAYOUT_PATH = re.compile(r"\b(?:output|models)/")


def _run(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *arguments], cwd=REPO, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300, check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def _entry(help_text: str, flag: str) -> str:
    """The help text of one flag: its line to the next flag's."""
    found = re.search(rf"^ {{2}}{re.escape(flag)}\b.*?(?=^ {{2}}-|\Z)", help_text,
                      re.MULTILINE | re.DOTALL)
    assert found, f"{flag} is not in --help: {help_text}"
    return found.group(0)


@pytest.mark.parametrize("name", COMMANDS)
def test_started_without_its_path_flags_the_command_refuses_naming_them(name):
    command, flags = COMMANDS[name]
    result = _run(*command)
    assert result.returncode == 2, result.stderr[-600:]
    for flag in REFUSED_FIRST.get(name, flags):
        assert flag in result.stderr, f"{flag} not named in: {result.stderr[-600:]}"


@pytest.mark.parametrize("name", COMMANDS)
def test_help_names_no_default_under_output_or_models_for_those_flags(name):
    command, flags = COMMANDS[name]
    result = _run(*command, "--help")
    assert result.returncode == 0, result.stderr[-600:]
    for flag in flags:
        entry = _entry(result.stdout, flag)
        assert not _PRE_LAYOUT_PATH.search(entry), entry
