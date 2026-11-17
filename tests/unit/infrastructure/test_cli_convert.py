"""``python -m price_predictor convert``: the flags it hands ``ConvertMain``."""

from __future__ import annotations

import subprocess

import pytest

from price_predictor.infrastructure import cli, forge_jvm

PATHS = [
    "--cards-path", "forge/cardsfolder",
    "--tokens-path", "forge/tokenscripts",
    "--output-path", "output/effects/gen3/converted/cardsfolder",
    "--tokens-output-path", "output/effects/gen3/converted/tokenscripts",
]


@pytest.fixture
def launched(monkeypatch) -> dict:
    """The JVM command ``run_convert`` builds, recorded instead of run."""
    seen: dict = {}

    def build(**kwargs):
        seen.update(kwargs)
        return ["java", kwargs["main_class"], *kwargs["main_args"]]

    monkeypatch.setattr(forge_jvm, "build_forge_classpath", lambda **_kw: "forge.jar")
    monkeypatch.setattr(forge_jvm, "build_jvm_command", build)
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, check: subprocess.CompletedProcess(cmd, 3),
    )
    return seen


def _convert(argv: list[str]) -> int:
    return cli.run_convert(cli.build_parser().parse_args(["convert", *argv]))


def test_convert_passes_every_path_and_returns_the_mains_exit(launched):
    assert _convert(PATHS) == 3

    assert launched["main_class"] == "com.pricepredictor.connector.ConvertMain"
    assert launched["main_args"] == [
        "--cards-path", "forge/cardsfolder",
        "--output-path", "output/effects/gen3/converted/cardsfolder",
        "--tokens-path", "forge/tokenscripts",
        "--tokens-output-path", "output/effects/gen3/converted/tokenscripts",
    ]


def test_convert_passes_replace_through(launched):
    _convert([*PATHS, "--replace"])

    assert launched["main_args"][-1] == "--replace"
    assert launched["main_args"].count("--replace") == 1


@pytest.mark.parametrize("flag", ["--output-path", "--tokens-output-path"])
def test_convert_has_no_default_output_tree(flag, launched):
    index = PATHS.index(flag)
    without = PATHS[:index] + PATHS[index + 2:]

    with pytest.raises(SystemExit):
        _convert(without)
    assert launched == {}
