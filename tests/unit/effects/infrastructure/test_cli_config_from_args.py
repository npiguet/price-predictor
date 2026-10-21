"""``config_from_args``: a command's config read off its parsed namespace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pytest

from effects.infrastructure.cli import build_parser, config_from_args


@dataclass(frozen=True, slots=True)
class _Config:
    records: Path
    limit: int = 3
    label: str = "unset"


class TestConfigFromArgs:
    def test_a_field_takes_the_flag_of_its_name(self):
        args = argparse.Namespace(records=Path("r"), limit=7, func=print)
        assert config_from_args(_Config, args) == _Config(Path("r"), 7)

    def test_a_field_with_no_flag_keeps_its_default(self):
        assert config_from_args(
            _Config, argparse.Namespace(records=Path("r")),
        ).limit == 3

    def test_an_override_wins_over_the_flag(self):
        args = argparse.Namespace(records=Path("r"), limit=7)
        assert config_from_args(_Config, args, limit=9).limit == 9

    def test_an_override_naming_no_field_is_refused(self):
        """A misspelt override would otherwise fall back to the default."""
        with pytest.raises(TypeError, match="limt"):
            config_from_args(_Config, argparse.Namespace(records=Path("r")), limt=9)


class TestBuildCorpusKeepsTheTypedStrings:
    """The manifest records ``--vocab-path`` and ``--variant-scripts`` as
    typed, so they stay strings: a ``Path`` would write Windows separators into
    a manifest whose digest a checkpoint is pinned to."""

    def test_they_reach_the_config_as_strings(self, monkeypatch):
        from effects.infrastructure.cli import run_build_corpus

        seen = {}

        def capture(config):
            seen["config"] = config
            return 0

        monkeypatch.setattr("effects.application.build_corpus.build", capture)
        args = build_parser().parse_args([
            "build-corpus", "--variant-scripts", "output/effects/variant-scripts/",
        ])
        assert run_build_corpus(args) == 0
        config = seen["config"]
        assert config.vocab_path == "models/effects/vocab.txt"
        assert config.variant_scripts == "output/effects/variant-scripts/"
        assert config.records_dir == Path("output/effects/records/")
