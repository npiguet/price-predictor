"""A command's config read off its parsed namespace, and flags read off a config."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from effects.application.build_corpus_config import BuildCorpusConfig
from effects.application.collect_coverage import CollectCoverageConfig
from effects.application.collect_variants import CollectVariantsConfig
from effects.application.corpus_validation.findings import Thresholds
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.infrastructure import cli
from effects.infrastructure.cli import (
    add_config_flags,
    collector_config,
    config_from_args,
)
from tests.unit.effects.cli_args import parse_cli


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
        args = parse_cli([
            "build-corpus", "--variant-scripts", "output/effects/variant-scripts/",
        ])
        assert run_build_corpus(args) == 0
        config = seen["config"]
        assert config.vocab_path == "vocab.txt"
        assert config.variant_scripts == "output/effects/variant-scripts/"
        assert config.records_dir == Path("records")


class TestAddConfigFlags:
    def test_a_flag_takes_its_fields_default_type_and_help(self):
        @dataclass
        class Config:
            limit: int | None = field(default=None, metadata={"help": "cap"})
            share: float = 0.5
            dry_run: bool = False

        parser = argparse.ArgumentParser()
        add_config_flags(parser, Config)
        assert parser.parse_args([]) == argparse.Namespace(
            limit=None, share=0.5, dry_run=False,
        )
        assert parser.parse_args(["--limit", "4", "--dry-run"]).limit == 4
        assert "cap" in parser.format_help()

    def test_without_only_it_skips_fields_a_flag_already_claims(self):
        @dataclass
        class Config:
            records: Path = Path("r")
            limit: int = 3

        parser = argparse.ArgumentParser()
        parser.add_argument("--records", type=Path, default="elsewhere")
        add_config_flags(parser, Config)
        assert parser.parse_args([]).records == Path("elsewhere")

    def test_a_field_no_scalar_type_fits_is_refused(self):
        """Its flag needs declaring by hand rather than a guessed ``str``."""
        @dataclass
        class Config:
            folders: tuple[Path, ...] = ()

        with pytest.raises(TypeError, match="folders"):
            add_config_flags(argparse.ArgumentParser(), Config)


class TestTheParsedDefaultsAreTheConfigsOwn:
    """The parser reproduces each config's defaults, so a command run with
    no flags builds the config a caller building it directly would."""

    def test_build_corpus(self):
        args = parse_cli(["build-corpus"])
        assert config_from_args(BuildCorpusConfig, args) == BuildCorpusConfig(
            records_dir=Path("records"), output=Path("corpus"), vocab_path="vocab.txt",
            keyword_definitions=Path("keywords.json"),
        )

    def test_train_effect_model(self):
        args = parse_cli(["train-effect-model", "--corpus", "c"])
        assert config_from_args(TrainEffectModelConfig, args) == (
            TrainEffectModelConfig.unpathed(
                corpus="c", vocab_path=Path("vocab.txt"),
                keyword_definitions=Path("keywords.json"), reports_dir=Path("reports"),
                model_output=Path("model"),
            )
        )

    def test_validate_corpus(self):
        args = parse_cli(["validate-corpus"])
        assert config_from_args(Thresholds, args) == Thresholds()

    @pytest.mark.parametrize(("command", "config"), [
        ("collect-coverage", CollectCoverageConfig),
        ("collect-variants", CollectVariantsConfig),
    ])
    def test_the_collectors(self, command, config):
        args = parse_cli([command])
        paths = {"effect_records": Path("records")}
        if command == "collect-variants":
            paths["variant_scripts"] = Path("variants")
        assert collector_config(config, args) == config(**paths)


class TestExitCodes:
    def test_every_class_named_by_string_exists(self):
        """A string key matches without importing, so a misspelt one would
        never match and its failure would surface as a traceback."""
        for handler in vars(cli).values():
            for kind in getattr(handler, "exit_codes", {}):
                if isinstance(kind, str):
                    module, _, name = kind.rpartition(".")
                    assert isinstance(
                        getattr(importlib.import_module(module), name), type,
                    ), kind
