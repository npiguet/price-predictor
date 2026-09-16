"""The stage-four CLI surface (T136).

``--surface script`` writes a vocabulary of its own, and every stage-four flag
defaults to absent on the commands stage one already ships — a stage-one
checkpoint must keep working unchanged after a stage-four rebuild exists.
"""

from __future__ import annotations

import pytest

from effects.application.build_vocab import (
    DEFAULT_SCRIPT_VOCAB_PATH,
    DEFAULT_VOCAB_PATH,
    SURFACE_SCRIPT,
    BuildVocabConfig,
)
from effects.infrastructure.cli import build_parser


def parse(*argv: str):
    return build_parser().parse_args(argv)


class TestCollectVariants:
    def test_the_defaults_are_the_contracts(self):
        args = parse("collect-variants")
        assert args.effect_records == "output/effects/records/"
        assert args.forge_cards_path == "../forge/forge-gui/res/cardsfolder/"
        assert args.variant_scripts == "output/effects/variant-scripts/"
        assert args.variant_volume == 0.2
        assert args.decks_per_round == 500

    def test_it_reads_forges_source_scripts_not_the_converted_ones(self):
        """A variant is a Forge script Forge must be able to load."""
        args = parse("collect-variants")
        assert "forge-gui/res" in args.forge_cards_path
        assert "output/" not in args.forge_cards_path

    def test_it_carries_the_shared_cap_flags(self):
        args = parse("collect-variants")
        assert args.mana_cap == 1
        assert args.probe_keywords == ""

    def test_the_volume_cap_is_overridable(self):
        assert parse(
            "collect-variants", "--variant-volume", "0.05",
        ).variant_volume == 0.05

    def test_it_accepts_a_split_to_inherit(self):
        args = parse("collect-variants", "--split-from", "models/x/latest.pt")
        assert args.split_from == "models/x/latest.pt"


class TestScriptSurface:
    def test_the_two_surfaces_write_different_files(self):
        """So a stage-four rebuild never overwrites the vocabulary a
        stage-one-to-three checkpoint recorded."""
        assert DEFAULT_VOCAB_PATH != DEFAULT_SCRIPT_VOCAB_PATH
        assert BuildVocabConfig(
            surface=SURFACE_SCRIPT,
        ).resolved_vocab_path() == DEFAULT_SCRIPT_VOCAB_PATH

    def test_the_flag_accepts_the_script_surface(self):
        assert parse("build-vocab", "--surface", "script").surface == "script"

    def test_prose_stays_the_default(self):
        assert parse("build-vocab").surface == "prose"


class TestVariantScriptsFlag:
    #: ``train-effect-model`` refuses to parse without a corpus (FR-125
    #: withdrawn), so every trainer parse here carries one.
    _TRAIN = ("train-effect-model", "--corpus", "output/effects/corpus")

    def test_the_trainer_accepts_it_and_defaults_to_absent(self):
        assert parse(*self._TRAIN).variant_scripts is None
        assert parse(
            *self._TRAIN, "--variant-scripts", "output/effects/v/",
        ).variant_scripts == "output/effects/v/"

    def test_encode_abilities_accepts_it_and_defaults_to_absent(self):
        assert parse("encode-abilities").variant_scripts is None
        assert parse(
            "encode-abilities", "--variant-scripts", "output/effects/v/",
        ).variant_scripts == "output/effects/v/"

    def test_the_evaluator_accepts_it_and_defaults_to_absent(self):
        assert parse("evaluate-effect-model").variant_scripts is None
        assert parse(
            "evaluate-effect-model", "--variant-scripts", "output/effects/v/",
        ).variant_scripts == "output/effects/v/"

    def test_absent_by_default_everywhere_it_appears(self):
        """A stage-one checkpoint keeps working unchanged."""
        for argv in (
            self._TRAIN, ("encode-abilities",), ("evaluate-effect-model",),
        ):
            assert parse(*argv).variant_scripts is None, argv


class TestSurfaceFollowsTheVocabulary:
    def test_a_script_vocabulary_path_selects_the_script_surface(self):
        """The surface follows the loaded --vocab-path rather than a flag of
        its own, so the two cannot disagree."""
        from effects.domain.ability_encoder import (
            SURFACE_PROSE,
            surface_of,
        )
        from effects.domain.ability_encoder import (
            SURFACE_SCRIPT as ENCODER_SCRIPT,
        )

        assert surface_of("models/effects/vocab-script.txt") == ENCODER_SCRIPT
        assert surface_of("models/effects/vocab.txt") == SURFACE_PROSE

    def test_an_unrecognised_path_reads_as_prose(self):
        from effects.domain.ability_encoder import SURFACE_PROSE, surface_of

        assert surface_of("models/effects/anything.txt") == SURFACE_PROSE


class TestSubcommandTable:
    def test_collect_variants_is_registered(self):
        parser = build_parser()
        action = next(a for a in parser._actions if isinstance(a.choices, dict))
        assert "collect-variants" in action.choices

    def test_every_subcommand_is_registered(self):
        parser = build_parser()
        action = next(a for a in parser._actions if isinstance(a.choices, dict))
        assert set(action.choices) == {
            "build-vocab", "extract-keyword-definitions", "collect-coverage",
            "build-corpus", "field-coverage", "validate-corpus",
            "collect-variants", "train-effect-model", "encode-abilities",
            "evaluate-effect-model", "holdout-cards",
        }

    def test_an_unknown_flag_is_rejected(self):
        with pytest.raises(SystemExit):
            parse("collect-variants", "--not-a-flag")
