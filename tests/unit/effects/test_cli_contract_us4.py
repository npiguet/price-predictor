"""The stage-four CLI surface (T136).

``--surface script`` writes a vocabulary of its own, and every stage-four flag
defaults to absent on the commands stage one already ships — a stage-one
checkpoint must keep working unchanged after a stage-four rebuild exists.
"""

from __future__ import annotations

from pathlib import Path

from effects.infrastructure.cli import build_parser


def parse(*argv: str):
    return build_parser().parse_args(argv)


class TestCollectVariants:
    def test_the_defaults_are_the_contracts(self):
        from effects.application.collect_variants import CollectVariantsConfig
        from effects.infrastructure.cli import collector_config

        args = parse("collect-variants")
        records = collector_config(CollectVariantsConfig, args).effect_records
        assert records == Path("output/effects/records/")
        assert args.forge_cards_path == Path("../forge/forge-gui/res/cardsfolder/")
        assert args.variant_scripts == Path("output/effects/variant-scripts/")
        assert args.variant_volume == 0.2
        assert args.decks_per_round == 500

    def test_it_carries_the_shared_cap_flags(self):
        args = parse("collect-variants")
        assert args.mana_cap == 1
        assert args.probe_keywords == (
            "first_strike,double_strike,deathtouch,lifelink,trample,"
            "indestructible,wither,infect"
        )


class TestVariantScriptsFlag:
    #: ``train-effect-model`` refuses to parse without a corpus (FR-125
    #: withdrawn), so every trainer parse here carries one.
    _TRAIN = ("train-effect-model", "--corpus", "output/effects/corpus")

    def test_absent_by_default_everywhere_it_appears(self):
        """A stage-one checkpoint keeps working unchanged."""
        for argv in (
            self._TRAIN, ("encode-abilities",), ("evaluate-effect-model",),
        ):
            assert parse(*argv).variant_scripts is None, argv
