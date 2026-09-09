"""The collecting supervisors' flags and their shared caps (T101).

``--effect-records`` has a default on the effects-owned collectors and **no**
default on ``sealed match-outcomes``. That asymmetry is the contract: on this
side collection is the whole point, on that side it is an opt-in.
"""

from __future__ import annotations

import pytest

from effects.infrastructure.cli import CAP_DEFAULTS, build_parser
from sealed.infrastructure.cli import build_parser as sealed_parser


def parse(*argv: str):
    return build_parser().parse_args(argv)


class TestEffectRecordsDefaultAsymmetry:
    def test_collect_coverage_defaults_to_the_records_directory(self):
        assert parse("collect-coverage").effect_records == "output/effects/records/"

    def test_match_outcomes_has_no_default(self):
        """A default there would turn every self-play run into a collection run."""
        assert sealed_parser().parse_args(
            ["match-outcomes"],
        ).effect_records is None


class TestCollectCoverage:
    def test_the_defaults_are_the_contracts(self):
        args = parse("collect-coverage")
        assert args.target_records == 50
        assert args.decks_per_round == 500
        assert args.no_progress_rounds == 3
        assert args.split_from is None
        assert args.cards_folders is None  # both converted trees

    def test_split_from_is_accepted(self):
        args = parse("collect-coverage", "--split-from", "models/x/latest.pt")
        assert args.split_from == "models/x/latest.pt"

    def test_cards_folder_is_repeatable(self):
        args = parse(
            "collect-coverage", "--cards-folder", "a/", "--cards-folder", "b/",
        )
        assert args.cards_folders == ["a/", "b/"]

    def test_the_worker_count_has_the_repos_usual_default(self):
        assert parse("collect-coverage").workers == 12


class TestSharedCapFlags:
    def test_the_defaults_are_the_contracts(self):
        args = parse("collect-coverage")
        assert args.mana_cap == 1
        assert args.playability_rate == 0.1
        assert args.interventions_per_game == 2
        assert args.probes_per_game == 2
        assert args.probe_keywords == ""
        assert args.legality_rate == 0.1
        assert args.snapshot_tiers == "1,2,3"

    def test_the_constants_and_the_parsed_defaults_agree(self):
        """Every cap in the table, not a list this test has to be told about.

        Spelled out one by one, a cap added to `CAP_DEFAULTS` and to only one
        of the two parsers passed here — which is how `snapshot_tiers` and
        `legality_rate` came to be readable by the JVM and settable by nobody.
        """
        args = parse("collect-coverage")
        for name, default in CAP_DEFAULTS.items():
            assert getattr(args, name) == default, name

    def test_probes_are_off_until_keywords_are_named(self):
        """No probe fork is taken at all with the flag unset."""
        assert parse("collect-coverage").probe_keywords == ""

    def test_the_same_flags_appear_on_match_outcomes(self):
        """One cap table, shared by every collecting supervisor."""
        theirs = sealed_parser().parse_args(["match-outcomes"])
        ours = parse("collect-coverage")
        for flag in CAP_DEFAULTS:
            assert getattr(theirs, flag) == getattr(ours, flag), flag

    def test_continuous_records_take_no_cap(self):
        """Coalescing per stable board is itself the cap (FR-029)."""
        args = parse("collect-coverage")
        assert not hasattr(args, "continuous_cap")

    def test_each_cap_is_overridable(self):
        args = parse(
            "collect-coverage",
            "--mana-cap", "10", "--playability-rate", "1.0",
            "--probe-keywords", "wither", "--legality-rate", "0.25",
            "--snapshot-tiers", "1,2,3,4",
        )
        assert args.mana_cap == 10
        assert args.playability_rate == 1.0
        assert args.probe_keywords == "wither"
        assert args.legality_rate == 0.25
        assert args.snapshot_tiers == "1,2,3,4"


class TestTheSnapshotTierFlag:
    """Stage three's hand-and-graveyard tier has to be reachable from a flag.

    It was not: the Java side read `-Deffect.snapshot.tiers` and documented it,
    and no supervisor could set it, so tier 4 needed a code edit to request.
    """

    def test_tier_four_is_reachable(self):
        assert parse(
            "collect-coverage", "--snapshot-tiers", "1,2,3,4",
        ).snapshot_tiers == "1,2,3,4"

    def test_it_stays_a_string_because_it_travels_as_a_property(self):
        args = parse("collect-coverage", "--snapshot-tiers", "1,2")
        assert args.snapshot_tiers == "1,2"

    @pytest.mark.parametrize("bad", ["1,3", "2,3", "1", "1,2,4", "4,3,2,1", "x"])
    def test_a_vector_the_builder_would_refuse_is_refused_here(self, bad):
        """Both JVM-side failures are worse than an argument error: an
        unparseable vector falls back to the default silently, and a parseable
        non-prefix throws once a game is already running."""
        with pytest.raises(SystemExit):
            parse("collect-coverage", "--snapshot-tiers", bad)

    def test_match_outcomes_refuses_the_same_vectors(self):
        with pytest.raises(SystemExit):
            sealed_parser().parse_args(
                ["match-outcomes", "--snapshot-tiers", "1,3"]
            )


class TestSubcommandTable:
    def test_collect_coverage_is_registered(self):
        parser = build_parser()
        action = next(a for a in parser._actions if isinstance(a.choices, dict))
        assert "collect-coverage" in action.choices

    def test_an_unknown_subcommand_is_rejected(self):
        with pytest.raises(SystemExit):
            parse("collect-everything")
