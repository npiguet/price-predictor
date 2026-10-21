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
        assert args.probe_keywords == (
            "first_strike,double_strike,deathtouch,lifelink,trample,"
            "indestructible,wither,infect"
        )
        assert args.legality_rate == 0.0

    def test_the_constants_and_the_parsed_defaults_agree(self):
        """Every cap in the table, not a list this test has to be told about.

        Spelled out one by one, a cap added to `CAP_DEFAULTS` and to only one
        of the two parsers passed here — which is how `legality_rate` came to
        be readable by the JVM and settable by nobody.
        """
        args = parse("collect-coverage")
        for name, default in CAP_DEFAULTS.items():
            assert getattr(args, name) == default, name

    def test_probes_are_on_by_default_and_an_empty_list_turns_them_off(self):
        """Every damage-step keyword is probed unless a run narrows the list,
        and an empty string takes no probe fork at all."""
        assert parse("collect-coverage").probe_keywords
        assert parse("collect-coverage", "--probe-keywords", "").probe_keywords == ""

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
        )
        assert args.mana_cap == 10
        assert args.playability_rate == 1.0
        assert args.probe_keywords == "wither"
        assert args.legality_rate == 0.25


class TestNoSnapshotTierFlag:
    """Every record collects all four tiers, so no command takes a depth."""

    def test_collect_coverage_refuses_it(self):
        with pytest.raises(SystemExit):
            parse("collect-coverage", "--snapshot-tiers", "1,2,3,4")

    def test_match_outcomes_refuses_it(self):
        with pytest.raises(SystemExit):
            sealed_parser().parse_args(
                ["match-outcomes", "--snapshot-tiers", "1,2,3,4"]
            )


class TestTheRandomSeatOnTheCollectors:
    """Coverage and variant decks reach the random seat like self-play does.

    The Forge AI never casts some cards at all; a random seat casts them for
    real, costs and choices included, where an intervention forks past both.
    """

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_the_flags_reach_the_worker_properties(self, command):
        from effects.domain.collection_caps import CollectionCaps

        args = parse(
            command, "--random-seat-share", "0.5",
            "--random-seat-probability", "0.25",
        )
        properties = CollectionCaps.from_args(args).as_system_properties()
        assert properties["effect.random.seat.share"] == "0.5"
        assert properties["effect.random.seat.probability"] == "0.25"

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_unset_flags_leave_the_seat_off(self, command):
        from effects.domain.collection_caps import CollectionCaps

        caps = CollectionCaps.from_args(parse(command))
        assert caps.random_seat_share == 0.0
        assert caps.random_seat_probability == 0.0

    @pytest.mark.parametrize("argv,message", [
        (("--random-seat-share", "0.5"), "needs --random-seat-probability"),
        (("--random-seat-share", "1.5", "--random-seat-probability", "0.5"),
         "between 0 and 1"),
        (("--random-seat-share", "0.5", "--random-seat-probability", "-0.1"),
         "between 0 and 1"),
    ])
    def test_a_bad_combination_is_refused(self, argv, message):
        from effects.infrastructure.cli import random_seat_error

        assert message in random_seat_error(parse("collect-coverage", *argv))

    def test_a_share_with_its_probability_is_accepted(self):
        from effects.infrastructure.cli import random_seat_error

        assert random_seat_error(parse(
            "collect-coverage", "--random-seat-share", "0.5",
            "--random-seat-probability", "0.5",
        )) is None

    @pytest.mark.parametrize("command,module", [
        ("collect-coverage", "effects.application.collect_coverage.run"),
        ("collect-variants", "effects.application.collect_variants.run"),
    ])
    def test_a_share_without_a_probability_starts_no_worker(
        self, command, module, capsys,
    ):
        from unittest.mock import patch

        from effects.infrastructure.cli import (
            run_collect_coverage,
            run_collect_variants,
        )

        runner = {
            "collect-coverage": run_collect_coverage,
            "collect-variants": run_collect_variants,
        }[command]
        with patch(module) as collect:
            assert runner(parse(command, "--random-seat-share", "0.5")) == 1
        collect.assert_not_called()
        assert "--random-seat-probability" in capsys.readouterr().out


class TestOneHoldoutSource:
    @pytest.mark.parametrize(("command", "module"), [
        ("collect-coverage", "effects.application.collect_coverage.run"),
        ("collect-variants", "effects.application.collect_variants.run"),
    ])
    def test_two_holdout_sources_are_refused_before_any_worker(
        self, command, module, capsys,
    ):
        """``--split-from`` beside ``--exclude-cards`` names one holdout twice;
        both collectors say so as a message, never a traceback."""
        from unittest.mock import patch

        from effects.infrastructure.cli import (
            run_collect_coverage,
            run_collect_variants,
        )

        runner = {
            "collect-coverage": run_collect_coverage,
            "collect-variants": run_collect_variants,
        }[command]
        with patch(module) as collect:
            assert runner(parse(
                command, "--split-from", "latest.pt", "--exclude-cards", "h.txt",
            )) == 1
        collect.assert_not_called()
        assert "one source of held-out cards" in capsys.readouterr().out


class TestSubcommandTable:
    def test_collect_coverage_is_registered(self):
        parser = build_parser()
        action = next(a for a in parser._actions if isinstance(a.choices, dict))
        assert "collect-coverage" in action.choices

    def test_an_unknown_subcommand_is_rejected(self):
        with pytest.raises(SystemExit):
            parse("collect-everything")


class TestTheHeldOutCoverageRound:
    """contracts/cli.md § collect-coverage: the two gen-2 flags and their defaults."""

    def test_only_cards_is_off_by_default_and_the_floor_is_five(self):
        args = parse("collect-coverage")
        assert args.only_cards is None
        assert args.min_text_games == 5

    def test_only_cards_beside_exclude_cards_is_refused(self):
        from effects.infrastructure.cli import coverage_config_from

        with pytest.raises(ValueError, match="--exclude-cards"):
            coverage_config_from(parse(
                "collect-coverage", "--only-cards", "h.txt", "--exclude-cards", "h.txt",
            ))
