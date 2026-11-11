"""The collecting supervisors' flags and their shared caps (T101).

``--effect-records`` has a default on the effects-owned collectors and **no**
default on ``sealed match-outcomes``. That asymmetry is the contract: on this
side collection is the whole point, on that side it is an opt-in.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from effects.domain.collection_caps import CollectionCaps
from effects.infrastructure.cli import (
    build_parser,
    run_collect_coverage,
    run_collect_variants,
)
from sealed.infrastructure.cli import build_parser as sealed_parser

#: Every cap and its default, read off the dataclass the workers receive.
#: ``random_seat_probability`` is left out: its flag defaults to None so that a
#: share above 0 without it can be refused (FR-026). So is ``aimed_at``, which
#: no cap flag sets: the collectors stamp it from ``--directed-fork-rules``.
#: So are the directed-fork caps, which only ``collect-variants`` takes
#: (``DIRECTED_DEFAULTS``): on ``collect-coverage`` the rules are a label.
DIRECTED_CAPS = ("directed_fork_rules", "directed_forks_per_game", "directed_fork_probability")
CAP_DEFAULTS = {
    spec.name: spec.default
    for spec in fields(CollectionCaps)
    if spec.name not in ("random_seat_probability", "aimed_at", *DIRECTED_CAPS)
}
DIRECTED_DEFAULTS = {
    spec.name: spec.default for spec in fields(CollectionCaps) if spec.name in DIRECTED_CAPS
}


def parse(*argv: str):
    return build_parser().parse_args(argv)


#: Stands in for the JAR, ``java`` and input-file check (python R2), which a
#: unit test must not make depend on a built connector.
PREREQUISITES = "price_predictor.infrastructure.forge_jvm.check_worker_prerequisites"


@pytest.fixture(autouse=True)
def _workers_could_start():
    from unittest.mock import patch

    with patch(PREREQUISITES) as check:
        yield check


#: Each collecting command, the application entry point its dispatcher calls,
#: and the dispatcher.
COLLECTORS = pytest.mark.parametrize(("command", "module", "runner"), [
    ("collect-coverage", "effects.application.collect_coverage.run", run_collect_coverage),
    ("collect-variants", "effects.application.collect_variants.run", run_collect_variants),
])


class TestEffectRecordsDefaultAsymmetry:
    def test_collect_coverage_defaults_to_the_records_directory(self):
        from effects.application.collect_coverage import CollectCoverageConfig
        from effects.infrastructure.cli import collector_config

        config = collector_config(CollectCoverageConfig, parse("collect-coverage"))
        assert config.effect_records == Path("output/effects/records/")

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
        assert args.seed == 42
        assert args.split_from is None
        assert args.cards_folders == (Path("output/cardsfolder"), Path("output/tokenscripts"))

    def test_the_worker_count_has_the_repos_usual_default(self):
        assert parse("collect-coverage").workers == 12


class TestSharedCapFlags:
    def test_the_defaults_are_the_contracts(self):
        args = parse("collect-coverage")
        assert args.mana_cap == 1
        assert args.playability_rate == 0.01
        assert args.zone_candidate_rate == 0.0001
        assert args.plain_candidate_rate == 0.001
        assert args.interventions_per_game == 2
        assert args.probes_per_game == 2
        assert args.probe_keywords == (
            "first_strike,double_strike,deathtouch,lifelink,trample,"
            "indestructible,wither,infect"
        )

    def test_the_constants_and_the_parsed_defaults_agree(self):
        """Every cap in the table, not a list this test has to be told about.

        Spelled out one by one, a cap added to `CollectionCaps` and to only one
        of the two parsers would pass here, readable by the JVM and settable by
        nobody.
        """
        args = parse("collect-coverage")
        for name, default in CAP_DEFAULTS.items():
            assert getattr(args, name) == default, name

    def test_probes_are_on_by_default_and_an_empty_list_turns_them_off(self):
        """Every damage-step keyword is probed unless a run narrows the list,
        and an empty string takes no probe fork at all."""
        assert parse("collect-coverage").probe_keywords
        assert parse("collect-coverage", "--probe-keywords", "").probe_keywords == ""

    def test_continuous_records_take_no_cap(self):
        """Coalescing per stable board is itself the cap (FR-029)."""
        args = parse("collect-coverage")
        assert not hasattr(args, "continuous_cap")

    def test_each_cap_is_overridable(self):
        args = parse(
            "collect-coverage",
            "--mana-cap", "10", "--playability-rate", "1.0",
            "--zone-candidate-rate", "0.25", "--plain-candidate-rate", "0.5",
            "--probe-keywords", "wither",
        )
        assert args.mana_cap == 10
        assert args.playability_rate == 1.0
        assert args.zone_candidate_rate == 0.25
        assert args.plain_candidate_rate == 0.5
        assert args.probe_keywords == "wither"


class TestDecisionSamplingOnTheCollectors:
    """FR-033b: the three sampling rates reach the worker, and a bucket rate
    above the stage-1 rate it is drawn from is refused before any worker."""

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_the_rates_reach_the_worker_properties(self, command):
        from effects.infrastructure.cli import collection_caps

        args = parse(
            command, "--playability-rate", "0.5",
            "--zone-candidate-rate", "0.125", "--plain-candidate-rate", "0.25",
        )
        properties = collection_caps(args).as_system_properties()
        assert properties["effect.playability.rate"] == "0.5"
        assert properties["effect.zone.candidate.rate"] == "0.125"
        assert properties["effect.plain.candidate.rate"] == "0.25"

    @pytest.mark.parametrize("argv,message", [
        (("--zone-candidate-rate", "0.5"), "exceeds --playability-rate"),
        (("--plain-candidate-rate", "0.02"), "exceeds --playability-rate"),
        (("--playability-rate", "1.5"), "between 0 and 1"),
        (("--zone-candidate-rate", "-0.1"), "between 0 and 1"),
    ])
    def test_a_bad_rate_is_refused(self, argv, message):
        from effects.infrastructure.cli import candidate_rate_error

        assert message in candidate_rate_error(parse("collect-coverage", *argv))

    def test_the_defaults_are_accepted(self):
        from effects.infrastructure.cli import candidate_rate_error

        assert candidate_rate_error(parse("collect-coverage")) is None

    @COLLECTORS
    def test_a_bad_rate_starts_no_worker(self, command, module, runner, caplog):
        from unittest.mock import patch

        with patch(module) as collect:
            assert runner(parse(command, "--zone-candidate-rate", "0.5")) == 1
        collect.assert_not_called()
        assert f"{command}: " in caplog.text
        assert "--zone-candidate-rate" in caplog.text


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
        from effects.infrastructure.cli import collection_caps

        args = parse(
            command, "--random-seat-share", "0.5",
            "--random-seat-probability", "0.25",
        )
        properties = collection_caps(args).as_system_properties()
        assert properties["effect.random.seat.share"] == "0.5"
        assert properties["effect.random.seat.probability"] == "0.25"

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_unset_flags_leave_the_seat_off(self, command):
        from effects.infrastructure.cli import collection_caps

        caps = collection_caps(parse(command))
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

    @COLLECTORS
    def test_a_share_without_a_probability_starts_no_worker(
        self, command, module, runner, caplog,
    ):
        from unittest.mock import patch

        with patch(module) as collect:
            assert runner(parse(command, "--random-seat-share", "0.5")) == 1
        collect.assert_not_called()
        assert f"{command}: " in caplog.text
        assert "--random-seat-probability" in caplog.text


class TestCountDroppedEventsOnTheCollectors:
    """FR-041: both collectors take ``--count-dropped-events`` as
    ``match-outcomes`` does, off by default, and send it to the worker."""

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_it_is_off_by_default(self, command):
        from effects.infrastructure.cli import collection_caps

        caps = collection_caps(parse(command))
        assert caps.count_dropped_events is False
        assert caps.as_system_properties()["effect.count.dropped.events"] == "false"

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_the_flag_reaches_the_worker_property(self, command):
        from effects.infrastructure.cli import collection_caps

        caps = collection_caps(parse(command, "--count-dropped-events"))
        assert caps.count_dropped_events is True
        assert caps.as_system_properties()["effect.count.dropped.events"] == "true"


class TestMaxRecordsPerGameOnTheCollectors:
    """FR-041a: both collectors take ``--max-records-per-game`` as
    ``match-outcomes`` does, 5000 by default, and send it to the worker."""

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_it_is_5000_by_default(self, command):
        from effects.infrastructure.cli import collection_caps

        caps = collection_caps(parse(command))
        assert caps.max_records_per_game == 5000
        assert caps.as_system_properties()["effect.max.records.per.game"] == "5000"

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_the_flag_reaches_the_worker_property(self, command):
        from effects.infrastructure.cli import collection_caps

        caps = collection_caps(parse(command, "--max-records-per-game", "300"))
        assert caps.max_records_per_game == 300
        assert caps.as_system_properties()["effect.max.records.per.game"] == "300"


class TestOneHoldoutSource:
    @COLLECTORS
    def test_two_holdout_sources_are_refused_before_any_worker(
        self, command, module, runner, caplog,
    ):
        """``--split-from`` beside ``--exclude-cards`` names one holdout twice;
        both collectors say so as a message, never a traceback."""
        from unittest.mock import patch

        with patch(module) as collect:
            assert runner(parse(
                command, "--split-from", "latest.pt", "--exclude-cards", "h.txt",
            )) == 1
        collect.assert_not_called()
        assert "one source of held-out cards" in caplog.text


class TestCollectVariantsHoldoutRule:
    """A6: the rule a variant printing a held-out text is dropped under is the
    one ``build-corpus`` is given, so the collector takes the same flags."""

    def test_the_defaults_are_build_corpuss(self):
        from effects.infrastructure.cli import holdout_rule

        assert holdout_rule(parse("collect-variants")) == holdout_rule(parse("build-corpus"))
        assert parse("collect-variants").cards_folders == parse("build-corpus").cards_folders

    def test_the_flags_reach_the_config(self):
        from unittest.mock import patch

        from effects.domain.split import HoldoutRule
        from effects.domain.text_holdout import HoldoutUnit
        from effects.infrastructure.cli import run_collect_variants

        with patch("effects.application.collect_variants.run", return_value=0) as collect:
            assert run_collect_variants(parse(
                "collect-variants", "--exclude-cards", "h.txt", "--cards-folder", "trees/cards",
                "--holdout-permille", "50", "--holdout-max-carriers", "3",
                "--holdout-unit", "text",
            )) == 0
        config = collect.call_args.args[0]
        assert config.holdout == HoldoutRule(50, 3, HoldoutUnit.TEXT)
        assert config.cards_folders == (Path("trees/cards"),)


class TestTheHeldOutCoverageRound:
    """contracts/cli.md § collect-coverage: the two gen-2 flags and their defaults."""

    def test_only_cards_is_off_by_default_and_the_floor_is_five(self):
        args = parse("collect-coverage")
        assert args.only_cards is None
        assert args.min_text_games == 5

    def test_only_cards_beside_exclude_cards_is_refused(self):
        from effects.application.collect_coverage import CollectCoverageConfig
        from effects.infrastructure.cli import collector_config

        with pytest.raises(ValueError, match="--exclude-cards"):
            collector_config(CollectCoverageConfig, parse(
                "collect-coverage", "--only-cards", "h.txt", "--exclude-cards", "h.txt",
            ))


class TestTheWordingRound:
    """contracts/cli.md § collect-coverage: ``--wording-cards PATH`` (FR-038)."""

    def test_it_is_off_by_default(self):
        assert parse("collect-coverage").wording_cards is None

    def test_the_list_reaches_the_config(self):
        from unittest.mock import patch

        with patch("effects.application.collect_coverage.run", return_value=0) as collect:
            assert run_collect_coverage(parse(
                "collect-coverage", "--wording-cards", "any-target.txt",
                "--exclude-cards", "h.txt",
            )) == 0
        assert collect.call_args.args[0].wording_cards == Path("any-target.txt")

    def test_beside_only_cards_it_is_refused_before_any_worker(self, caplog):
        from unittest.mock import patch

        with patch("effects.application.collect_coverage.run") as collect:
            assert run_collect_coverage(parse(
                "collect-coverage", "--wording-cards", "w.txt", "--only-cards", "h.txt",
            )) == 1
        collect.assert_not_called()
        assert "collect-coverage: " in caplog.text
        assert "--wording-cards or --only-cards" in caplog.text

    def test_collect_variants_takes_no_list(self):
        with pytest.raises(SystemExit):
            parse("collect-variants", "--wording-cards", "w.txt")


class TestEveryCollectorRecordIsAimed:
    """FR-009a, FR-012: every ``collect-coverage`` and ``collect-variants``
    record carries ``aimed_at``, empty where the run names no rule; on
    ``collect-coverage`` ``--directed-fork-rules`` is a label only."""

    @pytest.mark.parametrize("command", ["collect-coverage", "collect-variants"])
    def test_a_run_naming_no_rule_is_aimed_at_none(self, command):
        from effects.infrastructure.cli import collection_caps

        caps = collection_caps(parse(command))
        assert caps.aimed_at == ()
        assert caps.as_system_properties()["effect.aimed.at"] == ""

    def test_the_listed_rules_reach_the_worker_property(self):
        from effects.infrastructure.cli import collection_caps

        caps = collection_caps(parse(
            "collect-coverage", "--directed-fork-rules", "704.5f, 702.19b",
        ))
        assert caps.aimed_at == ("704.5f", "702.19b")
        assert caps.as_system_properties()["effect.aimed.at"] == "704.5f,702.19b"

    def test_an_empty_list_names_no_rule(self):
        assert parse("collect-coverage", "--directed-fork-rules", "").directed_fork_rules == ()

    def test_the_rules_reach_the_config_the_collector_runs(self):
        from unittest.mock import patch

        with patch("effects.application.collect_coverage.run", return_value=0) as collect:
            assert run_collect_coverage(parse(
                "collect-coverage", "--directed-fork-rules", "704.5f",
                "--exclude-cards", "h.txt",
            )) == 0
        assert collect.call_args.args[0].caps.aimed_at == ("704.5f",)

    def test_the_label_takes_no_startup_check(self):
        """A rule no detector knows still runs: no directed fork is taken."""
        from unittest.mock import patch

        with patch("effects.application.collect_coverage.run", return_value=0) as collect:
            assert run_collect_coverage(parse(
                "collect-coverage", "--directed-fork-rules", "no-such-rule",
                "--exclude-cards", "h.txt",
            )) == 0
        collect.assert_called_once()


class TestDirectedForksOnCollectVariants:
    """FR-012, contracts/cli.md § collect-variants: the three flags as on
    ``match-outcomes``, with the same startup check; the rules both take
    directed forks and aim every record."""

    def test_the_defaults_are_the_contracts(self):
        args = parse("collect-variants")
        for name, default in DIRECTED_DEFAULTS.items():
            assert getattr(args, name) == default, name
        assert args.directed_fork_rules == ()
        assert args.directed_forks_per_game == 2
        assert args.directed_fork_probability == 0.25

    def test_the_flags_reach_the_worker_properties(self):
        from effects.infrastructure.cli import collection_caps

        caps = collection_caps(parse(
            "collect-variants", "--directed-fork-rules", "704.5f,702.19b",
            "--directed-forks-per-game", "3", "--directed-fork-probability", "1",
        ))
        properties = caps.as_system_properties()
        assert properties["effect.directed.fork.rules"] == "704.5f,702.19b"
        assert properties["effect.directed.forks.per.game"] == "3"
        assert properties["effect.directed.fork.probability"] == "1.0"
        assert caps.aimed_at == ("704.5f", "702.19b")

    def test_collect_coverage_takes_only_the_label(self):
        """No directed fork runs on a coverage round: its rules aim the
        records and reach no fork property."""
        from effects.infrastructure.cli import collection_caps

        with pytest.raises(SystemExit):
            parse("collect-coverage", "--directed-forks-per-game", "3")
        with pytest.raises(SystemExit):
            parse("collect-coverage", "--directed-fork-probability", "1")
        caps = collection_caps(parse("collect-coverage", "--directed-fork-rules", "704.5f"))
        assert caps.directed_fork_rules == ()
        assert caps.as_system_properties()["effect.directed.fork.rules"] == ""
        assert caps.aimed_at == ("704.5f",)

    def test_the_rules_reach_the_config_the_collector_checks(self):
        from unittest.mock import patch

        with patch("effects.application.collect_variants.run", return_value=0) as collect:
            assert run_collect_variants(parse(
                "collect-variants", "--directed-fork-rules", "704.5f",
                "--exclude-cards", "h.txt",
            )) == 0
        assert collect.call_args.args[0].caps.directed_fork_rules == ("704.5f",)


class TestAHoldoutSourceIsRequired:
    """T190, FR-058g (python D3): without --exclude-cards, --split-from or
    --training-corpus a collector decks and varies held-out cards, so it exits
    1 before any worker starts."""

    @COLLECTORS
    def test_no_holdout_source_exits_1_before_any_worker(
        self, command, module, runner, caplog,
    ):
        from unittest.mock import patch

        with patch(module) as collect:
            assert runner(parse(command)) == 1
        collect.assert_not_called()
        assert f"{command}: " in caplog.text
        assert "--exclude-cards, --split-from or --training-corpus" in caplog.text

    @COLLECTORS
    @pytest.mark.parametrize("source", [
        ("--exclude-cards", "h.txt"), ("--split-from", "latest.pt"),
    ])
    def test_any_one_source_is_enough(self, command, module, runner, source):
        from unittest.mock import patch

        with patch(module, return_value=0) as collect:
            assert runner(parse(command, *source)) == 0
        collect.assert_called_once()

    @COLLECTORS
    def test_a_training_corpus_is_a_source(self, command, module, runner, tmp_path):
        from unittest.mock import patch

        (tmp_path / "holdout-cards.txt").write_text("fire\n", encoding="utf-8")
        with patch(module, return_value=0) as collect:
            assert runner(parse(command, "--training-corpus", str(tmp_path))) == 0
        collect.assert_called_once()

    def test_the_held_out_round_needs_none(self):
        """``--only-cards`` decks the held-out cards on purpose."""
        from unittest.mock import patch

        with patch("effects.application.collect_coverage.run", return_value=0) as collect:
            assert run_collect_coverage(parse("collect-coverage", "--only-cards", "h.txt")) == 0
        collect.assert_called_once()


class TestWorkersMustBeAbleToStart:
    """python R2: the JAR, ``java`` and the list files are checked before any
    worker starts, and a pool whose every slot fails to start ends the run."""

    @COLLECTORS
    def test_a_missing_prerequisite_exits_1_before_any_worker(
        self, command, module, runner, caplog, _workers_could_start,
    ):
        from unittest.mock import patch

        _workers_could_start.side_effect = FileNotFoundError("Connector JAR not found")
        with patch(module) as collect:
            assert runner(parse(command, "--exclude-cards", "h.txt")) == 1
        collect.assert_not_called()
        assert "Connector JAR not found" in caplog.text

    @COLLECTORS
    def test_the_list_files_are_checked(
        self, command, module, runner, _workers_could_start,
    ):
        from unittest.mock import patch

        with patch(module, return_value=0):
            runner(parse(command, "--exclude-cards", "h.txt"))
        files = _workers_could_start.call_args.kwargs["files"]
        assert files["--exclude-cards"] == Path("h.txt")

    @COLLECTORS
    def test_a_pool_that_cannot_start_exits_1(self, command, module, runner, caplog):
        from unittest.mock import patch

        from price_predictor.infrastructure.forge_jvm import WorkerStartError

        with patch(module, side_effect=WorkerStartError("every slot failed")):
            assert runner(parse(command, "--exclude-cards", "h.txt")) == 1
        assert "every slot failed" in caplog.text


class TestTheVariantRounds:
    """FR-058a, FR-058c, contracts/cli.md § collect-variants: the round flags
    and their defaults; --variant-volume and --corpus-records are gone."""

    def test_the_defaults_are_the_contract_s(self):
        args = parse("collect-variants")
        assert str(args.round) == "untargeted"
        assert (args.variant_games, args.no_progress_rounds) == (2, 3)
        assert args.rule_counts is None
        assert (args.steer_floor, args.steer_variants, args.steer_min_gain,
                args.steer_iterations) == (100, 2000, 1, 5)

    @pytest.mark.parametrize("flag", ["--variant-volume", "--corpus-records"])
    def test_the_volume_flags_are_removed(self, flag):
        with pytest.raises(SystemExit):
            parse("collect-variants", flag, "1")

    def test_an_unknown_round_is_refused(self):
        with pytest.raises(SystemExit):
            parse("collect-variants", "--round", "sideways")

    def test_the_flags_reach_the_config(self, tmp_path):
        from unittest.mock import patch

        counts = tmp_path / "counts.json"
        counts.write_text('{"generation": "gen-3", "rules": {}}', encoding="utf-8")
        with patch("effects.application.collect_variants.run", return_value=0) as collect:
            assert run_collect_variants(parse(
                "collect-variants", "--exclude-cards", "h.txt", "--round", "steered",
                "--rule-counts", str(counts), "--variant-games", "3",
                "--no-progress-rounds", "4", "--steer-floor", "50", "--steer-variants", "10",
                "--steer-min-gain", "2", "--steer-iterations", "6",
            )) == 0
        config = collect.call_args.args[0]
        assert str(config.round) == "steered" and config.rule_counts == counts
        assert (config.variant_games, config.no_progress_rounds, config.steer_floor,
                config.steer_variants, config.steer_min_gain, config.steer_iterations) == (
            3, 4, 50, 10, 2, 6)


class TestTheSteeredRoundNeedsGen3Counts:
    """FR-058g: --round steered exits 1 at startup without --rule-counts, or
    with a counts file holding gen-2 records."""

    def _refused(self, caplog, *argv) -> None:
        from unittest.mock import patch

        with patch("effects.application.collect_variants.run") as collect:
            assert run_collect_variants(parse(
                "collect-variants", "--exclude-cards", "h.txt", "--round", "steered", *argv,
            )) == 1
        collect.assert_not_called()
        assert "collect-variants: " in caplog.text

    def test_without_rule_counts(self, caplog):
        self._refused(caplog)
        assert "--rule-counts" in caplog.text

    @pytest.mark.parametrize("document, message", [
        ('{"generation": "gen-2", "rules": {}}', "gen-2"),
        ('{"rules": {}}', "names no generation"),
    ])
    def test_with_counts_of_another_generation(self, tmp_path, caplog, document, message):
        counts = tmp_path / "counts.json"
        counts.write_text(document, encoding="utf-8")
        self._refused(caplog, "--rule-counts", str(counts))
        assert message in caplog.text

    def test_with_a_counts_file_that_is_missing(self, tmp_path, caplog):
        self._refused(caplog, "--rule-counts", str(tmp_path / "none.json"))

    def test_the_untargeted_round_reads_no_counts(self):
        from unittest.mock import patch

        with patch("effects.application.collect_variants.run", return_value=0):
            assert run_collect_variants(parse("collect-variants", "--exclude-cards", "h.txt")) == 0
