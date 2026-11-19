"""The instrumentation opt-in on ``sealed match-outcomes`` (T044).

``--effect-records`` has **no default** here, unlike on the effects-owned
collectors where collection is the whole point. That asymmetry is the contract
(FR-031): on this command the flag is an opt-in, and a default would quietly
turn every sealed self-play run into a collection run.
"""

from __future__ import annotations

import pytest

from sealed.infrastructure.cli import (
    _effect_collection_caps,
    build_parser,
    check_random_seat_flags,
    run_match_outcomes,
)
from sealed.infrastructure.effect_collection_flags import (
    EFFECT_PROBE_KEYWORDS,
    announce_probe_state,
    candidate_rate_error,
)


def parse(*argv: str):
    return build_parser().parse_args(argv)


class TestEffectRecordsOptIn:
    def test_it_has_no_default(self):
        """Collection is the opt-in here, not the point."""
        assert parse("match-outcomes").effect_records is None

    def test_it_takes_a_destination_directory(self):
        args = parse("match-outcomes", "--effect-records", "output/effects/records/")
        assert args.effect_records == "output/effects/records/"

    def test_the_commands_own_flags_are_untouched(self):
        args = parse("match-outcomes")
        assert args.workers == 12
        assert args.best_of == 7
        assert args.side_a_decks is None
        assert args.side_b_decks is None


class TestCapAndBudgetFlags:
    def test_the_defaults_are_the_contracts(self):
        args = parse("match-outcomes")
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

    def test_an_empty_keyword_list_turns_probes_off(self):
        """No probe fork is taken at all with the list emptied."""
        assert parse("match-outcomes", "--probe-keywords", "").probe_keywords == ""

    def test_each_cap_is_overridable(self):
        args = parse(
            "match-outcomes",
            "--mana-cap", "50",
            "--playability-rate", "0.5",
            "--zone-candidate-rate", "0.125",
            "--plain-candidate-rate", "0.25",
            "--interventions-per-game", "4",
            "--probes-per-game", "1",
        )
        assert args.mana_cap == 50
        assert args.playability_rate == 0.5
        assert args.zone_candidate_rate == 0.125
        assert args.plain_candidate_rate == 0.25
        assert args.interventions_per_game == 4
        assert args.probes_per_game == 1

class TestCountDroppedEvents:
    """FR-041: ``--count-dropped-events`` is off by default and travels to the
    worker as ``effect.count.dropped.events``."""

    def test_it_is_off_by_default(self):
        args = parse("match-outcomes")
        assert args.count_dropped_events is False
        assert _effect_collection_caps(args)["effect.count.dropped.events"] == "false"

    def test_the_flag_reaches_the_worker_property(self):
        args = parse("match-outcomes", "--count-dropped-events")
        assert args.count_dropped_events is True
        assert _effect_collection_caps(args)["effect.count.dropped.events"] == "true"


class TestProbesAnnounceThemselves:
    """The probe state is announced, because the flag that reads like the
    switch is the budget.

    ``--probes-per-game`` buys nothing on its own: the budget is spent only on
    keywords ``--probe-keywords`` names. A run that empties that list collects
    a corpus with zero probe forks in it, and the 55,296-record smoke corpus
    was exactly that — nothing said so until gate 2 had no engine-side branch
    to check against.
    """

    def test_the_disabled_line_names_the_flag_that_would_enable_probes(self):
        line = announce_probe_state("", 2)
        assert "DISABLED" in line
        assert "--probe-keywords" in line

    def test_the_disabled_line_says_the_budget_bought_nothing(self):
        """The budget's number is in the line, or it reads as a stray warning."""
        assert "--probes-per-game 2" in announce_probe_state("", 2)

    def test_whitespace_only_keywords_still_read_as_disabled(self):
        assert "DISABLED" in announce_probe_state("  , ,", 2)

    def test_an_enabled_run_names_the_keywords_and_the_budget(self):
        line = announce_probe_state("trample, deathtouch", 3)
        assert "DISABLED" not in line
        assert "trample, deathtouch" in line
        assert "3 per game" in line

    def test_the_default_flag_value_probes_all_eight_keywords(self):
        """Parsed defaults, not a hand-written keyword list."""
        args = parse("match-outcomes", "--effect-records", "out/")
        line = announce_probe_state(args.probe_keywords, args.probes_per_game)
        assert "DISABLED" not in line
        for keyword in EFFECT_PROBE_KEYWORDS.split(","):
            assert keyword in line


class TestRandomSeatFlags:
    """FR-021 and FR-026 (spec Story 3 scenario 11)."""

    def test_the_share_defaults_to_off_and_p_has_no_default(self):
        args = parse("match-outcomes")
        assert args.random_seat_share == 0.0
        assert args.random_seat_probability is None
        assert check_random_seat_flags(args) is None

    def test_a_complete_request_is_accepted(self):
        args = parse(
            "match-outcomes", "--effect-records", "out/",
            "--random-seat-share", "0.125", "--random-seat-probability", "0.25",
        )
        assert check_random_seat_flags(args) is None

    @pytest.mark.parametrize(
        ("argv", "complaint"),
        [
            (("--effect-records", "out/", "--random-seat-share", "0.125"),
             "needs --random-seat-probability"),
            (("--random-seat-share", "0.125", "--random-seat-probability", "0.5"),
             "needs --effect-records"),
            (("--effect-records", "out/", "--random-seat-share", "1.5",
              "--random-seat-probability", "0.5"), "between 0 and 1"),
            (("--effect-records", "out/", "--random-seat-share", "-0.1",
              "--random-seat-probability", "0.5"), "between 0 and 1"),
            (("--effect-records", "out/", "--random-seat-share", "0.5",
              "--random-seat-probability", "1.2"), "between 0 and 1"),
        ],
    )
    def test_an_incomplete_or_out_of_range_request_is_refused(self, argv, complaint):
        assert complaint in check_random_seat_flags(parse("match-outcomes", *argv))

    def test_the_refusal_happens_before_any_game(self, monkeypatch, capsys):
        """No supervisor is constructed, so no worker spawns."""
        import sealed.application.match_outcomes as module

        def no_supervisor(*args, **kwargs):
            raise AssertionError("a refused run must not start a supervisor")

        monkeypatch.setattr(module, "MatchOutcomeSupervisor", no_supervisor)
        args = parse("match-outcomes", "--random-seat-share", "0.125")
        assert run_match_outcomes(args) == 2
        assert "--random-seat-probability" in capsys.readouterr().err


class TestDecisionSamplingRates:
    """FR-033b: the rates travel as worker properties, and a bucket rate above
    the stage-1 rate is refused before any game."""

    def test_the_rates_reach_the_worker_properties(self):
        args = parse(
            "match-outcomes", "--playability-rate", "0.5",
            "--zone-candidate-rate", "0.125", "--plain-candidate-rate", "0.25",
        )
        properties = _effect_collection_caps(args)
        assert properties["effect.playability.rate"] == "0.5"
        assert properties["effect.zone.candidate.rate"] == "0.125"
        assert properties["effect.plain.candidate.rate"] == "0.25"

    @pytest.mark.parametrize(
        ("argv", "complaint"),
        [
            (("--plain-candidate-rate", "0.5"), "exceeds --playability-rate"),
            (("--playability-rate", "0.0005"), "exceeds --playability-rate"),
            (("--playability-rate", "2"), "between 0 and 1"),
            (("--plain-candidate-rate", "-1"), "between 0 and 1"),
        ],
    )
    def test_a_bad_rate_is_refused(self, argv, complaint):
        assert complaint in candidate_rate_error(parse("match-outcomes", *argv))

    def test_the_defaults_are_accepted(self):
        assert candidate_rate_error(parse("match-outcomes")) is None

    def test_the_refusal_happens_before_any_game(self, monkeypatch, capsys):
        import sealed.application.match_outcomes as module

        def no_supervisor(*args, **kwargs):
            raise AssertionError("a refused run must not start a supervisor")

        monkeypatch.setattr(module, "MatchOutcomeSupervisor", no_supervisor)
        args = parse("match-outcomes", "--zone-candidate-rate", "0.5")
        assert run_match_outcomes(args) == 2
        assert "--zone-candidate-rate" in capsys.readouterr().err


class TestDirectedForkFlags:
    """FR-012, FR-014: the three directed-fork flags, their properties, and
    the startup check that exits 1 naming the rule before any game."""

    def test_the_defaults_are_the_contracts(self):
        args = parse("match-outcomes")
        assert args.directed_fork_rules == ()
        assert args.directed_forks_per_game == 2
        assert args.directed_fork_probability == 0.25

    def test_the_rules_are_comma_separated_and_blanks_dropped(self):
        args = parse("match-outcomes", "--directed-fork-rules", "704.5f, ,702.19b")
        assert args.directed_fork_rules == ("704.5f", "702.19b")

    def test_the_flags_reach_the_worker_properties(self):
        properties = _effect_collection_caps(parse(
            "match-outcomes", "--directed-fork-rules", "704.5f",
            "--directed-forks-per-game", "1", "--directed-fork-probability", "0.5",
        ))
        assert properties["effect.directed.fork.rules"] == "704.5f"
        assert properties["effect.directed.forks.per.game"] == "1"
        assert properties["effect.directed.fork.probability"] == "0.5"

    def test_no_rule_sends_a_blank_list(self):
        assert _effect_collection_caps(parse("match-outcomes"))[
            "effect.directed.fork.rules"
        ] == ""

    @staticmethod
    def _refused(monkeypatch, capsys, *argv) -> str:
        import sealed.application.match_outcomes as module

        def no_supervisor(*args, **kwargs):
            raise AssertionError("a refused run must not start a supervisor")

        monkeypatch.setattr(module, "MatchOutcomeSupervisor", no_supervisor)
        assert run_match_outcomes(parse("match-outcomes", *argv)) == 1
        return capsys.readouterr().err

    def test_a_probability_outside_the_interval_exits_1(self, monkeypatch, capsys):
        err = self._refused(monkeypatch, capsys, "--directed-fork-probability", "0")
        assert "--directed-fork-probability must be in (0, 1]" in err

    def test_an_unknown_rule_exits_1_naming_it(self, monkeypatch, capsys):
        err = self._refused(monkeypatch, capsys, "--directed-fork-rules", "no-such-rule")
        assert "no-such-rule" in err

    def test_an_unregistered_rule_exits_1_naming_it(self, monkeypatch, capsys):
        """704.5f passes the catalogue's checks; the JVM, stubbed, lists none."""
        import sealed.infrastructure.effect_collection_flags as flags

        monkeypatch.setattr(flags, "registered_directed_rules", lambda: frozenset())
        err = self._refused(monkeypatch, capsys, "--directed-fork-rules", "704.5f")
        assert "704.5f has no board precondition registered" in err


class TestProbeKeywordSpelling:
    """python R6: the JVM compares each listed keyword exactly against Forge's
    keyword as ``KeywordEdit.normalize`` spells it (lower case, spaces as
    ``_``), so ``Trample`` or ``first strike`` was never probed, silently."""

    def test_a_spelling_the_jvm_would_miss_is_normalised(self):
        args = parse("match-outcomes", "--probe-keywords", "first strike, Trample")
        assert args.probe_keywords == "first_strike,trample"

    def test_a_keyword_gate_two_does_not_check_is_refused(self, capsys):
        with pytest.raises(SystemExit) as stopped:
            parse("match-outcomes", "--probe-keywords", "trample,flying")
        assert stopped.value.code == 2
        assert "flying" in capsys.readouterr().err

    def test_the_default_and_the_empty_list_pass_unchanged(self):
        assert parse("match-outcomes").probe_keywords == EFFECT_PROBE_KEYWORDS
        assert parse("match-outcomes", "--probe-keywords", "").probe_keywords == ""

    def test_the_effects_collectors_share_it(self):
        from effects.infrastructure.cli import build_parser as effects_parser

        args = effects_parser().parse_args(
            ["collect-coverage", "--probe-keywords", "Double Strike"],
        )
        assert args.probe_keywords == "double_strike"


class TestWorkersAndTheEnd:
    """V4 #9 on match-outcomes: ``--workers`` is 1 to 12; and a run collecting
    effect records that completed matches and recorded nothing exits 1."""

    @pytest.mark.parametrize("workers", ["0", "13", "-1"])
    def test_an_out_of_range_worker_count_is_a_usage_error(self, workers):
        with pytest.raises(SystemExit):
            parse("match-outcomes", "--workers", workers)

    def test_the_edges_are_accepted(self):
        assert parse("match-outcomes", "--workers", "1").workers == 1
        assert parse("match-outcomes", "--workers", "12").workers == 12

    @pytest.mark.parametrize(("collected_nothing", "code"), [(True, 1), (False, 0)])
    def test_a_run_that_recorded_nothing_exits_1(self, monkeypatch, collected_nothing, code):
        from unittest.mock import MagicMock

        import sealed.application.match_outcomes as module
        from price_predictor.infrastructure import forge_jvm

        supervisor = MagicMock(collected_nothing=collected_nothing)
        monkeypatch.setattr(module, "MatchOutcomeSupervisor", lambda **_: supervisor)
        monkeypatch.setattr(forge_jvm, "check_worker_prerequisites", lambda **_: None)
        args = parse("match-outcomes", "--effect-records", "records", "--workers", "1")

        assert run_match_outcomes(args) == code
        supervisor.run.assert_called_once()


class TestTheWorkerStopFlags:
    """V4 #1 / P1 #1: with effect records the run's end asks each worker to
    stop at its game's end; the flags bound the wait and set the recycle."""

    def test_unset_they_take_the_defaults(self, monkeypatch):
        from price_predictor.infrastructure.forge_jvm import WorkerStopSettings

        assert self._settings(monkeypatch, "--effect-records", "records") == (
            WorkerStopSettings(timeout=600, recycle_seconds=600))

    def test_the_values_reach_the_supervisor(self, monkeypatch):
        from price_predictor.infrastructure.forge_jvm import WorkerStopSettings

        assert self._settings(
            monkeypatch, "--effect-records", "records", "--stop-timeout", "45",
            "--recycle-seconds", "900",
        ) == WorkerStopSettings(timeout=45, recycle_seconds=900)

    @pytest.mark.parametrize(("argv", "bound"), [
        (("--stop-timeout", "0"), "between 1 and 3600 seconds, got 0"),
        (("--stop-timeout", "3601"), "between 1 and 3600 seconds, got 3601"),
        (("--recycle-seconds", "299"), "between 300 and 86400 seconds, got 299"),
        (("--recycle-seconds", "86401"), "between 300 and 86400 seconds, got 86401"),
    ])
    def test_a_value_out_of_bounds_is_a_usage_error_naming_the_bound(
        self, argv, bound, capsys,
    ):
        """The bound's own message: an unknown flag would exit 2 too."""
        with pytest.raises(SystemExit) as exit_info:
            parse("match-outcomes", "--effect-records", "records", *argv)
        assert exit_info.value.code == 2
        assert f"{argv[0]}: must be {bound}" in capsys.readouterr().err

    @pytest.mark.parametrize(("flag", "value"), [
        ("--stop-timeout", 1), ("--stop-timeout", 3600),
        ("--recycle-seconds", 300), ("--recycle-seconds", 86400),
    ])
    def test_the_bounds_are_inclusive(self, flag, value):
        args = parse("match-outcomes", "--effect-records", "records", flag, str(value))
        assert getattr(args, flag[2:].replace("-", "_")) == value

    @pytest.mark.parametrize("flag", ["--stop-timeout", "--recycle-seconds"])
    def test_without_effect_records_they_are_refused(self, monkeypatch, capsys, flag):
        import sealed.application.match_outcomes as module

        monkeypatch.setattr(module, "MatchOutcomeSupervisor", lambda **_: pytest.fail("ran"))
        assert run_match_outcomes(parse("match-outcomes", flag, "600")) == 2
        assert "need --effect-records" in capsys.readouterr().err

    @staticmethod
    def _settings(monkeypatch, *argv):
        from unittest.mock import MagicMock

        import sealed.application.match_outcomes as module
        from price_predictor.infrastructure import forge_jvm

        made = {}

        def supervisor(**kwargs):
            made.update(kwargs)
            return MagicMock(collected_nothing=False)

        monkeypatch.setattr(module, "MatchOutcomeSupervisor", supervisor)
        monkeypatch.setattr(forge_jvm, "check_worker_prerequisites", lambda **_: None)
        assert run_match_outcomes(parse("match-outcomes", "--workers", "1", *argv)) == 0
        return made["worker_stop"]


class TestWhereTheTwoSealedCorporaGo:
    """``--outcomes-dir`` names the directory ``match-outcomes.txt`` and
    ``cards-played.txt`` are appended to. Unset, a run collecting effect
    records keeps them in its own run directory, and a plain run appends to
    ``output/sealed/`` exactly as it always did."""

    def test_it_has_no_default(self):
        assert parse("match-outcomes").outcomes_dir is None

    @pytest.mark.parametrize(("argv", "expected"), [
        ((), "output/sealed/match-outcomes.txt"),
        (("--effect-records", "records/run"), "records/run/match-outcomes.txt"),
        (("--outcomes-dir", "sealed/run"), "sealed/run/match-outcomes.txt"),
        (("--effect-records", "records/run", "--outcomes-dir", "sealed/run"),
         "sealed/run/match-outcomes.txt"),
    ])
    def test_the_supervisor_appends_where_the_flags_say(self, monkeypatch, argv, expected):
        from pathlib import Path

        assert self._made(monkeypatch, *argv)["output_path"] == Path(expected)

    @pytest.mark.parametrize(("argv", "expected"), [
        ((), "output/sealed"),
        (("--outcomes-dir", "sealed/run"), "sealed/run"),
        (("--effect-records", "records/run"), "records/run"),
        (("--effect-records", "records/run", "--outcomes-dir", "sealed/run"), "records/run"),
    ])
    def test_the_run_directory_follows(self, argv, expected):
        from pathlib import Path

        from sealed.infrastructure import cli

        args = parse("match-outcomes", *argv)
        assert cli.RUN_MANIFESTS["match-outcomes"].run_dir(args) == Path(expected)

    @staticmethod
    def _made(monkeypatch, *argv):
        from unittest.mock import MagicMock

        import sealed.application.match_outcomes as module
        from price_predictor.infrastructure import forge_jvm

        made = {}

        def supervisor(**kwargs):
            made.update(kwargs)
            return MagicMock(collected_nothing=False)

        monkeypatch.setattr(module, "MatchOutcomeSupervisor", supervisor)
        monkeypatch.setattr(forge_jvm, "check_worker_prerequisites", lambda **_: None)
        assert run_match_outcomes(parse("match-outcomes", "--workers", "1", *argv)) == 0
        return made


class TestTheEngineRefusal:
    """Finding 11.1 (task K01c): a run collecting effect records whose workers'
    engine is not whole exits 1, its reason in ``run.json``'s outcome, unless
    ``--allow-degraded``; the flag needs ``--effect-records``."""

    def test_it_is_off_by_default_and_reaches_the_supervisor(self, monkeypatch):
        from unittest.mock import MagicMock

        import sealed.application.match_outcomes as module
        from price_predictor.infrastructure import forge_jvm

        made = []
        monkeypatch.setattr(module, "MatchOutcomeSupervisor", lambda **kwargs: (
            made.append(kwargs["allow_degraded"]) or MagicMock(collected_nothing=False)))
        monkeypatch.setattr(forge_jvm, "check_worker_prerequisites", lambda **_: None)
        for extra in ((), ("--allow-degraded",)):
            run_match_outcomes(parse("match-outcomes", "--effect-records", "records", *extra))

        assert made == [False, True]

    def test_without_effect_records_it_is_refused(self, monkeypatch, capsys):
        import sealed.application.match_outcomes as module

        monkeypatch.setattr(module, "MatchOutcomeSupervisor", lambda **_: pytest.fail("ran"))
        assert run_match_outcomes(parse("match-outcomes", "--allow-degraded")) == 2
        assert "--allow-degraded needs --effect-records" in capsys.readouterr().err

    def test_a_refused_engine_exits_1_and_its_reason_reaches_the_outcome(
        self, monkeypatch, capsys,
    ):
        from unittest.mock import MagicMock

        import sealed.application.match_outcomes as module
        from price_predictor.domain.worker_status import WorkerEngineRefused
        from price_predictor.infrastructure import forge_jvm
        from price_predictor.infrastructure.cli_run_manifest import REFUSAL_DEST

        supervisor = MagicMock()
        supervisor.run.side_effect = WorkerEngineRefused("the engine is not whole: mode degraded")
        monkeypatch.setattr(module, "MatchOutcomeSupervisor", lambda **_: supervisor)
        monkeypatch.setattr(forge_jvm, "check_worker_prerequisites", lambda **_: None)
        args = parse("match-outcomes", "--effect-records", "records")

        assert run_match_outcomes(args) == 1
        assert getattr(args, REFUSAL_DEST) == "the engine is not whole: mode degraded"
        assert "Error: the engine is not whole" in capsys.readouterr().err
