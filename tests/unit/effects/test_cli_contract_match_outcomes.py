"""The instrumentation opt-in on ``sealed match-outcomes`` (T044).

``--effect-records`` has **no default** here, unlike on the effects-owned
collectors where collection is the whole point. That asymmetry is the contract
(FR-031): on this command the flag is an opt-in, and a default would quietly
turn every sealed self-play run into a collection run.

US1 ships this surface, so its fast-suite coverage belongs with US1 rather than
with the collectors US2 adds.
"""

from __future__ import annotations

from sealed.infrastructure.cli import (
    EFFECT_INTERVENTIONS_PER_GAME,
    EFFECT_MANA_CAP,
    EFFECT_PLAYABILITY_RATE,
    EFFECT_PROBES_PER_GAME,
    announce_probe_state,
    build_parser,
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
    def test_the_five_defaults_are_the_contracts(self):
        args = parse("match-outcomes")
        assert args.mana_cap == 1
        assert args.playability_rate == 0.1
        assert args.interventions_per_game == 2
        assert args.probes_per_game == 2
        assert args.probe_keywords == ""

    def test_the_constants_and_the_defaults_agree(self):
        args = parse("match-outcomes")
        assert args.mana_cap == EFFECT_MANA_CAP
        assert args.playability_rate == EFFECT_PLAYABILITY_RATE
        assert args.interventions_per_game == EFFECT_INTERVENTIONS_PER_GAME
        assert args.probes_per_game == EFFECT_PROBES_PER_GAME

    def test_probes_are_off_unless_keywords_are_named(self):
        """No probe fork is taken at all with the flag unset (US3 acceptance 3)."""
        assert parse("match-outcomes").probe_keywords == ""

    def test_probe_keywords_is_comma_separated(self):
        args = parse("match-outcomes", "--probe-keywords", "wither,infect")
        assert args.probe_keywords == "wither,infect"

    def test_each_cap_is_overridable(self):
        args = parse(
            "match-outcomes",
            "--mana-cap", "50",
            "--playability-rate", "0.5",
            "--interventions-per-game", "4",
            "--probes-per-game", "1",
        )
        assert args.mana_cap == 50
        assert args.playability_rate == 0.5
        assert args.interventions_per_game == 4
        assert args.probes_per_game == 1


class TestProbesAnnounceThemselves:
    """The absence has to be loud, because the flag that reads like the switch
    is the budget.

    ``--probes-per-game`` defaults to 2 and buys nothing: the budget is spent
    only on keywords ``--probe-keywords`` names, and it names none. A run
    launched without it collects a corpus with zero probe forks in it, and the
    55,296-record smoke corpus was exactly that — nothing said so until gate 2
    had no engine-side branch to check against. So the default is unchanged
    (probes are the expensive mechanism) and the state is announced instead.
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

    def test_the_default_flag_value_produces_the_disabled_line(self):
        """Parsed defaults, not a hand-written empty string."""
        args = parse("match-outcomes", "--effect-records", "out/")
        assert "DISABLED" in announce_probe_state(
            args.probe_keywords, args.probes_per_game,
        )


class TestOtherSealedCommandsAreUnaffected:
    def test_build_decks_gains_no_effect_flags(self):
        args = parse("build-decks", "--pools", "p.txt", "--label", "gen-1")
        assert not hasattr(args, "effect_records")

    def test_train_scorer_gains_no_effect_flags(self):
        assert not hasattr(parse("train-scorer"), "effect_records")
