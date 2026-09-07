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
        assert args.mana_cap == 2000
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


class TestOtherSealedCommandsAreUnaffected:
    def test_build_decks_gains_no_effect_flags(self):
        args = parse("build-decks", "--pools", "p.txt", "--label", "gen-1")
        assert not hasattr(args, "effect_records")

    def test_train_scorer_gains_no_effect_flags(self):
        assert not hasattr(parse("train-scorer"), "effect_records")
