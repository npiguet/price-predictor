"""The effect-record cap flags every collecting supervisor carries.

Shared by ``sealed match-outcomes`` and the ``effects`` collectors, so the two
cannot drift into collecting under different defaults. Loads no torch: the
``effects`` parser imports it. The defaults restate ``effects``'
``CollectionCaps`` because ``sealed`` never imports ``effects``;
``tests/unit/effects/domain/test_collection_caps.py`` pins the two together.
"""

from __future__ import annotations

import argparse

EFFECT_MANA_CAP = 1
EFFECT_PLAYABILITY_RATE = 0.1
EFFECT_INTERVENTIONS_PER_GAME = 2
EFFECT_PROBES_PER_GAME = 2
#: Every damage-step keyword gate 2 checks, in the spelling the JVM compares a
#: normalized Forge keyword against.
EFFECT_PROBE_KEYWORDS = (
    "first_strike,double_strike,deathtouch,lifelink,trample,indestructible,"
    "wither,infect"
)
EFFECT_RANDOM_SEAT_SHARE = 0.0
EFFECT_RANDOM_SEAT_PROBABILITY = 0.0


def add_cap_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mana-cap", type=int, default=EFFECT_MANA_CAP,
        help=(
            "Records per unique mana ability and mana produced, per game "
            "(default: %(default)s): repeat taps observe a board that barely moved"
        ),
    )
    parser.add_argument(
        "--playability-rate", type=float, default=EFFECT_PLAYABILITY_RATE,
        help=(
            "Fraction of decision-subkind logging points sampled (default: "
            "%(default)s); legality records are coalesced instead"
        ),
    )
    parser.add_argument(
        "--interventions-per-game", type=int,
        default=EFFECT_INTERVENTIONS_PER_GAME,
        help="Interventional resolutions per game (default: %(default)s)",
    )
    parser.add_argument(
        "--probes-per-game", type=int, default=EFFECT_PROBES_PER_GAME,
        help=(
            "Damage-step probe forks per game (default: %(default)s); a budget "
            "spent only on keywords --probe-keywords names"
        ),
    )
    parser.add_argument(
        "--probe-keywords", default=EFFECT_PROBE_KEYWORDS,
        help=(
            "Comma-separated damage-step keywords to probe (default: all "
            "eight); an empty string takes no probe fork at all"
        ),
    )
    parser.add_argument(
        "--random-seat-share", type=float, default=EFFECT_RANDOM_SEAT_SHARE,
        help=(
            "Share of matches where one seat plays at random among the legal "
            "options, casting cards the Forge AI never would (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--random-seat-probability", type=float, default=None,
        help=(
            "P, the chance the random seat draws at random at a decision "
            "point; required when --random-seat-share is above 0"
        ),
    )


def random_seat_error(args: argparse.Namespace) -> str | None:
    """Why the random-seat flags cannot run, or None when they can (FR-026).

    A share with no ``P`` is a seat whose behaviour nobody chose.
    """
    share = getattr(args, "random_seat_share", EFFECT_RANDOM_SEAT_SHARE)
    probability = getattr(args, "random_seat_probability", None)
    if not 0.0 <= share <= 1.0:
        return f"--random-seat-share must be between 0 and 1, got {share}"
    if probability is not None and not 0.0 <= probability <= 1.0:
        return (
            f"--random-seat-probability must be between 0 and 1, got {probability}"
        )
    if share > 0 and probability is None:
        return "--random-seat-share above 0 needs --random-seat-probability"
    return None


def announce_probe_state(probe_keywords: str, probes_per_game: int) -> str:
    """The startup line every collecting supervisor prints about probes.

    ``--probes-per-game`` reads like the switch and is not, so a run that
    empties ``--probe-keywords`` collects no probe evidence and nothing else
    says so until gate 2 has nothing to check.
    """
    named = [word.strip() for word in probe_keywords.split(",") if word.strip()]
    if not named:
        return (
            "Damage-step probes DISABLED: --probe-keywords is empty, so the "
            f"--probes-per-game {probes_per_game} budget buys no fork at all. "
            "Pass --probe-keywords <keyword>[,<keyword>...] to take any."
        )
    return (
        f"Damage-step probes: up to {probes_per_game} per game on "
        f"{', '.join(named)}"
    )
