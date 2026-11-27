"""The effect-record cap flags every collecting supervisor carries.

Shared by ``sealed match-outcomes`` and the ``effects`` collectors, so the two
cannot drift into collecting under different defaults. Loads no torch: the
``effects`` parser imports it. The values, names and defaults of the caps are
defined once, in ``sealed.domain.collection_caps``; this module adds the flags
that set them, the properties they travel as and the startup checks over the
parsed values.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from price_predictor.domain.dataclass_defaults import field_default
from sealed.domain.collection_caps import (
    DEFAULT_DIRECTED_FORK_PROBABILITY,
    DEFAULT_DIRECTED_FORKS_PER_GAME,
    DEFAULT_INTERVENTIONS_PER_GAME,
    DEFAULT_MANA_CAP,
    DEFAULT_MAX_RECORDS_PER_GAME,
    DEFAULT_PLAIN_CANDIDATE_RATE,
    DEFAULT_PLAYABILITY_RATE,
    DEFAULT_PROBE_KEYWORDS,
    DEFAULT_PROBES_PER_GAME,
    DEFAULT_RANDOM_SEAT_PROBABILITY,
    DEFAULT_RANDOM_SEAT_SHARE,
    DEFAULT_ZONE_CANDIDATE_RATE,
    DIRECTED_RULE_PROBABILITY_PREFIX,
    CollectionCaps,
    property_name,
)


def positive_int(text: str) -> int:
    """An integer of at least 1: a cap, a round of no decks, a target of no
    records or no rounds before retiring is no run."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def non_negative_int(text: str) -> int:
    """An integer of at least 0, for a per-game budget: a negative one is no
    budget the worker can spend."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def probe_keyword_list(text: str) -> str:
    """``--probe-keywords``: each keyword spelled as the JVM compares it
    (``KeywordEdit.normalize``: lower case, spaces as ``_``), blanks dropped.

    The worker matches the list exactly against a Forge keyword so spelled, so
    ``Trample`` or ``first strike`` would never be probed and nothing would
    say so; a keyword gate 2 does not check is refused.
    """
    known = DEFAULT_PROBE_KEYWORDS.split(",")
    named = [
        word.strip().lower().replace(" ", "_") for word in text.split(",") if word.strip()
    ]
    unknown = [word for word in named if word not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"not a damage-step keyword: {', '.join(unknown)} (known: {', '.join(known)})"
        )
    return ",".join(named)


def rule_list(text: str) -> tuple[str, ...]:
    """``--directed-fork-rules``: comma-separated rule ids, blanks dropped, so
    an empty value names no rule."""
    return tuple(rule.strip() for rule in text.split(",") if rule.strip())


def _add_sampling_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mana-cap", type=non_negative_int, default=DEFAULT_MANA_CAP,
        help=(
            "Records per unique mana ability and mana produced, per game "
            "(default: %(default)s): repeat taps observe a board that barely moved"
        ),
    )
    parser.add_argument(
        "--playability-rate", type=float, default=DEFAULT_PLAYABILITY_RATE,
        help=(
            "Stage 1 of decision sampling: the share of decision candidates "
            "evaluated (default: %(default)s); legality records are coalesced "
            "instead"
        ),
    )
    parser.add_argument(
        "--zone-candidate-rate", type=float, default=DEFAULT_ZONE_CANDIDATE_RATE,
        help=(
            "Share of decision candidates failing only zone (and cost or "
            "timing) written, both sampling stages together (default: "
            "%(default)s); at most --playability-rate"
        ),
    )
    parser.add_argument(
        "--plain-candidate-rate", type=float, default=DEFAULT_PLAIN_CANDIDATE_RATE,
        help=(
            "Share of plain decision candidates (usual zone, failing at most "
            "cost or timing) written, both sampling stages together (default: "
            "%(default)s); at most --playability-rate"
        ),
    )


def _add_fork_budget_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--interventions-per-game", type=non_negative_int,
        default=DEFAULT_INTERVENTIONS_PER_GAME,
        help="Interventional resolutions per game (default: %(default)s)",
    )
    parser.add_argument(
        "--probes-per-game", type=non_negative_int, default=DEFAULT_PROBES_PER_GAME,
        help=(
            "Damage-step probe forks per game (default: %(default)s); a budget "
            "spent only on keywords --probe-keywords names"
        ),
    )
    parser.add_argument(
        "--probe-keywords", type=probe_keyword_list, default=DEFAULT_PROBE_KEYWORDS,
        help=(
            "Comma-separated damage-step keywords to probe (default: all "
            "eight); an empty string takes no probe fork at all"
        ),
    )


def _add_random_seat_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--random-seat-share", type=float, default=DEFAULT_RANDOM_SEAT_SHARE,
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


def _add_record_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-records-per-game", type=positive_int, default=DEFAULT_MAX_RECORDS_PER_GAME,
        help=(
            "Records one game may write, its forks' included (default: "
            "%(default)s); past it the worker withholds the rest, counts them "
            "and prints one line naming the game, the turn and the acting card "
            "most of them shared"
        ),
    )
    parser.add_argument(
        "--count-dropped-events", action="store_true",
        default=CollectionCaps().count_dropped_events,
        help=(
            "Count the events that reach no record, by event type and engine "
            "phase, and print the table summed over every worker when the "
            "run ends"
        ),
    )


def add_cap_flags(parser: argparse.ArgumentParser) -> None:
    _add_sampling_flags(parser)
    _add_fork_budget_flags(parser)
    _add_random_seat_flags(parser)
    _add_record_flags(parser)


def add_directed_fork_flags(parser: argparse.ArgumentParser) -> None:
    """The directed-fork flags of ``match-outcomes`` and ``collect-variants``
    (FR-012). ``collect-coverage`` takes ``--directed-fork-rules`` as a label
    only and none of these."""
    parser.add_argument(
        "--directed-fork-rules", type=rule_list, default=CollectionCaps().directed_fork_rules,
        help=(
            "Comma-separated rule ids to take directed forks for, each with a "
            "board precondition in forge-connector; the order breaks a tie "
            "between rules matching one boundary (default: none)"
        ),
    )
    parser.add_argument(
        "--directed-forks-per-game", type=non_negative_int,
        default=DEFAULT_DIRECTED_FORKS_PER_GAME,
        help=(
            "Directed forks per game, kept or discarded, a budget beside the "
            "other forks' (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--directed-fork-probability", type=float,
        default=DEFAULT_DIRECTED_FORK_PROBABILITY,
        help=(
            "Chance a phase boundary where an enabled rule's precondition holds "
            "forks, in (0, 1] (default: %(default)s)"
        ),
    )


def caps_from_args(args: argparse.Namespace) -> CollectionCaps:
    """The caps a command's parsed flags name; a flag the command lacks takes
    the cap's default, which for the directed forks names no rule.

    ``aimed_at`` is the effects collectors' alone and stays unset here. An unset
    ``--random-seat-probability`` is None until here, so a share above 0
    without one can be refused first; with the share at 0 the worker never
    reads it.
    """
    values = {
        spec.name: getattr(args, spec.name, field_default(spec))
        for spec in fields(CollectionCaps)
        if spec.name != "aimed_at"
    }
    if values["random_seat_probability"] is None:
        values["random_seat_probability"] = DEFAULT_RANDOM_SEAT_PROBABILITY
    values["directed_fork_rules"] = tuple(values["directed_fork_rules"])
    return CollectionCaps(**values)


def worker_cap_properties(args: argparse.Namespace) -> dict[str, str]:
    """The cap flags as the ``-D`` properties a ``match-outcomes`` worker reads,
    the directed rules' own probabilities ahead of the directed-fork rules."""
    properties = caps_from_args(args).as_system_properties()
    probabilities = {
        name: properties.pop(name)
        for name in list(properties)
        if name.startswith(DIRECTED_RULE_PROBABILITY_PREFIX)
    }
    ahead = property_name("directed_fork_rules")
    ordered: dict[str, str] = {}
    for name, value in properties.items():
        if name == ahead:
            ordered.update(probabilities)
        ordered[name] = value
    return ordered


def random_seat_error(args: argparse.Namespace) -> str | None:
    """Why the random-seat flags cannot run, or None when they can (FR-026).

    A share with no ``P`` is a seat whose behaviour nobody chose.
    """
    share = getattr(args, "random_seat_share", DEFAULT_RANDOM_SEAT_SHARE)
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


def _rate_range_error(rates: dict[str, float]) -> str | None:
    for flag, rate in rates.items():
        if not 0.0 <= rate <= 1.0:
            return f"{flag} must be between 0 and 1, got {rate}"
    return None


def _bucket_error(rates: dict[str, float]) -> str | None:
    stage_one = rates["--playability-rate"]
    for flag in ("--zone-candidate-rate", "--plain-candidate-rate"):
        if rates[flag] > stage_one:
            return (
                f"{flag} {rates[flag]} exceeds --playability-rate {stage_one}: "
                "a bucket cannot keep more candidates than stage 1 evaluates"
            )
    return None


def candidate_rate_error(args: argparse.Namespace) -> str | None:
    """Why the decision-sampling rates cannot run, or None when they can
    (FR-033b).

    A bucket's rate is a share of every candidate enumerated, both stages
    together, so it cannot exceed the stage-1 share it is drawn from.
    """
    rates = {
        "--playability-rate": getattr(args, "playability_rate", DEFAULT_PLAYABILITY_RATE),
        "--zone-candidate-rate": getattr(
            args, "zone_candidate_rate", DEFAULT_ZONE_CANDIDATE_RATE
        ),
        "--plain-candidate-rate": getattr(
            args, "plain_candidate_rate", DEFAULT_PLAIN_CANDIDATE_RATE
        ),
    }
    return _rate_range_error(rates) or _bucket_error(rates)


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
