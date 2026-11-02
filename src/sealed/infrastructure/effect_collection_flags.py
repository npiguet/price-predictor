"""The effect-record cap flags every collecting supervisor carries.

Shared by ``sealed match-outcomes`` and the ``effects`` collectors, so the two
cannot drift into collecting under different defaults. Loads no torch: the
``effects`` parser imports it. The defaults restate ``effects``'
``CollectionCaps`` because ``sealed`` never imports ``effects``;
``tests/unit/effects/domain/test_collection_caps.py`` pins the two together.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

EFFECT_MANA_CAP = 1
#: Stage 1 of decision sampling (FR-033b): the share of enumerated decision
#: candidates evaluated.
EFFECT_PLAYABILITY_RATE = 0.01
#: Stage 2's buckets, each the share of every enumerated candidate written,
#: both stages together: a candidate failing ``zone`` and otherwise only
#: ``cost``/``casting_speed``, and a plain one in its usual zone. A special
#: candidate is written whenever stage 1 keeps it.
EFFECT_ZONE_CANDIDATE_RATE = 0.0001
EFFECT_PLAIN_CANDIDATE_RATE = 0.001
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
EFFECT_COUNT_DROPPED_EVENTS = False
#: The property ``--count-dropped-events`` travels to the worker as.
COUNT_DROPPED_EVENTS_PROPERTY = "effect.count.dropped.events"

#: The file each worker lifetime appends its games' dropped-event counts to,
#: beside its shard: ``{run_id}.{worker}-{lifetime}`` then this. The JVM's
#: ``DroppedEvents.SUFFIX``; a test pins the two.
DROPPED_EVENTS_SUFFIX = ".dropped-events.tsv"


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
            "Stage 1 of decision sampling: the share of decision candidates "
            "evaluated (default: %(default)s); legality records are coalesced "
            "instead"
        ),
    )
    parser.add_argument(
        "--zone-candidate-rate", type=float, default=EFFECT_ZONE_CANDIDATE_RATE,
        help=(
            "Share of decision candidates failing only zone (and cost or "
            "timing) written, both sampling stages together (default: "
            "%(default)s); at most --playability-rate"
        ),
    )
    parser.add_argument(
        "--plain-candidate-rate", type=float, default=EFFECT_PLAIN_CANDIDATE_RATE,
        help=(
            "Share of plain decision candidates (usual zone, failing at most "
            "cost or timing) written, both sampling stages together (default: "
            "%(default)s); at most --playability-rate"
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
    parser.add_argument(
        "--count-dropped-events", action="store_true",
        default=EFFECT_COUNT_DROPPED_EVENTS,
        help=(
            "Count the events that reach no record, by event type and engine "
            "phase, and print the table summed over every worker when the "
            "run ends"
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


def candidate_rate_error(args: argparse.Namespace) -> str | None:
    """Why the decision-sampling rates cannot run, or None when they can
    (FR-033b).

    A bucket's rate is a share of every candidate enumerated, both stages
    together, so it cannot exceed the stage-1 share it is drawn from.
    """
    rates = {
        "--playability-rate": getattr(args, "playability_rate", EFFECT_PLAYABILITY_RATE),
        "--zone-candidate-rate": getattr(
            args, "zone_candidate_rate", EFFECT_ZONE_CANDIDATE_RATE
        ),
        "--plain-candidate-rate": getattr(
            args, "plain_candidate_rate", EFFECT_PLAIN_CANDIDATE_RATE
        ),
    }
    for flag, rate in rates.items():
        if not 0.0 <= rate <= 1.0:
            return f"{flag} must be between 0 and 1, got {rate}"
    stage_one = rates["--playability-rate"]
    for flag in ("--zone-candidate-rate", "--plain-candidate-rate"):
        if rates[flag] > stage_one:
            return (
                f"{flag} {rates[flag]} exceeds --playability-rate {stage_one}: "
                "a bucket cannot keep more candidates than stage 1 evaluates"
            )
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


def sum_dropped_events(records_dir: Path, run_id: str) -> tuple[Counter[tuple[str, str]], int]:
    """The run's dropped-event counts summed over its worker lifetimes, and
    how many lifetimes' files they came from.

    Each lifetime appends one ``event_type<TAB>phase<TAB>count`` line per pair
    and game. A worker is killed rather than stopped, so its last line may be
    cut short; a line that does not parse is skipped, as a shard's trailing
    partial line is.
    """
    counts: Counter[tuple[str, str]] = Counter()
    files = sorted(Path(records_dir).glob(f"{run_id}.*{DROPPED_EVENTS_SUFFIX}"))
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Printed from the supervisors' ``finally``: one unreadable file
            # must not hide the run's report and return code.
            continue
        for line in text.splitlines():
            fields = line.split("\t")
            if len(fields) != 3 or not fields[2].isdigit():
                continue
            counts[(fields[0], fields[1])] += int(fields[2])
    return counts, len(files)


def dropped_events_report(records_dir: Path, run_id: str) -> str:
    """The table a run under ``--count-dropped-events`` prints when it ends
    (FR-041): every (event type, engine phase) pair, most dropped first."""
    counts, lifetimes = sum_dropped_events(records_dir, run_id)
    if not counts:
        return (
            f"Dropped events: none counted over {lifetimes} worker lifetime"
            f"{'' if lifetimes == 1 else 's'}"
        )
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    type_width = max(len("event type"), *(len(t) for (t, _), _ in rows))
    phase_width = max(len("phase"), *(len(p) for (_, p), _ in rows))
    total = sum(counts.values())
    count_width = max(len("count"), len(str(total)))
    lines = [
        f"Dropped events (no record open), summed over {lifetimes} worker "
        f"lifetime{'' if lifetimes == 1 else 's'}:",
        f"  {'event type':<{type_width}}  {'phase':<{phase_width}}  {'count':>{count_width}}",
    ]
    lines.extend(
        f"  {event_type:<{type_width}}  {phase:<{phase_width}}  {count:>{count_width}}"
        for (event_type, phase), count in rows
    )
    lines.append(f"  {'total':<{type_width}}  {'':<{phase_width}}  {total:>{count_width}}")
    return "\n".join(lines)
