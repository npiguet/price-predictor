"""The effect-record cap flags every collecting supervisor carries.

Shared by ``sealed match-outcomes`` and the ``effects`` collectors, so the two
cannot drift into collecting under different defaults. Loads no torch: the
``effects`` parser imports it. The defaults restate ``effects``'
``CollectionCaps`` because ``sealed`` never imports ``effects``;
``tests/unit/effects/domain/test_collection_caps.py`` pins the two together.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Sequence
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
#: Directed forks (FR-012): no rule unless a run names one, at most two a
#: game beside the other fork budgets, and a quarter of the boundaries where
#: an enabled rule's precondition holds.
EFFECT_DIRECTED_FORK_RULES: tuple[str, ...] = ()
EFFECT_DIRECTED_FORKS_PER_GAME = 2
EFFECT_DIRECTED_FORK_PROBABILITY = 0.25
DIRECTED_FORK_RULES_PROPERTY = "effect.directed.fork.rules"
DIRECTED_FORKS_PER_GAME_PROPERTY = "effect.directed.forks.per.game"
DIRECTED_FORK_PROBABILITY_PROPERTY = "effect.directed.fork.probability"
#: The rules catalogue the startup check reads (``rules/README.md``).
DEFAULT_RULES_DIR = Path("rules")
#: Prints the rule ids the worker holds a board precondition for.
DIRECTED_RULES_MAIN = "com.pricepredictor.connector.DirectedRulesMain"

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
        "--probe-keywords", type=probe_keyword_list, default=EFFECT_PROBE_KEYWORDS,
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


def probe_keyword_list(text: str) -> str:
    """``--probe-keywords``: each keyword spelled as the JVM compares it
    (``KeywordEdit.normalize``: lower case, spaces as ``_``), blanks dropped.

    The worker matches the list exactly against a Forge keyword so spelled, so
    ``Trample`` or ``first strike`` would never be probed and nothing would
    say so; a keyword gate 2 does not check is refused.
    """
    known = EFFECT_PROBE_KEYWORDS.split(",")
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


def add_directed_fork_flags(parser: argparse.ArgumentParser) -> None:
    """The directed-fork flags of ``match-outcomes`` and ``collect-variants``
    (FR-012). ``collect-coverage`` takes ``--directed-fork-rules`` as a label
    only and none of these."""
    parser.add_argument(
        "--directed-fork-rules", type=rule_list, default=EFFECT_DIRECTED_FORK_RULES,
        help=(
            "Comma-separated rule ids to take directed forks for, each with a "
            "board precondition in forge-connector; the order breaks a tie "
            "between rules matching one boundary (default: none)"
        ),
    )
    parser.add_argument(
        "--directed-forks-per-game", type=int, default=EFFECT_DIRECTED_FORKS_PER_GAME,
        help=(
            "Directed forks per game, kept or discarded, a budget beside the "
            "other forks' (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--directed-fork-probability", type=float,
        default=EFFECT_DIRECTED_FORK_PROBABILITY,
        help=(
            "Chance a phase boundary where an enabled rule's precondition holds "
            "forks, in (0, 1] (default: %(default)s)"
        ),
    )


def directed_fork_properties(args: argparse.Namespace) -> dict[str, str]:
    """The directed-fork flags as the ``-D`` properties the worker reads; a
    command without them sends the defaults, which name no rule."""
    rules = getattr(args, "directed_fork_rules", EFFECT_DIRECTED_FORK_RULES)
    return {
        DIRECTED_FORK_RULES_PROPERTY: ",".join(rules),
        DIRECTED_FORKS_PER_GAME_PROPERTY: str(
            getattr(args, "directed_forks_per_game", EFFECT_DIRECTED_FORKS_PER_GAME)
        ),
        DIRECTED_FORK_PROBABILITY_PROPERTY: str(
            getattr(args, "directed_fork_probability", EFFECT_DIRECTED_FORK_PROBABILITY)
        ),
    }


_DETECTOR_RANK = {"missing": 0, "none": 1, "declared": 2}


def rule_readiness(rules_dir: Path = DEFAULT_RULES_DIR) -> dict[str, tuple[str, bool]]:
    """Per rule id of the catalogue: its detector state, and whether a
    contrasting variant of one of its scenarios is classified into a fork
    operation.

    The detector state is ``"declared"`` when some scenario listing the rule
    declares a detector family, else ``"none"`` when one declares
    ``{"none": reason}``, else ``"missing"``. The base variant, the first,
    makes nothing, and an unclassifiable variant's operation is an object.

    Read here rather than through ``effects.infrastructure.rules_catalogue_io``,
    which this package may not import; ``test_directed_fork_flags.py`` pins the
    two readings together over the real catalogue.
    """
    detector: dict[str, str] = {}
    forkable: dict[str, bool] = {}
    for path in sorted(Path(rules_dir, "scenarios").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for scenario in doc.get("scenarios") or ():
            declaration = scenario.get("detector")
            if declaration is None:
                state = "missing"
            elif "none" in declaration:
                state = "none"
            else:
                state = "declared"
            classified = any(
                isinstance(variant.get("operation"), str)
                for variant in (scenario.get("variants") or ())[1:]
            )
            for rule in scenario.get("rules") or ():
                if _DETECTOR_RANK[state] >= _DETECTOR_RANK[detector.get(rule, "missing")]:
                    detector[rule] = state
                forkable[rule] = forkable.get(rule, False) or classified
    return {rule: (detector[rule], forkable[rule]) for rule in detector}


def registered_directed_rules() -> frozenset[str]:
    """The rule ids one run of ``DirectedRulesMain`` prints: those the worker
    holds a board precondition for (FR-014).

    Raises ``RuntimeError`` when the query fails, and ``FileNotFoundError``
    when the connector JAR or java is missing.
    """
    from price_predictor.infrastructure.forge_jvm import run_forge_worker

    result = run_forge_worker(DIRECTED_RULES_MAIN, input_text="")
    if result.returncode != 0:
        raise RuntimeError(
            f"DirectedRulesMain exited {result.returncode}: {(result.stderr or '').strip()}"
        )
    return frozenset(line.strip() for line in result.stdout.splitlines() if line.strip())


def directed_fork_error(
    rules: Sequence[str],
    probability: float,
    *,
    rules_dir: Path = DEFAULT_RULES_DIR,
    registered: Callable[[], frozenset[str]] | None = None,
) -> str | None:
    """Why the directed-fork flags cannot run, naming the rule, or None when
    they can (FR-014): the startup check of ``match-outcomes`` and
    ``collect-variants``, never of ``collect-coverage``, whose
    ``--directed-fork-rules`` is a label only.

    A listed rule needs a detector declared in its scenarios (not ``none``),
    a contrasting variant classified into a fork operation, and a board
    precondition registered in ``forge-connector``, which one run of
    ``DirectedRulesMain`` lists. The JVM is asked once the catalogue's checks
    pass, and not at all when no rule is listed; ``registered`` replaces that
    query, for a test.
    """
    if not 0.0 < probability <= 1.0:
        return f"--directed-fork-probability must be in (0, 1], got {probability}"
    if not rules:
        return None
    readiness = rule_readiness(rules_dir)
    for rule in rules:
        detector, forkable = readiness.get(rule, ("missing", False))
        if detector == "missing":
            return (
                f"--directed-fork-rules: {rule} has no detector declaration "
                f"in {Path(rules_dir, 'scenarios')}"
            )
        if detector == "none":
            return (
                f"--directed-fork-rules: {rule} declares its detector none, so "
                "no record of its fork could be tagged"
            )
        if not forkable:
            return (
                f"--directed-fork-rules: {rule} has no contrasting variant "
                "classified into a fork operation"
            )
    try:
        known = (registered or registered_directed_rules)()
    except (OSError, RuntimeError) as exc:
        return f"--directed-fork-rules: cannot list the registered rules: {exc}"
    for rule in rules:
        if rule not in known:
            return (
                f"--directed-fork-rules: {rule} has no board precondition "
                "registered in forge-connector (DirectedRulesMain does not list it)"
            )
    return None


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
