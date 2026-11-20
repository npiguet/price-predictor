"""The ``match-outcomes`` command and its flag checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_predictor.domain.worker_status import ALLOW_DEGRADED_HELP
from price_predictor.infrastructure.cli_run_manifest import REFUSAL_DEST
from price_predictor.infrastructure.forge_worker_pool import worker_count as forge_worker_count
from price_predictor.infrastructure.worker_stops import add_worker_stop_flags
from sealed.domain.collection_caps import (
    DEFAULT_DIRECTED_FORK_PROBABILITY,
    DEFAULT_PROBE_KEYWORDS,
    DEFAULT_PROBES_PER_GAME,
    DEFAULT_RANDOM_SEAT_SHARE,
    CollectionCaps,
)
from sealed.infrastructure.collection_conversion import (
    add_cards_folder_flag,
    cards_folder_error,
    check_conversion,
)
from sealed.infrastructure.directed_fork_check import directed_fork_error
from sealed.infrastructure.directed_fork_probabilities_file import (
    add_directed_fork_probabilities_flag,
    read_directed_fork_needs,
)
from sealed.infrastructure.effect_collection_flags import (
    add_cap_flags,
    add_directed_fork_flags,
    announce_probe_state,
    candidate_rate_error,
    random_seat_error,
    worker_cap_properties,
)
from sealed.infrastructure.match_worker_connector import DEFAULT_SIDE_B_DECKS_WEIGHT, WorkerDecks


def _build_match_outcomes_parser(subparsers) -> None:
    match_parser = subparsers.add_parser(
        "match-outcomes",
        help="Generate sealed match outcome training data using Forge AI",
    )
    match_parser.set_defaults(func=run_match_outcomes)
    _add_pool_and_worker_flags(match_parser)
    _add_side_deck_flags(match_parser)
    _add_match_and_corpus_flags(match_parser)
    _add_effect_record_flags(match_parser)


def _add_pool_and_worker_flags(match_parser: argparse.ArgumentParser) -> None:
    """The cards no pool may contain and the worker count."""
    match_parser.add_argument(
        "--exclude-cards",
        default=None,
        help=(
            "File of card names, one per line, that no pool may contain. "
            "Written by 'python -m effects holdout-cards'. The worker opens "
            "its own pool per match, so a depleted collection run needs this "
            "here rather than on generate-pools; without it the run plays "
            "full-strength pools and every game it collects holds held-out "
            "cards."
        ),
    )
    match_parser.add_argument(
        "--workers",
        type=forge_worker_count,
        default=12,
        help="Number of parallel Java worker processes to spawn, 1 to 12 (default: 12)",
    )


def _add_side_deck_flags(match_parser: argparse.ArgumentParser) -> None:
    """The generated-decks files each side samples from."""
    match_parser.add_argument(
        "--side-a-decks",
        default=None,
        help=(
            "Optional path to a generated-decks file (built by build-decks)."
            " When given, deck A in every match is sampled from this file"
            " and the 4 Forge methods are NOT used to produce side A. When"
            " omitted, side A is built by the 4 Forge methods (weights"
            " 4:3:2:1) on a freshly-generated pool from a random sealed-legal"
            " set, like Phase 0."
        ),
    )
    match_parser.add_argument(
        "--side-b-decks",
        default=None,
        help=(
            "Optional path to a generated-decks file. When given, deck B is"
            " produced by the 4 Forge methods (weights 4:3:2:1) PLUS sampling"
            " from this file (weight --side-b-decks-weight). When omitted,"
            " deck B is produced by the 4 Forge methods only. The Forge"
            " methods always run, regardless of this flag — same-set pairing"
            " on deck A's set code is preserved."
        ),
    )
    match_parser.add_argument(
        "--side-b-decks-weight",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Weight of the file-sampled side-B method relative to the 4 Forge"
            f" methods (which carry weights 4:3:2:1, total 10). Default:"
            f" {DEFAULT_SIDE_B_DECKS_WEIGHT}. Set higher to oversample"
            " scorer-built decks against Forge methods (e.g. 8 for a gen3 run"
            " that needs strong gen1+gen2 representation on side B). Only"
            " meaningful when --side-b-decks is given; supplying it without"
            " --side-b-decks is an error."
        ),
    )


def _add_match_and_corpus_flags(match_parser: argparse.ArgumentParser) -> None:
    """The games per match and where the two corpora go."""
    match_parser.add_argument(
        "--best-of",
        type=int,
        default=7,
        help=(
            "Number of games per match (best-of-N). Must be a positive odd integer."
            " Default: 7."
        ),
    )
    match_parser.add_argument(
        "--outcomes-dir",
        default=None,
        help=(
            "Directory the two sealed corpora, match-outcomes.txt and"
            " cards-played.txt, are appended to. Default: the --effect-records"
            " directory when given, so a collection keeps them in its own run"
            " directory; one of the two flags is required."
        ),
    )


# ── effect-record instrumentation (spec 023) ────────────────────────────
#
# The opt-in lives here rather than in `effects` because these matches were
# going to be played anyway: the instrumentation rides them and costs no extra
# simulation. Absent the flag, nothing changes — no effect records are written
# and match-outcomes.txt and cards-played.txt keep their exact format and
# content.


#: The match-outcomes corpus; the worker writes cards-played.txt beside it.
MATCH_OUTCOMES_FILE = "match-outcomes.txt"


def outcomes_dir(args: argparse.Namespace) -> Path | None:
    """The directory ``match-outcomes.txt`` and ``cards-played.txt`` go to:
    ``--outcomes-dir``, else the ``--effect-records`` run directory, else None."""
    for value in (getattr(args, "outcomes_dir", None), getattr(args, "effect_records", None)):
        if value:
            return Path(value)
    return None


def outcomes_dir_error(args: argparse.Namespace) -> str | None:
    """Why ``match-outcomes`` has nowhere to append its corpora, or None."""
    if outcomes_dir(args) is None:
        return "--outcomes-dir or --effect-records is required: no default directory"
    return None


def _effect_collection_caps(args: argparse.Namespace) -> dict[str, str]:
    """The cap flags as the ``-D`` properties the worker reads them from, built
    from the one ``CollectionCaps`` the ``effects`` collectors build theirs
    from."""
    return worker_cap_properties(args)


def _add_effect_record_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--effect-records",
        default=None,
        help=(
            "Opt in to effect-record collection, writing JSONL shards to this"
            " directory. There is NO default: on this command collection is the"
            " opt-in, not the point. Absent, no effect records are written and"
            " this command behaves exactly as it does today."
        ),
    )
    add_cards_folder_flag(parser)
    add_cap_flags(parser)
    add_directed_fork_flags(parser)
    add_directed_fork_probabilities_flag(parser)
    # The stop protocol needs the worker logs, which only a run collecting
    # effect records keeps; without them the pool kills as it always did.
    add_worker_stop_flags(parser, only_with="--effect-records")
    parser.add_argument(
        "--allow-degraded", action="store_true",
        help=ALLOW_DEGRADED_HELP + "; only with --effect-records",
    )


def worker_stop_settings(args: argparse.Namespace):
    """The workers' stop settings, or None (exit 2 named by
    :func:`worker_stop_error`) when a stop flag is given without
    ``--effect-records``."""
    from price_predictor.infrastructure.worker_stops import (
        DEFAULT_RECYCLE_SECONDS,
        DEFAULT_STOP_TIMEOUT,
        WorkerStopSettings,
    )

    timeout = getattr(args, "stop_timeout", None)
    recycle = getattr(args, "recycle_seconds", None)
    return WorkerStopSettings(
        timeout=DEFAULT_STOP_TIMEOUT if timeout is None else timeout,
        recycle_seconds=DEFAULT_RECYCLE_SECONDS if recycle is None else recycle,
    )


def worker_stop_error(args: argparse.Namespace) -> str | None:
    """Why the stop flags cannot apply, or None: a worker prints its stop
    line to the log only a run collecting effect records keeps."""
    given = [
        flag for flag, dest in (("--stop-timeout", "stop_timeout"),
                                ("--recycle-seconds", "recycle_seconds"))
        if getattr(args, dest, None) is not None
    ]
    if given and not getattr(args, "effect_records", None):
        return (
            f"{' and '.join(given)} need --effect-records: without it the workers "
            "keep no log, and the pool recycles and ends them by the kill"
        )
    return None


def allow_degraded_error(args: argparse.Namespace) -> str | None:
    """Why ``--allow-degraded`` cannot apply, or None: only a run collecting
    effect records keeps the worker logs whose engine it judges."""
    if getattr(args, "allow_degraded", False) and not getattr(args, "effect_records", None):
        return (
            "--allow-degraded needs --effect-records: without it nothing judges "
            "the workers' engine"
        )
    return None


def check_random_seat_flags(args: argparse.Namespace) -> str | None:
    """Why the random-seat flags cannot run, or None when they can (FR-026).

    Beyond the collectors' shared rules: a random-seat match writes effect
    records only, so one with nowhere to write them plays for nothing.
    """
    error = random_seat_error(args)
    if error is not None:
        return error
    share = getattr(args, "random_seat_share", DEFAULT_RANDOM_SEAT_SHARE)
    if share > 0 and not getattr(args, "effect_records", None):
        return (
            "--random-seat-share above 0 needs --effect-records: a random-seat"
            " match writes effect records only"
        )
    return None


def _directed_startup_error(args: argparse.Namespace) -> str | None:
    """Why the directed-fork flags cannot run, or None (FR-014): the
    ``--directed-fork-probabilities`` file, read into ``args.directed_fork_needs``
    and its starting ``args.directed_rule_probabilities``, then the rules'
    startup check."""
    rules = getattr(args, "directed_fork_rules", CollectionCaps().directed_fork_rules)
    path = getattr(args, "directed_fork_probabilities", None)
    if path is not None:
        try:
            args.directed_fork_needs = read_directed_fork_needs(path, rules)
        except ValueError as exc:
            return str(exc)
        args.directed_rule_probabilities = args.directed_fork_needs.probabilities()
    return directed_fork_error(
        rules, getattr(args, "directed_fork_probability", DEFAULT_DIRECTED_FORK_PROBABILITY),
    )


def _effect_flags_usage_error(args: argparse.Namespace) -> str | None:
    """The first exit-2 refusal of the random-seat, worker-stop,
    ``--allow-degraded`` and sampling-rate flags, in that order, or None."""
    for check in (outcomes_dir_error, cards_folder_error, check_random_seat_flags,
                  worker_stop_error, allow_degraded_error, candidate_rate_error):
        error = check(args)
        if error is not None:
            return error
    return None


def run_match_outcomes(args: argparse.Namespace) -> int:
    """Execute the match-outcomes command."""
    if args.best_of < 1 or args.best_of % 2 == 0:
        print(
            f"Error: --best-of must be a positive odd integer, got: {args.best_of}",
            file=sys.stderr,
        )
        return 2

    # --side-b-decks-weight uses argparse.SUPPRESS so the attribute is absent
    # from `args` when the user doesn't pass the flag. This lets us reject
    # "weight without --side-b-decks" while still supplying the runtime
    # default (DEFAULT_SIDE_B_DECKS_WEIGHT) when only --side-b-decks is set.
    explicit_side_b_weight = getattr(args, "side_b_decks_weight", None)
    if args.side_b_decks is None and explicit_side_b_weight is not None:
        print(
            "Error: --side-b-decks-weight is only valid with --side-b-decks",
            file=sys.stderr,
        )
        return 2

    usage_error = _effect_flags_usage_error(args)
    if usage_error is not None:
        print(f"Error: {usage_error}", file=sys.stderr)
        return 2

    # FR-014: exit 1 naming the rule, before any game. One JVM query, and
    # only when a rule is listed.
    directed_error = _directed_startup_error(args)
    if directed_error is not None:
        print(f"Error: {directed_error}", file=sys.stderr)
        return 1

    return _supervise_match_outcomes(args, explicit_side_b_weight)


def _supervise_match_outcomes(
    args: argparse.Namespace, explicit_side_b_weight: int | None,
) -> int:
    """Run the match-outcomes supervisor over the checked flags: 2 when a
    worker cannot start, 1 when the converted tree's conversion or the
    workers' engine was refused (the reason reaches ``run.json``) or when
    matches recorded nothing, else 0."""
    from price_predictor.domain.converted_provenance import ConversionRefused
    from price_predictor.domain.worker_status import WorkerEngineRefused
    from price_predictor.infrastructure import forge_classpath, forge_worker_pool
    from sealed.application.match_outcomes import MatchOutcomeSupervisor, MatchRunOptions

    output_path = outcomes_dir(args) / MATCH_OUTCOMES_FILE
    side_a_decks = Path(args.side_a_decks) if args.side_a_decks else None
    side_b_decks = Path(args.side_b_decks) if args.side_b_decks else None
    side_b_decks_weight = (
        explicit_side_b_weight
        if explicit_side_b_weight is not None
        else DEFAULT_SIDE_B_DECKS_WEIGHT
    )
    # Absent, the supervisor and its workers behave exactly as they do today:
    # no shard is opened and match-outcomes.txt and cards-played.txt keep their
    # format and content untouched (FR-030).
    effect_records = getattr(args, "effect_records", None)
    effect_records_dir = Path(effect_records) if effect_records else None
    if effect_records_dir is not None:
        print(f"Collecting effect records into {effect_records_dir}")
        print(
            announce_probe_state(
                getattr(args, "probe_keywords", DEFAULT_PROBE_KEYWORDS),
                getattr(args, "probes_per_game", DEFAULT_PROBES_PER_GAME),
            )
        )

    exclude_cards = Path(args.exclude_cards) if args.exclude_cards else None
    try:
        # Before any worker: a missing JAR, java or list file would otherwise
        # surface only inside each worker's start.
        forge_classpath.check_worker_prerequisites(files={"--exclude-cards": exclude_cards})
        check_conversion(args, effect_records_dir)
        supervisor = MatchOutcomeSupervisor(
            worker_count=args.workers, output_path=output_path, best_of=args.best_of,
            options=MatchRunOptions(
                decks=WorkerDecks(
                    side_a=side_a_decks,
                    side_b=side_b_decks,
                    side_b_weight=side_b_decks_weight,
                    exclude_cards=exclude_cards,
                ),
                effect_records_dir=effect_records_dir,
                collection_caps=_effect_collection_caps(args),
                worker_stop=worker_stop_settings(args),
                allow_degraded=getattr(args, "allow_degraded", False),
            ),
        )
        supervisor.steer_directed_forks(getattr(args, "directed_fork_needs", None))
        supervisor.run()
    except (FileNotFoundError, forge_worker_pool.WorkerStartError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (ConversionRefused, WorkerEngineRefused) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        setattr(args, REFUSAL_DEST, str(exc))
        return 1

    # Matches played and no effect record written: the run's output is unusable.
    return 1 if supervisor.collected_nothing else 0
