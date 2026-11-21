"""The ``play-draft-games`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_predictor.infrastructure.forge_worker_pool import worker_count


def _build_play_draft_games_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "play-draft-games",
        help=(
            "Play Forge matches between decks drafted in the same pod of a "
            "drafts.jsonl corpus, appending rows in the sealed match-outcome "
            "format. Tally them with scripts/analyze_winrates.py."
        ),
    )
    parser.set_defaults(func=run_play_draft_games)
    parser.add_argument(
        "--drafts-path", required=True,
        help="Corpus to sample from.",
    )
    parser.add_argument(
        "--run-id", action="append", dest="run_ids", default=None,
        help=(
            "Restrict sampling to records with this run_id; repeatable. "
            "Absent, the whole corpus is in scope."
        ),
    )
    parser.add_argument(
        "--output-path", required=True,
        help="Append target (a run directory's draft-games.txt).",
    )
    parser.add_argument(
        "--n-pairings", type=int, default=None,
        help=(
            "Stop once this many matches have been recorded by this run. "
            "Absent, play until interrupted."
        ),
    )
    parser.add_argument(
        "--best-of", type=int, default=3,
        help="Games per match; must be a positive odd integer (default: 3)",
    )
    parser.add_argument(
        "--include-mirrors", action="store_true",
        help="Keep pairings whose two seats carry the same agent label",
    )
    parser.add_argument(
        "--forge-native-fraction", type=float, default=0.0,
        help=(
            "Share of forge-full seats whose deck Forge rebuilds from their "
            "drafted pool; those seats are labelled forge-native (default: 0)"
        ),
    )
    parser.add_argument(
        "--workers", type=worker_count, default=12,
        help="Concurrent Forge worker processes, 1 to 12 (default: 12)",
    )


def run_play_draft_games(args: argparse.Namespace) -> int:
    """Play same-pod matches and append them in the sealed match-outcome format."""
    from draft.application.play_draft_games import (
        PlayDraftGamesConfig,
        PlayDraftGamesUseCase,
    )
    from draft.domain.seat_table import project
    from draft.infrastructure.draft_record_io import read_records

    drafts_path = Path(args.drafts_path)
    run_ids = tuple(args.run_ids or ())

    # ── Startup validation: everything checkable before a game is played ──
    if args.best_of < 1 or args.best_of % 2 == 0:
        print(
            f"Error: --best-of must be a positive odd integer, got {args.best_of}",
            file=sys.stderr,
        )
        return 2
    if not 0.0 <= args.forge_native_fraction <= 1.0:
        print(
            "Error: --forge-native-fraction must be between 0 and 1, got "
            f"{args.forge_native_fraction}",
            file=sys.stderr,
        )
        return 2
    if args.n_pairings is not None and args.n_pairings < 1:
        print(
            f"Error: --n-pairings must be a positive integer, got {args.n_pairings}",
            file=sys.stderr,
        )
        return 2
    if args.workers < 1:
        print(
            f"Error: --workers must be a positive integer, got {args.workers}",
            file=sys.stderr,
        )
        return 2
    if not drafts_path.exists():
        print(f"Error: corpus not found: {drafts_path}", file=sys.stderr)
        return 2

    records = [r for r in read_records(drafts_path) if not run_ids or r.run_id in run_ids]
    if not records:
        scope = f" matching --run-id {', '.join(run_ids)}" if run_ids else ""
        print(f"Error: no records in {drafts_path}{scope}", file=sys.stderr)
        return 2

    rows = list(project(records))
    labels = sorted({row.label for row in rows})
    if not _has_playable_pair(rows, include_mirrors=args.include_mirrors):
        print(
            "Error: no pair survives the --run-id and mirror filters. "
            f"Labels present: {', '.join(labels) or '(none)'}. "
            "Pass --include-mirrors to play same-label pairings.",
            file=sys.stderr,
        )
        return 2

    # ── Startup echo ──
    scope = ", ".join(run_ids) if run_ids else "whole corpus"
    print(f"Corpus:      {drafts_path} ({len(records)} records, {len(rows)} seats)")
    print(f"Run ids:     {scope}")
    print(f"Labels:      {', '.join(labels)}")
    print(
        "Pairings:    "
        + (f"{args.n_pairings}" if args.n_pairings is not None else "unbounded (Ctrl-C to stop)")
    )
    print(
        f"Matches:     best-of-{args.best_of}, {args.workers} workers, "
        f"mirrors {'included' if args.include_mirrors else 'excluded'}"
    )
    print(f"Output:      {args.output_path}")

    config = PlayDraftGamesConfig(
        drafts_path=drafts_path,
        output_path=Path(args.output_path),
        run_ids=run_ids,
        n_pairings=args.n_pairings,
        best_of=args.best_of,
        include_mirrors=args.include_mirrors,
        forge_native_fraction=args.forge_native_fraction,
        workers=args.workers,
    )

    from price_predictor.infrastructure.forge_worker_pool import WorkerStartError

    try:
        summary = PlayDraftGamesUseCase().execute(config)
    except WorkerStartError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    # 130 only for an interrupt that arrived before the first match completed.
    # A clean bounded finish, and an interrupt after any match, are both 0.
    if summary.interrupted and summary.matches_played == 0:
        return 130
    return 0


def _has_playable_pair(rows, *, include_mirrors: bool) -> bool:
    """True when some pod holds two seats that could legally be paired."""
    by_pod: dict[str, list[str]] = {}
    for row in rows:
        by_pod.setdefault(row.draft_id, []).append(row.label)
    for labels in by_pod.values():
        if len(labels) < 2:
            continue
        if include_mirrors or len(set(labels)) > 1:
            return True
    return False
