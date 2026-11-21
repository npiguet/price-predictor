"""The ``analyze-generated-decks`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_analyze_generated_decks_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "analyze-generated-decks",
        help=(
            "Aggregate composition stats (colors, curve, types, lands, pips, "
            "rarity) over the per-seat decks in a drafts.jsonl corpus, plus a "
            "per-label deck-score summary (agent vs Forge). Per-label breakdowns "
            "follow the global report."
        ),
    )
    parser.set_defaults(func=run_analyze_generated_decks)
    parser.add_argument(
        "--agent", required=True,
        help=(
            "Restrict the analysis to seats piloted by this agent/mix label "
            "(e.g. draft-agent, forge-full). Run once per agent to compare."
        ),
    )
    parser.add_argument(
        "--drafts-path", required=True,
        help="Drafts corpus to analyze.",
    )
    parser.add_argument(
        "--cards-path", required=True,
        help="Directory with converted card files.",
    )
    parser.add_argument(
        "--no-rarity", action="store_true",
        help="Skip the rarity-distribution section (no MTGJSON load).",
    )


def run_analyze_generated_decks(args: argparse.Namespace) -> int:
    """Composition + per-label deck-score stats over a drafts.jsonl corpus."""
    from draft.application.analyze_generated_decks import (
        available_agents,
        deck_score_summary,
        format_deck_score_section,
        generated_decks_from_drafts,
    )
    from draft.infrastructure.draft_record_io import read_records
    from sealed.application.analyze_generated_decks import analyze_decks

    drafts_path = Path(args.drafts_path)
    records = list(read_records(drafts_path))
    if not records:
        print(f"Error: no records in {drafts_path}", file=sys.stderr)
        return 2

    decks = generated_decks_from_drafts(records, agent=args.agent)
    if not decks:
        available = available_agents(records)
        print(
            f"Error: no built decks for agent {args.agent!r} in {drafts_path}. "
            f"Available agents: {', '.join(available) or '(none)'}",
            file=sys.stderr,
        )
        return 2

    print(format_deck_score_section(deck_score_summary(records, agent=args.agent)))
    print()

    analyze_decks(
        decks, Path(args.cards_path),
        no_rarity=args.no_rarity,
        source_summary=[(f"{drafts_path} [agent={args.agent}]", len(decks))],
    )
    return 0
