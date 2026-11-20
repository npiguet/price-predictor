"""The generated-decks commands: ``build-decks``, ``pick-decks`` and ``analyze-generated-decks``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sealed.domain.greedy_deck_builder import COLOR_PAIRS_STRATEGY
from sealed.infrastructure.cli_common import _add_cards_path, _add_embeddings_path, _embeddings_path


def _parse_restarts(value: str) -> int | str:
    """Parse --restarts: a positive integer, or the literal 'color-pairs'."""
    if value == COLOR_PAIRS_STRATEGY:
        return value
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--restarts must be a positive integer or "
            f"{COLOR_PAIRS_STRATEGY!r}, got: {value!r}"
        ) from None
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--restarts must be >= 1, got: {n}"
        )
    return n


def _parse_label(value: str) -> str:
    """Parse --label: a non-empty string with no ';', '|', or whitespace."""
    if not value:
        raise argparse.ArgumentTypeError("--label must be a non-empty string")
    forbidden = {";", "|"}
    if any(ch in forbidden for ch in value):
        raise argparse.ArgumentTypeError(
            f"--label must not contain ';' or '|' (file-format separators), "
            f"got: {value!r}"
        )
    if any(ch.isspace() for ch in value):
        raise argparse.ArgumentTypeError(
            f"--label must not contain whitespace, got: {value!r}"
        )
    return value


def _build_analyze_generated_decks_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "analyze-generated-decks",
        help=(
            "Aggregate composition stats (colors, curve, types, lands, pips, "
            "rarity) over one or more generated-decks files, with per-label "
            "breakdowns."
        ),
    )
    parser.set_defaults(func=run_analyze_generated_decks)
    parser.add_argument(
        "decks_files", nargs="+", type=Path,
        help="Generated-decks files.",
    )
    _add_cards_path(parser)
    parser.add_argument(
        "--no-rarity", action="store_true",
        help="Skip the rarity-distribution section (no MTGJSON load).",
    )


def _build_build_decks_parser(subparsers) -> None:
    build_parser = subparsers.add_parser(
        "build-decks",
        help="Build scorer-guided 40-card decks from a sealed pools file",
    )
    build_parser.set_defaults(func=run_build_decks)
    build_parser.add_argument(
        "--pools-path",
        required=True,
        help="Input pools file (with SET_CODE; prefixes)",
    )
    build_parser.add_argument(
        "--label",
        type=_parse_label,
        required=True,
        help=(
            "Generation-method tag written as the first column of every output "
            "line (e.g. 'gen-3'). Recorded by match-outcomes self-play as the "
            "method tag for any match where this deck is sampled. Must be a "
            "non-empty string without ';', '|', or whitespace."
        ),
    )
    build_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Scorer model checkpoint",
    )
    _add_cards_path(build_parser)
    _add_embeddings_path(build_parser)
    build_parser.add_argument(
        "--output",
        required=True,
        help="Output generated-decks file",
    )
    build_parser.add_argument(
        "--sa-temperature",
        type=float,
        default=0.8,
        help=(
            "Initial temperature for simulated annealing (default: 0.8). "
            "Set to 0 for pure greedy. Try 0.1-1.0 for SA; reasonable values "
            "depend on the magnitude of typical score differences for the model."
        ),
    )
    build_parser.add_argument(
        "--sa-cooling",
        type=float,
        default=0.85,
        help=(
            "Per-iteration temperature multiplier (default: 0.85). "
            "Ignored if --sa-temperature is 0. Lower cooling = faster decay = "
            "shorter exploration window. Below ~0.85 SA may flee good basins "
            "without finding new ones, which can produce worse decks than greedy."
        ),
    )
    build_parser.add_argument(
        "--sa-max-iterations",
        type=int,
        default=200,
        help=(
            "Hard cap on swap iterations (default: 200). "
            "Pure greedy stops earlier on convergence."
        ),
    )
    build_parser.add_argument(
        "--restarts",
        type=_parse_restarts,
        default=1,
        help=(
            "Restart strategy. Either a positive integer N (run the greedy/SA "
            "search N times per pool from independent random inits; default: 1) "
            "or the literal 'color-pairs' (run one search per MTG two-color "
            "pair, each seeded with an initial 23-spell deck filtered to "
            "on-color cards; the search is unconstrained after init). The "
            "highest-scoring deck across runs is kept."
        ),
    )
    build_parser.add_argument(
        "--print-decks",
        action="store_true",
        help=(
            "After building, also print each deck to stdout in the human-"
            "readable format used by evaluate-scorer (=== Deck N  score=... === "
            "header + one card per line with mana cost)."
        ),
    )
    build_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from a partial run. Counts complete lines already in "
            "--output, skips that many pools from the front of --pools-path, "
            "and appends remaining decks to the existing file. Without this "
            "flag the output file is truncated."
        ),
    )


def _build_pick_decks_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "pick-decks",
        help="Build 40-card decks from a pools file using a trained picker (one forward per pool)",
    )
    parser.set_defaults(func=run_pick_decks)
    parser.add_argument(
        "--pools-path", required=True,
        help="Input pools file (with SET_CODE; prefixes).",
    )
    parser.add_argument(
        "--picker-checkpoint", required=True,
        help="Picker weights.",
    )
    _add_cards_path(parser)
    _add_embeddings_path(parser)
    parser.add_argument(
        "--label", type=_parse_label, required=True,
        help=(
            "Generation-method tag written as the first column of every output "
            "line. Must be non-empty without ';', '|', or whitespace."
        ),
    )
    parser.add_argument(
        "--output", required=True,
        help="Output generated-decks file.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Append-and-skip resume: count complete lines already in --output, "
            "skip that many pools, append the rest. Without it --output is truncated."
        ),
    )


def run_build_decks(args: argparse.Namespace) -> int:
    """Execute the build-decks command."""
    from sealed.application.build_decks import BuildDecksConfig, BuildDecksUseCase

    config = BuildDecksConfig(
        pools_path=Path(args.pools_path),
        label=args.label,
        checkpoint=Path(args.checkpoint),
        cards_path=Path(args.cards_path),
        embeddings_path=_embeddings_path(args),
        output=Path(args.output),
        sa_temperature=args.sa_temperature,
        sa_cooling=args.sa_cooling,
        sa_max_iterations=args.sa_max_iterations,
        restarts=args.restarts,
        print_decks=args.print_decks,
        resume=args.resume,
    )

    try:
        use_case = BuildDecksUseCase()
        written = use_case.execute(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {written} decks to {config.output}")
    return 0


def run_pick_decks(args: argparse.Namespace) -> int:
    """Execute the pick-decks command."""
    from sealed.application.pick_decks import PickDecksConfig, PickDecksUseCase

    config = PickDecksConfig(
        pools_path=Path(args.pools_path),
        picker_checkpoint=Path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
        embeddings_path=_embeddings_path(args),
        label=args.label,
        output=Path(args.output),
        resume=args.resume,
    )

    try:
        written = PickDecksUseCase().execute(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {written} decks to {config.output}")
    return 0


def run_analyze_generated_decks(args: argparse.Namespace) -> int:
    """Print composition stats over generated-decks files."""
    from sealed.application.analyze_generated_decks import analyze_decks
    from sealed.infrastructure.pool_file_reader import parse_generated_decks

    deck_paths: list[Path] = list(args.decks_files)
    all_decks = []
    source_summary: list[tuple[str, int]] = []
    for path in deck_paths:
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 2
        decks = parse_generated_decks(path)
        all_decks.extend(decks)
        source_summary.append((str(path), len(decks)))

    if not all_decks:
        print("No decks loaded.", file=sys.stderr)
        return 2

    analyze_decks(
        all_decks, Path(args.cards_path),
        no_rarity=args.no_rarity, source_summary=source_summary,
    )
    return 0
