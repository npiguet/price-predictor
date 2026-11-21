"""The ``validate-builder`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from draft.infrastructure.cli_common import _add_embeddings_path, _optional_path


def _build_validate_builder_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-builder",
        help=(
            "Picker-vs-SA builder-validation diagnostic (FR-042): build each pool "
            "with the picker and SA, score both, and print the gating "
            "picker-vs-SA Spearman, the SA-picker score-gap median/IQR, and the "
            "SA-vs-SA reference ceiling. Run once per picker/scorer pair to choose "
            "--build-method."
        ),
    )
    parser.set_defaults(func=run_validate_builder)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pools-from",
        help="drafts.jsonl whose per-seat drafted pools are the eval pools.",
    )
    source.add_argument(
        "--fresh-pools", action="store_true",
        help=(
            "Run a fresh draft session and use its drafted pools (the 45-card "
            "shape the picker labels), instead of an existing corpus. Needs Forge."
        ),
    )
    parser.add_argument(
        "--set", dest="set_code", default=None,
        help="Set code for --fresh-pools drafts (random sealed-legal set if omitted).",
    )
    parser.add_argument(
        "--n-pools", type=int, default=300,
        help="Number of pools to evaluate (default: 300).",
    )
    parser.add_argument(
        "--scorer-checkpoint", required=True,
        help="Frozen scorer.",
    )
    parser.add_argument(
        "--picker-checkpoint", required=True,
        help="Picker.",
    )
    parser.add_argument(
        "--cards-path", required=True,
        help=".npz cache.",
    )
    _add_embeddings_path(parser)


def run_validate_builder(args: argparse.Namespace) -> int:
    """Execute the validate-builder diagnostic (FR-042)."""
    from draft.application.validate_builder import (
        ValidateBuilderConfig,
        format_diagnostic,
        run_validate,
    )

    config = ValidateBuilderConfig(
        pools_from=Path(args.pools_from) if args.pools_from else None,
        fresh_pools=args.fresh_pools,
        set_code=args.set_code,
        n_pools=args.n_pools,
        scorer_checkpoint=Path(args.scorer_checkpoint),
        picker_checkpoint=Path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
        embeddings_path=_optional_path(args.embeddings_path),
    )
    try:
        diagnostic = run_validate(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    print(format_diagnostic(diagnostic))
    return 0
