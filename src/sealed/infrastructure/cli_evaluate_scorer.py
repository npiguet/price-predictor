"""The ``evaluate-scorer`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_predictor.infrastructure.cli_helpers import add_dataclass_arg
from sealed.application.evaluate_scorer import EvaluateScorerConfig
from sealed.infrastructure.cli_common import _add_cards_path, _add_embeddings_path, _embeddings_path


def _build_evaluate_scorer_parser(subparsers) -> None:
    eval_parser = subparsers.add_parser(
        "evaluate-scorer",
        help="Evaluate the trained scorer against Forge's deck builder",
    )
    eval_parser.set_defaults(func=run_evaluate_scorer)
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Model checkpoint to evaluate (e.g. best_l2_h4_s4_ff1088_mlp256.pt)",
    )
    _add_cards_path(eval_parser)
    _add_embeddings_path(eval_parser)
    add_dataclass_arg(
        eval_parser, EvaluateScorerConfig, "pools",
        help_text="Number of sealed pools to generate for evaluation",
    )
    add_dataclass_arg(
        eval_parser, EvaluateScorerConfig, "best_of",
        help_text="Number of games per match",
    )
    add_dataclass_arg(
        eval_parser, EvaluateScorerConfig, "workers",
        help_text="Number of parallel Java worker processes",
    )
    eval_parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for match files (default: temp dir)",
    )
    eval_parser.add_argument(
        "--set",
        default=None,
        dest="set_code",
        help=(
            "MTG set code to evaluate on (e.g. RVR, MH3, BLB). When omitted,"
            " a random sealed-legal set is selected."
        ),
    )
    eval_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run from --work-dir. Skips pool/deck/matches"
            " generation and re-enters the worker phase against the existing"
            " validation-matches-*.txt files (workers skip already-completed"
            " matches based on outcomes line count). Requires --work-dir."
        ),
    )


def run_evaluate_scorer(args: argparse.Namespace) -> int:
    """Execute the evaluate-scorer command."""
    from sealed.application.evaluate_scorer import EvaluateScorerUseCase

    if args.resume and not args.work_dir:
        print("Error: --resume requires --work-dir", file=sys.stderr)
        return 2
    # The Java worker refuses an even best-of at startup, and the evaluation
    # supervisor restarts a worker that exits: caught here, or never caught.
    if args.best_of < 1 or args.best_of % 2 == 0:
        print(
            f"Error: --best-of must be a positive odd integer, got: {args.best_of}",
            file=sys.stderr,
        )
        return 2

    config = EvaluateScorerConfig(
        checkpoint=Path(args.checkpoint),
        cards_path=Path(args.cards_path),
        embeddings_path=_embeddings_path(args),
        pools=args.pools,
        best_of=args.best_of,
        workers=args.workers,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        set_code=args.set_code,
        resume=args.resume,
    )

    try:
        use_case = EvaluateScorerUseCase()
        use_case.execute(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0
