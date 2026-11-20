"""The ``generate-pools`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_generate_pools_parser(subparsers) -> None:
    generate_parser = subparsers.add_parser(
        "generate-pools",
        help="Generate sealed pools using Forge's booster generation logic",
    )
    generate_parser.set_defaults(func=run_generate_pools)
    generate_parser.add_argument(
        "--set",
        default=None,
        dest="set_code",
        help=(
            "MTG set code to generate boosters from (e.g. RVR, MH3, BLB). "
            "When omitted, each pool uses a randomly selected sealed-legal set."
        ),
    )
    generate_parser.add_argument(
        "--exclude-cards",
        default=None,
        help=(
            "File of card names, one per line, that no booster may contain. "
            "Written by 'python -m effects holdout-cards'; a plain file rather "
            "than a computed rule, so sealed needs no import of effects. Pools "
            "built with it are the depleted training corpus for the effect "
            "model; without it, the full-strength validation corpus."
        ),
    )
    generate_parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Number of sealed pools to generate",
    )
    generate_parser.add_argument(
        "--pools-path",
        required=True,
        help=(
            "Output directory; pools.txt is written here. "
            "'{set}' in it is replaced by --set when that is given."
        ),
    )


def run_generate_pools(args: argparse.Namespace) -> int:
    """Execute the generate-pools command."""
    from sealed.application.generate_pools import GeneratePoolsUseCase
    from sealed.infrastructure.pool_connector import PoolConnector

    set_code = args.set_code
    pool_count = args.size

    pools_path = _pools_dir(args)

    connector = PoolConnector()
    use_case = GeneratePoolsUseCase()

    try:
        use_case.execute(
            set_code, pool_count, pools_path, connector,
            exclude_cards=(
                Path(args.exclude_cards) if args.exclude_cards else None
            ),
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


def _pools_dir(args: argparse.Namespace) -> Path:
    """``generate-pools``' output directory: ``--pools-path``, ``{set}`` filled in."""
    return Path(args.pools_path.replace("{set}", args.set_code) if args.set_code
                else args.pools_path)
