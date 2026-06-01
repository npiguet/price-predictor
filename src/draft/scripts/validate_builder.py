"""One-off picker-vs-SA builder-validation diagnostic (FR-042, not a CLI subcommand).

Usage::

    python -m draft.scripts.validate_builder --pools-from output/draft/drafts.jsonl
    python -m draft.scripts.validate_builder --fresh-pools --set BLB --n-pools 300

Prints the picker-vs-SA Spearman (gating), the SA−picker score-gap median/IQR,
and the SA-vs-SA reference ceiling. Run once per picker/scorer checkpoint pair
before a large corpus run to choose ``--build-method picker`` vs ``greedy``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from draft.application.validate_builder import (
    ValidateBuilderConfig,
    format_diagnostic,
    run_validate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="draft.scripts.validate_builder",
        description="Picker-vs-SA builder-validation diagnostic (FR-042).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pools-from",
        help="drafts.jsonl whose per-seat drafted pools are the eval pools.",
    )
    source.add_argument(
        "--fresh-pools", action="store_true",
        help="Generate fresh sealed pools instead (needs Forge).",
    )
    parser.add_argument(
        "--set", dest="set_code", default=None,
        help="Set code for --fresh-pools (random sealed-legal set if omitted).",
    )
    parser.add_argument(
        "--n-pools", type=int, default=300,
        help="Number of pools to evaluate (default: 300).",
    )
    parser.add_argument(
        "--scorer-checkpoint", default="models/sealed/scorer/latest.pt",
        help="Frozen scorer (default: models/sealed/scorer/latest.pt).",
    )
    parser.add_argument(
        "--picker-checkpoint", default="models/sealed/picker/latest.pt",
        help="Picker (default: models/sealed/picker/latest.pt).",
    )
    parser.add_argument(
        "--cards-path", default="output/cardsfolder/",
        help=".npz cache (default: output/cardsfolder/).",
    )
    args = parser.parse_args(argv)

    config = ValidateBuilderConfig(
        pools_from=Path(args.pools_from) if args.pools_from else None,
        fresh_pools=args.fresh_pools,
        set_code=args.set_code,
        n_pools=args.n_pools,
        scorer_checkpoint=Path(args.scorer_checkpoint),
        picker_checkpoint=Path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
    )
    diagnostic = run_validate(config)
    print(format_diagnostic(diagnostic))
    return 0


if __name__ == "__main__":
    sys.exit(main())
