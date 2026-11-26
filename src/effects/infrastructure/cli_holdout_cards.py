"""``holdout-cards``: the depletion list."""

from __future__ import annotations

import argparse
from pathlib import Path

from effects.application.holdout_cards import HoldoutCardsConfig, HoldoutCardsError
from effects.infrastructure.cli_common import (
    _add_cards_folder,
    _add_holdout_flags,
    add_config_flags,
    config_from_args,
    exits_with,
    holdout_rule,
)


def _holdout_cards_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "holdout-cards",
        help=(
            "Write the depletion list: every card carrying a held-out ability "
            "text, for 'sealed generate-pools --exclude-cards'"
        ),
    )
    parser.set_defaults(func=run_holdout_cards)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Destination file, one Forge canonical card name per line",
    )
    _add_cards_folder(parser)
    add_config_flags(parser, HoldoutCardsConfig, only=("printings_path",))
    _add_holdout_flags(parser)


@exits_with({HoldoutCardsError: 1})
def run_holdout_cards(args: argparse.Namespace) -> int:
    """Write the list, and report the share of the corpus it removes: what a
    depleted pool loses, and the knob the two holdout flags turn."""
    from effects.application.holdout_cards import run as holdout_cards

    config = config_from_args(HoldoutCardsConfig, args, holdout=holdout_rule(args))
    report = holdout_cards(config)
    print(f"held-out templates: {report.keys}")
    print(f"held-out texts: {report.texts}")
    print(
        f"held-out cards: {len(report.names)} of {report.converted_cards} converted "
        f"({100.0 * report.depleted_share:.1f}% depleted) -> {config.out}"
    )
    if not report.names:
        print(
            "Nothing to deplete. Raise --holdout-permille, or check that the "
            "converted tree has sidecars with script text."
        )
        return 1
    return 0
