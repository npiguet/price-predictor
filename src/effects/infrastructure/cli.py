"""CLI interface for the effects module (``python -m effects``).

Mirrors ``draft/infrastructure/cli.py``: ``build_parser`` adds one subparser per
command with ``set_defaults(func=…)`` dispatch, and ``main`` parses and calls the
resolved ``func``. Application imports are lazy (inside each ``run_*``) so
``python -m effects --help`` and the unit suite don't pay the torch import cost.

The subcommand table starts empty and is filled in stage by stage; every
subcommand's flags and defaults are contract, pinned by
``specs/023-ability-effect-model/contracts/cli.md``.
"""

from __future__ import annotations

import argparse
import sys

# Each entry is a ``_build_<name>_parser`` function taking the subparsers action.
# Subcommands register here as they land, so the table is the single list of
# what ``python -m effects`` offers.
_SUBCOMMAND_BUILDERS: tuple = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="effects",
        description="Ability effect model: collection, training, encoding, evaluation",
    )
    subparsers = parser.add_subparsers(help="Available commands")
    for build in _SUBCOMMAND_BUILDERS:
        build(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
