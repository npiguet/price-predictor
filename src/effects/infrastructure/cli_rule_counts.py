"""``rule-counts``: per-rule record counts of the catalogue's detectors."""

from __future__ import annotations

import argparse

from effects.application.rule_counts_flags import RuleCountsFlags
from effects.domain.shard_generation import UnacceptedGenerationError
from effects.infrastructure.cli_common import (
    _NO_SHARDS,
    _SHARD_REFUSED,
    _add_cards_folder,
    _add_variant_scripts,
    add_config_flags,
    config_from_args,
    exits_with,
)


def _rule_counts_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "rule-counts",
        help=(
            "Count, per rule of the catalogue, the records showing each side of it, "
            "split by random-seat, Forge-AI, aimed and fork records"
        ),
    )
    parser.set_defaults(func=run_rule_counts)
    add_config_flags(parser, RuleCountsFlags, only=("records_dir", "output"))
    _add_cards_folder(parser)
    _add_variant_scripts(
        parser, required=True, help_text="Variant sidecars the detectors read; empty reads none",
    )
    add_config_flags(parser, RuleCountsFlags, only=("rules_dir", "workers"))


@exits_with({_NO_SHARDS: 1, UnacceptedGenerationError: 1, **_SHARD_REFUSED})
def run_rule_counts(args: argparse.Namespace) -> int:
    from effects.application.rule_counts import run as rule_counts

    return rule_counts(config_from_args(RuleCountsFlags, args).config())
