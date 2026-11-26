"""``field-coverage``: the record fields a corpus never varied."""

from __future__ import annotations

import argparse

from effects.application.field_coverage import FieldCoverageConfig
from effects.domain.shard_generation import UnacceptedGenerationError
from effects.infrastructure.cli_common import (
    _NO_SHARDS,
    _SHARD_REFUSED,
    add_config_flags,
    config_from_args,
    exits_with,
)


def _field_coverage_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "field-coverage",
        help=(
            "Report which record fields a collected corpus has ever actually "
            "carried, and which are always the same value"
        ),
    )
    parser.set_defaults(func=run_field_coverage)
    add_config_flags(parser, FieldCoverageConfig)


@exits_with({_NO_SHARDS: 1, UnacceptedGenerationError: 1, **_SHARD_REFUSED})
def run_field_coverage(args: argparse.Namespace) -> int:
    from effects.application.field_coverage import field_coverage, report_lines
    from effects.infrastructure.shard_io import (
        DecodeTally,
        read_shard,
        require_generation,
        require_shards,
    )

    config = config_from_args(FieldCoverageConfig, args)
    root = config.effect_records
    shards = require_shards(root)
    require_generation(shard.path(root) for shard in shards)
    tally = DecodeTally()
    coverage = field_coverage(
        record for shard in shards for record in read_shard(shard.path(root), tally=tally)
    )
    if not coverage:
        print(f"no records under {root}")
        return 1
    print(f"{len(coverage)} fields over {root}\n")
    for line in report_lines(coverage, all_fields=config.all):
        print(line)
    for line in tally.report_lines():
        print(line)
    return 0
