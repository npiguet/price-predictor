"""``extract-keyword-definitions``: Forge's keyword table as JSON."""

from __future__ import annotations

import argparse

from effects.application.extract_keyword_definitions import ExtractKeywordDefinitionsConfig
from effects.infrastructure.cli_common import (
    add_config_flags,
    config_from_args,
    exits_with,
)


def _extract_keyword_definitions_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "extract-keyword-definitions",
        help="Write Forge's keyword table (reminder-text templates) as JSON",
    )
    parser.set_defaults(func=run_extract_keyword_definitions)
    add_config_flags(parser, ExtractKeywordDefinitionsConfig)


@exits_with({"effects.application.extract_keyword_definitions.KeywordExtractionError": 1})
def run_extract_keyword_definitions(args: argparse.Namespace) -> int:
    from effects.application.extract_keyword_definitions import run as extract

    return extract(config_from_args(ExtractKeywordDefinitionsConfig, args))
