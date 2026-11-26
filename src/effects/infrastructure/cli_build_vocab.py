"""``build-vocab``: the effects tokenizer vocabulary."""

from __future__ import annotations

import argparse
from pathlib import Path

from effects.application.build_vocab import DEFAULT_TARGET_SIZE, EmptyCardsFolderError
from effects.domain.ability_tokenizer import Surface
from effects.infrastructure.cli_common import (
    _add_cards_folder,
    config_from_args,
    exits_with,
)


def _build_vocab_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "build-vocab",
        help="Build the effects tokenizer vocabulary from the converted corpus",
    )
    parser.set_defaults(func=run_build_vocab)
    parser.add_argument(
        "--surface", type=Surface, choices=list(Surface), default=Surface.PROSE,
        help=(
            "prose reads the converted text; script adds the sidecars' script "
            "lines (default: %(default)s)"
        ),
    )
    _add_cards_folder(parser)
    parser.add_argument(
        "--vocab-path", type=Path, required=True,
        help="Output path, inside the vocabulary's inputs run",
    )
    parser.add_argument(
        "--keyword-definitions", type=Path, required=True,
        help="Keyword definitions to scan",
    )
    parser.add_argument(
        "--target-size", type=int, default=DEFAULT_TARGET_SIZE,
        help="Post-truncate the corpus-frequency vocabulary (default: %(default)s)",
    )


# A --keyword-definitions that names no file is refused, never read as none.
@exits_with({EmptyCardsFolderError: 1, ValueError: 1, FileNotFoundError: 1})
def run_build_vocab(args: argparse.Namespace) -> int:
    from effects.application.build_vocab import BuildVocabConfig
    from effects.application.build_vocab import run as build_vocab

    build_vocab(config_from_args(BuildVocabConfig, args))
    return 0
