"""The ``build-vocab`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_build_vocab_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "build-vocab",
        help=(
            "Build the sealed-side tokenizer vocabulary from the converted corpus"
        ),
    )
    parser.set_defaults(func=run_build_vocab)
    parser.add_argument(
        "--cards-folder", required=True,
        help="Directory of converted .txt files",
    )
    parser.add_argument(
        "--vocab-path", required=True,
        help="Output vocab file",
    )
    parser.add_argument(
        "--target-size", type=int, default=5000,
        help="Approximate vocab size (default: 5000). Seeded tokens preserved.",
    )


def run_build_vocab(args: argparse.Namespace) -> int:
    """Execute the build-vocab command."""
    from sealed.application.build_vocab import (
        BuildVocabConfig,
        EmptyCardsFolderError,
    )
    from sealed.application.build_vocab import (
        run as run_build_vocab_pipeline,
    )

    config = BuildVocabConfig(
        cards_folder=Path(args.cards_folder),
        vocab_path=Path(args.vocab_path),
        target_size=args.target_size,
    )
    try:
        run_build_vocab_pipeline(config)
    except EmptyCardsFolderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
