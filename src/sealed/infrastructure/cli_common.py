"""Flags and path readings several sealed commands share."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_cards_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cards-path", required=True,
        help="Directory with the converted cards and, beside them, their .npz embeddings.",
    )


def _add_embeddings_path(parser: argparse.ArgumentParser, *, resumable: bool = False) -> None:
    """Where a reader finds the ``.npz`` card embeddings ``encode-cards`` wrote."""
    parser.add_argument(
        "--embeddings-path", default=None,
        help=(
            "Directory of the .npz card embeddings, laid out like --cards-path "
            "(encode-cards --embeddings-path); default: beside the cards in "
            "--cards-path." + (" Resumable." if resumable else "")
        ),
    )


def _embeddings_path(args: argparse.Namespace) -> Path | None:
    """``--embeddings-path``, None when unset (the embeddings beside the cards)."""
    return Path(args.embeddings_path) if args.embeddings_path else None


def _missing_paths_error(
    args: argparse.Namespace, resumed_train_config: dict | None, names: tuple[str, ...],
) -> int | None:
    """Exit 2 naming each of ``names`` that was not given and cannot be inherited
    from the resumed run, or None."""
    recorded = resumed_train_config or {}
    missing = [
        "--" + name.replace("_", "-") for name in names
        if getattr(args, name) is None and not recorded.get(name)
    ]
    if not missing:
        return None
    print(
        f"Error: {', '.join(missing)} required (a resumed run inherits them from "
        "its checkpoint).",
        file=sys.stderr,
    )
    return 2
