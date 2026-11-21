"""Flags and checks several draft commands share."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_embeddings_path(parser: argparse.ArgumentParser, *, resumable: bool = False) -> None:
    """Where the ``.npz`` card embeddings ``sealed encode-cards`` wrote are read."""
    parser.add_argument(
        "--embeddings-path", default=None,
        help=(
            "Directory of the .npz card embeddings, laid out like --cards-path "
            "(sealed encode-cards --embeddings-path); default: beside the cards "
            "in --cards-path." + (" Resumable." if resumable else "")
        ),
    )


def _add_checkpoint_dir(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--checkpoint-dir", default=None, help=help_text)


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _picker_checkpoint_missing(args: argparse.Namespace) -> bool:
    """Print the error and return True when --build-method picker lacks --picker-checkpoint."""
    if args.build_method != "picker" or args.picker_checkpoint is not None:
        return False
    print("Error: --picker-checkpoint is required by --build-method picker.", file=sys.stderr)
    return True


_TRAINER_REQUIRED_PATHS: tuple[str, ...] = ("drafts_path", "cards_path", "checkpoint_dir")


def _require_trainer_paths(args: argparse.Namespace, resumed_train_config: dict | None) -> None:
    """Exit 2 naming each of --drafts-path, --cards-path and --checkpoint-dir that was
    not given and cannot be inherited from a resumed run."""
    recorded = resumed_train_config or {}
    missing = [
        "--" + name.replace("_", "-") for name in _TRAINER_REQUIRED_PATHS
        if getattr(args, name) is None and not recorded.get(name)
    ]
    if not missing:
        return
    print(
        f"Error: {', '.join(missing)} required (a resumed run inherits them from its "
        "checkpoint).",
        file=sys.stderr,
    )
    raise SystemExit(2)
