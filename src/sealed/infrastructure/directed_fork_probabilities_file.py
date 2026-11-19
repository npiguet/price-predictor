"""The ``--directed-fork-probabilities`` file of ``sealed match-outcomes``.

A JSON object from rule id to the probability its directed forks are taken
with, in [0, 1], as ``scripts/rules_catalogue.py directed-rules
--probabilities-output`` writes it. A command that counts no shard cannot
compute the probabilities itself, so it reads them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path


def add_directed_fork_probabilities_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--directed-fork-probabilities", type=Path, default=None, metavar="FILE",
        help=(
            "JSON object from rule id to its own directed-fork probability in "
            "[0, 1] (`scripts/rules_catalogue.py directed-rules "
            "--probabilities-output`); a listed rule without one takes "
            "--directed-fork-probability, and one at 0 is never forked. Every "
            "rule named must be in --directed-fork-rules (default: none)"
        ),
    )


def read_directed_fork_probabilities(
    path: Path, rules: Sequence[str],
) -> dict[str, float]:
    """The file's rule probabilities.

    Raises ``ValueError`` naming the file, and the rule where one is at fault,
    for a file that is missing or not a JSON object, a value that is no number
    in [0, 1], or a rule ``rules`` (``--directed-fork-rules``) does not list.
    """
    document = _read_object(Path(path))
    probabilities: dict[str, float] = {}
    for rule, value in document.items():
        if rule not in rules:
            raise ValueError(
                f"--directed-fork-probabilities {path}: rule {rule} is not in "
                "--directed-fork-rules"
            )
        probabilities[rule] = _probability(path, rule, value)
    return probabilities


def _read_object(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"--directed-fork-probabilities {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(
            f"--directed-fork-probabilities {path}: expected a JSON object from rule "
            f"to probability, got {type(document).__name__}"
        )
    return document


def _probability(path: Path, rule: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"--directed-fork-probabilities {path}: rule {rule}: expected a number, "
            f"got {value!r}"
        )
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"--directed-fork-probabilities {path}: rule {rule}: the probability "
            f"must be in [0, 1], got {value}"
        )
    return float(value)
