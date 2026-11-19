"""The ``--directed-fork-probabilities`` file of ``sealed match-outcomes``.

``{"count_per_side": N, "rules": {rule: {side: records}}}``, as
``scripts/rules_catalogue.py directed-rules --probabilities-output`` writes it:
the records each side needs, and for each listed rule the one side its
registered variants make (``fires`` or ``does_not_fire``) and the records the
counts give it there. A command that counts no shard cannot compute the
probabilities itself: it computes them from these at start and again at
every log scan, adding the forks its workers kept
(:class:`sealed.domain.directed_fork_needs.DirectedForkNeeds`).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sealed.domain.directed_fork_needs import DirectedForkNeeds, DirectedRuleNeed

#: The sides a rule's registered variants can make.
SIDES = ("fires", "does_not_fire")

_FORMAT = '{"count_per_side": N, "rules": {rule: {side: records}}}'


def add_directed_fork_probabilities_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--directed-fork-probabilities", type=Path, default=None, metavar="FILE",
        help=(
            "JSON file of the records each listed rule's registered side has and "
            "the count a side needs (`scripts/rules_catalogue.py directed-rules "
            "--probabilities-output`): each rule's directed-fork probability, "
            "recomputed at every log scan from the forks the workers kept and "
            "given to every worker started after it. A listed rule without an "
            "entry takes --directed-fork-probability. Every rule named must be "
            "in --directed-fork-rules (default: none)"
        ),
    )


def read_directed_fork_needs(path: Path, rules: Sequence[str]) -> DirectedForkNeeds:
    """The file's count per side and each rule's need.

    Raises ``ValueError`` naming the file, and the rule where one is at fault,
    for a file that is missing, not in this format (the older flat map from
    rule to probability among them), a rule ``rules``
    (``--directed-fork-rules``) does not list, or a rule whose registered
    variants span two sides: the workers' tally counts a rule's kept forks,
    not each side's.
    """
    document = _read_object(Path(path))
    return DirectedForkNeeds(
        count_per_side=_count_per_side(path, document["count_per_side"]),
        rules={rule: _need(path, rule, entry, rules)
               for rule, entry in _rules(path, document["rules"]).items()},
    )


def _read_object(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"--directed-fork-probabilities {path}: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"count_per_side", "rules"}:
        raise ValueError(
            f"--directed-fork-probabilities {path}: expected {_FORMAT}; a file in the "
            "old format (a map from rule to probability) has no counts to recompute "
            "from: write it again with scripts/rules_catalogue.py directed-rules "
            "--probabilities-output"
        )
    return document


def _count_per_side(path: Path, value: object) -> int:
    if not _is_count(value) or value < 1:
        raise ValueError(
            f"--directed-fork-probabilities {path}: count_per_side must be a whole "
            f"number of at least 1, got {value!r}"
        )
    return value


def _rules(path: Path, value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError(
            f"--directed-fork-probabilities {path}: rules must be a JSON object, "
            f"got {type(value).__name__}"
        )
    return value


def _need(path: Path, rule: str, entry: object, rules: Sequence[str]) -> DirectedRuleNeed:
    if rule not in rules:
        raise ValueError(
            f"--directed-fork-probabilities {path}: rule {rule} is not in "
            "--directed-fork-rules"
        )
    side, records = _one_side(path, rule, entry)
    if side not in SIDES or not _is_record_count(records):
        raise ValueError(
            f"--directed-fork-probabilities {path}: rule {rule}: expected one of "
            f"{', '.join(SIDES)} with a record count of at least 0, got {side}: {records!r}"
        )
    return DirectedRuleNeed(side, records)


def _one_side(path: Path, rule: str, entry: object) -> tuple[str, object]:
    """The entry's one ``(side, records)``."""
    if not isinstance(entry, dict) or not entry:
        raise ValueError(
            f"--directed-fork-probabilities {path}: rule {rule}: expected "
            f"{{side: records}}, got {entry!r}"
        )
    if len(entry) > 1:
        raise ValueError(
            f"--directed-fork-probabilities {path}: rule {rule}: its registered "
            f"variants span two sides ({', '.join(entry)}); the workers' tally counts "
            "a rule's kept forks, not each side's, so its probability cannot be "
            "recomputed"
        )
    (side, records), = entry.items()
    return side, records


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_record_count(value: object) -> bool:
    return _is_count(value) and value >= 0
