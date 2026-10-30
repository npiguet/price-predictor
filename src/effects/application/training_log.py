"""The training run's log lines, kept apart from the loop that computes the
numbers so a line's shape can be tested without a run behind it."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from effects.application.training_schedule import MAX_GRAD_NORM
from effects.domain.loss_floor import FLOOR_EPSILON


def format_norms(norms: Mapping[str, float]) -> str:
    """The pre-clip gradient norms, or nothing before the first backward."""
    if not norms:
        return ""
    body = ", ".join(f"{name} {value:.2f}" for name, value in norms.items())
    return f"\n  |g| {body} (clipped per group at {MAX_GRAD_NORM:g})"


def explained(name: str, value: float, floor: Mapping[str, float] | None) -> str:
    """``1 - loss/floor`` against the best constant prediction for the field
    (FR-127c): zero is the base rate, negative is worse than predicting it. A
    field whose targets never vary reads ``n/a``."""
    if floor is None:
        return ""
    reference = floor.get(name)
    if reference is None or reference < FLOOR_EPSILON:
        return " (n/a)"
    return f" ({round(100.0 * (1.0 - value / reference))}%)"


def format_parts(
    parts: Mapping[str, float], floor: Mapping[str, float] | None = None,
) -> str:
    """Each field's loss term, largest first so a blow-up stands out, and what
    it explains. ``floor`` is passed only for the card-disjoint breakdown: a
    shard's training line has no floor, its records changing every shard."""
    if not parts:
        return ""
    ranked = sorted(parts.items(), key=lambda kv: -kv[1])
    body = ", ".join(
        f"{name} {value:.3f}{explained(name, value, floor)}"
        for name, value in ranked
    )
    return f"\n  fields: {body}"


def format_floor(floor: Mapping[str, float]) -> str:
    """The floor itself, ordered like the breakdown it scales."""
    ranked = sorted(floor.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{name} {value:.3f}" for name, value in ranked)


def format_shares(counts: Mapping[str, int], order: Sequence[str] = ()) -> str:
    """``label share%`` per label, in ``order`` first and then largest first."""
    total = sum(counts.values())
    if not total:
        return "none"
    ordered = [label for label in order if label in counts] + sorted(
        (label for label in counts if label not in order),
        key=lambda label: -counts[label],
    )
    return ", ".join(
        f"{label} {100.0 * counts[label] / total:.1f}%" for label in ordered
    )


def format_terms(terms: Mapping[str, float]) -> str:
    """The head terms of the loss, or nothing when there were none."""
    if not terms:
        return ""
    body = ", ".join(f"{name} {value:.3f}" for name, value in terms.items())
    return f"\n  heads: {body}"


_EPOCH_LINE = re.compile(
    r"epoch (\d+) \| train ([\d.]+) \| card-disjoint ([\d.]+) \| "
    r"game-disjoint ([\d.]+) \| gate F1 ([\d.]+) \| zone acc ([\d.]+) \| "
    r"deviance ([\d.]+)"
)
_FIELD_ITEM = re.compile(r"(\w+) ([\d.]+) \((-?\d+%|n/a)\)")


@dataclass(frozen=True)
class EpochSummary:
    """One epoch as the run's log reports it, read back by the analysis scripts.

    ``fields`` maps each output field to its card-disjoint loss and the share
    of the constant prediction's loss it removes, in percent (None: ``n/a``).
    """

    epoch: int
    train: float
    card_disjoint: float
    game_disjoint: float
    gate_f1: float
    zone_accuracy: float
    deviance: float
    fields: dict[str, tuple[float, float | None]]


def parse_epoch_summaries(lines: Iterable[str]) -> list[EpochSummary]:
    """The epoch summaries of a training log, in epoch order of first appearance.

    A resumed run logs the epochs after its last checkpoint again; the later
    line is the one the resumed weights produced, so it replaces the earlier.
    """
    lines = list(lines)
    found: dict[int, EpochSummary] = {}
    for i, line in enumerate(lines):
        m = _EPOCH_LINE.search(line)
        if not m:
            continue
        fields: dict[str, tuple[float, float | None]] = {}
        following = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if following.startswith("fields:"):
            for name, loss, pct in _FIELD_ITEM.findall(following):
                fields[name] = (float(loss), None if pct == "n/a" else float(pct[:-1]))
        epoch = int(m[1])
        found[epoch] = EpochSummary(epoch, *(float(v) for v in m.groups()[1:]), fields)
    return list(found.values())
