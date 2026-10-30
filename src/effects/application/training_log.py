"""The training run's log lines, kept apart from the loop that computes the
numbers so a line's shape can be tested without a run behind it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
